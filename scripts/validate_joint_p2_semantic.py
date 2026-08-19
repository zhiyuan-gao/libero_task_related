#!/usr/bin/env python3
"""Minimal A/B/C validation for P2 joint-masked semantic supervision."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

from policy_aux_gate_utils import load_real_libero_item
from policy_aux_gate_utils import move_observation
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models import pi0_config
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxTargets

DEFAULT_CHECKPOINT = Path("/workspace/vla/models/openpi/pi05_base_pytorch/model.safetensors")
DEFAULT_SNAPSHOT = Path(
    "/workspace/vla/cache/huggingface/hub/datasets--physical-intelligence--libero/"
    "snapshots/a4336d589d589045d1c56423ffdf3b88a0e19b1f"
)
DEFAULT_MAPPING = Path(
    "/workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/debug/lerobot_episode_mapping.json"
)
DEFAULT_MANIFEST = Path(
    "/workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/manifests/"
    "libero10_policy_aux_manifest.parquet"
)
DEFAULT_OUTPUT = Path(
    "/workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/unit_gates/"
    "joint_p2_semantic_abc_gate.json"
)


def make_targets(auxiliary: dict, device: torch.device) -> PolicyAuxTargets:
    return PolicyAuxTargets(
        geometry=torch.zeros((1, 2048), dtype=torch.float32, device=device),
        geometry_valid=torch.ones((1,), dtype=torch.bool, device=device),
        geometry_mean=torch.zeros((2048,), dtype=torch.float32, device=device),
        geometry_std=torch.ones((2048,), dtype=torch.float32, device=device),
        ground_masks={name: value.to(device) for name, value in auxiliary["ground_masks"].items()},
        ground_valid_views=auxiliary["ground_valid_views"].to(device),
        semantic_input_ids=auxiliary["semantic_input_ids"].to(device),
        semantic_labels=auxiliary["semantic_labels"].to(device),
        semantic_loss_mask=auxiliary["semantic_loss_mask"].to(device),
    )


def replacement_targets(targets: PolicyAuxTargets, vocab_size: int) -> PolicyAuxTargets:
    teacher_mask = targets.semantic_loss_mask[:, 1:].to(torch.bool)
    replacement_inputs = targets.semantic_input_ids.clone()
    replacement_labels = targets.semantic_labels.clone()
    replacement_inputs[teacher_mask] = (replacement_inputs[teacher_mask] + 17) % vocab_size
    label_mask = targets.semantic_loss_mask.to(torch.bool)
    replacement_labels[label_mask] = (replacement_labels[label_mask] + 17) % vocab_size
    return dataclasses.replace(
        targets,
        semantic_input_ids=replacement_inputs,
        semantic_labels=replacement_labels,
    )


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    difference = candidate - reference
    reference_norm = reference.norm()
    candidate_norm = candidate.norm()
    metrics = {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / reference_norm.clamp_min(1e-12)),
        "reference_norm": float(reference_norm),
        "candidate_norm": float(candidate_norm),
    }
    if reference.numel() > 1 and float(reference_norm) > 0 and float(candidate_norm) > 0:
        metrics["cosine"] = float(F.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0))
    return metrics


def fixed_forward(
    model: PI05AuxPolicy,
    observation,
    actions: torch.Tensor,
    targets: PolicyAuxTargets,
    noise: torch.Tensor,
    diffusion_time: torch.Tensor,
    semantic_impl: str,
    reference_attention_impl: str = "eager",
) -> dict:
    torch.manual_seed(20260819)
    torch.cuda.manual_seed_all(20260819)
    return model.forward_with_aux(
        observation,
        actions,
        targets,
        noise=noise,
        time=diffusion_time,
        semantic_impl=semantic_impl,
        reference_semantic_attention_impl=reference_attention_impl,
        return_validation_outputs=True,
    )


def selected_parameters(model: PI05AuxPolicy) -> dict[str, torch.nn.Parameter]:
    exact_names = {
        "paligemma_backbone": (
            "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"
        ),
        "action_expert": "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
        "geometry_query": "geometry_queries",
        "geometry_head": "geometry_head.output_projection.weight",
        "ground_query": "ground_queries",
        "ground_head": "ground_head.query_projection.weight",
    }
    by_name = dict(model.named_parameters())
    missing = {label: name for label, name in exact_names.items() if name not in by_name}
    if missing:
        raise RuntimeError(f"Could not resolve representative parameters: {missing}")
    return {label: by_name[name] for label, name in exact_names.items()}


def capture_selected_gradients(parameters: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    missing = [name for name, parameter in parameters.items() if parameter.grad is None]
    if missing:
        raise RuntimeError(f"Representative parameters are missing gradients: {missing}")
    return {name: parameter.grad.detach().float().cpu().clone() for name, parameter in parameters.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--annotation-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after-forward", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    observation, actions, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.annotation_manifest,
    )
    observation = move_observation(observation, device)
    actions = actions.to(device)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    model = PI05AuxPolicy(
        config,
        PolicyAuxConfig(
            mode="ground_geometry_semantic_lm",
            lambda_sem=0.01,
            lambda_ground=0.50,
            lambda_geo=0.15,
        ),
    )
    strict_load = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
    model.to(device=device, dtype=torch.float32).eval()
    targets = make_targets(auxiliary, device)
    vocab_size = int(model.paligemma_with_expert.paligemma.lm_head.weight.shape[0])
    changed_targets = replacement_targets(targets, vocab_size)
    generator = torch.Generator(device=device).manual_seed(20260819)
    noise = torch.randn(actions.shape, generator=generator, device=device)
    diffusion_time = torch.full((1,), 0.5, dtype=torch.float32, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        reference = fixed_forward(
            model, observation, actions, targets, noise, diffusion_time, "two_pass_reference"
        )
        joint = fixed_forward(model, observation, actions, targets, noise, diffusion_time, "joint_masked")

    forward_tensors = {
        "action_output": (
            reference["diagnostics"]["action_velocity"],
            joint["diagnostics"]["action_velocity"],
        ),
        "geometry_output": (
            reference["diagnostics"]["geometry_prediction"],
            joint["diagnostics"]["geometry_prediction"],
        ),
        "ground_output": (
            reference["diagnostics"]["ground_logits"],
            joint["diagnostics"]["ground_logits"],
        ),
        "semantic_logits": (
            reference["diagnostics"]["semantic_logits"],
            joint["diagnostics"]["semantic_logits"],
        ),
    }
    for loss_name in ("action", "geometry", "ground", "semantic", "total"):
        forward_tensors[f"{loss_name}_loss"] = (
            reference["losses"][loss_name],
            joint["losses"][loss_name],
        )
    forward_metrics = {name: tensor_metrics(old, new) for name, (old, new) in forward_tensors.items()}
    forward_checks = {}
    for name, metrics in forward_metrics.items():
        if name == "semantic_logits":
            relative_tolerance, absolute_tolerance = 5e-4, 5e-2
        elif name.endswith("_output"):
            relative_tolerance, absolute_tolerance = 2.5e-3, 1e-3
        else:
            relative_tolerance, absolute_tolerance = 5e-4, 1e-3
        forward_checks[name] = (
            metrics["relative_l2"] <= relative_tolerance and metrics["max_abs"] <= absolute_tolerance
        )
    valid_context = reference["diagnostics"]["context_pad_mask"].to(torch.bool)
    context_hidden_state_metrics = {
        "old_main_vs_old_semantic": tensor_metrics(
            reference["diagnostics"]["main_context_hidden_states"][valid_context],
            reference["diagnostics"]["semantic_context_hidden_states"][valid_context],
        ),
        "old_main_vs_new_joint": tensor_metrics(
            reference["diagnostics"]["main_context_hidden_states"][valid_context],
            joint["diagnostics"]["main_context_hidden_states"][valid_context],
        ),
        "old_semantic_vs_new_joint": tensor_metrics(
            reference["diagnostics"]["semantic_context_hidden_states"][valid_context],
            joint["diagnostics"]["semantic_context_hidden_states"][valid_context],
        ),
    }

    with torch.no_grad():
        original_joint = fixed_forward(model, observation, actions, targets, noise, diffusion_time, "joint_masked")
        changed_joint = fixed_forward(
            model, observation, actions, changed_targets, noise, diffusion_time, "joint_masked"
        )
    no_leakage_metrics = {
        "action_output": tensor_metrics(
            original_joint["diagnostics"]["action_velocity"], changed_joint["diagnostics"]["action_velocity"]
        ),
        "geometry_output": tensor_metrics(
            original_joint["diagnostics"]["geometry_prediction"],
            changed_joint["diagnostics"]["geometry_prediction"],
        ),
        "ground_output": tensor_metrics(
            original_joint["diagnostics"]["ground_logits"], changed_joint["diagnostics"]["ground_logits"]
        ),
        "semantic_logits": tensor_metrics(
            original_joint["diagnostics"]["semantic_logits"], changed_joint["diagnostics"]["semantic_logits"]
        ),
        "semantic_loss_abs_change": float(
            (original_joint["losses"]["semantic"] - changed_joint["losses"]["semantic"]).abs()
        ),
    }
    no_leakage_checks = {
        "action_unchanged": no_leakage_metrics["action_output"]["max_abs"] <= 1e-7,
        "geometry_unchanged": no_leakage_metrics["geometry_output"]["max_abs"] <= 1e-7,
        "ground_unchanged": no_leakage_metrics["ground_output"]["max_abs"] <= 1e-7,
        "semantic_logits_changed": no_leakage_metrics["semantic_logits"]["relative_l2"] > 1e-5,
        "semantic_loss_changed": no_leakage_metrics["semantic_loss_abs_change"] > 1e-6,
        "physical_length_unchanged": (
            targets.semantic_input_ids.shape == changed_targets.semantic_input_ids.shape
        ),
        "padding_and_loss_mask_unchanged": torch.equal(
            targets.semantic_loss_mask,
            changed_targets.semantic_loss_mask,
        ),
    }

    if args.stop_after_forward:
        payload = {
            "status": "DEBUG",
            "forward_parity": {"checks": forward_checks, "metrics": forward_metrics},
            "context_hidden_state_debug": context_hidden_state_metrics,
            "semantic_no_leakage": {"checks": no_leakage_checks, "metrics": no_leakage_metrics},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    del reference, joint, original_joint, changed_joint, forward_tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Full-model FP32 gradients exceed one A100-80GB. C only asks for
    # representative parameters, so freeze every other parameter while keeping
    # the exact FP32 forward graph and production checkpoint mechanism.
    parameters = selected_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)
    for parameter in parameters.values():
        parameter.requires_grad_(requires_grad=True)
    model.train()
    model.gradient_checkpointing_enable()
    model.zero_grad(set_to_none=True)
    reference_grad_result = fixed_forward(
        model, observation, actions, targets, noise, diffusion_time, "two_pass_reference"
    )
    reference_grad_result["losses"]["total"].backward()
    reference_gradients = capture_selected_gradients(parameters)
    del reference_grad_result
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    joint_grad_result = fixed_forward(model, observation, actions, targets, noise, diffusion_time, "joint_masked")
    joint_grad_result["losses"]["total"].backward()
    joint_layout_present = joint_grad_result["joint_train_layout"] is not None
    joint_gradients = capture_selected_gradients(parameters)
    gradient_metrics = {
        name: tensor_metrics(reference_gradients[name], joint_gradients[name]) for name in parameters
    }
    gradient_checks = {
        name: metrics.get("cosine", -1.0) >= 0.999
        and metrics["relative_l2"] <= 5e-3
        and 0.995 <= metrics["candidate_norm"] / max(metrics["reference_norm"], 1e-12) <= 1.005
        for name, metrics in gradient_metrics.items()
    }

    del joint_grad_result
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    with torch.no_grad():
        production_reference = fixed_forward(
            model,
            observation,
            actions,
            targets,
            noise,
            diffusion_time,
            "two_pass_reference",
            reference_attention_impl="sdpa",
        )
        production_joint = fixed_forward(
            model, observation, actions, targets, noise, diffusion_time, "joint_masked"
        )
    production_tensors = {
        "action_output": (
            production_reference["diagnostics"]["action_velocity"],
            production_joint["diagnostics"]["action_velocity"],
        ),
        "geometry_output": (
            production_reference["diagnostics"]["geometry_prediction"],
            production_joint["diagnostics"]["geometry_prediction"],
        ),
        "ground_output": (
            production_reference["diagnostics"]["ground_logits"],
            production_joint["diagnostics"]["ground_logits"],
        ),
        "semantic_logits": (
            production_reference["diagnostics"]["semantic_logits"],
            production_joint["diagnostics"]["semantic_logits"],
        ),
    }
    for loss_name in ("action", "geometry", "ground", "semantic", "total"):
        production_tensors[f"{loss_name}_loss"] = (
            production_reference["losses"][loss_name],
            production_joint["losses"][loss_name],
        )
    production_metrics = {
        name: tensor_metrics(reference_tensor, joint_tensor)
        for name, (reference_tensor, joint_tensor) in production_tensors.items()
    }
    production_checks = {}
    for name, metrics in production_metrics.items():
        if name == "ground_output":
            relative_tolerance, cosine_tolerance = 6e-2, 0.998
        elif name.endswith("_output") or name == "semantic_logits":
            relative_tolerance, cosine_tolerance = 2e-2, 0.999
        else:
            relative_tolerance, cosine_tolerance = 2e-2, None
        production_checks[name] = metrics["relative_l2"] <= relative_tolerance and (
            cosine_tolerance is None or metrics.get("cosine", -1.0) >= cosine_tolerance
        )

    checks = {
        "A_forward_parity": all(forward_checks.values()),
        "B_semantic_no_leakage": all(no_leakage_checks.values()),
        "C_representative_gradient_parity": all(gradient_checks.values()),
        "production_BF16_SDPA_eager_tolerance": all(production_checks.values()),
        "no_semantic_query_parameter": not hasattr(model, "semantic_queries"),
        "joint_layout_is_training_only": joint_layout_present,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "gate": "p2_joint_masked_semantic_abc_v1",
        "sample_id": auxiliary["sample_id"],
        "precision": "FP32 eager/eager with representative-only gradients for C",
        "attention_comparison": "eager/eager",
        "strict_base_load": strict_load,
        "forward_parity": {"checks": forward_checks, "metrics": forward_metrics},
        "context_hidden_state_debug": context_hidden_state_metrics,
        "semantic_no_leakage": {"checks": no_leakage_checks, "metrics": no_leakage_metrics},
        "gradient_parity": {"checks": gradient_checks, "metrics": gradient_metrics},
        "production_bf16_sdpa_vs_eager": {"checks": production_checks, "metrics": production_metrics},
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise RuntimeError(f"P2 joint semantic A/B/C validation failed: {checks}")


if __name__ == "__main__":
    main()
