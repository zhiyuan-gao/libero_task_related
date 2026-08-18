#!/usr/bin/env python3
"""Validate P2 native semantic LM supervision and structural action-path isolation."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

from policy_aux_gate_utils import load_real_libero_item
from policy_aux_gate_utils import move_observation
import torch

from openpi.models import pi0_config
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxTargets
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.training.policy_aux_dataset import PolicySemanticTokenizer


def semantic_targets(auxiliary: dict, device: torch.device) -> PolicyAuxTargets:
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


def replacement_semantic_targets(
    base: PolicyAuxTargets,
    replacement_text: str,
    device: torch.device,
) -> PolicyAuxTargets:
    replacement = PolicySemanticTokenizer().batch([replacement_text])
    return dataclasses.replace(
        base,
        semantic_input_ids=torch.from_numpy(replacement.input_ids).to(device),
        semantic_labels=torch.from_numpy(replacement.labels).to(device),
        semantic_loss_mask=torch.from_numpy(replacement.loss_mask).to(device),
    )


def run_full_forward(
    model: PI05AuxPolicy,
    observation,
    actions: torch.Tensor,
    targets: PolicyAuxTargets,
    noise: torch.Tensor,
    diffusion_time: torch.Tensor,
) -> dict:
    torch.manual_seed(20260818)
    torch.cuda.manual_seed_all(20260818)
    with torch.no_grad():
        return model.forward_with_aux(
            observation,
            actions,
            targets,
            noise=noise,
            time=diffusion_time,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--semantic-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)

    observation, actions, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.annotation_manifest,
    )
    inventory = json.loads(args.semantic_inventory.read_text())
    replacement_text = next(
        row["canonical_original_string"]
        for row in inventory["targets"]
        if row["canonical_original_string"] != auxiliary["semantic_text"]
    )
    observation = move_observation(observation, device)
    actions = actions.to(device)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    aux_config = PolicyAuxConfig(
        mode="ground_geometry_semantic_lm",
        lambda_sem=1.0,
        lambda_ground=1.0,
        lambda_geo=1.0,
    )
    model = PI05AuxPolicy(config, aux_config)
    strict_load = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
    model.to(device).eval()
    targets = semantic_targets(auxiliary, device)
    replacement_targets = replacement_semantic_targets(targets, replacement_text, device)
    changed_ground_targets = dataclasses.replace(
        targets,
        ground_masks={name: torch.zeros_like(mask) for name, mask in targets.ground_masks.items()},
    )
    changed_geometry_targets = dataclasses.replace(
        targets,
        geometry=torch.full_like(targets.geometry, 17.0),
    )
    generator = torch.Generator(device=device).manual_seed(20260818)
    noise = torch.randn(actions.shape, generator=generator, device=device)
    diffusion_time = torch.full((1,), 0.5, dtype=torch.float32, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    first = run_full_forward(model, observation, actions, targets, noise, diffusion_time)
    second = run_full_forward(model, observation, actions, targets, noise, diffusion_time)
    changed_target = run_full_forward(model, observation, actions, replacement_targets, noise, diffusion_time)
    changed_ground = run_full_forward(model, observation, actions, changed_ground_targets, noise, diffusion_time)
    changed_geometry = run_full_forward(model, observation, actions, changed_geometry_targets, noise, diffusion_time)
    first_action = first["action_loss_per_element"]
    second_action = second["action_loss_per_element"]
    changed_action = changed_target["action_loss_per_element"]
    first_semantic = first["losses"]["semantic"]
    second_semantic = second["losses"]["semantic"]
    changed_semantic = changed_target["losses"]["semantic"]
    layout = first["layout"]

    # Run the semantic pass alone so its gradient provenance cannot include the
    # action expert or either auxiliary query group.
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)
    selected_parameters = {}
    selectors = (
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "multi_modal_projector.linear.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    )
    for name, parameter in model.named_parameters():
        if "paligemma_with_expert.paligemma." in name and any(selector in name for selector in selectors):
            parameter.requires_grad_(requires_grad=True)
            selected_parameters[name] = parameter
    if not any("vision_tower" in name for name in selected_parameters):
        raise RuntimeError("Could not resolve a shared VLM vision parameter for the gradient gate")
    if not any("language_model.layers.0" in name for name in selected_parameters):
        raise RuntimeError("Could not resolve a shared VLM language-layer parameter for the gradient gate")

    expert_forward_calls = 0

    def count_expert_forward(_module, _inputs, _output) -> None:
        nonlocal expert_forward_calls
        expert_forward_calls += 1

    hook = model.paligemma_with_expert.gemma_expert.model.register_forward_hook(count_expert_forward)
    processed = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    context, context_pad, _, _, _, language_span = model._embed_context_with_layout(  # noqa: SLF001
        processed.images,
        processed.image_masks,
        processed.tokenized_prompt,
        processed.tokenized_prompt_mask,
    )
    semantic_only = model._native_semantic_lm_decode(  # noqa: SLF001
        context,
        context_pad,
        language_span,
        targets.semantic_input_ids,
        targets.semantic_labels,
        targets.semantic_loss_mask,
    )
    semantic_only["loss"].backward()
    hook.remove()
    gradient_norms = {
        name: 0.0 if parameter.grad is None else float(parameter.grad.float().norm())
        for name, parameter in selected_parameters.items()
    }
    vision_grad_nonzero = any(value > 0 for name, value in gradient_norms.items() if "vision_tower" in name)
    language_grad_nonzero = any(value > 0 for name, value in gradient_norms.items() if "language_model" in name)

    checks = {
        "official_pi05_libero_semantics": (
            config.pi05 is True and config.action_horizon == 10 and config.discrete_state_input is False
        ),
        "primary_p2_has_no_semantic_query_parameter": not hasattr(model, "semantic_queries"),
        "primary_p2_layout_has_only_ground_and_geometry_queries": set(layout.query_groups) == {"ground", "geometry"},
        "semantic_loss_is_finite": bool(torch.isfinite(first_semantic)),
        "fixed_input_action_is_bitwise_deterministic": bool(torch.equal(first_action, second_action)),
        "fixed_input_semantic_ce_is_bitwise_deterministic": bool(torch.equal(first_semantic, second_semantic)),
        "changing_only_semantic_teacher_does_not_change_action": bool(torch.equal(first_action, changed_action)),
        "changing_only_ground_teacher_does_not_change_action": bool(
            torch.equal(first_action, changed_ground["action_loss_per_element"])
        ),
        "changing_only_geometry_teacher_does_not_change_action": bool(
            torch.equal(first_action, changed_geometry["action_loss_per_element"])
        ),
        "changing_semantic_teacher_changes_semantic_ce": bool(not torch.equal(first_semantic, changed_semantic)),
        "semantic_ce_reaches_shared_vision_path": vision_grad_nonzero,
        "semantic_ce_reaches_shared_language_path": language_grad_nonzero,
        "semantic_only_pass_never_calls_action_expert": expert_forward_calls == 0,
        "ground_and_geometry_queries_receive_no_semantic_only_gradient": all(
            getattr(model, name).grad is None for name in ("ground_queries", "geometry_queries")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Native semantic LM gate failed: {checks}")
    payload = {
        "status": "PASS",
        "gate": "pi05_p2_native_semantic_lm_and_no_action_leakage_v1",
        "sample_id": auxiliary["sample_id"],
        "overall_instruction": auxiliary["prompt"],
        "semantic_target": auxiliary["semantic_text"],
        "replacement_semantic_target": replacement_text,
        "strict_load": strict_load,
        "model_config": dataclasses.asdict(config),
        "architecture": {
            "action_prefix_query_groups": list(layout.query_groups),
            "semantic_query_count": 0,
            "semantic_objective": "separate native PaliGemma autoregressive LM pass",
            "teacher_tokens_present_in_action_prefix": False,
            "action_expert_forward_calls_in_semantic_only_pass": expert_forward_calls,
        },
        "metrics": {
            "semantic_ce": float(first_semantic),
            "replacement_semantic_ce": float(changed_semantic),
            "semantic_token_accuracy": float(first["diagnostics"]["semantic_token_accuracy"]),
            "semantic_supervised_token_count": int(first["diagnostics"]["semantic_supervised_token_count"]),
            "semantic_only_ce": float(semantic_only["loss"]),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "elapsed_seconds": time.monotonic() - started,
        },
        "shared_vlm_gradient_norms": gradient_norms,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
