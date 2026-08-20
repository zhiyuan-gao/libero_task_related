#!/usr/bin/env python3
"""Validate P1/P2 per-loss gradient connectivity on one real policy sample."""

from __future__ import annotations

import argparse
import gc
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
from openpi.training.policy_aux_dataset import PolicyAuxTargetIndex
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig


def tensor(value, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, device=device)


def targets_from_index(item: dict, mode: str, device: torch.device) -> PolicyAuxTargets:
    return PolicyAuxTargets(
        geometry=tensor(item["geometry"], device)[None],
        geometry_valid=tensor(item["geometry_valid"], device)[None],
        geometry_mean=tensor(item["geometry_mean"], device),
        geometry_std=tensor(item["geometry_std"], device),
        ground_masks=(
            {name: tensor(value, device)[None] for name, value in item["ground_masks"].items()}
            if mode == "ground_geometry_semantic_lm"
            else None
        ),
        ground_valid_views=(
            tensor(item["ground_valid_views"], device)[None] if mode == "ground_geometry_semantic_lm" else None
        ),
        semantic_input_ids=(
            tensor(item["semantic_input_ids"], device)[None]
            if mode in ("semantic_geometry", "ground_geometry_semantic_lm")
            else None
        ),
        semantic_labels=(
            tensor(item["semantic_labels"], device)[None]
            if mode in ("semantic_geometry", "ground_geometry_semantic_lm")
            else None
        ),
        semantic_loss_mask=(
            tensor(item["semantic_loss_mask"], device)[None]
            if mode in ("semantic_geometry", "ground_geometry_semantic_lm")
            else None
        ),
    )


def selected_parameters(model: PI05AuxPolicy) -> dict[str, torch.nn.Parameter]:
    requested = {
        "shared_vision": "paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "shared_language": "paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
        "action_expert": "gemma_expert.model.layers.0.self_attn.q_proj.weight",
        "action_output": "action_out_proj.weight",
        "geometry_queries": "geometry_queries",
        "geometry_head": "geometry_head.output_projection.weight",
    }
    if hasattr(model, "ground_queries"):
        requested.update(
            {
                "ground_queries": "ground_queries",
                "ground_head": "ground_head.query_projection.weight",
            }
        )
    resolved = {}
    named = dict(model.named_parameters())
    for label, suffix in requested.items():
        matches = [parameter for name, parameter in named.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Could not uniquely resolve {label}: suffix={suffix}, matches={len(matches)}")
        resolved[label] = matches[0]
    return resolved


def gradient_norms(
    loss: torch.Tensor,
    parameters: dict[str, torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> dict[str, float]:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters.values()),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return {
        label: 0.0 if gradient is None else float(gradient.float().norm())
        for label, gradient in zip(parameters, gradients, strict=True)
    }


def positive(norms: dict[str, float], *names: str) -> bool:
    return all(torch.isfinite(torch.tensor(norms[name])) and norms[name] > 0.0 for name in names)


def zero(norms: dict[str, float], *names: str) -> bool:
    return all(norms[name] == 0.0 for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--geometry-index", type=Path, required=True)
    parser.add_argument("--geometry-normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    observation, actions, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.policy_manifest,
    )
    observation = move_observation(observation, device)
    actions = actions.to(device)
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    reports = {}
    for mode in ("geometry", "semantic_geometry", "ground_geometry_semantic_lm"):
        train_config = PolicyAuxTrainConfig(
            mode=mode,
            policy_manifest_path=str(args.policy_manifest),
            episode_mapping_path=str(args.mapping),
            geometry_target_index_path=str(args.geometry_index),
            geometry_normalization_path=str(args.geometry_normalization),
            lambda_geo=1.0,
            lambda_sem=1.0 if mode in ("semantic_geometry", "ground_geometry_semantic_lm") else None,
            lambda_ground=1.0 if mode == "ground_geometry_semantic_lm" else None,
            num_ground_queries=0 if mode == "semantic_geometry" else 8,
        )
        target_index = PolicyAuxTargetIndex(train_config)
        target_item = target_index.item(0)
        targets = targets_from_index(target_item, mode, device)
        aux_config = PolicyAuxConfig(
            mode=mode,
            num_ground_queries=0 if mode == "semantic_geometry" else 8,
            lambda_geo=1.0,
            lambda_sem=1.0 if mode in ("semantic_geometry", "ground_geometry_semantic_lm") else None,
            lambda_ground=1.0 if mode == "ground_geometry_semantic_lm" else None,
        )
        model = PI05AuxPolicy(model_config, aux_config)
        strict_load = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
        model.to(device).train()
        model.gradient_checkpointing_enable()
        parameters = selected_parameters(model)
        generator = torch.Generator(device=device).manual_seed(20260818)
        noise = torch.randn(actions.shape, generator=generator, device=device)
        diffusion_time = torch.full((1,), 0.5, dtype=torch.float32, device=device)
        torch.manual_seed(20260818)
        torch.cuda.reset_peak_memory_stats(device)
        result = model.forward_with_aux(
            observation,
            actions,
            targets,
            noise=noise,
            time=diffusion_time,
        )
        losses = result["losses"]
        loss_order = ["action", "geometry"]
        if mode == "semantic_geometry":
            loss_order.append("semantic")
        elif mode == "ground_geometry_semantic_lm":
            loss_order.extend(["ground", "semantic"])
        norms = {
            name: gradient_norms(losses[name], parameters, retain_graph=index < len(loss_order) - 1)
            for index, name in enumerate(loss_order)
        }
        checks = {
            "all_losses_finite": all(bool(torch.isfinite(losses[name])) for name in loss_order),
            "action_reaches_action_expert_and_output": positive(norms["action"], "action_expert", "action_output"),
            "action_consumes_geometry_queries": positive(norms["action"], "geometry_queries"),
            "action_does_not_use_geometry_decoder": zero(norms["action"], "geometry_head"),
            "geometry_reaches_query_head_and_shared_vlm": positive(
                norms["geometry"],
                "geometry_queries",
                "geometry_head",
                "shared_language",
                "shared_vision",
            ),
            "geometry_does_not_reach_action_expert_or_output": zero(
                norms["geometry"], "action_expert", "action_output"
            ),
        }
        if mode == "ground_geometry_semantic_lm":
            checks.update(
                {
                    "action_consumes_ground_queries": positive(norms["action"], "ground_queries"),
                    "action_does_not_use_ground_decoder": zero(norms["action"], "ground_head"),
                    "ground_reaches_query_head_and_shared_vlm": positive(
                        norms["ground"],
                        "ground_queries",
                        "ground_head",
                        "shared_language",
                        "shared_vision",
                    ),
                    "ground_isolated_from_geometry_and_action_modules": zero(
                        norms["ground"],
                        "geometry_queries",
                        "geometry_head",
                        "action_expert",
                        "action_output",
                    ),
                    "geometry_isolated_from_ground_modules": zero(norms["geometry"], "ground_queries", "ground_head"),
                    "semantic_reaches_only_expected_shared_vlm_path": (
                        positive(norms["semantic"], "shared_language", "shared_vision")
                        and zero(
                            norms["semantic"],
                            "ground_queries",
                            "ground_head",
                            "geometry_queries",
                            "geometry_head",
                            "action_expert",
                            "action_output",
                        )
                    ),
                }
            )
        if mode == "semantic_geometry":
            checks.update(
                {
                    "semantic_reaches_only_expected_shared_vlm_path": (
                        positive(norms["semantic"], "shared_language", "shared_vision")
                        and zero(
                            norms["semantic"],
                            "geometry_queries",
                            "geometry_head",
                            "action_expert",
                            "action_output",
                        )
                    ),
                    "semantic_geometry_has_no_ground_parameters": (
                        "ground_queries" not in parameters and "ground_head" not in parameters
                    ),
                }
            )
        if not all(checks.values()):
            raise RuntimeError(f"{mode} component-gradient gate failed: {checks}; norms={norms}")
        reports[mode] = {
            "strict_load": strict_load,
            "losses": {name: float(losses[name].detach()) for name in loss_order},
            "gradient_norms": norms,
            "checks": checks,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        del result, losses, targets, parameters, model, target_index
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "status": "PASS",
        "gate": "pi05_p1_p2_component_gradient_connectivity_v1",
        "sample_id": auxiliary["sample_id"],
        "engineering_coefficients_only": {
            "lambda_sem": 1.0,
            "lambda_ground": 1.0,
            "lambda_geo": 1.0,
            "approved_for_training": False,
        },
        "modes": reports,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
