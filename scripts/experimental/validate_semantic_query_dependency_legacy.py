#!/usr/bin/env python3
"""SUPERSEDED legacy experiment: semantic CE dependence on Q_sem.

This script and its archived result preserve the pre-2026-08-18 Semantic-Query
prototype. It is intentionally outside the primary P2 pipeline and is not
compatible with the current ``PI05AuxPolicy`` primary configuration, which has
no Semantic Query and uses the native autoregressive VLM language path.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-updates", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    args = parser.parse_args()
    device = torch.device(args.device)
    observation, actions, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.annotation_manifest,
    )
    observation = move_observation(observation, device)
    actions = actions.to(device)
    aux_targets = PolicyAuxTargets(
        geometry=torch.zeros((1, 2048), device=device),
        geometry_valid=torch.zeros(1, dtype=torch.bool, device=device),
        geometry_mean=torch.zeros(2048, device=device),
        geometry_std=torch.ones(2048, device=device),
        ground_masks={key: value.to(device) for key, value in auxiliary["ground_masks"].items()},
        ground_valid_views=auxiliary["ground_valid_views"].to(device),
        semantic_input_ids=auxiliary["semantic_input_ids"].to(device),
        semantic_labels=auxiliary["semantic_labels"].to(device),
        semantic_loss_mask=auxiliary["semantic_loss_mask"].to(device),
    )
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    model = PI05AuxPolicy(
        config,
        PolicyAuxConfig(
            mode="semantic_ground_geometry",
            lambda_sem=1.0,
            lambda_ground=0.0,
            lambda_geo=0.0,
        ),
    )
    load_result = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)
    model.semantic_queries.requires_grad_(requires_grad=True)
    model.to(device).train()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([model.semantic_queries], lr=args.learning_rate, weight_decay=0.0)
    noise = torch.full_like(actions, 0.125)
    timestep = torch.full((1,), 0.5, device=device)

    losses = []
    started = time.perf_counter()
    for update in range(args.max_updates):
        torch.manual_seed(20260818)
        result = model(
            observation,
            actions,
            aux_targets,
            noise=noise,
            time=timestep,
        )
        loss = result["losses"]["semantic"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite semantic loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(model.semantic_queries.grad.float().norm())
        if gradient_norm <= 0.0:
            raise RuntimeError(f"Zero Q_sem gradient at update {update}")
        torch.nn.utils.clip_grad_norm_([model.semantic_queries], 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if update % 10 == 0:
            print(json.dumps({"update": update, "semantic_ce": losses[-1]}), flush=True)
        if update >= 9 and losses[-1] < min(1.0, losses[0] * 0.10):
            break

    model.eval()
    torch.manual_seed(20260818)
    with torch.no_grad():
        evaluation = model(
            observation,
            actions,
            aux_targets,
            noise=noise,
            time=timestep,
        )
        semantic_state = evaluation["diagnostics"]["semantic_state"]
        normal = model._semantic_decode(  # noqa: SLF001
            semantic_state,
            aux_targets.semantic_input_ids,
            aux_targets.semantic_labels,
            aux_targets.semantic_loss_mask,
        )["loss"]
        zero = model._semantic_decode(  # noqa: SLF001
            torch.zeros_like(semantic_state),
            aux_targets.semantic_input_ids,
            aux_targets.semantic_labels,
            aux_targets.semantic_loss_mask,
        )["loss"]
    normal_value = float(normal)
    zero_value = float(zero)
    checks = {
        "semantic_loss_decreased_by_at_least_80_percent": losses[-1] <= losses[0] * 0.20,
        "zero_q_sem_ce_worsens_absolute": zero_value > normal_value + 0.25,
        "zero_q_sem_ce_worsens_relative": zero_value > normal_value * 1.25,
        "only_q_sem_was_trainable": [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        == ["semantic_queries"],
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "gate": "pi05_p2_semantic_query_dependency_v1",
        "engineering_scope": "single-real-sample Q_sem-only tiny overfit; not a research result",
        "sample_id": auxiliary["sample_id"],
        "target_text": auxiliary["semantic_text"],
        "strict_base_load": load_result,
        "updates": len(losses),
        "learning_rate": args.learning_rate,
        "initial_semantic_ce": losses[0],
        "final_training_semantic_ce": losses[-1],
        "evaluation_semantic_ce": normal_value,
        "zero_q_sem_semantic_ce": zero_value,
        "zero_minus_normal_ce": zero_value - normal_value,
        "zero_over_normal_ce": zero_value / max(normal_value, 1e-12),
        "loss_curve": losses,
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise RuntimeError(
            f"Semantic dependency gate failed: checks={checks}, normal={normal_value}, zero={zero_value}"
        )


if __name__ == "__main__":
    main()
