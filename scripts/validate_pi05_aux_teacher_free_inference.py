#!/usr/bin/env python3
"""Validate enabled P1/P2 action inference without auxiliary teacher artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

from policy_aux_gate_utils import load_real_libero_item
from policy_aux_gate_utils import move_observation
import torch

from openpi.models import pi0_config
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    observation, _, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.annotation_manifest,
    )
    observation = move_observation(observation, device)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    modes = ("geometry", "ground_geometry_semantic_lm")
    reports = {}
    vggt_modules_before = sorted(name for name in sys.modules if name == "vggt" or name.startswith("vggt."))
    for mode in modes:
        sentinel_root = f"/nonexistent/teacher_artifacts_must_not_be_read/{mode}"
        aux_config = PolicyAuxConfig(
            mode=mode,
            semantic_annotation_root=sentinel_root,
            ground_mask_root=sentinel_root,
            geometry_cache_root=sentinel_root,
            geometry_normalization_path=sentinel_root,
        )
        model = PI05AuxPolicy(config, aux_config)
        strict_load = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
        model.to(device).eval()
        generator = torch.Generator(device=device).manual_seed(20260818)
        noise = torch.randn(
            (1, config.action_horizon, config.action_dim),
            generator=generator,
            device=device,
        )
        torch.cuda.reset_peak_memory_stats(device)
        first = model.sample_actions(device, observation, noise=noise.clone(), num_steps=args.num_steps)
        second = model.sample_actions(device, observation, noise=noise.clone(), num_steps=args.num_steps)
        checks = {
            "action_shape_is_official_horizon": list(first.shape) == [1, 10, 32],
            "actions_are_finite": bool(torch.isfinite(first).all()),
            "fixed_noise_is_bitwise_deterministic": bool(torch.equal(first, second)),
            "no_auxiliary_target_argument_required": True,
            "sentinel_teacher_paths_do_not_exist": not Path(sentinel_root).exists(),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{mode} teacher-free inference gate failed: {checks}")
        reports[mode] = {
            "strict_load": strict_load,
            "aux_config": dataclasses.asdict(aux_config),
            "checks": checks,
            "action_shape": list(first.shape),
            "action_mean": float(first.float().mean()),
            "action_std": float(first.float().std()),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        del model
        torch.cuda.empty_cache()
    vggt_modules_after = sorted(name for name in sys.modules if name == "vggt" or name.startswith("vggt."))
    checks = {
        "p1_and_p2_pass": set(reports) == set(modes),
        "vggt_not_imported_by_inference": vggt_modules_after == vggt_modules_before,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Teacher-free inference gate failed: {checks}")
    payload = {
        "status": "PASS",
        "gate": "pi05_p1_p2_teacher_free_action_inference_v1",
        "sample_id": auxiliary["sample_id"],
        "model_config": dataclasses.asdict(config),
        "num_denoising_steps_for_gate": args.num_steps,
        "modes": reports,
        "global_checks": checks,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
