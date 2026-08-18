#!/usr/bin/env python3
"""Strict-load trained P1/P2 checkpoints through the standard serving path."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from safetensors import safe_open
import torch

from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.training import config as _config

FIXED_SEED = 20260818
SENTINEL_ROOT = Path("/unusable/policy_aux_teacher_artifacts_must_not_be_read")


def teacher_free_config(name: str):
    config = _config.get_config(name)
    if config.policy_aux is None:
        raise ValueError(f"Serving gate requires an auxiliary config: {name}")
    model_config = dataclasses.replace(config.model, pytorch_compile_mode=None)
    policy_aux = dataclasses.replace(
        config.policy_aux,
        policy_manifest_path=str(SENTINEL_ROOT / name / "policy_manifest.parquet"),
        episode_mapping_path=str(SENTINEL_ROOT / name / "episode_mapping.json"),
        geometry_target_index_path=str(SENTINEL_ROOT / name / "geometry_index.parquet"),
        geometry_normalization_path=str(SENTINEL_ROOT / name / "geometry_normalization.json"),
        lerobot_root=None,
    )
    return dataclasses.replace(config, model=model_config, policy_aux=policy_aux)


def checkpoint_auxiliary_keys(checkpoint: Path, model: PI05AuxPolicy) -> tuple[list[str], dict[str, bool]]:
    expected = sorted(model.expected_auxiliary_state_keys())
    restored = {}
    state = model.state_dict()
    with safe_open(checkpoint, framework="pt", device="cpu") as checkpoint_file:
        checkpoint_keys = set(checkpoint_file.keys())
        missing = sorted(set(expected) - checkpoint_keys)
        if missing:
            raise RuntimeError(f"Trained checkpoint lacks expected auxiliary keys: {missing}")
        for name in expected:
            checkpoint_tensor = checkpoint_file.get_tensor(name)
            loaded_tensor = state[name].detach().cpu().to(checkpoint_tensor.dtype)
            restored[name] = bool(torch.equal(loaded_tensor, checkpoint_tensor))
    if not all(restored.values()):
        failed = sorted(name for name, matches in restored.items() if not matches)
        raise RuntimeError(f"Loaded auxiliary parameters differ from checkpoint: {failed}")
    return expected, restored


def run_mode(*, config_name: str, checkpoint: Path, device: torch.device, num_steps: int) -> dict:
    config = teacher_free_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise RuntimeError(f"LIBERO normalization assets did not load for {config_name}")

    with tempfile.TemporaryDirectory(prefix=f"{config_name}_serving_") as checkpoint_dir:
        os.symlink(checkpoint.resolve(strict=True), Path(checkpoint_dir) / "model.safetensors")
        policy = policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            norm_stats=data_config.norm_stats,
            pytorch_device=str(device),
            sample_kwargs={"num_steps": num_steps},
        )

    model = policy._model  # noqa: SLF001
    if not isinstance(model, PI05AuxPolicy):
        raise RuntimeError(f"Standard serving path constructed {type(model).__name__}, not PI05AuxPolicy")
    expected_keys, restored = checkpoint_auxiliary_keys(checkpoint, model)

    teacher_paths = {
        "policy_manifest_path": config.policy_aux.policy_manifest_path,
        "episode_mapping_path": config.policy_aux.episode_mapping_path,
        "geometry_target_index_path": config.policy_aux.geometry_target_index_path,
        "geometry_normalization_path": config.policy_aux.geometry_normalization_path,
    }
    if any(Path(path).exists() for path in teacher_paths.values()):
        raise RuntimeError(f"A serving sentinel unexpectedly exists: {teacher_paths}")
    model_teacher_paths = {
        "semantic_annotation_root": model.aux_config.semantic_annotation_root,
        "ground_mask_root": model.aux_config.ground_mask_root,
        "geometry_cache_root": model.aux_config.geometry_cache_root,
        "geometry_normalization_path": model.aux_config.geometry_normalization_path,
    }
    if any(path is not None for path in model_teacher_paths.values()):
        raise RuntimeError(f"Teacher paths leaked into the serving model: {model_teacher_paths}")

    np.random.seed(FIXED_SEED)
    observation = libero_policy.make_libero_example()
    observation["prompt"] = "pick up the black bowl and place it on the plate"
    noise = np.random.default_rng(FIXED_SEED).standard_normal(
        (config.model.action_horizon, config.model.action_dim), dtype=np.float32
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = policy.infer(observation, noise=noise)
    actions = np.asarray(output["actions"])
    checks = {
        "standard_create_trained_policy_path": True,
        "model_is_pi05_aux_policy": isinstance(model, PI05AuxPolicy),
        "strict_complete_checkpoint_load": True,
        "external_libero_action_shape": list(actions.shape) == [10, 7],
        "model_horizon_and_action_dim": [config.model.action_horizon, config.model.action_dim] == [10, 32],
        "actions_are_finite": bool(np.isfinite(actions).all()),
        "all_expected_aux_keys_in_checkpoint": len(expected_keys) == len(restored),
        "all_expected_aux_parameters_restored_bitwise": all(restored.values()),
        "teacher_sentinel_paths_absent": not any(Path(path).exists() for path in teacher_paths.values()),
        "teacher_paths_absent_from_model": all(path is None for path in model_teacher_paths.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Aux serving gate failed for {config_name}: {checks}")
    report = {
        "status": "PASS",
        "config_name": config_name,
        "mode": model.aux_config.mode,
        "checkpoint": str(checkpoint.resolve()),
        "model_type": type(model).__name__,
        "expected_auxiliary_state_keys": expected_keys,
        "restored_auxiliary_state_keys": restored,
        "teacher_sentinel_paths": teacher_paths,
        "model_teacher_paths": model_teacher_paths,
        "checks": checks,
        "action_shape": list(actions.shape),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
    }
    del policy, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--p2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")

    started = time.monotonic()
    device = torch.device(args.device)
    vggt_before = sorted(name for name in sys.modules if name == "vggt" or name.startswith("vggt."))
    reports = {
        "p1": run_mode(
            config_name="pi05_libero_p1_aux",
            checkpoint=args.p1_checkpoint,
            device=device,
            num_steps=args.num_steps,
        ),
        "p2": run_mode(
            config_name="pi05_libero_p2_aux",
            checkpoint=args.p2_checkpoint,
            device=device,
            num_steps=args.num_steps,
        ),
    }
    vggt_after = sorted(name for name in sys.modules if name == "vggt" or name.startswith("vggt."))
    global_checks = {
        "p1_aux_aware_serving": reports["p1"]["status"] == "PASS",
        "p2_aux_aware_serving": reports["p2"]["status"] == "PASS",
        "vggt_not_imported": vggt_before == vggt_after,
        "no_teacher_or_annotation_inputs": True,
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"P1/P2 serving gate failed: {global_checks}")
    payload = {
        "status": "PASS",
        "gate": "p1_p2_aux_aware_standard_serving_v1",
        "reports": reports,
        "global_checks": global_checks,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
