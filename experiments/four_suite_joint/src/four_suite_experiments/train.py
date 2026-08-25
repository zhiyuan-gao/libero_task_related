"""Explicitly gated entrypoint for full four-suite optimizer runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from .action_access import install_action_access_policy
from .configs import blocked_action_groups
from .configs import build_train_config
from .configs import expected_target_scope
from .data_overlay import install_data_overlay
from .paths import ArtifactPaths
from .paths import SourcePaths
from .validate import validate_artifacts
from .validate import validate_lerobot_snapshot


def _load_upstream_trainer(openpi_root: Path):
    path = openpi_root / "scripts/train_pytorch.py"
    spec = importlib.util.spec_from_file_location(
        "four_suite_upstream_train_pytorch", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("main", "supervision_only"), required=True
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--base-weight-path", type=Path)
    parser.add_argument("--libero-assets-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--disable-wandb", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.environ.get("FOUR_SUITE_FULL_TRAINING_APPROVED") != "YES":
        raise PermissionError(
            "Refusing optimizer work. Freeze the budget, then set FOUR_SUITE_FULL_TRAINING_APPROVED=YES."
        )
    artifacts = ArtifactPaths(args.artifact_dir.resolve())
    preflight = validate_artifacts(
        artifacts, target_scope=expected_target_scope(args.variant)
    )
    source_paths = SourcePaths.defaults(args.artifact_dir)
    validate_lerobot_snapshot(source_paths.lerobot_root, require_complete=True)
    config = build_train_config(
        variant=args.variant,
        artifacts=artifacts,
        exp_name=args.exp_name,
        num_train_steps=args.num_train_steps,
        warmup_steps=args.warmup_steps,
        checkpoint_base_dir=args.checkpoint_base_dir,
        lerobot_root=source_paths.lerobot_root,
        base_weight_path=args.base_weight_path,
        libero_assets_dir=args.libero_assets_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_enabled=not args.disable_wandb,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    base_model_file = Path(config.pytorch_weight_path) / "model.safetensors"
    if not base_model_file.is_file():
        raise FileNotFoundError(f"strict FP32-converted base is missing: {base_model_file}")
    norm_stats_file = (
        Path(config.data.assets.assets_dir)
        / "physical-intelligence/libero/norm_stats.json"
    )
    if not norm_stats_file.is_file():
        raise FileNotFoundError(f"LIBERO normalization assets are missing: {norm_stats_file}")
    install_data_overlay()
    install_action_access_policy(blocked_action_groups(args.variant))
    launch_manifest = {
        "preflight": preflight,
        "variant": args.variant,
        "exp_name": args.exp_name,
        "num_train_steps": args.num_train_steps,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "global_batch_size": args.batch_size,
        "blocked_action_groups": sorted(blocked_action_groups(args.variant)),
        "openpi_root": str(source_paths.openpi_root),
        "base_weight_path": str(config.pytorch_weight_path),
        "libero_assets_dir": str(config.data.assets.assets_dir),
        "checkpoint_dir": str(config.checkpoint_dir),
    }
    print(json.dumps(launch_manifest, indent=2, sort_keys=True), flush=True)
    trainer = _load_upstream_trainer(source_paths.openpi_root)
    trainer.init_logging()
    trainer.train_loop(config)


if __name__ == "__main__":
    main()
