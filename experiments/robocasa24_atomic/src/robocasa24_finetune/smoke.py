"""Gated real 8-GPU optimizer smoke for an already validated configuration."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from .configs import build_train_config
from .constants import TASKS
from .integration import install_robocasa_overlays
from .train import _configure_performance_environment
from .train import _load_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("baseline", "task_relevant", "whole_scene"), required=True
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--policy-assets-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--base-weight-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.updates <= 5:
        raise ValueError("optimizer smoke is restricted to 1..5 updates")
    if os.environ.get("ROBOCASA24_SMOKE_APPROVED") != "YES":
        raise PermissionError(
            "set ROBOCASA24_SMOKE_APPROVED=YES only when all eight GPUs are available"
        )
    install_robocasa_overlays()
    formal = build_train_config(
        variant=args.variant,
        exp_name=args.exp_name,
        data_root=args.data_root,
        manifest_root=args.manifest_root,
        policy_assets_root=args.policy_assets_root,
        artifact_dir=args.artifact_dir,
        base_weight_dir=args.base_weight_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        num_workers=args.num_workers,
        wandb_enabled=False,
        tasks=tuple(args.tasks) if args.tasks else TASKS,
    )
    smoke = dataclasses.replace(
        formal,
        num_train_steps=args.updates,
        save_interval=10_000,
        save_final_checkpoint=False,
        max_checkpoints_to_keep=1,
        overwrite=True,
        resume=False,
    )
    _configure_performance_environment()
    trainer = _load_trainer(args.openpi_root.resolve(strict=True))
    trainer.init_logging()
    trainer.train_loop(smoke)


if __name__ == "__main__":
    main()
