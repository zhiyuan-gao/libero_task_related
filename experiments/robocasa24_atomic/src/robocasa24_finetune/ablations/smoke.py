"""Gated 8-GPU optimizer smoke for one written ablation configuration."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from ..constants import TASKS
from ..train import _configure_performance_environment
from ..train import _load_trainer
from .configs import build_ablation_train_config
from .integration import install_ablation_overlays
from .specs import ABLATION_SPECS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=tuple(ABLATION_SPECS), required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--policy-assets-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--base-weight-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.updates <= 5:
        raise ValueError("ablation optimizer smoke is restricted to 1..5 updates")
    if os.environ.get("ROBOCASA24_ABLATION_SMOKE_APPROVED") != "YES":
        raise PermissionError(
            "set ROBOCASA24_ABLATION_SMOKE_APPROVED=YES only when all eight GPUs are available"
        )

    install_ablation_overlays()
    formal = build_ablation_train_config(
        ablation=args.ablation,
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
        log_interval=1,
        save_interval=10_000,
        save_final_checkpoint=False,
        max_checkpoints_to_keep=None,
        overwrite=True,
        resume=False,
    )
    _configure_performance_environment()
    trainer = _load_trainer(args.openpi_root.resolve(strict=True))
    trainer.init_logging()
    trainer.train_loop(smoke)


if __name__ == "__main__":
    main()
