"""Gated 8-GPU entrypoint for the isolated RoboCasa ablation suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..train import _configure_performance_environment
from ..train import _load_trainer
from .configs import build_ablation_train_config
from .integration import install_ablation_overlays
from .specs import ABLATION_SPECS
from .specs import get_ablation_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=tuple(ABLATION_SPECS), required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--policy-assets-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--base-weight-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-micro-batch", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = get_ablation_spec(args.ablation)
    install_ablation_overlays()
    config = build_ablation_train_config(
        ablation=args.ablation,
        exp_name=args.exp_name,
        data_root=args.data_root,
        manifest_root=args.manifest_root,
        policy_assets_root=args.policy_assets_root,
        artifact_dir=args.artifact_dir,
        base_weight_dir=args.base_weight_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        seed=args.seed,
        global_micro_batch=args.global_micro_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        resume=args.resume,
        overwrite=args.overwrite,
        wandb_enabled=not args.disable_wandb,
    )
    manifest = {
        "ablation": spec.variant,
        "display_name": spec.display_name,
        "target_scope": spec.target_scope,
        "semantic_enabled": spec.semantic_enabled,
        "geometry_enabled": spec.geometry_enabled,
        "motion_enabled": spec.motion_enabled,
        "action_conditioning": list(spec.action_conditioning),
        "lambda_geo": spec.lambda_geo,
        "lambda_sem": spec.lambda_sem,
        "lambda_motion": spec.lambda_motion,
        "action_horizon": config.model.action_horizon,
        "global_micro_batch": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_global_batch": (
            config.batch_size * config.gradient_accumulation_steps
        ),
        "updates": config.num_train_steps,
        "warmup_updates": config.lr_schedule.warmup_steps,
        "peak_lr": config.lr_schedule.peak_lr,
        "ema": config.ema_decay,
        "checkpoint_save_interval": config.save_interval,
        "checkpoint_keep_period": config.keep_period,
        "ordinary_checkpoints_to_keep": 1,
        "resume_state": "all retained checkpoints",
        "checkpoint_dir": str(config.checkpoint_dir),
    }
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.ablation == "full":
        raise ValueError(
            "Full SGeM-VLA is the main experiment result and must not be "
            "retrained through the formal ablation entrypoint"
        )
    if os.environ.get("ROBOCASA24_ABLATION_TRAINING_APPROVED") != "YES":
        raise PermissionError(
            "Ablation optimizer work is gated. Complete the dedicated 8-GPU "
            "smoke and set ROBOCASA24_ABLATION_TRAINING_APPROVED=YES only "
            "after explicit approval."
        )
    _configure_performance_environment()
    trainer = _load_trainer(args.openpi_root.resolve(strict=True))
    trainer.init_logging()
    trainer.train_loop(config)


if __name__ == "__main__":
    main()
