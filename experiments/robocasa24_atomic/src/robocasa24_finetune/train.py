"""Gated RoboCasa Atomic-24 PyTorch training entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from .configs import build_train_config
from .integration import install_robocasa_overlays


def _configure_performance_environment() -> None:
    """Match the allocator/logging profile used by validated LIBERO runs."""

    os.environ.setdefault("OPENPI_USE_DEFAULT_CUDA_ALLOCATOR", "1")
    os.environ.setdefault("OPENPI_LOG_MEMORY_STATS", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_trainer(openpi_root: Path):
    path = openpi_root / "scripts" / "train_pytorch.py"
    spec = importlib.util.spec_from_file_location(
        "robocasa24_upstream_train_pytorch", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream PyTorch trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("baseline", "task_relevant", "whole_scene"), required=True
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--policy-assets-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--base-weight-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-micro-batch", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-checkpoints-to-keep", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    install_robocasa_overlays()
    config = build_train_config(
        variant=args.variant,
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
        save_interval=args.save_interval,
        max_checkpoints_to_keep=args.max_checkpoints_to_keep,
        resume=args.resume,
        overwrite=args.overwrite,
        wandb_enabled=not args.disable_wandb,
    )
    manifest = {
        "variant": args.variant,
        "exp_name": args.exp_name,
        "action_horizon": config.model.action_horizon,
        "model_action_dim": config.model.action_dim,
        "global_micro_batch": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_global_batch": config.batch_size
        * config.gradient_accumulation_steps,
        "updates": config.num_train_steps,
        "warmup_updates": config.lr_schedule.warmup_steps,
        "peak_lr": config.lr_schedule.peak_lr,
        "ema": config.ema_decay,
        "save_interval": config.save_interval,
        "max_checkpoints_to_keep": config.max_checkpoints_to_keep,
        "checkpoint_dir": str(config.checkpoint_dir),
    }
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if os.environ.get("ROBOCASA24_FULL_TRAINING_APPROVED") != "YES":
        raise PermissionError(
            "Full optimizer work is gated. Set ROBOCASA24_FULL_TRAINING_APPROVED=YES only after the 8-GPU smoke."
        )
    _configure_performance_environment()
    trainer = _load_trainer(args.openpi_root.resolve(strict=True))
    trainer.init_logging()
    trainer.train_loop(config)


if __name__ == "__main__":
    main()
