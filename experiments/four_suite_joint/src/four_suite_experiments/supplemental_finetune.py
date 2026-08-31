"""Warm-start TRQC on an audited additive official-demo population."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys

from .action_access import install_action_access_policy
from .configs import build_train_config
from .constants import AUGMENTED_EPISODES
from .constants import AUGMENTED_FRAMES
from .constants import AUGMENTED_GEOMETRY_VALID
from .constants import AUGMENTED_MOTION_VALID
from .constants import COMPLETED_EPISODES
from .constants import COMPLETED_FRAMES
from .constants import COMPLETED_GEOMETRY_VALID
from .constants import COMPLETED_MOTION_VALID
from .data_overlay import install_data_overlay
from .paths import ArtifactPaths


def _load_trainer(openpi_root: Path):
    path = openpi_root / "scripts/train_pytorch.py"
    spec = importlib.util.spec_from_file_location("four_suite_supplemental_train_pytorch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate(
    lerobot_root: Path,
    artifacts: ArtifactPaths,
    *,
    expected_episodes: int,
    expected_frames: int,
    expected_geometry_valid: int,
    expected_motion_valid: int,
) -> dict:
    missing = [str(path) for path in artifacts.all_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"supplemental artifact bundle is incomplete: {missing}")
    validation_path = artifacts.root / "validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = json.loads(validation_path.read_text())
    provenance = json.loads(artifacts.provenance.read_text())
    info = json.loads((lerobot_root / "meta/info.json").read_text())
    episodes = [
        json.loads(line) for line in (lerobot_root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    if (
        validation.get("status") != "PASS"
        or provenance.get("status") != "PASS"
        or int(info.get("total_episodes", -1)) != expected_episodes
        or int(info.get("total_frames", -1)) != expected_frames
        or len(episodes) != expected_episodes
        or int(provenance["population"]["geometry_valid"]) != expected_geometry_valid
        or int(provenance["population"]["motion_valid"]) != expected_motion_valid
    ):
        raise ValueError("supplemental LeRobot/artifact population failed the frozen gate")
    chunk_size = int(info["chunks_size"])
    missing_episodes = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        path = (
            lerobot_root / "data" / f"chunk-{episode_index // chunk_size:03d}" / f"episode_{episode_index:06d}.parquet"
        )
        if not path.is_file():
            missing_episodes.append(str(path))
    if missing_episodes:
        raise FileNotFoundError(
            f"supplemental LeRobot root lacks {len(missing_episodes)} episodes; first={missing_episodes[0]}"
        )
    return {
        "status": "PASS",
        "episodes": expected_episodes,
        "frames": expected_frames,
        "geometry_valid": expected_geometry_valid,
        "motion_valid": expected_motion_valid,
        "normalization": provenance["normalization"],
        "observation_recipe": provenance["supplemental_observation_recipe"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-assets-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num-updates",
        type=int,
        default=3_000,
        help="Number of optimizer updates in this warm-start stage.",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help="Optional absolute final step, primarily for exact continuation of an existing stage.",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=None,
        help="Override the LR schedule horizon while retaining an absolute target step.",
    )
    parser.add_argument(
        "--official-completion",
        action="store_true",
        help="Use the frozen 1,932-episode Object/Goal/LIBERO-10 completion population.",
    )
    args = parser.parse_args()

    if os.environ.get("FOUR_SUITE_SUPPLEMENTAL_FINETUNE_APPROVED") != "YES":
        raise PermissionError("Refusing optimizer work without FOUR_SUITE_SUPPLEMENTAL_FINETUNE_APPROVED=YES")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.num_updates <= 0:
        raise ValueError("--num-updates must be positive")
    if args.target_step is not None and args.target_step <= 0:
        raise ValueError("--target-step must be positive")
    if args.decay_steps is not None and args.decay_steps <= 0:
        raise ValueError("--decay-steps must be positive")
    openpi_root = args.openpi_root.resolve(strict=True)
    lerobot_root = args.lerobot_root.resolve(strict=True)
    artifacts = ArtifactPaths(args.artifact_dir.resolve(strict=True))
    parent = args.parent_checkpoint.resolve(strict=True)
    if not (parent / "model.safetensors").is_file():
        raise FileNotFoundError(f"parent model is missing: {parent / 'model.safetensors'}")
    assets = args.libero_assets_dir.resolve(strict=True)
    if not (assets / "physical-intelligence/libero/norm_stats.json").is_file():
        raise FileNotFoundError("frozen LIBERO normalization assets are missing")
    expected = (
        (
            COMPLETED_EPISODES,
            COMPLETED_FRAMES,
            COMPLETED_GEOMETRY_VALID,
            COMPLETED_MOTION_VALID,
        )
        if args.official_completion
        else (
            AUGMENTED_EPISODES,
            AUGMENTED_FRAMES,
            AUGMENTED_GEOMETRY_VALID,
            AUGMENTED_MOTION_VALID,
        )
    )
    preflight = _validate(
        lerobot_root,
        artifacts,
        expected_episodes=expected[0],
        expected_frames=expected[1],
        expected_geometry_valid=expected[2],
        expected_motion_valid=expected[3],
    )

    stage_updates = 2 if args.smoke and not args.resume else args.num_updates
    steps = args.target_step or stage_updates
    warmup = 200 if args.resume else 1 if args.smoke else 200
    decay_steps = args.decay_steps or stage_updates
    if warmup >= decay_steps:
        raise ValueError("stage length must exceed warmup steps")
    config = build_train_config(
        variant="trqc",
        artifacts=artifacts,
        exp_name=args.exp_name,
        num_train_steps=steps,
        warmup_steps=warmup,
        checkpoint_base_dir=args.checkpoint_base_dir,
        lerobot_root=lerobot_root,
        base_weight_path=parent,
        libero_assets_dir=assets,
        seed=42,
        batch_size=256,
        num_workers=args.num_workers,
        wandb_enabled=not args.disable_wandb,
        resume=args.resume,
        overwrite=args.overwrite,
        supplemental_augmentation=not args.official_completion,
        official_completion=args.official_completion,
        peak_lr=1e-5,
        decay_steps=decay_steps,
        decay_lr=1e-6,
        save_interval=(steps + 1 if args.smoke else 500),
        late_save_interval=None,
        late_save_start_step=None,
        max_checkpoints_to_keep=(1 if args.smoke else max(1, (steps + 499) // 500)),
        max_resume_checkpoints_to_keep=(1 if args.smoke else 2),
    )
    config = dataclasses.replace(
        config,
        name=(
            "pi05_libero40_trqc_official_completion_finetune"
            if args.official_completion
            else "pi05_libero40_trqc_supplemental_finetune"
        ),
        save_final_checkpoint=not args.smoke,
        pytorch_weight_load_mode="strict",
        policy_metadata={
            "training_population": (
                "libero40_object_goal_10_completed_to_50_spatial_432"
                if args.official_completion
                else "libero40_1693_plus_action_final_success_115"
            ),
            "parent_checkpoint": str(parent),
            "optimizer_initialization": "new_adamw",
            "normalization": "frozen_step30000_base_population",
            "supplemental_observation_recipe": preflight["observation_recipe"],
        },
    )
    install_data_overlay()
    install_action_access_policy(frozenset())
    launch = {
        "status": "READY",
        "mode": "smoke" if args.smoke else "formal_warm_start_stage",
        "preflight": preflight,
        "exp_name": args.exp_name,
        "parent_checkpoint": str(parent),
        "new_optimizer": not args.resume,
        "schedule_steps": decay_steps,
        "requested_updates": stage_updates,
        "final_step": steps,
        "warmup_steps": warmup,
        "global_batch_size": 256,
        "seed": 42,
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "loss_coefficients": {"semantic": 0.01, "geometry": 0.05, "motion": 0.05},
        "save_steps": ([] if args.smoke else list(range(500, steps + 1, 500))),
        "checkpoint_dir": str(config.checkpoint_dir),
    }
    print(json.dumps(launch, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    trainer = _load_trainer(openpi_root)
    trainer.init_logging()
    trainer.train_loop(config)


if __name__ == "__main__":
    main()
