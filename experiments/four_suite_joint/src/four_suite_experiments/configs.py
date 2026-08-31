"""Construct the three frozen LIBERO-40 TRQC experiment variants."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Literal

from openpi.training import config as openpi_config

from .constants import AUGMENTED_EPISODES
from .constants import AUGMENTED_FRAMES
from .constants import AUGMENTED_MOTION_VALID
from .constants import COMPLETED_EPISODES
from .constants import COMPLETED_FRAMES
from .constants import COMPLETED_MOTION_VALID
from .constants import FOUR_SUITE_EPISODES
from .constants import FOUR_SUITE_FRAMES
from .constants import FOUR_SUITE_MOTION_VALID
from .data_overlay import FourSuitePolicyAuxTrainConfig
from .paths import ArtifactPaths

Variant = Literal["trqc", "whole_scene", "no_query_access"]
BASE_CONFIG_NAME = "pi05_libero3_semantic_geometry_motion_aux"
CONFIG_NAMES: dict[Variant, str] = {
    "trqc": "pi05_libero40_trqc",
    "whole_scene": "pi05_libero40_trqc_whole_scene_geometry_motion",
    "no_query_access": "pi05_libero40_trqc_no_query_access",
}


def blocked_action_groups(variant: Variant) -> frozenset[str]:
    if variant == "no_query_access":
        return frozenset({"geometry", "motion"})
    if variant in ("trqc", "whole_scene"):
        return frozenset()
    raise ValueError(f"unsupported four-suite variant: {variant}")


def expected_target_scope(variant: Variant) -> str:
    blocked_action_groups(variant)
    return "whole_scene" if variant == "whole_scene" else "task_relevant"


def build_train_config(
    *,
    variant: Variant,
    artifacts: ArtifactPaths,
    exp_name: str,
    num_train_steps: int,
    warmup_steps: int,
    checkpoint_base_dir: str | Path,
    lerobot_root: str | Path,
    base_weight_path: str | Path | None = None,
    libero_assets_dir: str | Path | None = None,
    seed: int = 42,
    batch_size: int = 256,
    num_workers: int = 8,
    wandb_enabled: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    supplemental_augmentation: bool = False,
    official_completion: bool = False,
    peak_lr: float | None = None,
    decay_steps: int | None = None,
    decay_lr: float | None = None,
    save_interval: int = 1_000,
    late_save_interval: int | None = 500,
    late_save_start_step: int | None = 20_000,
    max_checkpoints_to_keep: int = 30,
    max_resume_checkpoints_to_keep: int = 2,
):
    if num_train_steps <= 0:
        raise ValueError("num_train_steps must be explicitly positive")
    if warmup_steps < 0 or warmup_steps >= num_train_steps:
        raise ValueError(
            "warmup_steps must satisfy 0 <= warmup_steps < num_train_steps"
        )
    if not exp_name or exp_name.strip() != exp_name:
        raise ValueError("exp_name must be a non-empty, trimmed identifier")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size and num_workers are invalid")

    base = openpi_config.get_config(BASE_CONFIG_NAME)
    configured_base_weights = base_weight_path or os.environ.get(
        "FOUR_SUITE_BASE_WEIGHTS", base.pytorch_weight_path
    )
    if configured_base_weights is None:
        raise ValueError("a strict FP32-converted base weight path is required")
    configured_assets = libero_assets_dir or os.environ.get(
        "FOUR_SUITE_LIBERO_ASSETS", base.data.assets.assets_dir
    )
    if configured_assets is None:
        raise ValueError("a LIBERO normalization-assets directory is required")
    data = dataclasses.replace(
        base.data,
        assets=dataclasses.replace(
            base.data.assets,
            assets_dir=str(Path(configured_assets).expanduser().resolve()),
        ),
    )
    if supplemental_augmentation and official_completion:
        raise ValueError("supplemental and official-completion populations are mutually exclusive")
    policy_aux = FourSuitePolicyAuxTrainConfig(
        expected_episodes=(
            COMPLETED_EPISODES
            if official_completion
            else AUGMENTED_EPISODES if supplemental_augmentation else FOUR_SUITE_EPISODES
        ),
        expected_frames=(
            COMPLETED_FRAMES
            if official_completion
            else AUGMENTED_FRAMES if supplemental_augmentation else FOUR_SUITE_FRAMES
        ),
        supplemental_augmentation=supplemental_augmentation,
        official_completion=official_completion,
        target_scope=expected_target_scope(variant),
        mode="semantic_geometry_motion",
        policy_manifest_path=str(artifacts.policy_manifest.resolve()),
        episode_mapping_path=str(artifacts.episode_mapping.resolve()),
        geometry_target_index_path=str(artifacts.geometry_index.resolve()),
        geometry_normalization_path=str(artifacts.geometry_normalization.resolve()),
        motion_target_index_path=str(artifacts.motion_index.resolve()),
        motion_normalization_path=str(artifacts.motion_normalization.resolve()),
        motion_target_count=(
            COMPLETED_MOTION_VALID
            if official_completion
            else AUGMENTED_MOTION_VALID if supplemental_augmentation else FOUR_SUITE_MOTION_VALID
        ),
        lambda_sem=0.01,
        lambda_geo=0.05,
        lambda_motion=0.05,
        num_ground_queries=0,
        num_geometry_queries=8,
        num_motion_queries=8,
        lerobot_root=str(Path(lerobot_root).expanduser().resolve()),
        loss_coefficients_approved=True,
    )
    # The minimal manifest travels with the additive metadata bundle; large target
    # stores stay immutable in their source caches and are referenced by the indices.
    schedule = dataclasses.replace(
        base.lr_schedule,
        warmup_steps=warmup_steps,
        **({"peak_lr": peak_lr} if peak_lr is not None else {}),
        **({"decay_steps": decay_steps} if decay_steps is not None else {}),
        **({"decay_lr": decay_lr} if decay_lr is not None else {}),
    )
    return dataclasses.replace(
        base,
        name=CONFIG_NAMES[variant],
        exp_name=exp_name,
        data=data,
        policy_aux=policy_aux,
        pytorch_weight_path=str(Path(configured_base_weights).expanduser().resolve()),
        lr_schedule=schedule,
        checkpoint_base_dir=str(Path(checkpoint_base_dir).expanduser().resolve()),
        num_train_steps=num_train_steps,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        save_interval=save_interval,
        late_save_interval=late_save_interval,
        late_save_start_step=late_save_start_step,
        keep_period=None,
        checkpoint_keep_steps=(),
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        max_resume_checkpoints_to_keep=max_resume_checkpoints_to_keep,
        wandb_enabled=wandb_enabled,
        resume=resume,
        overwrite=overwrite,
    )
