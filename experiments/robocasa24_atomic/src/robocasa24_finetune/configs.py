"""Construct the three matched RoboCasa Atomic-24 training variants."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np
from safetensors import safe_open

from openpi.models import pi0_config
from openpi.training import config as openpi_config
from openpi.training import optimizer

from .auxiliary import PreparedAuxiliaryPaths
from .auxiliary import RoboCasaAuxTrainConfig
from .auxiliary import require_prepared_target_scope
from .constants import ACTION_HORIZON
from .constants import DATASET_REPO_ID
from .constants import EXECUTION_HORIZON
from .constants import MODEL_ACTION_DIM
from .constants import TASKS
from .integration import RoboCasaDataConfigFactory

Variant = Literal["baseline", "task_relevant", "whole_scene"]


def _validate_fp32_base(base_weight_dir: Path) -> None:
    checkpoint = base_weight_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    non_fp32: list[tuple[str, str]] = []
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"empty base checkpoint: {checkpoint}")
        for key in keys:
            dtype = str(handle.get_slice(key).get_dtype())
            if dtype != "F32":
                non_fp32.append((key, dtype))
                if len(non_fp32) == 8:
                    break
    if non_fp32:
        raise ValueError(
            "RoboCasa requires the FP32-converted pi0.5 base; "
            f"found non-F32 tensors: {non_fp32}"
        )


def _aux_config(variant: Variant, artifact_dir: Path) -> RoboCasaAuxTrainConfig | None:
    if variant == "baseline":
        return None
    scope = "task_relevant" if variant == "task_relevant" else "whole_scene"
    paths = PreparedAuxiliaryPaths(artifact_dir.resolve(strict=True))
    require_prepared_target_scope(paths, scope)
    motion_valid = np.load(paths.motion_valid, mmap_mode="r")
    motion_count = int(motion_valid.sum())
    return RoboCasaAuxTrainConfig(
        mode="semantic_geometry_motion",
        artifact_dir=str(paths.root),
        target_scope=scope,
        geometry_target_index_path=str(paths.index),
        geometry_normalization_path=str(paths.geometry_normalization),
        motion_target_index_path=str(paths.index),
        motion_normalization_path=str(paths.motion_normalization),
        motion_target_count=motion_count,
        policy_manifest_path=str(paths.index),
        episode_mapping_path=str(paths.index),
    )


def build_train_config(
    *,
    variant: Variant,
    exp_name: str,
    data_root: str | Path,
    manifest_root: str | Path,
    policy_assets_root: str | Path,
    artifact_dir: str | Path | None,
    base_weight_dir: str | Path,
    checkpoint_base_dir: str | Path,
    seed: int = 42,
    global_micro_batch: int = 128,
    gradient_accumulation_steps: int = 1,
    num_workers: int = 2,
    num_train_steps: int = 30_000,
    warmup_steps: int = 10_000,
    save_interval: int = 1_000,
    max_checkpoints_to_keep: int | None = 4,
    resume: bool = False,
    overwrite: bool = False,
    wandb_enabled: bool = True,
    tasks: tuple[str, ...] = TASKS,
) -> openpi_config.TrainConfig:
    if variant not in ("baseline", "task_relevant", "whole_scene"):
        raise ValueError(f"unsupported RoboCasa variant: {variant}")
    if global_micro_batch <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("micro-batch and accumulation must be positive")
    if num_workers < 0 or save_interval <= 0:
        raise ValueError("worker count/save interval is invalid")
    if max_checkpoints_to_keep is not None and max_checkpoints_to_keep <= 0:
        raise ValueError("checkpoint retention must be positive when set")
    if global_micro_batch * gradient_accumulation_steps != 128:
        raise ValueError("effective global batch must remain exactly 128")
    if global_micro_batch % 8 != 0:
        raise ValueError("global micro-batch must be divisible by eight GPUs")
    if num_train_steps != 30_000 or warmup_steps != 10_000:
        raise ValueError("formal RoboCasa recipe is frozen at 30k updates / 10k warmup")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if variant != "baseline" and artifact_dir is None:
        raise ValueError("an auxiliary artifact directory is required")
    tasks = tuple(tasks)
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in TASKS for task in tasks)
    ):
        raise ValueError("invalid RoboCasa training task subset")

    base_weight_dir = Path(base_weight_dir).resolve(strict=True)
    _validate_fp32_base(base_weight_dir)
    policy_assets_root = Path(policy_assets_root).resolve(strict=True)
    data_factory = RoboCasaDataConfigFactory(
        repo_id=DATASET_REPO_ID,
        assets=openpi_config.AssetsConfig(
            assets_dir=str(policy_assets_root),
            asset_id=DATASET_REPO_ID,
        ),
        data_root=str(Path(data_root).resolve(strict=True)),
        manifest_root=str(Path(manifest_root).resolve(strict=True)),
        tasks=tasks,
        sampling_seed=seed,
    )
    aux = _aux_config(variant, Path(artifact_dir)) if artifact_dir is not None else None
    # Reuse the reviewed PyTorch trainer/checkpointer fields, replacing every
    # scientific field that differs from LIBERO explicitly below.
    # Use a committed auxiliary-query carrier only for TrainConfig defaults;
    # every scientific RoboCasa field is replaced explicitly below. This keeps
    # the experiment independent of any later LIBERO-only named config.
    base = openpi_config.get_config("pi05_libero3_p3_binary_ground_aux")
    population_name = "robocasa24" if tasks == TASKS else f"robocasa{len(tasks)}"
    return dataclasses.replace(
        base,
        name=f"pi05_{population_name}_{variant}",
        project_name="robocasa24-pi05",
        exp_name=exp_name,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=MODEL_ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            discrete_state_input=False,
            pytorch_compile_mode=None,
        ),
        data=data_factory,
        policy_aux=aux,
        pytorch_weight_path=str(base_weight_dir),
        pytorch_training_precision="bfloat16",
        batch_size=global_micro_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_workers=num_workers,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=seed,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=save_interval,
        late_save_interval=None,
        late_save_start_step=None,
        save_final_checkpoint=True,
        keep_period=None,
        checkpoint_keep_steps=(),
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        max_resume_checkpoints_to_keep=max_checkpoints_to_keep,
        checkpoint_base_dir=str(Path(checkpoint_base_dir).resolve()),
        policy_metadata={
            "benchmark": "RoboCasa Atomic-24",
            "variant": variant,
            "tasks": len(tasks),
            "task_names": list(tasks),
            "episodes_per_task": 50,
            "task_sampling": "P(task) proportional to task_frames^0.4",
            "episode_sampling": "uniform within task",
            "timestep_sampling": "uniform within episode",
            "normalization_task_weighting": "equal task weights over raw frames",
            "policy_views": 3,
            "raw_state_dim": 16,
            "raw_action_dim": 12,
            "predicted_action_horizon": ACTION_HORIZON,
            "execution_horizon": EXECUTION_HORIZON,
            "base_checkpoint": "official pi05_base FP32-converted PyTorch",
        },
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
        fsdp_devices=1,
    )
