"""Build matched 30k-update configs without changing the main experiment code."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from openpi.training import config as openpi_config

from ..auxiliary import PreparedAuxiliaryPaths
from ..configs import build_train_config
from ..constants import TASKS
from .specs import AblationVariant
from .specs import get_ablation_spec


@dataclasses.dataclass(frozen=True)
class AblationAuxTrainConfig:
    """Trainer-facing auxiliary contract for one isolated ablation."""

    ablation_variant: AblationVariant
    action_conditioning: tuple[str, ...]
    mode: str
    artifact_dir: str
    target_scope: str
    geometry_target_index_path: str
    geometry_normalization_path: str
    motion_target_index_path: str
    motion_normalization_path: str
    motion_target_count: int
    lambda_geo: float | None
    lambda_sem: float | None
    lambda_motion: float | None
    lambda_ground: float | None = None
    semantic_max_target_len: int = 32
    num_ground_queries: int = 0
    num_geometry_queries: int = 8
    num_motion_queries: int = 0
    ground_mask_dim: int = 256
    ground_focal_alpha: float = 0.25
    ground_focal_gamma: float = 2.0
    ground_objective: str = "coverage_focal_dice"
    ground_positive_weight: float | None = None
    loss_coefficients_approved: bool = True
    diagnostic_skip_semantic_lm: bool = False
    query_topology: str = "independent"

    # Compatibility fields are consumed only by the shared loader signature;
    # the process-local RoboCasa dataset dispatch never selects LeRobot data.
    policy_manifest_path: str = ""
    episode_mapping_path: str = ""
    lerobot_revision: str = "robocasa24-base50"
    lerobot_root: str | None = None
    lerobot_task_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        spec = get_ablation_spec(self.ablation_variant)
        expected = {
            "action_conditioning": spec.action_conditioning,
            "mode": spec.internal_mode,
            "target_scope": spec.target_scope,
            "lambda_geo": spec.lambda_geo,
            "lambda_sem": spec.lambda_sem,
            "lambda_motion": spec.lambda_motion,
            "lambda_ground": None,
            "num_ground_queries": 8 if spec.internal_mode == "geometry" else 0,
            "num_geometry_queries": 8,
            "num_motion_queries": (
                8 if spec.internal_mode == "semantic_geometry_motion" else 0
            ),
            "query_topology": "independent",
        }
        observed = {name: getattr(self, name) for name in expected}
        if observed != expected:
            raise ValueError(
                f"{self.ablation_variant}: config differs from its frozen spec; "
                f"expected={expected}, observed={observed}"
            )
        if not self.loss_coefficients_approved:
            raise ValueError("ablation loss coefficients must be explicitly frozen")
        if self.ground_objective != "coverage_focal_dice" or self.ground_positive_weight is not None:
            raise ValueError("RoboCasa ablations do not include Ground supervision")
        if self.diagnostic_skip_semantic_lm:
            raise ValueError("Semantic must not use a diagnostic shortcut")
        if self.motion_target_count <= 0:
            raise ValueError("prepared Motion population must be non-empty")
        root = Path(self.artifact_dir).resolve(strict=True)
        for value in (
            self.geometry_target_index_path,
            self.geometry_normalization_path,
            self.motion_target_index_path,
            self.motion_normalization_path,
        ):
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError(path)


def _ablation_aux_config(
    variant: AblationVariant,
    artifact_dir: Path,
    motion_target_count: int,
) -> AblationAuxTrainConfig:
    spec = get_ablation_spec(variant)
    paths = PreparedAuxiliaryPaths(artifact_dir.resolve(strict=True))
    return AblationAuxTrainConfig(
        ablation_variant=variant,
        action_conditioning=spec.action_conditioning,
        mode=spec.internal_mode,
        artifact_dir=str(paths.root),
        target_scope=spec.target_scope,
        geometry_target_index_path=str(paths.index),
        geometry_normalization_path=str(paths.geometry_normalization),
        motion_target_index_path=str(paths.index),
        motion_normalization_path=str(paths.motion_normalization),
        motion_target_count=motion_target_count,
        lambda_geo=spec.lambda_geo,
        lambda_sem=spec.lambda_sem,
        lambda_motion=spec.lambda_motion,
        num_ground_queries=8 if spec.internal_mode == "geometry" else 0,
        num_geometry_queries=8,
        num_motion_queries=(
            8 if spec.internal_mode == "semantic_geometry_motion" else 0
        ),
        policy_manifest_path=str(paths.index),
        episode_mapping_path=str(paths.index),
    )


def build_ablation_train_config(
    *,
    ablation: AblationVariant,
    exp_name: str,
    data_root: str | Path,
    manifest_root: str | Path,
    policy_assets_root: str | Path,
    artifact_dir: str | Path,
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
    """Reuse every main-recipe field, then replace only the ablation contract."""

    spec = get_ablation_spec(ablation)
    scope_variant = "whole_scene" if spec.target_scope == "whole_scene" else "task_relevant"
    shared = build_train_config(
        variant=scope_variant,
        exp_name=exp_name,
        data_root=data_root,
        manifest_root=manifest_root,
        policy_assets_root=policy_assets_root,
        artifact_dir=artifact_dir,
        base_weight_dir=base_weight_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        seed=seed,
        global_micro_batch=global_micro_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        warmup_steps=warmup_steps,
        save_interval=save_interval,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
        tasks=tasks,
    )
    if shared.policy_aux is None:
        raise AssertionError("shared auxiliary config was not constructed")
    aux = _ablation_aux_config(
        ablation,
        Path(artifact_dir),
        motion_target_count=int(shared.policy_aux.motion_target_count),
    )
    metadata = dict(shared.policy_metadata)
    metadata.update(
        variant=f"ablation:{ablation}",
        ablation_variant=ablation,
        ablation_display_name=spec.display_name,
        target_scope=spec.target_scope,
        semantic_enabled=spec.semantic_enabled,
        geometry_enabled=spec.geometry_enabled,
        motion_enabled=spec.motion_enabled,
        action_conditioning=list(spec.action_conditioning),
        lambda_geo=spec.lambda_geo,
        lambda_sem=spec.lambda_sem,
        lambda_motion=spec.lambda_motion,
    )
    population_name = "robocasa24" if tuple(tasks) == TASKS else f"robocasa{len(tasks)}"
    return dataclasses.replace(
        shared,
        name=f"pi05_{population_name}_ablation_{ablation}",
        project_name="robocasa24-pi05-ablations",
        policy_aux=aux,
        policy_metadata=metadata,
    )
