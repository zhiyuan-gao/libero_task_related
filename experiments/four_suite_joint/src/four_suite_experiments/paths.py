"""Local path resolution with HPC-friendly environment overrides."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .constants import LIBERO_REVISION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]
TARGET_SCOPES = ("task_relevant", "whole_scene")


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class SourcePaths:
    target_scope: str
    openpi_root: Path
    lerobot_root: Path
    joint_manifest: Path
    geometry_indices: tuple[Path, ...]
    geometry_normalizations: tuple[Path, ...]
    motion_indices: tuple[Path, ...]
    motion_normalizations: tuple[Path, ...]
    artifact_dir: Path

    @classmethod
    def defaults(
        cls,
        artifact_dir: str | Path | None = None,
        *,
        target_scope: str = "task_relevant",
    ) -> SourcePaths:
        if target_scope not in TARGET_SCOPES:
            raise ValueError(f"unsupported target scope: {target_scope}")
        external_root = PROJECT_ROOT / "external_assets"
        data_root = _env_path(
            "FOUR_SUITE_ANNOTATION_ROOT",
            external_root / "annotation",
        )
        aux_root = data_root / "policy_aux_v1"
        if target_scope == "task_relevant":
            geometry_roots = (
                aux_root / "geometry_libero10",
                aux_root / "geometry_libero_goal_object_spatial_v1",
            )
            motion_roots = (
                aux_root / "motion_libero10_full_v1",
                aux_root / "motion_libero_goal_object_spatial_v1",
            )
        else:
            geometry_roots = (
                _env_path(
                    "FOUR_SUITE_WHOLE_SCENE_GEOMETRY_ROOT",
                    aux_root / "geometry_whole_scene_four_suite_v1",
                ),
            )
            motion_roots = (
                _env_path(
                    "FOUR_SUITE_WHOLE_SCENE_MOTION_ROOT",
                    aux_root / "motion_whole_scene_four_suite_v1",
                ),
            )
        if artifact_dir is None:
            scope_env = (
                "FOUR_SUITE_TASK_RELEVANT_ARTIFACT_DIR"
                if target_scope == "task_relevant"
                else "FOUR_SUITE_WHOLE_SCENE_ARTIFACT_DIR"
            )
            artifact_dir = os.environ.get(
                scope_env,
                os.environ.get(
                    "FOUR_SUITE_ARTIFACT_DIR",
                    str(PROJECT_ROOT / f"artifacts/{target_scope}"),
                ),
            )
        return cls(
            target_scope=target_scope,
            openpi_root=_env_path("OPENPI_ROOT", REPO_ROOT),
            lerobot_root=_env_path(
                "FOUR_SUITE_LEROBOT_ROOT",
                external_root / "hf/hub/datasets--physical-intelligence--libero/snapshots" / LIBERO_REVISION,
            ),
            joint_manifest=_env_path(
                "FOUR_SUITE_JOINT_MANIFEST",
                external_root / "runtime_metadata/four_suite_policy_geometry_manifest.parquet",
            ),
            geometry_indices=tuple(root / "target_index.parquet" for root in geometry_roots),
            geometry_normalizations=tuple(root / "normalization/train_mean_std.json" for root in geometry_roots),
            motion_indices=tuple(root / "index.parquet" for root in motion_roots),
            motion_normalizations=tuple(root / "target_statistics_train.json" for root in motion_roots),
            artifact_dir=Path(artifact_dir).expanduser().resolve(),
        )

    def required_sources(self) -> tuple[Path, ...]:
        return (
            self.openpi_root,
            self.lerobot_root,
            self.joint_manifest,
            *self.geometry_indices,
            *self.geometry_normalizations,
            *self.motion_indices,
            *self.motion_normalizations,
        )


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def policy_manifest(self) -> Path:
        return self.root / "policy_manifest.parquet"

    @property
    def episode_mapping(self) -> Path:
        return self.root / "episode_mapping.json"

    @property
    def geometry_index(self) -> Path:
        return self.root / "geometry_index.parquet"

    @property
    def geometry_normalization(self) -> Path:
        return self.root / "geometry_normalization.json"

    @property
    def motion_index(self) -> Path:
        return self.root / "motion_index.parquet"

    @property
    def motion_targets(self) -> Path:
        return self.root / "motion_targets_fp32.npy"

    @property
    def motion_normalization(self) -> Path:
        return self.root / "motion_normalization.json"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    def all_files(self) -> tuple[Path, ...]:
        return (
            self.policy_manifest,
            self.episode_mapping,
            self.geometry_index,
            self.geometry_normalization,
            self.motion_index,
            self.motion_targets,
            self.motion_normalization,
            self.provenance,
        )
