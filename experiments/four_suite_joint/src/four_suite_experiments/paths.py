"""Local path resolution with HPC-friendly environment overrides."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .constants import LIBERO_REVISION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class SourcePaths:
    openpi_root: Path
    lerobot_root: Path
    joint_manifest: Path
    geometry_indices: tuple[Path, Path]
    geometry_normalizations: tuple[Path, Path]
    motion_indices: tuple[Path, Path]
    motion_normalizations: tuple[Path, Path]
    artifact_dir: Path

    @classmethod
    def defaults(cls, artifact_dir: str | Path | None = None) -> SourcePaths:
        data_root = _env_path(
            "FOUR_SUITE_ANNOTATION_ROOT",
            "/workspace/vla/p3/workspace/data/libero_four_suite_annotation",
        )
        aux_root = data_root / "policy_aux_v1"
        return cls(
            openpi_root=_env_path("OPENPI_ROOT", REPO_ROOT),
            lerobot_root=_env_path(
                "FOUR_SUITE_LEROBOT_ROOT",
                "/workspace/vla/p3/workspace/cache/huggingface/hub/"
                f"datasets--physical-intelligence--libero/snapshots/{LIBERO_REVISION}",
            ),
            joint_manifest=_env_path(
                "FOUR_SUITE_JOINT_MANIFEST",
                "/workspace/vla/p3/runtime_metadata/four_suite_policy_geometry_manifest.parquet",
            ),
            geometry_indices=(
                aux_root / "geometry_libero10/target_index.parquet",
                aux_root
                / "geometry_libero_goal_object_spatial_v1/target_index.parquet",
            ),
            geometry_normalizations=(
                aux_root / "geometry_libero10/normalization/train_mean_std.json",
                aux_root
                / "geometry_libero_goal_object_spatial_v1/normalization/train_mean_std.json",
            ),
            motion_indices=(
                aux_root / "motion_libero10_full_v1/index.parquet",
                aux_root / "motion_libero_goal_object_spatial_v1/index.parquet",
            ),
            motion_normalizations=(
                aux_root / "motion_libero10_full_v1/target_statistics_train.json",
                aux_root
                / "motion_libero_goal_object_spatial_v1/target_statistics_train.json",
            ),
            artifact_dir=(
                Path(artifact_dir).expanduser().resolve()
                if artifact_dir is not None
                else _env_path(
                    "FOUR_SUITE_ARTIFACT_DIR",
                    PROJECT_ROOT / "artifacts/task_relevant",
                )
            ),
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
            self.motion_normalization,
            self.provenance,
        )
