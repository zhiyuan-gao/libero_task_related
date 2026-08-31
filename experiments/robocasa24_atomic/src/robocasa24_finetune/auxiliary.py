"""Portable, memory-mapped auxiliary targets aligned to policy frame order."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .constants import GEOMETRY_DIM
from .constants import MOTION_DIM

TargetScope = Literal["task_relevant", "whole_scene"]


@dataclasses.dataclass(frozen=True)
class RoboCasaAuxTrainConfig:
    """Model/loss fields plus paths for one prepared Atomic-24 target scope."""

    mode: str
    artifact_dir: str
    target_scope: TargetScope
    geometry_target_index_path: str
    geometry_normalization_path: str
    motion_target_index_path: str
    motion_normalization_path: str
    motion_target_count: int
    lambda_geo: float = 0.05
    lambda_sem: float = 0.01
    lambda_motion: float = 0.05
    lambda_ground: float | None = None
    semantic_max_target_len: int = 32
    num_ground_queries: int = 0
    num_geometry_queries: int = 8
    num_motion_queries: int = 8
    ground_mask_dim: int = 256
    ground_focal_alpha: float = 0.25
    ground_focal_gamma: float = 2.0
    ground_objective: str = "coverage_focal_dice"
    ground_positive_weight: float | None = None
    loss_coefficients_approved: bool = True
    diagnostic_skip_semantic_lm: bool = False
    query_topology: str = "independent"

    # Compatibility-only names. The RoboCasa overlay never invokes LIBERO
    # episode selection or consumes either path.
    policy_manifest_path: str = ""
    episode_mapping_path: str = ""
    lerobot_revision: str = "robocasa24-base50"
    lerobot_root: str | None = None
    lerobot_task_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.mode != "semantic_geometry_motion":
            raise ValueError("RoboCasa main/ablation requires Semantic+Geometry+Motion")
        if self.target_scope not in ("task_relevant", "whole_scene"):
            raise ValueError(f"unsupported target scope: {self.target_scope}")
        if (
            self.lambda_geo != 0.05
            or self.lambda_sem != 0.01
            or self.lambda_motion != 0.05
            or self.lambda_ground is not None
        ):
            raise ValueError(
                "RoboCasa auxiliary loss coefficients differ from the frozen method"
            )
        if (
            self.num_geometry_queries,
            self.num_motion_queries,
            self.num_ground_queries,
        ) != (8, 8, 0):
            raise ValueError("RoboCasa query counts differ from the frozen method")
        if self.query_topology != "independent":
            raise ValueError("RoboCasa query topology must remain independent")
        if self.motion_target_count <= 0:
            raise ValueError("motion_target_count must be positive")
        root = Path(self.artifact_dir)
        for value in (
            self.geometry_target_index_path,
            self.geometry_normalization_path,
            self.motion_target_index_path,
            self.motion_normalization_path,
        ):
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                raise FileNotFoundError(path)


@dataclasses.dataclass(frozen=True)
class PreparedAuxiliaryPaths:
    root: Path

    @property
    def index(self) -> Path:
        return self.root / "index.parquet"

    @property
    def semantic_input_ids(self) -> Path:
        return self.root / "semantic_input_ids.npy"

    @property
    def semantic_labels(self) -> Path:
        return self.root / "semantic_labels.npy"

    @property
    def semantic_loss_mask(self) -> Path:
        return self.root / "semantic_loss_mask.npy"

    @property
    def geometry_targets(self) -> Path:
        return self.root / "geometry_targets.npy"

    @property
    def geometry_valid(self) -> Path:
        return self.root / "geometry_valid.npy"

    @property
    def geometry_normalization(self) -> Path:
        return self.root / "geometry_normalization.json"

    @property
    def motion_targets(self) -> Path:
        return self.root / "motion_targets.npy"

    @property
    def motion_valid(self) -> Path:
        return self.root / "motion_valid.npy"

    @property
    def motion_normalization(self) -> Path:
        return self.root / "motion_normalization.json"


def require_prepared_target_scope(
    paths: PreparedAuxiliaryPaths, expected_scope: TargetScope
) -> None:
    """Reject a prepared cache whose scientific target scope is mismatched."""

    report_path = paths.root / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_scope = str(payload.get("target_scope", ""))
    if payload.get("status") != "PASS" or report_scope != expected_scope:
        raise ValueError(
            "prepared artifact report target scope differs: "
            f"expected={expected_scope}, observed={report_scope!r}"
        )
    index_scopes = set(
        pd.read_parquet(paths.index, columns=["target_scope"])["target_scope"].astype(str)
    )
    if index_scopes != {expected_scope}:
        raise ValueError(
            "prepared artifact index target scope differs: "
            f"expected={expected_scope}, observed={sorted(index_scopes)}"
        )


def _normalization(path: Path, expected_dim: int) -> tuple[np.ndarray, np.ndarray, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = int(payload.get("count", -1))
    dim = int(payload.get("feature_dim", -1))
    mean = np.asarray(payload.get("mean"), dtype=np.float32)
    std = np.asarray(payload.get("std"), dtype=np.float32)
    if (
        count <= 0
        or dim != expected_dim
        or mean.shape != (expected_dim,)
        or std.shape != (expected_dim,)
    ):
        raise ValueError(f"invalid target normalization metadata: {path}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError(f"non-finite target normalization metadata: {path}")
    return mean, std, count


def _validate_prepared_index(
    paths: PreparedAuxiliaryPaths, expected_sample_ids: Sequence[str] | None
) -> tuple[int, str]:
    if expected_sample_ids is None:
        report_path = paths.root / "report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        length = int(payload.get("frame_count", -1))
        scope = str(payload.get("target_scope", ""))
        if (
            payload.get("status") != "PASS"
            or length <= 0
            or scope not in {"task_relevant", "whole_scene"}
        ):
            raise ValueError(f"invalid prepared artifact report: {report_path}")
        return length, scope

    index = pd.read_parquet(
        paths.index, columns=["dataset_index", "sample_id", "target_scope"]
    ).sort_values("dataset_index")
    actual_indices = index["dataset_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_indices, np.arange(len(index), dtype=np.int64)):
        raise ValueError("prepared RoboCasa target index is not contiguous")
    if len(index) != len(expected_sample_ids):
        raise ValueError("prepared auxiliary/policy frame counts differ")
    actual_ids = index["sample_id"].astype(str).to_numpy()
    for start in range(0, len(index), 8192):
        stop = min(start + 8192, len(index))
        expected = np.asarray(
            [expected_sample_ids[i] for i in range(start, stop)], dtype=str
        )
        if not np.array_equal(actual_ids[start:stop], expected):
            raise ValueError(
                "prepared auxiliary targets do not match policy sample order"
            )
    scopes = set(index["target_scope"].astype(str))
    if len(scopes) != 1:
        raise ValueError("prepared target scope is missing or mixed")
    scope = scopes.pop()
    if scope not in {"task_relevant", "whole_scene"}:
        raise ValueError("prepared target scope is invalid")
    return len(index), scope


def _resolve_prepared_rows(
    paths: PreparedAuxiliaryPaths, expected_sample_ids: Sequence[str]
) -> np.ndarray:
    """Resolve an ordered policy subset onto immutable full-artifact rows."""

    index = pd.read_parquet(
        paths.index, columns=["dataset_index", "sample_id", "target_scope"]
    ).sort_values("dataset_index")
    actual_indices = index["dataset_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_indices, np.arange(len(index), dtype=np.int64)):
        raise ValueError("prepared RoboCasa target index is not contiguous")
    actual_ids = pd.Index(index["sample_id"].astype(str))
    if not actual_ids.is_unique:
        raise ValueError("prepared RoboCasa target sample IDs are not unique")
    expected_ids = pd.Index([str(value) for value in expected_sample_ids])
    if not expected_ids.is_unique:
        raise ValueError("policy subset sample IDs are not unique")
    rows = actual_ids.get_indexer(expected_ids)
    if np.any(rows < 0):
        missing = expected_ids[rows < 0][:8].tolist()
        raise ValueError(
            f"policy subset contains samples absent from prepared targets: {missing}"
        )
    scopes = set(index["target_scope"].astype(str))
    if len(scopes) != 1 or next(iter(scopes)) not in {
        "task_relevant",
        "whole_scene",
    }:
        raise ValueError("prepared target scope is missing, invalid, or mixed")
    return np.asarray(rows, dtype=np.int64)


class PreparedAuxiliaryStore:
    """Lazy per-worker memory maps for one completely aligned target scope."""

    def __init__(
        self, root: str | Path, expected_sample_ids: Sequence[str] | None = None
    ) -> None:
        self.paths = PreparedAuxiliaryPaths(Path(root).resolve(strict=True))
        self.length, self.target_scope = _validate_prepared_index(
            self.paths, expected_sample_ids
        )
        self.semantic_input_ids = np.load(self.paths.semantic_input_ids, mmap_mode="r")
        self.semantic_labels = np.load(self.paths.semantic_labels, mmap_mode="r")
        self.semantic_loss_mask = np.load(self.paths.semantic_loss_mask, mmap_mode="r")
        self.geometry_targets = np.load(self.paths.geometry_targets, mmap_mode="r")
        self.geometry_valid = np.load(self.paths.geometry_valid, mmap_mode="r")
        self.motion_targets = np.load(self.paths.motion_targets, mmap_mode="r")
        self.motion_valid = np.load(self.paths.motion_valid, mmap_mode="r")
        expected_shapes = {
            "semantic_input_ids": (self.length, 31),
            "semantic_labels": (self.length, 32),
            "semantic_loss_mask": (self.length, 32),
            "geometry_targets": (self.length, GEOMETRY_DIM),
            "geometry_valid": (self.length,),
            "motion_targets": (self.length, MOTION_DIM),
            "motion_valid": (self.length,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"prepared {name} shape differs: {getattr(self, name).shape} != {shape}"
                )
        if (
            self.semantic_input_ids.dtype != np.int64
            or self.semantic_labels.dtype != np.int64
        ):
            raise ValueError("prepared Semantic token dtype differs")
        if self.semantic_loss_mask.dtype != np.bool_:
            raise ValueError("prepared Semantic mask dtype differs")
        if (
            self.geometry_targets.dtype != np.float32
            or self.motion_targets.dtype != np.float32
        ):
            raise ValueError("prepared teacher target dtype differs")
        if self.geometry_valid.dtype != np.bool_ or self.motion_valid.dtype != np.bool_:
            raise ValueError("prepared target validity dtype differs")
        self.geometry_mean, self.geometry_std, geometry_count = _normalization(
            self.paths.geometry_normalization, GEOMETRY_DIM
        )
        self.motion_mean, self.motion_std, motion_count = _normalization(
            self.paths.motion_normalization, MOTION_DIM
        )
        if geometry_count != int(self.geometry_valid.sum()):
            raise ValueError("Geometry normalization/valid counts differ")
        if motion_count != int(self.motion_valid.sum()):
            raise ValueError("Motion normalization/valid counts differ")

    def item(self, index: int) -> dict:
        i = int(index)
        if i < 0 or i >= self.length:
            raise IndexError(i)
        geometry_valid = bool(self.geometry_valid[i])
        motion_valid = bool(self.motion_valid[i])
        geometry = np.asarray(self.geometry_targets[i], dtype=np.float32)
        motion = np.asarray(self.motion_targets[i], dtype=np.float32)
        if (not geometry_valid and np.any(geometry)) or (
            not motion_valid and np.any(motion)
        ):
            raise ValueError(
                "invalid prepared target row is not the required zero vector"
            )
        return {
            "semantic_input_ids": np.asarray(
                self.semantic_input_ids[i], dtype=np.int64
            ),
            "semantic_labels": np.asarray(self.semantic_labels[i], dtype=np.int64),
            "semantic_loss_mask": np.asarray(self.semantic_loss_mask[i], dtype=bool),
            "geometry": geometry,
            "geometry_valid": np.asarray(geometry_valid, dtype=bool),
            "geometry_mean": self.geometry_mean,
            "geometry_std": self.geometry_std,
            "motion": motion,
            "motion_valid": np.asarray(motion_valid, dtype=bool),
            "motion_mean": self.motion_mean,
            "motion_std": self.motion_std,
        }


class RoboCasaPolicyAuxTransformedDataset:
    """Attach prepared teacher targets after policy transforms."""

    def __init__(self, dataset, config: RoboCasaAuxTrainConfig) -> None:
        self.dataset = dataset
        self.config = config
        source = dataset
        while not hasattr(source, "sample_ids"):
            if hasattr(source, "dataset"):
                source = source.dataset
            elif hasattr(source, "_dataset"):
                source = source._dataset  # noqa: SLF001 - unwrap upstream wrapper
            else:
                break
        if not hasattr(source, "sample_ids"):
            raise TypeError("RoboCasa base dataset does not expose stable sample IDs")
        if not hasattr(source, "resolve_sample_index"):
            raise TypeError("RoboCasa base dataset does not expose sampled raw rows")
        self._source = source
        sample_ids = source.sample_ids
        if len(self.dataset) != len(sample_ids):
            raise ValueError(
                "transformed policy dataset/sample identity lengths differ"
            )
        # Resolve the ordered policy population onto immutable full-artifact
        # rows once in each DDP parent. This is identity for the 24-task run and
        # a strict sample-ID mapping for a reviewed task subset.
        paths = PreparedAuxiliaryPaths(Path(config.artifact_dir).resolve(strict=True))
        self._artifact_rows = _resolve_prepared_rows(paths, sample_ids)
        if len(self._artifact_rows) != len(self.dataset):
            raise ValueError("prepared subset mapping/policy lengths differ")
        self._store: PreparedAuxiliaryStore | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        if self._store is None:
            self._store = PreparedAuxiliaryStore(self.config.artifact_dir)
        item = self.dataset[int(index)]
        if "policy_aux" in item:
            raise ValueError("base RoboCasa dataset unexpectedly contains policy_aux")
        source_row = int(self._source.resolve_sample_index(int(index)))
        artifact_row = int(self._artifact_rows[source_row])
        return {**item, "policy_aux": self._store.item(artifact_row)}
