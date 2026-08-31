"""Four-suite auxiliary target loader installed without changing OpenPI sources."""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openpi.training import policy_aux_dataset as upstream

from .constants import AUGMENTED_EPISODES
from .constants import AUGMENTED_FRAMES
from .constants import AUGMENTED_GEOMETRY_INVALID
from .constants import AUGMENTED_GEOMETRY_VALID
from .constants import AUGMENTED_MOTION_VALID
from .constants import COMPLETED_EPISODES
from .constants import COMPLETED_FRAMES
from .constants import COMPLETED_GEOMETRY_INVALID
from .constants import COMPLETED_GEOMETRY_VALID
from .constants import COMPLETED_MOTION_VALID
from .constants import FOUR_SUITE_EPISODES
from .constants import FOUR_SUITE_FRAMES
from .constants import FOUR_SUITE_GEOMETRY_INVALID
from .constants import FOUR_SUITE_GEOMETRY_VALID
from .constants import FOUR_SUITE_MOTION_VALID
from .constants import GEOMETRY_DIM
from .constants import LIBERO_REPO_ID
from .constants import LIBERO_REVISION
from .constants import MOTION_DIM

_UpstreamPolicyAuxTransformedDataset = upstream.PolicyAuxTransformedDataset


@dataclasses.dataclass(frozen=True)
class FourSuitePolicyAuxTrainConfig(upstream.PolicyAuxTrainConfig):
    """The existing model/data contract with an exact 40-task population."""

    expected_episodes: int = FOUR_SUITE_EPISODES
    expected_frames: int = FOUR_SUITE_FRAMES
    target_scope: str = "task_relevant"
    supplemental_augmentation: bool = False
    official_completion: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lerobot_task_indices is not None:
            raise ValueError("four-suite training requires all 40 tasks")
        if self.supplemental_augmentation and self.official_completion:
            raise ValueError("supplemental and official-completion populations are mutually exclusive")
        expected_episodes = (
            COMPLETED_EPISODES
            if self.official_completion
            else AUGMENTED_EPISODES if self.supplemental_augmentation else FOUR_SUITE_EPISODES
        )
        expected_frames = (
            COMPLETED_FRAMES
            if self.official_completion
            else AUGMENTED_FRAMES if self.supplemental_augmentation else FOUR_SUITE_FRAMES
        )
        expected_motion = (
            COMPLETED_MOTION_VALID
            if self.official_completion
            else AUGMENTED_MOTION_VALID if self.supplemental_augmentation else FOUR_SUITE_MOTION_VALID
        )
        if self.expected_episodes != expected_episodes or self.expected_frames != expected_frames:
            raise ValueError("four-suite population counts differ from the selected frozen population")
        if self.motion_target_count != expected_motion:
            raise ValueError(f"four-suite Motion count must be {expected_motion}")
        if (self.supplemental_augmentation or self.official_completion) and self.target_scope != "task_relevant":
            raise ValueError("the supplemental continuation is defined only for task-relevant targets")
        if self.target_scope not in ("task_relevant", "whole_scene"):
            raise ValueError(f"unsupported four-suite target scope: {self.target_scope}")

    def _validated_mapping_records(self) -> list[dict]:
        mapping = json.loads(Path(self.episode_mapping_path).read_text())
        records = mapping.get("episodes", [])
        if (
            mapping.get("status") != "PASS"
            or mapping.get("hf_repo_id") != LIBERO_REPO_ID
            or mapping.get("hf_revision") != LIBERO_REVISION
            or int(mapping.get("mapped_episode_count", -1)) != self.expected_episodes
            or int(mapping.get("mapped_frame_count", -1)) != self.expected_frames
            or len(records) != self.expected_episodes
            or sum(int(row["episode_length"]) for row in records) != self.expected_frames
        ):
            raise ValueError("four-suite episode mapping is not the frozen official population")
        records = sorted(records, key=lambda row: int(row["lerobot_episode_index"]))
        if [int(row["lerobot_episode_index"]) for row in records] != list(range(self.expected_episodes)):
            raise ValueError("four-suite episode mapping is not contiguous")
        return records

    def lerobot_episode_indices(self) -> list[int]:
        return [int(row["lerobot_episode_index"]) for row in self._validated_mapping_records()]

    def lerobot_dataset_indices(self) -> list[int]:
        indices: list[int] = []
        for row in self._validated_mapping_records():
            start = int(row["dataset_from_index"])
            stop = int(row["dataset_to_index_exclusive"])
            if stop - start != int(row["episode_length"]):
                raise ValueError(f"invalid dataset range in episode mapping: {row}")
            indices.extend(range(start, stop))
        if indices != list(range(self.expected_frames)):
            raise ValueError("four-suite dataset frame identities are not exactly contiguous")
        return indices


class FourSuiteAnnotationIndex:
    """Minimal direct lookup of stable sample identity and Semantic target."""

    def __init__(self, manifest_path: str | Path, *, expected_frames: int = FOUR_SUITE_FRAMES) -> None:
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        frame = pd.read_parquet(
            self.manifest_path,
            columns=["lerobot_dataset_index", "sample_id", "semantic_subtask"],
        ).sort_values("lerobot_dataset_index")
        if len(frame) != expected_frames or not frame["sample_id"].is_unique:
            raise ValueError("four-suite annotation manifest has the wrong frame population")
        if frame["lerobot_dataset_index"].astype(int).tolist() != list(range(expected_frames)):
            raise ValueError("four-suite annotation manifest is not in exact LeRobot order")
        if frame["semantic_subtask"].isna().any():
            raise ValueError("four-suite annotation manifest has missing Semantic targets")
        self._frame = frame.reset_index(drop=True)
        self.expected_frames = expected_frames

    def row_by_dataset_index(self, dataset_index: int) -> pd.Series:
        if dataset_index < 0 or dataset_index >= self.expected_frames:
            raise IndexError(dataset_index)
        row = self._frame.iloc[int(dataset_index)]
        if int(row["lerobot_dataset_index"]) != int(dataset_index):
            raise ValueError("annotation dataset index identity mismatch")
        return row


class FourSuiteGeometryTargetIndex:
    """Read multiple immutable Geometry memmaps through one ordered index."""

    def __init__(
        self,
        target_index_path: str | Path,
        normalization_path: str | Path,
        *,
        expected_frames: int = FOUR_SUITE_FRAMES,
        expected_valid: int = FOUR_SUITE_GEOMETRY_VALID,
        expected_invalid: int = FOUR_SUITE_GEOMETRY_INVALID,
    ) -> None:
        self.target_index_path = Path(target_index_path).resolve(strict=True)
        self.normalization_path = Path(normalization_path).resolve(strict=True)
        frame = pd.read_parquet(
            self.target_index_path,
            columns=[
                "lerobot_dataset_index",
                "sample_id",
                "geometry_valid",
                "target_memmap_path",
                "target_memmap_row",
                "target_dim",
                "target_dtype",
            ],
        ).sort_values("lerobot_dataset_index")
        if len(frame) != expected_frames or not frame["sample_id"].is_unique:
            raise ValueError("four-suite Geometry index has the wrong frame population")
        if frame["lerobot_dataset_index"].astype(int).tolist() != list(range(expected_frames)):
            raise ValueError("four-suite Geometry index is not in exact LeRobot order")
        valid = frame["geometry_valid"].astype(bool)
        if int(valid.sum()) != expected_valid or int((~valid).sum()) != expected_invalid:
            raise ValueError("four-suite Geometry validity counts differ")
        if frame.loc[valid, "target_memmap_path"].isna().any() or frame.loc[valid, "target_memmap_row"].isna().any():
            raise ValueError("valid four-suite Geometry rows lack target locations")
        if not frame.loc[valid, "target_dim"].eq(GEOMETRY_DIM).all():
            raise ValueError("four-suite Geometry target dimensions differ")
        if not frame.loc[valid, "target_dtype"].eq("float32").all():
            raise ValueError("four-suite Geometry target dtypes differ")
        self._frame = frame.reset_index(drop=True)
        self._targets: dict[str, np.ndarray] = {}
        self.expected_frames = expected_frames

        normalization = json.loads(self.normalization_path.read_text())
        if (
            normalization.get("status") != "PASS"
            or normalization.get("split") != "train"
            or int(normalization.get("sample_count", -1)) != expected_valid
            or int(normalization.get("feature_dim", -1)) != GEOMETRY_DIM
        ):
            raise ValueError("four-suite Geometry normalization metadata differs")
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        if self.mean.shape != (GEOMETRY_DIM,) or self.std.shape != (GEOMETRY_DIM,):
            raise ValueError("four-suite Geometry normalization shape differs")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("four-suite Geometry normalization is invalid")

    def _memmap(self, path_value: object) -> np.ndarray:
        path = str(Path(str(path_value)).resolve(strict=True))
        if path not in self._targets:
            targets = np.load(path, mmap_mode="r")
            if targets.ndim != 2 or targets.shape[1] != GEOMETRY_DIM or targets.dtype != np.float32:
                raise ValueError(f"unexpected Geometry memmap shape/dtype: {path}")
            self._targets[path] = targets
        return self._targets[path]

    def target_by_dataset_index(self, dataset_index: int) -> tuple[np.ndarray | None, bool, str]:
        if dataset_index < 0 or dataset_index >= self.expected_frames:
            raise IndexError(dataset_index)
        row = self._frame.iloc[int(dataset_index)]
        if int(row["lerobot_dataset_index"]) != int(dataset_index):
            raise ValueError("Geometry dataset index identity mismatch")
        sample_id = str(row["sample_id"])
        if not bool(row["geometry_valid"]):
            return None, False, sample_id
        targets = self._memmap(row["target_memmap_path"])
        target = np.asarray(targets[int(row["target_memmap_row"])], dtype=np.float32)
        if target.shape != (GEOMETRY_DIM,) or not np.isfinite(target).all():
            raise ValueError(f"invalid Geometry target for {sample_id}")
        return target, True, sample_id


class FourSuiteMotionTargetIndex:
    """Read one immutable Motion target memmap by stable sample ID."""

    def __init__(
        self,
        target_index_path: str | Path,
        normalization_path: str | Path,
        *,
        expected_count: int = FOUR_SUITE_MOTION_VALID,
    ) -> None:
        if expected_count <= 0:
            raise ValueError("Motion expected_count must be positive")
        self.target_index_path = Path(target_index_path).resolve(strict=True)
        self.normalization_path = Path(normalization_path).resolve(strict=True)
        frame = pd.read_parquet(
            self.target_index_path,
            columns=[
                "sample_id",
                "target_memmap_path",
                "target_memmap_row",
                "target_dim",
                "target_dtype",
            ],
        )
        if len(frame) != expected_count or not frame["sample_id"].is_unique:
            raise ValueError(f"Motion target index must cover {expected_count} unique valid samples")
        if not frame["target_dim"].eq(MOTION_DIM).all() or not frame["target_dtype"].eq("float32").all():
            raise ValueError("Motion targets must all be float32[256]")
        rows = frame["target_memmap_row"].astype(np.int64).to_numpy()
        if not np.array_equal(np.sort(rows), np.arange(expected_count, dtype=np.int64)):
            raise ValueError("Motion memmap rows must be a permutation of the valid population")
        paths = frame["target_memmap_path"].drop_duplicates().tolist()
        if len(paths) != 1:
            raise ValueError("Motion index must resolve to one immutable target memmap")
        memmap_path = Path(str(paths[0])).expanduser()
        if not memmap_path.is_absolute():
            memmap_path = self.target_index_path.parent / memmap_path
        self._memmap_path = memmap_path.resolve(strict=True)
        self._targets = np.load(self._memmap_path, mmap_mode="r")
        if self._targets.shape != (expected_count, MOTION_DIM) or self._targets.dtype != np.float32:
            raise ValueError("Unexpected Motion target memmap shape/dtype")
        self._row_by_sample_id = dict(zip(frame["sample_id"].astype(str), rows, strict=True))

        normalization = json.loads(self.normalization_path.read_text())
        if (
            int(normalization.get("count", -1)) != expected_count
            or normalization.get("dtype") != "float32"
            or int(normalization.get("feature_dim", -1)) != MOTION_DIM
            or normalization.get("finite") is not True
        ):
            raise ValueError("Motion train normalization metadata differs")
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        if self.mean.shape != (MOTION_DIM,) or self.std.shape != (MOTION_DIM,):
            raise ValueError("Motion normalization shape differs")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("Motion normalization is invalid")

    def target_by_sample_id(self, sample_id: str) -> tuple[np.ndarray | None, bool]:
        row = self._row_by_sample_id.get(sample_id)
        if row is None:
            return None, False
        target = np.asarray(self._targets[row], dtype=np.float32)
        if target.shape != (MOTION_DIM,) or not np.isfinite(target).all():
            raise ValueError(f"Invalid Motion target for {sample_id}")
        return target, True


class FourSuitePolicyAuxTargetIndex:
    """Join Semantic, Geometry, and Motion targets by immutable sample ID."""

    def __init__(self, config: FourSuitePolicyAuxTrainConfig) -> None:
        self.config = config
        expected_geometry_valid = (
            COMPLETED_GEOMETRY_VALID
            if config.official_completion
            else AUGMENTED_GEOMETRY_VALID if config.supplemental_augmentation
            else FOUR_SUITE_GEOMETRY_VALID
        )
        expected_geometry_invalid = (
            COMPLETED_GEOMETRY_INVALID
            if config.official_completion
            else AUGMENTED_GEOMETRY_INVALID if config.supplemental_augmentation
            else FOUR_SUITE_GEOMETRY_INVALID
        )
        expected_motion = (
            COMPLETED_MOTION_VALID
            if config.official_completion
            else AUGMENTED_MOTION_VALID if config.supplemental_augmentation
            else FOUR_SUITE_MOTION_VALID
        )
        self.annotations = FourSuiteAnnotationIndex(
            config.policy_manifest_path,
            expected_frames=config.expected_frames,
        )
        self.geometry = FourSuiteGeometryTargetIndex(
            config.geometry_target_index_path,
            config.geometry_normalization_path,
            expected_frames=config.expected_frames,
            expected_valid=expected_geometry_valid,
            expected_invalid=expected_geometry_invalid,
        )
        self.motion = (
            FourSuiteMotionTargetIndex(
                config.motion_target_index_path,
                config.motion_normalization_path,
                expected_count=expected_motion,
            )
            if config.mode in ("semantic_geometry_motion", "semantic_geometry_motion_binary_ground")
            else None
        )
        self.semantic_tokenizer = upstream.PolicySemanticTokenizer(config.semantic_max_target_len)

    def item(self, dataset_index: int) -> dict:
        row = self.annotations.row_by_dataset_index(int(dataset_index))
        sample_id = str(row["sample_id"])
        geometry, geometry_valid, geometry_sample_id = self.geometry.target_by_dataset_index(int(dataset_index))
        if sample_id != geometry_sample_id:
            raise ValueError(f"four-suite target identity mismatch at dataset index {dataset_index}")
        result = {
            "geometry": geometry if geometry is not None else np.zeros((GEOMETRY_DIM,), dtype=np.float32),
            "geometry_valid": np.asarray(geometry_valid, dtype=bool),
            "geometry_mean": self.geometry.mean,
            "geometry_std": self.geometry.std,
        }
        if self.motion is not None:
            motion, motion_valid = self.motion.target_by_sample_id(sample_id)
            result.update(
                {
                    "motion": motion if motion is not None else np.zeros((MOTION_DIM,), dtype=np.float32),
                    "motion_valid": np.asarray(motion_valid, dtype=bool),
                    "motion_mean": self.motion.mean,
                    "motion_std": self.motion.std,
                }
            )
        if self.config.mode in ("semantic_geometry", "semantic_geometry_motion"):
            semantic = self.semantic_tokenizer.fixed(str(row["semantic_subtask"]))
            result.update(
                {
                    "semantic_input_ids": semantic.input_ids,
                    "semantic_labels": semantic.labels,
                    "semantic_loss_mask": semantic.loss_mask,
                }
            )
        return result


class FourSuitePolicyAuxTransformedDataset(_UpstreamPolicyAuxTransformedDataset):
    """Spawn-safe dataset overlay for the portable four-suite target manifest."""

    def _make_target_index(self) -> FourSuitePolicyAuxTargetIndex:
        return FourSuitePolicyAuxTargetIndex(self.config)


def install_data_overlay() -> None:
    """Install target and dataset classes used by the four-suite entry point.

    The dataset subclass is intentional: unlike a module monkeypatch, its class
    identity survives serialization into PyTorch DataLoader spawn workers.
    """

    FourSuitePolicyAuxTargetIndex.__four_suite_overlay__ = True
    FourSuitePolicyAuxTransformedDataset.__four_suite_overlay__ = True
    upstream.PolicyAuxTargetIndex = FourSuitePolicyAuxTargetIndex
    upstream.PolicyAuxTransformedDataset = FourSuitePolicyAuxTransformedDataset


@functools.lru_cache(maxsize=1)
def overlay_identity() -> str:
    return "four_suite_policy_aux_overlay_v2_spawn_safe"
