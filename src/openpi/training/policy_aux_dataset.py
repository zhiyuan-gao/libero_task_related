"""Stable LIBERO-10 auxiliary-target lookup utilities for P1/P2 training."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from openpi.models import tokenizer as _tokenizer


@dataclasses.dataclass(frozen=True)
class SemanticTeacherTensors:
    input_ids: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray


PolicyAuxMode = Literal["geometry", "ground_geometry_semantic_lm"]

CANONICAL_LIBERO_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
CANONICAL_LIBERO_EPISODES = 379
CANONICAL_LIBERO_FRAMES = 101_469


@dataclasses.dataclass(frozen=True)
class PolicyAuxTrainConfig:
    """Explicit P1/P2 target/model configuration used by the PyTorch trainer."""

    mode: PolicyAuxMode
    policy_manifest_path: str
    episode_mapping_path: str
    geometry_target_index_path: str
    geometry_normalization_path: str
    lambda_geo: float | None = None
    lambda_sem: float | None = None
    lambda_ground: float | None = None
    semantic_max_target_len: int = 32
    num_ground_queries: int = 8
    num_geometry_queries: int = 8
    ground_mask_dim: int = 256
    ground_focal_alpha: float = 0.25
    ground_focal_gamma: float = 2.0
    lerobot_revision: str = CANONICAL_LIBERO_REVISION
    lerobot_root: str | None = None
    loss_coefficients_approved: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("geometry", "ground_geometry_semantic_lm"):
            raise ValueError(f"Unsupported policy auxiliary training mode: {self.mode}")
        required_lambdas = ["lambda_geo"]
        if self.mode == "ground_geometry_semantic_lm":
            required_lambdas.extend(("lambda_sem", "lambda_ground"))
        if self.loss_coefficients_approved and any(getattr(self, name) is None for name in required_lambdas):
            raise ValueError(f"Approved P1/P2 config is missing a required loss coefficient: {required_lambdas}")
        if self.loss_coefficients_approved and any(getattr(self, name) <= 0 for name in required_lambdas):
            raise ValueError("Approved P1/P2 loss coefficients must be strictly positive")
        for name in ("lambda_geo", "lambda_sem", "lambda_ground"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.num_ground_queries != 8 or self.num_geometry_queries != 8:
            raise ValueError("P1/P2 v0 query counts are frozen at eight")
        if self.lerobot_revision != CANONICAL_LIBERO_REVISION:
            raise ValueError(
                f"P1/P2 policy training is frozen to official LeRobot revision {CANONICAL_LIBERO_REVISION}"
            )
        if self.lerobot_root is not None and Path(self.lerobot_root).name != self.lerobot_revision:
            raise ValueError("LeRobot snapshot directory does not match the frozen revision")

    def lerobot_episode_indices(self) -> list[int]:
        mapping = json.loads(Path(self.episode_mapping_path).read_text())
        if mapping.get("status") != "PASS":
            raise ValueError("Policy episode mapping must have PASS status")
        records = mapping["episodes"]
        if (
            mapping.get("hf_repo_id") != "physical-intelligence/libero"
            or mapping.get("hf_revision") != self.lerobot_revision
            or int(mapping.get("mapped_episode_count", -1)) != CANONICAL_LIBERO_EPISODES
            or int(mapping.get("mapped_frame_count", -1)) != CANONICAL_LIBERO_FRAMES
            or len(records) != CANONICAL_LIBERO_EPISODES
            or sum(int(row["episode_length"]) for row in records) != CANONICAL_LIBERO_FRAMES
        ):
            raise ValueError("Policy episode mapping is not the frozen official LeRobot population")
        episodes = sorted(int(row["lerobot_episode_index"]) for row in records)
        if episodes != list(range(CANONICAL_LIBERO_EPISODES)):
            raise ValueError("P1/P2 policy data must be exactly official LIBERO-10 episodes 0..378")
        return episodes


class PolicySemanticTokenizer:
    """Exact PaliGemma tokenization for concrete semantic-subtask targets.

    The first label is predicted from the final valid instruction state on the
    native VLM language path. Teacher inputs contain the target sequence shifted
    left by one position, with no BOS token; EOS remains explicitly supervised.
    """

    def __init__(self, max_target_len: int = 32) -> None:
        if max_target_len < 2:
            raise ValueError("max_target_len must be at least two")
        self.max_target_len = max_target_len
        official = _tokenizer.PaligemmaTokenizer(max_target_len)
        self.sentencepiece = official._tokenizer  # noqa: SLF001

    def encode_target(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("Semantic target must be a non-empty canonical string")
        token_ids = list(self.sentencepiece.encode(text, add_eos=True))
        if len(token_ids) > self.max_target_len:
            raise ValueError(f"Semantic target has {len(token_ids)} tokens, exceeding {self.max_target_len}: {text!r}")
        return token_ids

    def batch(self, texts: Sequence[str]) -> SemanticTeacherTensors:
        encoded = [self.encode_target(text) for text in texts]
        if not encoded:
            raise ValueError("Cannot tokenize an empty semantic target batch")
        target_length = max(len(tokens) for tokens in encoded)
        labels = np.zeros((len(encoded), target_length), dtype=np.int64)
        loss_mask = np.zeros((len(encoded), target_length), dtype=bool)
        for row, tokens in enumerate(encoded):
            labels[row, : len(tokens)] = tokens
            loss_mask[row, : len(tokens)] = True
        return SemanticTeacherTensors(
            input_ids=labels[:, :-1].copy(),
            labels=labels,
            loss_mask=loss_mask,
        )

    def fixed(self, text: str) -> SemanticTeacherTensors:
        """Tokenize one target into fixed-size arrays that collate without padding logic."""

        tokens = self.encode_target(text)
        labels = np.zeros((self.max_target_len,), dtype=np.int64)
        loss_mask = np.zeros((self.max_target_len,), dtype=bool)
        labels[: len(tokens)] = tokens
        loss_mask[: len(tokens)] = True
        return SemanticTeacherTensors(
            input_ids=labels[:-1].copy(),
            labels=labels,
            loss_mask=loss_mask,
        )

    def pieces(self, token_ids: Sequence[int]) -> list[str]:
        return [self.sentencepiece.id_to_piece(int(token_id)) for token_id in token_ids]


class Libero10AnnotationIndex:
    """Map stable LeRobot episode/frame identities to frozen annotation rows."""

    def __init__(self, manifest_path: str | Path, episode_mapping_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.episode_mapping_path = Path(episode_mapping_path).resolve(strict=True)
        manifest = pd.read_parquet(self.manifest_path)
        manifest = manifest.loc[manifest["suite"].eq("libero_10")].copy()

        mapping_payload = json.loads(self.episode_mapping_path.read_text())
        records = mapping_payload["episodes"]
        if (
            len(records) != CANONICAL_LIBERO_EPISODES
            or mapping_payload.get("status") != "PASS"
            or mapping_payload.get("hf_repo_id") != "physical-intelligence/libero"
            or mapping_payload.get("hf_revision") != CANONICAL_LIBERO_REVISION
            or int(mapping_payload.get("mapped_frame_count", -1)) != CANONICAL_LIBERO_FRAMES
            or sum(int(row["episode_length"]) for row in records) != CANONICAL_LIBERO_FRAMES
        ):
            raise ValueError("A complete PASS LeRobot-to-annotation episode mapping is required")
        self._episode_mapping = {int(row["lerobot_episode_index"]): row for row in records}
        if set(self._episode_mapping) != set(range(CANONICAL_LIBERO_EPISODES)):
            raise ValueError("LeRobot episode indices are not unique")

        mapped_episode_ids = {str(row["annotation_episode_id"]) for row in records}
        manifest = manifest.loc[manifest["episode_id"].isin(mapped_episode_ids)].copy()
        expected_frames = sum(int(row["episode_length"]) for row in records)
        if len(manifest) != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} mapped official LIBERO-10 policy samples, found {len(manifest)}"
            )
        if manifest["sample_id"].duplicated().any():
            raise ValueError("Mapped LIBERO-10 policy sample IDs are not unique")
        if set(manifest["episode_id"].astype(str)) != mapped_episode_ids:
            raise ValueError("Policy annotation manifest does not cover every mapped episode")

        manifest = manifest.set_index(["episode_id", "frame_idx"], verify_integrity=True)
        self._manifest = manifest

    def row(self, lerobot_episode_index: int, frame_index: int) -> pd.Series:
        episode = self._episode_mapping[int(lerobot_episode_index)]
        key = (episode["annotation_episode_id"], int(frame_index))
        try:
            row = self._manifest.loc[key]
        except KeyError as error:
            raise KeyError(f"No auxiliary annotation row for LeRobot item {key}") from error
        if str(row["action_sha256"]) != str(episode["action_sha256"]):
            raise ValueError(f"Action identity mismatch for {key}")
        return row

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _load_mask_shard(path: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as arrays:
            return {name: arrays[name].copy() for name in arrays.files}

    def load_upright_ground_masks(self, row: pd.Series) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Decode immutable raw masks and create upright policy-view inputs.

        Only the deterministic raw OpenGL 180-degree rotation is done here.
        Resize/crop/rotation augmentation is performed jointly with RGB inside
        ``policy_aux_preprocessing``.
        """

        shard = self._load_mask_shard(str(row["mask_shard_path"]))
        shard_row = int(row["mask_shard_row"])
        expected_frame = int(row.name[1])
        if int(shard["frame_index"][shard_row]) != expected_frame:
            raise ValueError(f"Mask shard frame mismatch for {row['sample_id']}")
        masks = {}
        valid = []
        for source_name, policy_name in (
            ("agent", "base_0_rgb"),
            ("wrist", "left_wrist_0_rgb"),
        ):
            raw = np.unpackbits(
                shard[f"mask_{source_name}_packed"][shard_row],
                count=128 * 128,
                bitorder="little",
            ).reshape(128, 128)
            masks[policy_name] = np.ascontiguousarray(raw[::-1, ::-1], dtype=np.float32)
            is_valid = bool(shard[f"mask_valid_{source_name}"][shard_row])
            if is_valid != bool(row[f"{source_name}_mask_valid"]):
                raise ValueError(f"Mask validity mismatch for {row['sample_id']} {source_name}")
            valid.append(is_valid)
        return masks, np.asarray(valid, dtype=bool)


class GeometryPolicyTargetIndex:
    """Read the finalized policy Geometry cache through one memory-mapped array."""

    def __init__(self, target_index_path: str | Path, normalization_path: str | Path) -> None:
        self.target_index_path = Path(target_index_path).resolve(strict=True)
        self.normalization_path = Path(normalization_path).resolve(strict=True)
        frame = pd.read_parquet(self.target_index_path).sort_values("lerobot_dataset_index")
        if len(frame) != 101_469 or not frame["sample_id"].is_unique:
            raise ValueError("Geometry policy target index must cover 101469 unique policy samples")
        if frame["lerobot_dataset_index"].tolist() != list(range(len(frame))):
            raise ValueError("Geometry policy target index is not in exact LeRobot dataset order")
        valid = frame["geometry_valid"].astype(bool).to_numpy()
        if int(valid.sum()) != 101_381 or int((~valid).sum()) != 88:
            raise ValueError("Unexpected Geometry valid/invalid counts")
        if frame.loc[valid, "target_memmap_row"].isna().any():
            raise ValueError("A valid Geometry sample lacks a target row")
        if frame.loc[~valid, "target_memmap_row"].notna().any():
            raise ValueError("An invalid Geometry sample has a target row")
        paths = frame.loc[valid, "target_memmap_path"].drop_duplicates().tolist()
        if len(paths) != 1:
            raise ValueError("Geometry index must resolve to one immutable target memmap")
        memmap_path = Path(paths[0])
        if not memmap_path.is_absolute():
            memmap_path = self.target_index_path.parent / memmap_path
        self._memmap_path = memmap_path.resolve(strict=True)
        self._targets = np.load(self._memmap_path, mmap_mode="r")
        if self._targets.shape != (101_381, 2048) or self._targets.dtype != np.float32:
            raise ValueError("Unexpected Geometry target memmap shape/dtype")
        self._frame = frame.reset_index(drop=True)

        normalization = json.loads(self.normalization_path.read_text())
        if normalization.get("status") != "PASS" or normalization.get("split") != "train":
            raise ValueError("Geometry train normalization must have PASS status")
        if int(normalization["sample_count"]) != 101_381:
            raise ValueError("Geometry normalization count differs from valid target count")
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        if self.mean.shape != (2048,) or self.std.shape != (2048,):
            raise ValueError("Unexpected Geometry normalization shape")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("Geometry normalization is non-finite or non-positive")

    def target_by_dataset_index(self, dataset_index: int) -> tuple[np.ndarray | None, bool, str]:
        row = self._frame.iloc[int(dataset_index)]
        if int(row["lerobot_dataset_index"]) != int(dataset_index):
            raise ValueError("Geometry dataset index identity mismatch")
        valid = bool(row["geometry_valid"])
        if not valid:
            return None, False, str(row["sample_id"])
        target = np.asarray(self._targets[int(row["target_memmap_row"])], dtype=np.float32)
        if target.shape != (2048,) or not np.isfinite(target).all():
            raise ValueError(f"Invalid Geometry target for {row['sample_id']}")
        return target, True, str(row["sample_id"])


class PolicyAuxTargetIndex:
    """Join current-frame semantic, Grounding, and Geometry targets by stable identity."""

    def __init__(self, config: PolicyAuxTrainConfig) -> None:
        self.config = config
        self.annotations = Libero10AnnotationIndex(config.policy_manifest_path, config.episode_mapping_path)
        self.geometry = GeometryPolicyTargetIndex(config.geometry_target_index_path, config.geometry_normalization_path)
        self.semantic_tokenizer = PolicySemanticTokenizer(config.semantic_max_target_len)
        mapping = json.loads(Path(config.episode_mapping_path).read_text())
        self._dataset_identity = {}
        for episode in mapping["episodes"]:
            start = int(episode["dataset_from_index"])
            length = int(episode["episode_length"])
            for frame_index in range(length):
                self._dataset_identity[start + frame_index] = (
                    int(episode["lerobot_episode_index"]),
                    frame_index,
                )
        if set(self._dataset_identity) != set(range(101_469)):
            raise ValueError("Policy auxiliary dataset identity is not exactly contiguous")

    def item(self, dataset_index: int) -> dict:
        episode_index, frame_index = self._dataset_identity[int(dataset_index)]
        row = self.annotations.row(episode_index, frame_index)
        geometry, geometry_valid, geometry_sample_id = self.geometry.target_by_dataset_index(dataset_index)
        sample_id = str(row["sample_id"])
        if sample_id != geometry_sample_id:
            raise ValueError(f"Policy auxiliary identity mismatch at dataset index {dataset_index}")
        result = {
            "geometry": (geometry if geometry is not None else np.zeros((2048,), dtype=np.float32)),
            "geometry_valid": np.asarray(geometry_valid, dtype=bool),
            "geometry_mean": self.geometry.mean,
            "geometry_std": self.geometry.std,
        }
        if self.config.mode == "ground_geometry_semantic_lm":
            masks, ground_valid = self.annotations.load_upright_ground_masks(row)
            semantic = self.semantic_tokenizer.fixed(str(row["semantic_subtask"]))
            result.update(
                {
                    "ground_masks": masks,
                    "ground_valid_views": ground_valid,
                    "semantic_input_ids": semantic.input_ids,
                    "semantic_labels": semantic.labels,
                    "semantic_loss_mask": semantic.loss_mask,
                }
            )
        return result


class PolicyAuxTransformedDataset:
    """Attach immutable auxiliary targets after official policy transforms."""

    def __init__(self, dataset, config: PolicyAuxTrainConfig) -> None:
        self.dataset = dataset
        self.config = config
        self._target_index: PolicyAuxTargetIndex | None = None
        if len(dataset) != 101_469:
            raise ValueError(f"Expected official LIBERO-10 dataset length 101469, found {len(dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        if self._target_index is None:
            # Construct lazily inside each spawned DataLoader worker. The large
            # target matrix remains a shared read-only OS memory map rather than
            # being serialized from the parent process.
            self._target_index = PolicyAuxTargetIndex(self.config)
        item = self.dataset[index]
        if "policy_aux" in item:
            raise ValueError("Base transformed dataset unexpectedly contains policy_aux")
        return {**item, "policy_aux": self._target_index.item(index)}
