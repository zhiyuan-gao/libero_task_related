"""Identity-strict HDF5 policy dataset for RoboCasa Atomic-24."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
import dataclasses
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .constants import ACTION_DIM
from .constants import ACTION_HORIZON
from .constants import CAMERAS
from .constants import EPISODES_PER_TASK
from .constants import STATE_DIM
from .constants import TASKS

TASK_SAMPLING_ALPHA = 0.4
_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    """Small deterministic integer mixer used for allocation-free sampling."""

    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _hierarchical_sample_index(
    virtual_index: int,
    seed: int,
    task_cdf: np.ndarray,
    task_episode_indices: tuple[np.ndarray, ...],
    episode_starts: np.ndarray,
    episode_lengths: np.ndarray,
) -> int:
    """Sample task -> episode -> timestep with the official RoboCasa policy."""

    mixed = _splitmix64(
        (int(seed) & _UINT64_MASK)
        ^ (((int(virtual_index) + 1) * 0xD1342543DE82EF95) & _UINT64_MASK)
    )
    uniform = mixed / float(1 << 64)
    task_index = min(int(np.searchsorted(task_cdf, uniform, side="right")), len(task_cdf) - 1)

    mixed = _splitmix64(mixed)
    episodes = task_episode_indices[task_index]
    episode_index = int(episodes[mixed % len(episodes)])

    mixed = _splitmix64(mixed)
    frame_index = int(mixed % int(episode_lengths[episode_index]))
    return int(episode_starts[episode_index]) + frame_index


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    task: str
    episode_name: str
    episode_length: int
    instruction: str
    hdf5_path: str


@dataclasses.dataclass(frozen=True)
class SampleIdSequence(Sequence[str]):
    """Compact deterministic view of the canonical frame identities."""

    episodes: tuple[EpisodeRecord, ...]
    episode_indices: np.ndarray
    frame_indices: np.ndarray

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        i = int(index)
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(index)
        episode = self.episodes[int(self.episode_indices[i])]
        frame_idx = int(self.frame_indices[i])
        return f"{episode.task}/base50__{episode.episode_name}/frame_{frame_idx:06d}"

    def __iter__(self) -> Iterator[str]:
        for index in range(len(self)):
            yield self[index]


class HDF5HandleCache:
    """Process-local read-only handles, safe when DataLoader workers are forked."""

    def __init__(self, limit: int = 24) -> None:
        self.limit = int(limit)
        self.pid = os.getpid()
        self.handles: OrderedDict[str, h5py.File] = OrderedDict()

    def get(self, path: str) -> h5py.File:
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r", swmr=True)
        handle = self.handles.pop(path)
        self.handles[path] = handle
        while len(self.handles) > self.limit:
            _, old = self.handles.popitem(last=False)
            old.close()
        return handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self) -> dict:
        # Spawned workers must never inherit or pickle live HDF5 handles.
        return {"limit": self.limit}

    def __setstate__(self, state: dict) -> None:
        self.__init__(limit=int(state["limit"]))


def _canonical_hdf5(data_root: Path, task: str) -> Path:
    filename = "success50_rgb_seg_256_upright.hdf5"
    candidates = (
        data_root / "data" / "base50" / task / filename,
        data_root / task / filename,
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            f"ambiguous RoboCasa data root contains two base50 layouts: {data_root}"
        )
    # Preserve a deterministic failure path for the caller's strict resolve.
    return candidates[0]


def _manifest_path(manifest_root: Path, task: str) -> Path:
    return manifest_root / task / "source" / "source_manifest.parquet"


def _read_index(
    data_root: Path, manifest_root: Path, tasks: tuple[str, ...]
) -> tuple[tuple[EpisodeRecord, ...], np.ndarray, np.ndarray]:
    episodes_all: list[EpisodeRecord] = []
    episode_indices_all: list[np.ndarray] = []
    frame_indices_all: list[np.ndarray] = []
    seen: set[str] = set()
    for task in tasks:
        hdf5_path = _canonical_hdf5(data_root, task).resolve(strict=True)
        manifest_path = _manifest_path(manifest_root, task).resolve(strict=True)
        frame = pd.read_parquet(
            manifest_path,
            columns=[
                "sample_id",
                "task",
                "episode_name",
                "episode_length",
                "frame_idx",
                "instruction",
                "source_role",
                "task_frame_index",
            ],
        ).sort_values("task_frame_index")
        if (
            not frame["task"].eq(task).all()
            or not frame["source_role"].eq("base50").all()
        ):
            raise ValueError(
                f"{task}: source manifest is not the fixed base50 population"
            )
        episodes = frame[["episode_name", "episode_length"]].drop_duplicates()
        if len(episodes) != EPISODES_PER_TASK:
            raise ValueError(
                f"{task}: expected {EPISODES_PER_TASK} episodes, found {len(episodes)}"
            )
        with h5py.File(hdf5_path, "r", swmr=True) as handle:
            if set(episodes["episode_name"]) != set(handle["data"].keys()):
                raise ValueError(f"{task}: manifest/HDF5 episode population mismatch")
            for episode_name, episode_length in episodes.itertuples(index=False):
                group = handle["data"][str(episode_name)]
                length = int(episode_length)
                if group["actions"].shape != (length, ACTION_DIM):
                    raise ValueError(f"{task}/{episode_name}: action schema mismatch")
                for camera in CAMERAS:
                    if group[f"rgb256/{camera}"].shape != (length, 256, 256, 3):
                        raise ValueError(
                            f"{task}/{episode_name}: RGB schema mismatch for {camera}"
                        )
        task_episode_map: dict[str, int] = {}
        for episode_name, episode_length in episodes.itertuples(index=False):
            episode_frame = frame[
                frame["episode_name"].astype(str).eq(str(episode_name))
            ]
            expected_frames = np.arange(int(episode_length), dtype=np.int64)
            if not np.array_equal(
                episode_frame["frame_idx"].to_numpy(dtype=np.int64), expected_frames
            ):
                raise ValueError(
                    f"{task}/{episode_name}: frame population/order mismatch"
                )
            instructions = episode_frame["instruction"].astype(str).unique()
            if len(instructions) != 1 or not instructions[0]:
                raise ValueError(
                    f"{task}/{episode_name}: instruction is empty or inconsistent"
                )
            task_episode_map[str(episode_name)] = len(episodes_all)
            episodes_all.append(
                EpisodeRecord(
                    task=task,
                    episode_name=str(episode_name),
                    episode_length=int(episode_length),
                    instruction=str(instructions[0]),
                    hdf5_path=str(hdf5_path),
                )
            )
        task_episode_indices = (
            frame["episode_name"]
            .astype(str)
            .map(task_episode_map)
            .to_numpy(dtype=np.uint16)
        )
        task_frame_indices = frame["frame_idx"].to_numpy(dtype=np.int32)
        for sample_id, episode_index, frame_idx in zip(
            frame["sample_id"].astype(str),
            task_episode_indices,
            task_frame_indices,
            strict=True,
        ):
            episode = episodes_all[int(episode_index)]
            expected_id = (
                f"{task}/base50__{episode.episode_name}/frame_{int(frame_idx):06d}"
            )
            if sample_id != expected_id:
                raise ValueError(
                    f"non-canonical sample ID: {sample_id} != {expected_id}"
                )
            if sample_id in seen:
                raise ValueError(f"duplicate RoboCasa sample ID: {sample_id}")
            seen.add(sample_id)
        episode_indices_all.append(task_episode_indices)
        frame_indices_all.append(task_frame_indices)
    return (
        tuple(episodes_all),
        np.concatenate(episode_indices_all),
        np.concatenate(frame_indices_all),
    )


class RoboCasa24HDF5Dataset:
    """Official-style hierarchical samples from the exact 24 x 50 population."""

    def __init__(
        self,
        data_root: str | Path,
        manifest_root: str | Path,
        *,
        action_horizon: int = ACTION_HORIZON,
        tasks: tuple[str, ...] = TASKS,
        sampling_seed: int = 42,
        task_sampling_alpha: float = TASK_SAMPLING_ALPHA,
    ) -> None:
        self.data_root = Path(data_root).resolve(strict=True)
        self.manifest_root = Path(manifest_root).resolve(strict=True)
        self.tasks = tuple(tasks)
        if not self.tasks or any(task not in TASKS for task in self.tasks):
            raise ValueError("RoboCasa dataset task selection is invalid")
        if action_horizon != ACTION_HORIZON:
            raise ValueError(
                f"RoboCasa policy action horizon is frozen at {ACTION_HORIZON}"
            )
        self.action_horizon = action_horizon
        self.sampling_seed = int(sampling_seed)
        if task_sampling_alpha != TASK_SAMPLING_ALPHA:
            raise ValueError(
                f"RoboCasa task sampling alpha is frozen at {TASK_SAMPLING_ALPHA}"
            )
        self.task_sampling_alpha = float(task_sampling_alpha)
        self.episodes, self.episode_indices, self.frame_indices = _read_index(
            self.data_root, self.manifest_root, self.tasks
        )
        episode_lengths = np.asarray(
            [episode.episode_length for episode in self.episodes], dtype=np.int64
        )
        self.episode_starts = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(episode_lengths[:-1])]
        )
        if int(episode_lengths.sum()) != len(self.frame_indices):
            raise ValueError("RoboCasa episode lengths do not cover the frame index")
        expected_episode_indices = np.repeat(
            np.arange(len(self.episodes), dtype=np.int64), episode_lengths
        )
        if not np.array_equal(
            self.episode_indices.astype(np.int64), expected_episode_indices
        ):
            raise ValueError("RoboCasa raw rows are not contiguous by episode")
        self.episode_lengths = episode_lengths
        self.task_episode_indices = tuple(
            np.asarray(
                [i for i, episode in enumerate(self.episodes) if episode.task == task],
                dtype=np.int64,
            )
            for task in self.tasks
        )
        if any(len(indices) != EPISODES_PER_TASK for indices in self.task_episode_indices):
            raise ValueError("RoboCasa hierarchical sampler requires 50 episodes per task")
        self.task_frame_counts = np.asarray(
            [self.episode_lengths[indices].sum() for indices in self.task_episode_indices],
            dtype=np.int64,
        )
        task_weights = self.task_frame_counts.astype(np.float64) ** self.task_sampling_alpha
        self.task_sampling_probabilities = task_weights / task_weights.sum()
        self.task_sampling_cdf = np.cumsum(self.task_sampling_probabilities)
        self.task_sampling_cdf[-1] = 1.0
        self.handles = HDF5HandleCache(limit=min(24, len(self.tasks)))

    def __len__(self) -> int:
        return len(self.frame_indices)

    @property
    def sample_ids(self) -> SampleIdSequence:
        """Canonical raw-row IDs used to resolve immutable auxiliary targets."""

        return SampleIdSequence(self.episodes, self.episode_indices, self.frame_indices)

    def resolve_sample_index(self, index: int) -> int:
        """Map a virtual row to task^0.4 / uniform-episode / uniform-step data."""

        i = int(index)
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(index)
        worker = torch.utils.data.get_worker_info()
        seed = self.sampling_seed if worker is None else int(worker.seed)
        return _hierarchical_sample_index(
            i,
            seed,
            self.task_sampling_cdf,
            self.task_episode_indices,
            self.episode_starts,
            self.episode_lengths,
        )

    def _state(self, group: h5py.Group, frame_idx: int) -> np.ndarray:
        # Match the public RoboCasa OpenPI ordering: relative EEF pose, base pose,
        # then the two gripper joints.
        state = np.concatenate(
            [
                np.asarray(
                    group["obs/robot0_base_to_eef_pos"][frame_idx], dtype=np.float32
                ),
                np.asarray(
                    group["obs/robot0_base_to_eef_quat"][frame_idx], dtype=np.float32
                ),
                np.asarray(group["obs/robot0_base_pos"][frame_idx], dtype=np.float32),
                np.asarray(group["obs/robot0_base_quat"][frame_idx], dtype=np.float32),
                np.asarray(
                    group["obs/robot0_gripper_qpos"][frame_idx], dtype=np.float32
                ),
            ]
        )
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"invalid RoboCasa state: {state.shape}")
        return state

    def _actions(
        self, group: h5py.Group, frame_idx: int, episode_length: int
    ) -> np.ndarray:
        # Standard sequence-dataset tail behavior: clamp future indices to the
        # final recorded action without crossing an episode boundary.
        indices = np.minimum(
            frame_idx + np.arange(self.action_horizon, dtype=np.int64),
            episode_length - 1,
        )
        # h5py rejects repeated fancy indices; tail clamping necessarily
        # repeats the final row, so index the small episode action matrix in
        # NumPy rather than issuing an invalid HDF5 selection.
        actions = np.asarray(group["actions"], dtype=np.float32)[indices]
        if (
            actions.shape != (self.action_horizon, ACTION_DIM)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"invalid RoboCasa action chunk: {actions.shape}")
        return actions

    def __getitem__(self, index: int) -> dict:
        i = self.resolve_sample_index(index)
        episode = self.episodes[int(self.episode_indices[i])]
        handle = self.handles.get(episode.hdf5_path)
        group = handle["data"][episode.episode_name]
        frame_idx = int(self.frame_indices[i])
        return {
            "observation/image_left": np.asarray(
                group[f"rgb256/{CAMERAS[0]}"][frame_idx], dtype=np.uint8
            ),
            "observation/wrist_image": np.asarray(
                group[f"rgb256/{CAMERAS[1]}"][frame_idx], dtype=np.uint8
            ),
            "observation/image_right": np.asarray(
                group[f"rgb256/{CAMERAS[2]}"][frame_idx], dtype=np.uint8
            ),
            "observation/state": self._state(group, frame_idx),
            "actions": self._actions(group, frame_idx, episode.episode_length),
            "prompt": episode.instruction,
            "sample_id": self.sample_ids[i],
        }

    def close(self) -> None:
        self.handles.close()
