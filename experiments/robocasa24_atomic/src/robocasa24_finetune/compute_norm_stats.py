"""Compute OpenPI normalization stats without decoding RoboCasa RGB frames."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd

from openpi.shared import normalize

from .constants import ACTION_DIM
from .constants import ACTION_HORIZON
from .constants import DATASET_REPO_ID
from .constants import EPISODES_PER_TASK
from .constants import STATE_DIM
from .constants import TASKS
from .data import _canonical_hdf5
from .data import _manifest_path


def _episode_state(group: h5py.Group) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(group["obs/robot0_base_to_eef_pos"], dtype=np.float32),
            np.asarray(group["obs/robot0_base_to_eef_quat"], dtype=np.float32),
            np.asarray(group["obs/robot0_base_pos"], dtype=np.float32),
            np.asarray(group["obs/robot0_base_quat"], dtype=np.float32),
            np.asarray(group["obs/robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=-1,
    )
    if state.ndim != 2 or state.shape[1] != STATE_DIM or not np.isfinite(state).all():
        raise ValueError(f"invalid state matrix: {state.shape}")
    return state


def _episode_actions(group: h5py.Group) -> np.ndarray:
    actions = np.asarray(group["actions"], dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[1] != ACTION_DIM
        or not np.isfinite(actions).all()
    ):
        raise ValueError(f"invalid action matrix: {actions.shape}")
    return actions


def _equal_task_stats(task_stats: list[normalize.NormStats]) -> normalize.NormStats:
    """Combine within-task raw-frame moments with one equal vote per task."""

    means = np.stack([np.asarray(stats.mean, dtype=np.float64) for stats in task_stats])
    stds = np.stack([np.asarray(stats.std, dtype=np.float64) for stats in task_stats])
    mean = means.mean(axis=0)
    second_moment = np.mean(stds**2 + means**2, axis=0)
    variance = np.maximum(second_moment - mean**2, 0.0)
    return normalize.NormStats(mean=mean, std=np.sqrt(variance))


def compute(
    *,
    data_root: Path,
    manifest_root: Path,
    output_root: Path,
    tasks: tuple[str, ...] = TASKS,
) -> dict:
    """Match OpenPI's pre-normalization stats, but omit irrelevant image I/O."""

    data_root = Path(data_root).resolve(strict=True)
    manifest_root = Path(manifest_root).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in TASKS for task in tasks)
    ):
        raise ValueError("invalid Atomic-24 task selection")
    destination = output_root / DATASET_REPO_ID / "norm_stats.json"
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite normalization stats: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    per_task_state_stats: list[normalize.NormStats] = []
    per_task_action_stats: list[normalize.NormStats] = []
    frame_count = 0
    episode_count = 0
    per_task: dict[str, dict[str, int]] = {}
    for task in tasks:
        task_state_stats = normalize.RunningStats()
        task_action_stats = normalize.RunningStats()
        manifest = pd.read_parquet(
            _manifest_path(manifest_root, task),
            columns=[
                "episode_name",
                "episode_length",
                "source_role",
                "task_frame_index",
            ],
        ).sort_values("task_frame_index")
        if not manifest["source_role"].eq("base50").all():
            raise ValueError(f"{task}: normalization source is not base50")
        episodes = manifest[["episode_name", "episode_length"]].drop_duplicates()
        if len(episodes) != EPISODES_PER_TASK:
            raise ValueError(
                f"{task}: expected {EPISODES_PER_TASK} episodes, found {len(episodes)}"
            )
        hdf5_path = _canonical_hdf5(data_root, task).resolve(strict=True)
        task_frames = 0
        with h5py.File(hdf5_path, "r", swmr=True) as handle:
            if set(episodes["episode_name"].astype(str)) != set(handle["data"].keys()):
                raise ValueError(f"{task}: manifest/HDF5 population mismatch")
            for episode_name, episode_length in episodes.itertuples(index=False):
                group = handle["data"][str(episode_name)]
                state = _episode_state(group)
                actions = _episode_actions(group)
                if len(state) != int(episode_length) or len(actions) != int(
                    episode_length
                ):
                    raise ValueError(f"{task}/{episode_name}: episode length mismatch")
                task_state_stats.update(state)
                task_action_stats.update(actions)
                task_frames += len(state)
                episode_count += 1
        if task_frames != len(manifest):
            raise ValueError(
                f"{task}: normalized frame population differs from source manifest"
            )
        frame_count += task_frames
        per_task[task] = {"episodes": len(episodes), "frames": task_frames}
        per_task_state_stats.append(task_state_stats.get_statistics())
        per_task_action_stats.append(task_action_stats.get_statistics())

    stats = {
        "state": _equal_task_stats(per_task_state_stats),
        "actions": _equal_task_stats(per_task_action_stats),
    }
    serialized = normalize.serialize_json(stats) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".norm_stats.", suffix=".json", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    report = {
        "status": "PASS",
        "dataset_repo_id": DATASET_REPO_ID,
        "tasks": len(tasks),
        "episodes": episode_count,
        "frames": frame_count,
        "state_vectors": frame_count,
        "action_vectors": frame_count,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "normalization": "z-score mean/std from raw frames, equal weight per task",
        "normalization_task_weighting": "uniform across selected tasks",
        "normalization_action_population": "each recorded action exactly once",
        "training_tail_rule": "clamp to final action within each episode",
        "output": str(destination),
        "per_task": per_task,
    }
    report_path = destination.parent / "norm_stats_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            compute(
                data_root=args.data_root,
                manifest_root=args.manifest_root,
                output_root=args.output_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
