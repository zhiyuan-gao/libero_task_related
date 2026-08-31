"""Read-only validation of RoboCasa policy data and prepared teacher assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from openpi.shared import normalize

from .auxiliary import PreparedAuxiliaryStore
from .constants import ACTION_DIM
from .constants import CAMERAS
from .constants import DATASET_REPO_ID
from .constants import EPISODES_PER_TASK
from .constants import STATE_DIM
from .constants import TASKS
from .data import _canonical_hdf5
from .data import _manifest_path


def validate(
    *,
    data_root: Path,
    manifest_root: Path,
    policy_assets_root: Path,
    artifact_dir: Path | None,
    tasks: tuple[str, ...] = TASKS,
) -> dict:
    data_root = Path(data_root).resolve(strict=True)
    manifest_root = Path(manifest_root).resolve(strict=True)
    policy_assets_root = Path(policy_assets_root).resolve(strict=True)
    sample_ids: list[str] = []
    task_reports: dict[str, dict[str, int]] = {}
    for task in tasks:
        manifest = pd.read_parquet(
            _manifest_path(manifest_root, task),
            columns=[
                "sample_id",
                "task",
                "task_frame_index",
                "episode_name",
                "episode_length",
                "frame_idx",
                "source_role",
            ],
        ).sort_values("task_frame_index")
        if not np.array_equal(
            manifest["task_frame_index"].to_numpy(dtype=np.int64),
            np.arange(len(manifest), dtype=np.int64),
        ):
            raise ValueError(f"{task}: source ordering differs")
        if (
            not manifest["task"].eq(task).all()
            or not manifest["source_role"].eq("base50").all()
        ):
            raise ValueError(f"{task}: source population differs")
        episodes = manifest[["episode_name", "episode_length"]].drop_duplicates()
        if len(episodes) != EPISODES_PER_TASK:
            raise ValueError(f"{task}: expected {EPISODES_PER_TASK} episodes")
        with h5py.File(_canonical_hdf5(data_root, task), "r", swmr=True) as handle:
            if set(episodes["episode_name"].astype(str)) != set(handle["data"].keys()):
                raise ValueError(f"{task}: HDF5 episode population differs")
            for episode_name, episode_length in episodes.itertuples(index=False):
                group = handle["data"][str(episode_name)]
                length = int(episode_length)
                if group["actions"].shape != (length, ACTION_DIM):
                    raise ValueError(f"{task}/{episode_name}: actions differ")
                state_dims = sum(
                    group[name].shape[-1]
                    for name in (
                        "obs/robot0_base_to_eef_pos",
                        "obs/robot0_base_to_eef_quat",
                        "obs/robot0_base_pos",
                        "obs/robot0_base_quat",
                        "obs/robot0_gripper_qpos",
                    )
                )
                if state_dims != STATE_DIM:
                    raise ValueError(f"{task}/{episode_name}: state dimension differs")
                for camera in CAMERAS:
                    if group[f"rgb256/{camera}"].shape != (length, 256, 256, 3):
                        raise ValueError(
                            f"{task}/{episode_name}: {camera} image schema differs"
                        )
        task_ids = manifest["sample_id"].astype(str).tolist()
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{task}: duplicate source sample IDs")
        sample_ids.extend(task_ids)
        task_reports[task] = {"episodes": len(episodes), "frames": len(manifest)}
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs overlap across tasks")

    norm = normalize.load(policy_assets_root / DATASET_REPO_ID)
    if set(norm) != {"state", "actions"}:
        raise ValueError("policy normalization keys differ")
    for key, dim in (("state", STATE_DIM), ("actions", ACTION_DIM)):
        stats = norm[key]
        for name in ("mean", "std"):
            value = np.asarray(getattr(stats, name))
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"{key}/{name}: normalization schema differs")
        if np.any(np.asarray(stats.std) < 0):
            raise ValueError(f"{key}: negative standard deviation")
        # Official RoboCasa OpenPI mixture stats contain only mean/std. If a
        # compatible asset also carries optional quantiles, validate both.
        if (stats.q01 is None) != (stats.q99 is None):
            raise ValueError(f"{key}: incomplete optional quantile statistics")
        if stats.q01 is not None:
            q01 = np.asarray(stats.q01)
            q99 = np.asarray(stats.q99)
            if (
                q01.shape != (dim,)
                or q99.shape != (dim,)
                or not np.isfinite(q01).all()
                or not np.isfinite(q99).all()
                or np.any(q99 < q01)
            ):
                raise ValueError(f"{key}: invalid optional quantile normalization")

    auxiliary = None
    if artifact_dir is not None:
        store = PreparedAuxiliaryStore(artifact_dir, tuple(sample_ids))
        auxiliary = {
            "scope": pd.read_parquet(store.paths.index, columns=["target_scope"])[
                "target_scope"
            ].iloc[0],
            "frames": store.length,
            "geometry_valid": int(store.geometry_valid.sum()),
            "motion_valid": int(store.motion_valid.sum()),
        }
    return {
        "status": "PASS",
        "tasks": len(tasks),
        "episodes": sum(row["episodes"] for row in task_reports.values()),
        "frames": len(sample_ids),
        "policy_views": len(CAMERAS),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "normalization_mode": "z-score",
        "normalization": str(policy_assets_root / DATASET_REPO_ID / "norm_stats.json"),
        "auxiliary": auxiliary,
        "per_task": task_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--policy-assets-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                data_root=args.data_root,
                manifest_root=args.manifest_root,
                policy_assets_root=args.policy_assets_root,
                artifact_dir=args.artifact_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
