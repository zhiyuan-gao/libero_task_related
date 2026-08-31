from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import robocasa24_finetune.compute_norm_stats as norm_module
from robocasa24_finetune.compute_norm_stats import compute
from robocasa24_finetune.constants import DATASET_REPO_ID
from robocasa24_finetune.constants import TASKS

from openpi.shared import normalize


def _write_episode(group: h5py.Group, actions: np.ndarray) -> None:
    group.create_dataset("actions", data=np.repeat(actions[:, None], 12, axis=1))
    for key, width in (
        ("robot0_base_to_eef_pos", 3),
        ("robot0_base_to_eef_quat", 4),
        ("robot0_base_pos", 3),
        ("robot0_base_quat", 4),
        ("robot0_gripper_qpos", 2),
    ):
        group.create_dataset(
            f"obs/{key}", data=np.ones((len(actions), width), dtype=np.float32)
        )


def test_normalization_uses_raw_actions_and_equal_task_weights(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(norm_module, "EPISODES_PER_TASK", 1)
    data_root = tmp_path / "data"
    manifest_root = tmp_path / "manifests"
    selected_tasks = TASKS[:2]
    actions_by_task = (
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
    )
    for task, actions in zip(selected_tasks, actions_by_task, strict=True):
        hdf5_path = (
            data_root
            / task
            / "success50_rgb_seg_256_upright.hdf5"
        )
        hdf5_path.parent.mkdir(parents=True)
        with h5py.File(hdf5_path, "w") as handle:
            _write_episode(handle.create_group("data/demo_0"), actions)
        manifest_path = manifest_root / task / "source/source_manifest.parquet"
        manifest_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "episode_name": ["demo_0"] * len(actions),
                "episode_length": [len(actions)] * len(actions),
                "source_role": ["base50"] * len(actions),
                "task_frame_index": np.arange(len(actions)),
            }
        ).to_parquet(manifest_path, index=False)

    output_root = tmp_path / "assets"
    report = compute(
        data_root=data_root,
        manifest_root=manifest_root,
        output_root=output_root,
        tasks=selected_tasks,
    )
    stats = normalize.load(output_root / DATASET_REPO_ID)
    # Task means are 0 and 20, so equal-task merging is 10. A global-frame
    # merge would be 15, and horizon-expanded tail padding would be larger.
    np.testing.assert_allclose(stats["actions"].mean, np.full(12, 10.0))
    np.testing.assert_allclose(
        stats["actions"].std, np.full(12, np.sqrt(400.0 / 3.0))
    )
    assert report["action_vectors"] == 5
    assert report["normalization_task_weighting"] == "uniform across selected tasks"
