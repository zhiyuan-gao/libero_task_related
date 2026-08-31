from __future__ import annotations

from pathlib import Path
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest
from robocasa24_finetune.constants import ACTION_HORIZON
from robocasa24_finetune.constants import CAMERAS
from robocasa24_finetune.constants import POLICY_VIEW_NAMES
from robocasa24_finetune.constants import TASKS
import robocasa24_finetune.data as data_module
from robocasa24_finetune.data import RoboCasa24HDF5Dataset
from robocasa24_finetune.data import _canonical_hdf5
from robocasa24_finetune.data import _hierarchical_sample_index
from robocasa24_finetune.policy import RoboCasaInputs
from robocasa24_finetune.policy import RoboCasaOutputs

from openpi.models import model as model_api


def _write_demo(group: h5py.Group, length: int, action_offset: float) -> None:
    group.create_dataset(
        "actions",
        data=np.arange(length * 12, dtype=np.float32).reshape(length, 12)
        + action_offset,
    )
    state_fields = {
        "robot0_base_to_eef_pos": 3,
        "robot0_base_to_eef_quat": 4,
        "robot0_base_pos": 3,
        "robot0_base_quat": 4,
        "robot0_gripper_qpos": 2,
    }
    for name, width in state_fields.items():
        group.create_dataset(
            f"obs/{name}", data=np.ones((length, width), dtype=np.float32)
        )
    for camera in CAMERAS:
        group.create_dataset(
            f"rgb256/{camera}", shape=(length, 256, 256, 3), dtype=np.uint8
        )


def test_hdf5_dataset_identity_and_horizon_tail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(data_module, "EPISODES_PER_TASK", 2)
    task = TASKS[0]
    data_root = tmp_path / "data_root"
    manifest_root = tmp_path / "manifest_root"
    hdf5_path = data_root / "data/base50" / task / "success50_rgb_seg_256_upright.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    rows = []
    task_index = 0
    with h5py.File(hdf5_path, "w") as handle:
        for episode_number, length in enumerate((2, 3)):
            episode_name = f"demo_{episode_number}"
            _write_demo(
                handle.create_group(f"data/{episode_name}"),
                length,
                episode_number * 1000.0,
            )
            for frame_idx in range(length):
                rows.append(
                    {
                        "sample_id": f"{task}/base50__{episode_name}/frame_{frame_idx:06d}",
                        "task": task,
                        "episode_name": episode_name,
                        "episode_length": length,
                        "frame_idx": frame_idx,
                        "instruction": "perform the atomic task",
                        "source_role": "base50",
                        "task_frame_index": task_index,
                    }
                )
                task_index += 1
    manifest_path = manifest_root / task / "source/source_manifest.parquet"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    dataset = RoboCasa24HDF5Dataset(data_root, manifest_root, tasks=(task,))
    assert _canonical_hdf5(data_root / "data/base50", task) == hdf5_path
    assert len(dataset) == 5
    assert tuple(dataset.sample_ids) == tuple(row["sample_id"] for row in rows)
    tail_virtual_index = next(
        index
        for index in range(len(dataset))
        if (
            dataset.frame_indices[dataset.resolve_sample_index(index)]
            == dataset.episodes[
                int(dataset.episode_indices[dataset.resolve_sample_index(index)])
            ].episode_length
            - 1
        )
    )
    last = dataset[tail_virtual_index]
    assert last["actions"].shape == (ACTION_HORIZON, 12)
    np.testing.assert_array_equal(
        last["actions"], np.repeat(last["actions"][-1:], ACTION_HORIZON, axis=0)
    )
    assert last["observation/state"].shape == (16,)
    assert all(
        last[key].shape == (256, 256, 3)
        for key in (
            "observation/image_left",
            "observation/wrist_image",
            "observation/image_right",
        )
    )
    # The trainer uses spawn. Open HDF5 handles are stripped and the compact
    # integer index survives worker serialization.
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[1]["sample_id"] == dataset[1]["sample_id"]
    restored.close()
    dataset.close()


def test_hierarchical_sampler_matches_task_alpha_and_is_deterministic() -> None:
    episode_lengths = np.asarray([2] * 50 + [200] * 50, dtype=np.int64)
    episode_starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(episode_lengths[:-1])]
    )
    task_episodes = (
        np.arange(0, 50, dtype=np.int64),
        np.arange(50, 100, dtype=np.int64),
    )
    frame_counts = np.asarray([100, 10_000], dtype=np.float64)
    probabilities = frame_counts**0.4
    probabilities /= probabilities.sum()
    cdf = np.cumsum(probabilities)

    sampled = np.asarray(
        [
            _hierarchical_sample_index(
                index,
                42,
                cdf,
                task_episodes,
                episode_starts,
                episode_lengths,
            )
            for index in range(50_000)
        ],
        dtype=np.int64,
    )
    observed_task_0 = np.mean(sampled < int(episode_starts[50]))
    assert observed_task_0 == pytest.approx(probabilities[0], abs=0.005)
    assert _hierarchical_sample_index(
        123,
        42,
        cdf,
        task_episodes,
        episode_starts,
        episode_lengths,
    ) == _hierarchical_sample_index(
        123,
        42,
        cdf,
        task_episodes,
        episode_starts,
        episode_lengths,
    )


def test_policy_transform_preserves_three_views_and_real_action_width() -> None:
    raw = {
        "observation/image_left": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((256, 256, 3), dtype=np.uint8),
        "observation/image_right": np.full((256, 256, 3), 2, dtype=np.uint8),
        "observation/state": np.arange(16, dtype=np.float32),
        "actions": np.zeros((50, 12), dtype=np.float32),
        "prompt": "close the drawer",
    }
    transformed = RoboCasaInputs(model_type=model_api.ModelType.PI05)(raw)
    assert tuple(transformed["image"]) == POLICY_VIEW_NAMES
    assert all(bool(value) for value in transformed["image_mask"].values())
    assert transformed["state"].shape == (16,)
    assert transformed["actions"].shape == (50, 12)
    output = RoboCasaOutputs()({"actions": np.zeros((50, 32), dtype=np.float32)})
    assert output["actions"].shape == (50, 12)
