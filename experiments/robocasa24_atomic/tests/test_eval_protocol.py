from __future__ import annotations

import json

import numpy as np
import pytest
from robocasa24_finetune.constants import TASKS
from robocasa24_finetune.eval_protocol import append_jsonl
from robocasa24_finetune.eval_protocol import completed_episodes
from robocasa24_finetune.eval_protocol import episode_shard_for_worker
from robocasa24_finetune.eval_protocol import policy_observation
from robocasa24_finetune.eval_protocol import tasks_for_worker
from robocasa24_finetune.eval_protocol import validate_action_chunk
from robocasa24_finetune.eval_protocol import validate_protocol
from robocasa24_finetune.summarize_eval import summarize


def test_eight_workers_partition_complete_tasks() -> None:
    shards = [tasks_for_worker(TASKS, 8, worker) for worker in range(8)]
    assert all(len(shard) == 3 for shard in shards)
    assert sorted(task for shard in shards for task in shard) == sorted(TASKS)
    assert len({task for shard in shards for task in shard}) == len(TASKS)


def test_three_tasks_twenty_four_workers_partition_episodes() -> None:
    tasks = TASKS[:3]
    assignments = [
        episode_shard_for_worker(tasks, 24, worker) for worker in range(24)
    ]
    for task in tasks:
        task_assignments = [assignment for assignment in assignments if assignment[0] == task]
        assert len(task_assignments) == 8
        assert sorted(shard_index for _, shard_index, _ in task_assignments) == list(range(8))
        assert {shard_count for _, _, shard_count in task_assignments} == {8}

    observed = {
        (task, episode_idx)
        for task, shard_index, shard_count in assignments
        for episode_idx in range(50)
        if episode_idx % shard_count == shard_index
    }
    expected = {(task, episode_idx) for task in tasks for episode_idx in range(50)}
    assert observed == expected


def test_episode_sharding_rejects_fewer_workers_than_tasks() -> None:
    with pytest.raises(ValueError, match="at least one worker per task"):
        episode_shard_for_worker(TASKS[:3], 2, 0)


def test_formal_protocol_freezes_execute_25() -> None:
    validate_protocol(
        tasks=TASKS,
        trials_per_task=50,
        execution_horizon=25,
        resize_size=224,
        max_episode_steps=1000,
        seed=7,
        formal=True,
    )
    with pytest.raises(ValueError, match="protocol differs"):
        validate_protocol(
            tasks=TASKS,
            trials_per_task=50,
            execution_horizon=10,
            resize_size=224,
            max_episode_steps=1000,
            seed=7,
            formal=True,
        )


def test_policy_observation_matches_training_schema() -> None:
    left = np.zeros((256, 256, 3), dtype=np.uint8)
    left[0] = 17
    left[-1] = 91
    obs = {
        "robot0_agentview_left_image": left,
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_agentview_right_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_base_to_eef_pos": np.arange(3, dtype=np.float32),
        "robot0_base_to_eef_quat": np.arange(4, dtype=np.float32),
        "robot0_base_pos": np.arange(3, dtype=np.float32),
        "robot0_base_quat": np.arange(4, dtype=np.float32),
        "robot0_gripper_qpos": np.arange(2, dtype=np.float32),
    }
    result = policy_observation(obs, "open the drawer")
    assert result["observation/state"].shape == (16,)
    assert result["observation/image_left"].shape == (256, 256, 3)
    assert np.all(result["observation/image_left"][0] == 91)
    assert np.all(result["observation/image_left"][-1] == 17)
    assert result["prompt"] == "open the drawer"


def test_action_chunk_and_durable_resume(tmp_path) -> None:
    actions = validate_action_chunk(np.zeros((50, 12), dtype=np.float32))
    assert actions.shape == (50, 12)
    with pytest.raises(ValueError, match="exact RoboCasa action chunk"):
        validate_action_chunk(np.zeros((25, 12), dtype=np.float32))

    output = tmp_path / "worker.jsonl"
    append_jsonl(output, {"task": TASKS[0], "episode_idx": 4})
    assert completed_episodes(output) == {(TASKS[0], 4)}
    assert json.loads(output.read_text().strip())["episode_idx"] == 4


def test_summary_requires_and_reports_each_task(tmp_path) -> None:
    output = tmp_path / "worker.jsonl"
    for task in TASKS[:2]:
        for episode_idx in range(2):
            append_jsonl(
                output,
                {
                    "task": task,
                    "episode_idx": episode_idx,
                    "success": episode_idx == 0,
                    "steps": 10 + episode_idx,
                    "elapsed_seconds": 1.0,
                    "predicted_action_horizon": 50,
                    "execution_horizon": 25,
                    "seed": 9,
                },
            )
    summary = summarize([output], TASKS[:2], trials_per_task=2, formal=False)
    assert summary["overall"]["successes"] == 2
    assert summary["overall"]["trials"] == 4
    assert summary["by_task"][TASKS[0]]["success_rate"] == 0.5


def test_summary_accepts_an_explicit_short_execution_horizon(tmp_path) -> None:
    output = tmp_path / "worker.jsonl"
    append_jsonl(
        output,
        {
            "task": TASKS[0],
            "episode_idx": 0,
            "success": True,
            "steps": 9,
            "elapsed_seconds": 1.0,
            "predicted_action_horizon": 50,
            "execution_horizon": 5,
            "seed": 7,
        },
    )
    summary = summarize(
        [output], TASKS[:1], trials_per_task=1, formal=False, execution_horizon=5
    )
    assert summary["execution_horizon"] == 5
    with pytest.raises(ValueError, match="unexpected execution horizon"):
        summarize(
            [output],
            TASKS[:1],
            trials_per_task=1,
            formal=False,
            execution_horizon=10,
        )
