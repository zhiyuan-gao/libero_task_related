from __future__ import annotations

import json

from four_suite_experiments.constants import SUITES
from four_suite_experiments.summarize_eval import summarize_batch
from four_suite_experiments.summarize_eval import summarize_checkpoint
from four_suite_experiments.summarize_eval import summarize_suite
import pytest


@pytest.mark.parametrize("num_shards", [8, 16])
def test_formal_suite_checkpoint_and_batch_summaries(tmp_path, num_shards: int) -> None:
    shard_paths = [tmp_path / f"shard_{index}.jsonl" for index in range(num_shards)]
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        for task_id in range(10):
            for episode_idx in range(50):
                shard = (task_id * 50 + episode_idx) % num_shards
                record = {
                    "task_suite_name": "libero_spatial",
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "shard_index": shard,
                    "num_shards": num_shards,
                    "seed": 7,
                    "success": episode_idx % 2 == 0,
                    "error": None,
                    "task_description": f"task {task_id}",
                }
                handles[shard].write(json.dumps(record) + "\n")
    finally:
        for handle in handles:
            handle.close()

    suite_summary = summarize_suite("libero_spatial", shard_paths)
    assert suite_summary["total"] == 500
    assert suite_summary["successes"] == 250

    suite_paths = []
    for suite in SUITES:
        path = tmp_path / f"{suite}.json"
        payload = dict(suite_summary)
        payload["protocol"] = dict(suite_summary["protocol"], suite=suite)
        path.write_text(json.dumps(payload), encoding="utf-8")
        suite_paths.append(path)
    checkpoint_summary = summarize_checkpoint(suite_paths, 30_000)
    assert checkpoint_summary["total"] == 2_000
    assert checkpoint_summary["success_rate"] == 0.5

    checkpoint_path = tmp_path / "step_30000.json"
    checkpoint_path.write_text(json.dumps(checkpoint_summary), encoding="utf-8")
    batch_summary = summarize_batch([checkpoint_path])
    assert batch_summary["checkpoint_steps"] == [30_000]
