#!/usr/bin/env python3
"""Validate and summarize the frozen 150-rollout LIBERO-3 evaluation."""

import argparse
import json
import pathlib

TASK_IDS = (4, 2, 3)
TRIALS_PER_TASK = 50
NUM_SHARDS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    records = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise SystemExit(f"Invalid JSON at {path}:{line_number}: {error}") from error

    expected = {(task_id, episode_idx) for task_id in TASK_IDS for episode_idx in range(TRIALS_PER_TASK)}
    observed = [(int(record["task_id"]), int(record["episode_idx"])) for record in records]
    observed_set = set(observed)
    if len(observed) != len(observed_set):
        raise SystemExit("Evaluation has duplicate (task_id, episode_idx) records")
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        extra = sorted(observed_set - expected)
        raise SystemExit(f"Evaluation is incomplete or invalid: missing={missing}, extra={extra}")

    by_task = {}
    for task_position, task_id in enumerate(TASK_IDS):
        task_records = [record for record in records if int(record["task_id"]) == task_id]
        for record in task_records:
            episode_idx = int(record["episode_idx"])
            expected_shard = (task_position * TRIALS_PER_TASK + episode_idx) % NUM_SHARDS
            if int(record["shard_index"]) != expected_shard or int(record["num_shards"]) != NUM_SHARDS:
                raise SystemExit(
                    f"Invalid shard assignment for task={task_id}, episode={episode_idx}: "
                    f"record={record['shard_index']}/{record['num_shards']}, expected={expected_shard}/{NUM_SHARDS}"
                )
            if int(record["seed"]) != 7 or record.get("error") is not None:
                raise SystemExit(f"Invalid formal record: {record}")
        successes = sum(bool(record["success"]) for record in task_records)
        by_task[str(task_id)] = {
            "description": task_records[0]["task_description"],
            "total": len(task_records),
            "successes": successes,
            "success_rate": successes / len(task_records),
        }

    successes = sum(bool(record["success"]) for record in records)
    summary = {
        "protocol": {
            "suite": "libero_10",
            "benchmark_task_ids": list(TASK_IDS),
            "lerobot_task_indices": [0, 3, 8],
            "seed": 7,
            "trials_per_task": TRIALS_PER_TASK,
            "num_shards": NUM_SHARDS,
            "max_steps": 520,
            "wait_steps": 10,
            "resize": 224,
            "replan_steps": 5,
            "policy_flow_steps": 10,
        },
        "total": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "by_task": by_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
