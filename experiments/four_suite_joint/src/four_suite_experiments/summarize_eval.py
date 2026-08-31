"""Validate and summarize formal four-suite LIBERO rollout records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import SUITES

TASK_IDS = tuple(range(10))
TRIALS_PER_TASK = 50
DEFAULT_SEED = 7
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise SystemExit(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def summarize_suite(suite: str, inputs: list[Path], seed: int = DEFAULT_SEED) -> dict:
    if suite not in SUITES:
        raise SystemExit(f"Unknown four-suite benchmark: {suite}")
    records = _load_records(inputs)
    expected = {(task_id, episode_idx) for task_id in TASK_IDS for episode_idx in range(TRIALS_PER_TASK)}
    observed = [(int(record["task_id"]), int(record["episode_idx"])) for record in records]
    if len(observed) != len(set(observed)):
        raise SystemExit(f"{suite} evaluation has duplicate (task_id, episode_idx) records")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise SystemExit(f"{suite} evaluation is incomplete: missing={missing}, extra={extra}")

    shard_counts = {int(record["num_shards"]) for record in records}
    if len(shard_counts) != 1:
        raise SystemExit(f"{suite} evaluation mixes shard counts: {sorted(shard_counts)}")
    num_shards = shard_counts.pop()
    if num_shards < 1:
        raise SystemExit(f"{suite} evaluation has an invalid shard count: {num_shards}")

    by_task = {}
    for task_position, task_id in enumerate(TASK_IDS):
        task_records = [record for record in records if int(record["task_id"]) == task_id]
        for record in task_records:
            episode_idx = int(record["episode_idx"])
            expected_shard = (task_position * TRIALS_PER_TASK + episode_idx) % num_shards
            if record.get("task_suite_name") != suite:
                raise SystemExit(f"Suite identity mismatch in record: {record}")
            if int(record["shard_index"]) != expected_shard or int(record["num_shards"]) != num_shards:
                raise SystemExit(f"Invalid shard assignment in record: {record}")
            if int(record["seed"]) != seed or record.get("error") is not None:
                raise SystemExit(f"Invalid formal record: {record}")
        successes = sum(bool(record["success"]) for record in task_records)
        by_task[str(task_id)] = {
            "description": task_records[0]["task_description"],
            "total": len(task_records),
            "successes": successes,
            "success_rate": successes / len(task_records),
        }

    successes = sum(bool(record["success"]) for record in records)
    return {
        "protocol": {
            "suite": suite,
            "benchmark_task_ids": list(TASK_IDS),
            "seed": seed,
            "trials_per_task": TRIALS_PER_TASK,
            "num_shards": num_shards,
            "max_steps": MAX_STEPS[suite],
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


def summarize_checkpoint(inputs: list[Path], checkpoint_step: int) -> dict:
    summaries = [json.loads(path.read_text()) for path in inputs]
    if len(summaries) != len(SUITES):
        raise SystemExit(f"Checkpoint summary requires {len(SUITES)} suite summaries")
    by_suite = {summary["protocol"]["suite"]: summary for summary in summaries}
    if set(by_suite) != set(SUITES):
        raise SystemExit(f"Checkpoint summary requires exactly {SUITES}; observed={tuple(by_suite)}")
    if any(summary["total"] != 500 for summary in summaries):
        raise SystemExit("Each suite summary must contain exactly 500 formal rollouts")
    successes = sum(int(summary["successes"]) for summary in summaries)
    total = sum(int(summary["total"]) for summary in summaries)
    return {
        "checkpoint_step": checkpoint_step,
        "total": total,
        "successes": successes,
        "success_rate": successes / total,
        "suite_macro_average": sum(float(summary["success_rate"]) for summary in summaries) / len(summaries),
        "by_suite": by_suite,
    }


def summarize_batch(inputs: list[Path]) -> dict:
    summaries = [json.loads(path.read_text()) for path in inputs]
    steps = [int(summary["checkpoint_step"]) for summary in summaries]
    if len(steps) != len(set(steps)):
        raise SystemExit("Batch summary contains duplicate checkpoint steps")
    if steps != sorted(steps, reverse=True):
        raise SystemExit(f"Batch checkpoints are not in descending order: {steps}")
    if any(int(summary["total"]) != 2_000 for summary in summaries):
        raise SystemExit("Each checkpoint must contain exactly 2,000 formal four-suite rollouts")
    return {
        "protocol": "formal_four_suite_reverse_checkpoint_sweep",
        "checkpoint_count": len(summaries),
        "checkpoint_steps": steps,
        "checkpoints": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite_parser = subparsers.add_parser("suite")
    suite_parser.add_argument("--suite", choices=SUITES, required=True)
    suite_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    suite_parser.add_argument("--output", type=Path, required=True)
    suite_parser.add_argument("inputs", nargs="+", type=Path)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--checkpoint-step", type=int, required=True)
    checkpoint_parser.add_argument("--output", type=Path, required=True)
    checkpoint_parser.add_argument("inputs", nargs="+", type=Path)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--output", type=Path, required=True)
    batch_parser.add_argument("inputs", nargs="+", type=Path)

    args = parser.parse_args()
    if args.command == "suite":
        if args.seed < 0:
            raise SystemExit("Evaluation seed must be non-negative")
        summary = summarize_suite(args.suite, args.inputs, seed=args.seed)
    elif args.command == "checkpoint":
        summary = summarize_checkpoint(args.inputs, args.checkpoint_step)
    else:
        summary = summarize_batch(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
