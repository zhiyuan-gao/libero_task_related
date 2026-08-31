"""Validate and summarize completed multi-worker Atomic-24 rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EXECUTION_HORIZON
from .constants import TASKS
from .eval_protocol import EVAL_SEED
from .eval_protocol import TRIALS_PER_TASK
from .eval_protocol import read_jsonl


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    trials = len(rows)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else None,
        "mean_steps": float(np.mean([int(row["steps"]) for row in rows])) if rows else None,
        "mean_elapsed_seconds": (
            float(np.mean([float(row["elapsed_seconds"]) for row in rows]))
            if rows
            else None
        ),
    }


def summarize(
    paths: list[Path],
    tasks: tuple[str, ...],
    *,
    trials_per_task: int,
    formal: bool,
    execution_horizon: int = EXECUTION_HORIZON,
) -> dict[str, Any]:
    rows = read_jsonl(paths)
    expected_keys = {
        (task, episode_idx)
        for task in tasks
        for episode_idx in range(trials_per_task)
    }
    observed_keys = {(str(row["task"]), int(row["episode_idx"])) for row in rows}
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"evaluation population is incomplete: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for row in rows:
        if int(row["predicted_action_horizon"]) != 50:
            raise ValueError("a rollout used a non-50 prediction horizon")
        if int(row["execution_horizon"]) != execution_horizon:
            raise ValueError(
                "a rollout used an unexpected execution horizon: "
                f"expected={execution_horizon}, observed={row['execution_horizon']}"
            )
        if formal and int(row["seed"]) != EVAL_SEED:
            raise ValueError("a formal rollout used a non-frozen evaluation seed")

    by_task = {
        task: _metrics([row for row in rows if row["task"] == task]) for task in tasks
    }
    categories = {
        "pick_and_place": TASKS[:8],
        "doors_and_drawers": TASKS[8:14],
        "other_atomic": TASKS[14:],
    }
    by_category = {}
    for name, category_tasks in categories.items():
        selected = [row for row in rows if row["task"] in category_tasks]
        if selected:
            by_category[name] = _metrics(selected)
    return {
        "schema": "robocasa24.atomic24.eval_summary.v1",
        "status": "PASS",
        "formal": formal,
        "tasks": list(tasks),
        "trials_per_task": trials_per_task,
        "prediction_horizon": 50,
        "execution_horizon": execution_horizon,
        "seed": EVAL_SEED if formal else sorted({int(row["seed"]) for row in rows}),
        "overall": _metrics(rows),
        "by_category": by_category,
        "by_task": by_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--trials-per-task", type=int, default=TRIALS_PER_TASK)
    parser.add_argument("--execution-horizon", type=int, default=EXECUTION_HORIZON)
    parser.add_argument("--formal", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary = summarize(
        args.inputs,
        tuple(args.tasks),
        trials_per_task=args.trials_per_task,
        formal=args.formal,
        execution_horizon=args.execution_horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    overall = summary["overall"]
    print(
        f"Overall: {overall['successes']}/{overall['trials']} "
        f"({100.0 * overall['success_rate']:.2f}%)"
    )
    for task, metrics in summary["by_task"].items():
        print(
            f"{task}: {metrics['successes']}/{metrics['trials']} "
            f"({100.0 * metrics['success_rate']:.2f}%)"
        )


if __name__ == "__main__":
    main()
