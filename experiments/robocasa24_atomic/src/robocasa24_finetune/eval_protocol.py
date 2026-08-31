"""Frozen and testable RoboCasa Atomic-24 closed-loop evaluation semantics."""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ACTION_DIM
from .constants import ACTION_HORIZON
from .constants import CAMERAS
from .constants import EXECUTION_HORIZON
from .constants import STATE_DIM
from .constants import TASKS

EVAL_SEED = 7
TRIALS_PER_TASK = 50
RESIZE_SIZE = 224
MAX_EPISODE_STEPS = 1_000
FLOW_STEPS = 10
OBJECT_INSTANCE_SPLIT = "B"
LAYOUT_AND_STYLE_IDS = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))

IMAGE_OBSERVATION_KEYS = {
    "observation/image_left": f"{CAMERAS[0]}_image",
    "observation/wrist_image": f"{CAMERAS[1]}_image",
    "observation/image_right": f"{CAMERAS[2]}_image",
}

STATE_OBSERVATION_KEYS = (
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_gripper_qpos",
)


def tasks_for_worker(
    tasks: tuple[str, ...], num_workers: int, worker_index: int
) -> tuple[str, ...]:
    """Assign complete tasks to workers so each task keeps its reset sequence."""

    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError(
            f"invalid worker assignment: index={worker_index}, count={num_workers}"
        )
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("evaluation tasks must be non-empty and unique")
    unknown = tuple(task for task in tasks if task not in TASKS)
    if unknown:
        raise ValueError(f"unknown Atomic-24 tasks: {unknown}")
    return tuple(task for position, task in enumerate(tasks) if position % num_workers == worker_index)


def episode_shard_for_worker(
    tasks: tuple[str, ...], num_workers: int, worker_index: int
) -> tuple[str, int, int]:
    """Assign one task and one episode shard to every simulator worker.

    Workers are divided into contiguous, near-equal groups per task. Within a
    task group, worker ``j`` executes episodes for which
    ``episode_idx % group_size == j``. Every worker still resets all episodes in
    canonical order; this preserves RoboCasa's seeded reset stream while making
    the expensive closed-loop rollouts parallel.
    """

    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError(
            f"invalid worker assignment: index={worker_index}, count={num_workers}"
        )
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("evaluation tasks must be non-empty and unique")
    unknown = tuple(task for task in tasks if task not in TASKS)
    if unknown:
        raise ValueError(f"unknown Atomic-24 tasks: {unknown}")
    if num_workers < len(tasks):
        raise ValueError(
            "episode sharding requires at least one worker per task: "
            f"workers={num_workers}, tasks={len(tasks)}"
        )

    workers_per_task, remainder = divmod(num_workers, len(tasks))
    first_worker = 0
    for task_position, task in enumerate(tasks):
        task_worker_count = workers_per_task + int(task_position < remainder)
        last_worker = first_worker + task_worker_count
        if first_worker <= worker_index < last_worker:
            return task, worker_index - first_worker, task_worker_count
        first_worker = last_worker
    raise AssertionError("worker-to-task episode shard assignment is incomplete")


def validate_protocol(
    *,
    tasks: tuple[str, ...],
    trials_per_task: int,
    execution_horizon: int,
    resize_size: int,
    max_episode_steps: int,
    seed: int,
    formal: bool,
) -> None:
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in TASKS for task in tasks):
        raise ValueError("invalid Atomic-24 task selection")
    if trials_per_task <= 0 or execution_horizon <= 0 or max_episode_steps <= 0:
        raise ValueError("trial count and horizons must be positive")
    if execution_horizon > ACTION_HORIZON:
        raise ValueError("execution horizon cannot exceed the predicted action horizon")
    if resize_size <= 0 or seed < 0:
        raise ValueError("resize size must be positive and seed non-negative")
    if not formal:
        return
    expected = {
        "tasks": TASKS,
        "trials_per_task": TRIALS_PER_TASK,
        "execution_horizon": EXECUTION_HORIZON,
        "resize_size": RESIZE_SIZE,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "seed": EVAL_SEED,
    }
    observed = {
        "tasks": tasks,
        "trials_per_task": trials_per_task,
        "execution_horizon": execution_horizon,
        "resize_size": resize_size,
        "max_episode_steps": max_episode_steps,
        "seed": seed,
    }
    if observed != expected:
        raise ValueError(
            "formal RoboCasa Atomic-24 evaluation protocol differs: "
            f"expected={expected}, observed={observed}"
        )


def policy_observation(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Map a native RoboSuite observation to the exact training-time schema."""

    images: dict[str, np.ndarray] = {}
    for policy_key, observation_key in IMAGE_OBSERVATION_KEYS.items():
        if observation_key not in obs:
            raise KeyError(f"missing RoboCasa camera observation: {observation_key}")
        image = np.asarray(obs[observation_key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"invalid {observation_key} image shape: {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Native RoboSuite camera observations retain OpenGL's bottom-left row
        # order. The frozen base50 training files record
        # ``orientation_transform=vertical_flip_axis_height`` and
        # ``pixel_row_order=top_left_opencv``. Apply that same single-axis
        # correction here (not a 180-degree rotation).
        images[policy_key] = np.ascontiguousarray(np.flipud(image))

    state_parts = []
    for key in STATE_OBSERVATION_KEYS:
        if key not in obs:
            raise KeyError(f"missing RoboCasa state observation: {key}")
        state_parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    state = np.concatenate(state_parts)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"invalid RoboCasa policy state: {state.shape}")
    if not prompt:
        raise ValueError("RoboCasa episode language must be non-empty")
    return {
        **images,
        "observation/state": state,
        "prompt": str(prompt),
    }


def validate_action_chunk(actions: Any) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] != ACTION_HORIZON or chunk.shape[1] != ACTION_DIM:
        raise ValueError(
            "policy must return the exact RoboCasa action chunk "
            f"[{ACTION_HORIZON},{ACTION_DIM}], found {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("policy returned non-finite RoboCasa actions")
    return chunk


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = (str(record["task"]), int(record["episode_idx"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid evaluation record at {path}:{line_number}") from error
                if key in keys:
                    raise ValueError(f"duplicate RoboCasa rollout record: {key}")
                keys.add(key)
                records.append(record)
    return records


def completed_episodes(path: Path) -> set[tuple[str, int]]:
    return {(str(row["task"]), int(row["episode_idx"])) for row in read_jsonl([path])}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Durably append one completed rollout before moving to the next one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
