"""One resumable RoboCasa Atomic-24 simulator shard."""

from __future__ import annotations

import argparse
import collections
import logging
from pathlib import Path
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from .constants import ACTION_DIM
from .constants import ACTION_HORIZON
from .constants import CAMERAS
from .constants import EXECUTION_HORIZON
from .constants import TASKS
from .eval_protocol import EVAL_SEED
from .eval_protocol import LAYOUT_AND_STYLE_IDS
from .eval_protocol import MAX_EPISODE_STEPS
from .eval_protocol import OBJECT_INSTANCE_SPLIT
from .eval_protocol import RESIZE_SIZE
from .eval_protocol import TRIALS_PER_TASK
from .eval_protocol import append_jsonl
from .eval_protocol import completed_episodes
from .eval_protocol import episode_shard_for_worker
from .eval_protocol import policy_observation
from .eval_protocol import tasks_for_worker
from .eval_protocol import validate_action_chunk
from .eval_protocol import validate_protocol


def _create_env(task: str, seed: int):
    # Import simulator dependencies only inside the dedicated client process.
    # This keeps MuJoCo/RoboSuite out of the policy-server environment.
    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=task,
        robots="PandaOmron",
        camera_names=list(CAMERAS),
        camera_widths=256,
        camera_heights=256,
        seed=seed,
        render_onscreen=False,
        obj_instance_split=OBJECT_INSTANCE_SPLIT,
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=LAYOUT_AND_STYLE_IDS,
    )


def _resize_policy_images(element: dict[str, Any], resize_size: int) -> dict[str, Any]:
    result = dict(element)
    for key in (
        "observation/image_left",
        "observation/wrist_image",
        "observation/image_right",
    ):
        result[key] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(result[key], resize_size, resize_size)
        )
    return result


def _episode_video_path(
    video_root: Path, task: str, episode_idx: int, *, success: bool
) -> Path:
    suffix = "success" if success else "failure"
    return video_root / task / f"episode_{episode_idx:03d}_{suffix}.mp4"


def run(args: argparse.Namespace) -> None:
    tasks = tuple(args.tasks)
    validate_protocol(
        tasks=tasks,
        trials_per_task=args.trials_per_task,
        execution_horizon=args.execution_horizon,
        resize_size=args.resize_size,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        formal=args.formal,
    )
    if args.shard_mode == "task":
        assigned_tasks = tasks_for_worker(tasks, args.num_workers, args.worker_index)
        episode_shards = dict.fromkeys(assigned_tasks, (0, 1))
    else:
        task, shard_index, shard_count = episode_shard_for_worker(
            tasks, args.num_workers, args.worker_index
        )
        assigned_tasks = (task,)
        episode_shards = {task: (shard_index, shard_count)}
    output_jsonl = Path(args.output_jsonl)
    done_keys = completed_episodes(output_jsonl)
    client = websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port, ping_interval=None
    )
    logging.info(
        "worker %d/%d shard_mode=%s assigned tasks/shards: %s; completed=%d",
        args.worker_index,
        args.num_workers,
        args.shard_mode,
        {task: episode_shards[task] for task in assigned_tasks},
        len(done_keys),
    )

    for task in assigned_tasks:
        env = _create_env(task, args.seed)
        try:
            low, high = env.action_spec
            if np.asarray(low).shape != (ACTION_DIM,) or np.asarray(high).shape != (ACTION_DIM,):
                raise ValueError(
                    f"{task}: simulator action space differs from training 12-D contract: "
                    f"{np.asarray(low).shape}/{np.asarray(high).shape}"
                )
            for episode_idx in range(args.trials_per_task):
                # Always reset in canonical order, including episodes assigned
                # to another worker and already completed episodes. This keeps
                # every episode on RoboCasa's seeded reset stream under both
                # task-level and episode-level sharding.
                obs = env.reset()
                episode_shard_index, episode_shard_count = episode_shards[task]
                if episode_idx % episode_shard_count != episode_shard_index:
                    continue
                key = (task, episode_idx)
                if key in done_keys:
                    continue
                ep_meta = env.get_ep_meta()
                prompt = str(ep_meta.get("lang", "")).strip()
                action_plan: collections.deque[np.ndarray] = collections.deque()
                replay_images: list[np.ndarray] = []
                infer_ms: list[float] = []
                success = bool(env._check_success())  # noqa: SLF001 - RoboCasa API
                started = time.monotonic()
                steps = 0

                while not success and steps < args.max_episode_steps:
                    if not action_plan:
                        element = _resize_policy_images(
                            policy_observation(obs, prompt), args.resize_size
                        )
                        result = client.infer(element)
                        chunk = validate_action_chunk(result["actions"])
                        action_plan.extend(chunk[: args.execution_horizon])
                        timing = result.get("policy_timing", {})
                        if "infer_ms" in timing:
                            infer_ms.append(float(timing["infer_ms"]))

                    action = action_plan.popleft()
                    obs, _, _, _ = env.step(action)
                    steps += 1
                    success = bool(env._check_success())  # noqa: SLF001 - RoboCasa API
                    if args.save_video and (steps % 2 == 0 or success):
                        replay_images.append(
                            np.ascontiguousarray(
                                np.flipud(obs[f"{CAMERAS[0]}_image"])
                            )
                        )

                elapsed = time.monotonic() - started
                if args.save_video:
                    video_path = _episode_video_path(
                        Path(args.video_root), task, episode_idx, success=success
                    )
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(video_path, replay_images, fps=10)

                record = {
                    "schema": "robocasa24.atomic24.eval_rollout.v1",
                    "task": task,
                    "task_position": TASKS.index(task),
                    "episode_idx": episode_idx,
                    "success": success,
                    "steps": steps,
                    "elapsed_seconds": elapsed,
                    "worker_index": args.worker_index,
                    "num_workers": args.num_workers,
                    "shard_mode": args.shard_mode,
                    "episode_shard_index": episode_shard_index,
                    "episode_shard_count": episode_shard_count,
                    "seed": args.seed,
                    "prompt": prompt,
                    "predicted_action_horizon": ACTION_HORIZON,
                    "execution_horizon": args.execution_horizon,
                    "policy_queries": len(infer_ms),
                    "mean_policy_infer_ms": (
                        float(np.mean(infer_ms)) if infer_ms else None
                    ),
                }
                append_jsonl(output_jsonl, record)
                done_keys.add(key)
                logging.info(
                    "%s episode=%d success=%s steps=%d elapsed=%.1fs",
                    task,
                    episode_idx,
                    success,
                    steps,
                    elapsed,
                )
        finally:
            env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--shard-mode", choices=("task", "episode"), default="task")
    parser.add_argument("--trials-per-task", type=int, default=TRIALS_PER_TASK)
    parser.add_argument("--execution-horizon", type=int, default=EXECUTION_HORIZON)
    parser.add_argument("--resize-size", type=int, default=RESIZE_SIZE)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--formal", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
