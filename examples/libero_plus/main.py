"""Evaluate an OpenPI policy on the LIBERO-Plus variants of the frozen three-task population."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import io
import json
import logging
import math
import pathlib
import time
import typing

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_PLUS_SUITE = "libero_10"
LIBERO_PLUS_EXPECTED_VARIANTS = 872
LIBERO3_TASKS = {
    "moka": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "bowl": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "mugs": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
}


@dataclasses.dataclass(frozen=True)
class Variant:
    benchmark_index: int
    classification_id: int
    name: str
    base_task: str
    category: str
    difficulty_level: int


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_steps: int = 520
    seed: int = 7

    num_shards: int = 1
    shard_index: int = 0
    # Classification IDs are one-based IDs from task_classification.json.
    # If omitted, evaluate every Plus variant derived from the three tasks.
    variant_ids: typing.Optional[typing.Tuple[int, ...]] = None  # noqa: UP006, UP007 -- Python 3.8 client

    output_jsonl: str = "data/libero_plus/results.jsonl"
    video_out_path: str = "data/libero_plus/videos"
    save_video: bool = True
    # Formal mode freezes the policy-facing protocol and requires all 872 variants.
    formal: bool = True


def _load_variants(task_suite, args: Args) -> list[Variant]:
    classification_path = pathlib.Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json"
    classification = json.loads(classification_path.read_text())[LIBERO_PLUS_SUITE]

    variants = []
    for entry in classification:
        base_task = next(
            (alias for alias, prefix in LIBERO3_TASKS.items() if entry["name"].startswith(prefix + "_")), None
        )
        if base_task is None:
            continue
        benchmark_index = int(entry["id"]) - 1
        task = task_suite.get_task(benchmark_index)
        if task.name != entry["name"]:
            raise RuntimeError(
                "LIBERO-Plus classification/benchmark order mismatch: "
                f"id={entry['id']}, classification={entry['name']}, benchmark={task.name}"
            )
        variants.append(
            Variant(
                benchmark_index=benchmark_index,
                classification_id=int(entry["id"]),
                name=entry["name"],
                base_task=base_task,
                category=entry["category"],
                difficulty_level=int(entry["difficulty_level"]),
            )
        )

    if args.variant_ids is not None:
        requested = tuple(args.variant_ids)
        by_id = {variant.classification_id: variant for variant in variants}
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"Requested IDs are not variants of the frozen three tasks: {missing}")
        variants = [by_id[variant_id] for variant_id in requested]

    return variants


def _validate_protocol(args: Args, variants: list[Variant]) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"Invalid shard {args.shard_index}/{args.num_shards}")
    if not args.formal:
        return
    expected = {
        "resize_size": 224,
        "replan_steps": 5,
        "num_steps_wait": 10,
        "max_steps": 520,
        "seed": 7,
        "variant_ids": None,
    }
    observed = {name: getattr(args, name) for name in expected}
    if observed != expected:
        raise ValueError(f"Formal LIBERO-Plus protocol mismatch: expected={expected}, observed={observed}")
    if len(variants) != LIBERO_PLUS_EXPECTED_VARIANTS:
        raise ValueError(
            f"Formal three-task LIBERO-Plus set must contain {LIBERO_PLUS_EXPECTED_VARIANTS} variants; "
            f"found {len(variants)}"
        )


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()


def eval_libero_plus(args: Args) -> None:
    np.random.seed(args.seed)
    # The upstream Plus benchmark prints the full 2,519-element task order on
    # every construction, which makes eight-shard logs unnecessarily huge.
    with contextlib.redirect_stdout(io.StringIO()):
        task_suite = benchmark.get_benchmark_dict()[LIBERO_PLUS_SUITE]()
    variants = _load_variants(task_suite, args)
    _validate_protocol(args, variants)
    shard_variants = [
        variant for position, variant in enumerate(variants) if position % args.num_shards == args.shard_index
    ]

    output_path = pathlib.Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = pathlib.Path(args.video_out_path)
    if args.save_video:
        video_path.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Selected variants: %d total, %d on shard %d/%d",
        len(variants),
        len(shard_variants),
        args.shard_index,
        args.num_shards,
    )
    logging.info("Output JSONL: %s", output_path)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    successes = 0
    for variant in tqdm.tqdm(shard_variants):
        task = task_suite.get_task(variant.benchmark_index)
        initial_states = task_suite.get_task_init_states(variant.benchmark_index)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        replay_images = []
        done = False
        error_text = None
        policy_steps = 0
        started = time.monotonic()
        try:
            env.reset()
            obs = env.set_init_state(initial_states[0])
            action_plan = collections.deque()
            t = 0
            while t < args.max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    if len(action_chunk) < args.replan_steps:
                        raise RuntimeError(
                            f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                policy_steps += 1
                t += 1
                if done:
                    successes += 1
                    break
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            logging.exception("Variant %d failed with an exception", variant.classification_id)
            if args.formal:
                raise
        finally:
            env.close()

        if args.save_video and replay_images:
            suffix = "success" if done else "failure"
            imageio.mimwrite(
                video_path / f"variant_{variant.classification_id:04d}_{variant.base_task}_{suffix}.mp4",
                replay_images,
                fps=10,
            )

        record = dataclasses.asdict(variant)
        record.update(
            {
                "task_description": str(task_description),
                "success": bool(done),
                "policy_steps": policy_steps,
                "elapsed_seconds": time.monotonic() - started,
                "seed": args.seed,
                "shard_index": args.shard_index,
                "error": error_text,
            }
        )
        _append_jsonl(output_path, record)
        logging.info(
            "variant=%d base=%s category=%s difficulty=%d success=%s steps=%d running=%d/%d",
            variant.classification_id,
            variant.base_task,
            variant.category,
            variant.difficulty_level,
            done,
            policy_steps,
            successes,
            len(shard_variants),
        )

    logging.info(
        "Shard success rate: %d/%d = %.2f%%", successes, len(shard_variants), 100 * successes / len(shard_variants)
    )


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    eval_libero_plus(tyro.cli(Args))
