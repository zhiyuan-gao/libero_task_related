#!/usr/bin/env python3
"""Build an exact four-suite LeRobot-to-frozen-annotation policy selector."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
from datetime import UTC
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from huggingface_hub import HfFileSystem
import numpy as np
import pandas as pd

EXPECTED_REPO = "physical-intelligence/libero"
EXPECTED_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
EXPECTED_EPISODES = 1_693
EXPECTED_FRAMES = 273_465
EXPECTED_ANNOTATION_EPISODES = 2_000
EXPECTED_ANNOTATION_FRAMES = 337_819
EXPECTED_PER_SUITE = {
    "libero_10": {"episodes": 379, "frames": 101_469},
    "libero_goal": {"episodes": 428, "frames": 52_042},
    "libero_object": {"episodes": 454, "frames": 66_984},
    "libero_spatial": {"episodes": 432, "frames": 52_970},
}

_THREAD_LOCAL = threading.local()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_sha256(actions: pd.Series) -> str:
    array = np.stack(actions.to_numpy()).astype(np.float32, copy=False)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def thread_filesystem() -> HfFileSystem:
    filesystem = getattr(_THREAD_LOCAL, "filesystem", None)
    if filesystem is None:
        filesystem = HfFileSystem()
        _THREAD_LOCAL.filesystem = filesystem
    return filesystem


def parquet_relative_path(episode_index: int, chunk_size: int) -> str:
    return f"data/chunk-{episode_index // chunk_size:03d}/episode_{episode_index:06d}.parquet"


def extract_episode_identity(
    metadata: dict[str, Any],
    *,
    snapshot: Path,
    repo_id: str,
    revision: str,
    chunk_size: int,
    task_index_by_instruction: dict[str, int],
    cache_root: Path,
) -> dict[str, Any]:
    episode_index = int(metadata["episode_index"])
    cache_path = cache_root / f"episode_{episode_index:06d}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("repo_id") == repo_id
            and cached.get("revision") == revision
            and int(cached.get("lerobot_episode_index", -1)) == episode_index
        ):
            return cached

    tasks = metadata["tasks"]
    if len(tasks) != 1 or tasks[0] not in task_index_by_instruction:
        raise ValueError(f"Unexpected task metadata for LeRobot episode {episode_index}")
    expected_task_index = task_index_by_instruction[tasks[0]]
    expected_length = int(metadata["length"])
    relative_path = parquet_relative_path(episode_index, chunk_size)
    local_path = snapshot / relative_path
    columns = ["actions", "episode_index", "frame_index", "index", "task_index"]

    last_error: Exception | None = None
    for attempt in range(5):
        try:
            if local_path.is_file():
                frame = pd.read_parquet(local_path, columns=columns)
                source = "local_snapshot"
            else:
                remote_path = f"datasets/{repo_id}@{revision}/{relative_path}"
                with thread_filesystem().open(remote_path, "rb") as handle:
                    frame = pd.read_parquet(handle, columns=columns)
                source = "hf_range_read_actions_only"
            break
        except Exception as error:
            last_error = error
            if attempt == 4:
                raise RuntimeError(f"Failed to read LeRobot episode {episode_index} after five attempts") from error
            time.sleep(2**attempt)
    else:  # pragma: no cover
        raise RuntimeError(f"Failed to read LeRobot episode {episode_index}") from last_error

    if len(frame) != expected_length:
        raise ValueError(f"Length mismatch for LeRobot episode {episode_index}")
    if not bool(frame["episode_index"].eq(episode_index).all()):
        raise ValueError(f"Episode identity mismatch in {relative_path}")
    if not bool(frame["task_index"].eq(expected_task_index).all()):
        raise ValueError(f"Task identity mismatch in {relative_path}")
    if not np.array_equal(frame["frame_index"].to_numpy(), np.arange(expected_length)):
        raise ValueError(f"Non-contiguous frame indices in {relative_path}")
    dataset_indices = frame["index"].to_numpy(dtype=np.int64)
    if not np.array_equal(dataset_indices, np.arange(dataset_indices[0], dataset_indices[0] + expected_length)):
        raise ValueError(f"Non-contiguous global dataset indices in {relative_path}")

    record = {
        "repo_id": repo_id,
        "revision": revision,
        "lerobot_episode_index": episode_index,
        "lerobot_task_index": expected_task_index,
        "dataset_from_index": int(dataset_indices[0]),
        "dataset_to_index_exclusive": int(dataset_indices[-1]) + 1,
        "episode_length": expected_length,
        "instruction": tasks[0],
        "action_sha256": action_sha256(frame["actions"]),
        "parquet_relative_path": relative_path,
        "identity_read_source": source,
    }
    atomic_write_json(cache_path, record)
    return record


def validate_source_inventory(
    *,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    release_episodes: pd.DataFrame,
    release_frames: pd.DataFrame,
) -> None:
    if (
        int(info["total_episodes"]) != EXPECTED_EPISODES
        or int(info["total_frames"]) != EXPECTED_FRAMES
        or int(info["total_tasks"]) != 40
        or len(episodes) != EXPECTED_EPISODES
        or sum(int(row["length"]) for row in episodes) != EXPECTED_FRAMES
        or len(tasks) != 40
    ):
        raise ValueError("LeRobot metadata is not the frozen 1693-episode population")
    if [int(row["episode_index"]) for row in episodes] != list(range(EXPECTED_EPISODES)):
        raise ValueError("LeRobot episode metadata is not in exact contiguous order")
    if (
        len(annotation_rows) != EXPECTED_ANNOTATION_EPISODES
        or len({row["action_sha256"] for row in annotation_rows}) != EXPECTED_ANNOTATION_EPISODES
        or len(release_episodes) != EXPECTED_ANNOTATION_EPISODES
        or len(release_frames) != EXPECTED_ANNOTATION_FRAMES
    ):
        raise ValueError("Frozen annotation release does not have the expected 2000/337819 population")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--annotation-index", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=EXPECTED_REPO)
    parser.add_argument("--revision", default=EXPECTED_REVISION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cross-check-libero10-mapping", type=Path)
    args = parser.parse_args()
    if args.repo_id != EXPECTED_REPO or args.revision != EXPECTED_REVISION:
        raise ValueError("This selector is frozen to the exact official LeRobot repository revision")
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be in [1, 32]")

    snapshot = args.snapshot.resolve(strict=True)
    annotation_path = args.annotation_index.resolve(strict=True)
    release_root = args.release_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    cache_root = args.cache_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    info_path = snapshot / "meta/info.json"
    episodes_path = snapshot / "meta/episodes.jsonl"
    tasks_path = snapshot / "meta/tasks.jsonl"
    info = json.loads(info_path.read_text())
    episodes = load_jsonl(episodes_path)
    tasks = load_jsonl(tasks_path)
    annotation_rows = load_jsonl(annotation_path)
    release_episode_path = release_root / "episode_shard_index.parquet"
    release_frame_path = release_root / "frame_manifest.parquet"
    release_task_path = release_root / "tasks.json"
    release_episodes = pd.read_parquet(release_episode_path)
    release_frames = pd.read_parquet(release_frame_path)
    release_tasks = json.loads(release_task_path.read_text())["tasks"]
    validate_source_inventory(
        info=info,
        episodes=episodes,
        tasks=tasks,
        annotation_rows=annotation_rows,
        release_episodes=release_episodes,
        release_frames=release_frames,
    )

    task_index_by_instruction = {row["task"]: int(row["task_index"]) for row in tasks}
    release_task_by_instruction = {row["instruction"]: row for row in release_tasks}
    if set(task_index_by_instruction) != set(release_task_by_instruction):
        raise ValueError("LeRobot and frozen release task instruction sets differ")
    annotation_by_hash = {row["action_sha256"]: row for row in annotation_rows}
    if len(annotation_by_hash) != EXPECTED_ANNOTATION_EPISODES:
        raise ValueError("Frozen annotation action hashes are not unique")

    started = time.monotonic()
    identities: list[dict[str, Any]] = []
    kwargs = {
        "snapshot": snapshot,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "chunk_size": int(info["chunks_size"]),
        "task_index_by_instruction": task_index_by_instruction,
        "cache_root": cache_root,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_episode_identity, metadata, **kwargs): int(metadata["episode_index"])
            for metadata in episodes
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            identities.append(future.result())
            if completed % 50 == 0 or completed == EXPECTED_EPISODES:
                print(
                    json.dumps(
                        {
                            "mapped_action_hashes": completed,
                            "total": EXPECTED_EPISODES,
                            "elapsed_seconds": round(time.monotonic() - started, 2),
                        }
                    ),
                    flush=True,
                )
    identities.sort(key=lambda row: int(row["lerobot_episode_index"]))
    if len({row["action_sha256"] for row in identities}) != EXPECTED_EPISODES:
        raise ValueError("LeRobot action hashes are not unique")
    unknown_hashes = [row for row in identities if row["action_sha256"] not in annotation_by_hash]
    if unknown_hashes:
        raise ValueError(f"LeRobot contains {len(unknown_hashes)} unknown action hashes")

    episode_records: list[dict[str, Any]] = []
    for identity in identities:
        annotation = annotation_by_hash[identity["action_sha256"]]
        release_task = release_task_by_instruction[identity["instruction"]]
        if (
            annotation["suite"] != release_task["suite"]
            or int(annotation["official_task_index"]) != int(release_task["task_index"])
            or int(annotation["episode_length"]) != int(identity["episode_length"])
        ):
            raise ValueError(f"Task/length identity mismatch for LeRobot episode {identity['lerobot_episode_index']}")
        episode_records.append(
            {
                "lerobot_repo_id": args.repo_id,
                "lerobot_revision": args.revision,
                "lerobot_episode_index": int(identity["lerobot_episode_index"]),
                "lerobot_task_index": int(identity["lerobot_task_index"]),
                "lerobot_dataset_from_index": int(identity["dataset_from_index"]),
                "lerobot_dataset_to_index_exclusive": int(identity["dataset_to_index_exclusive"]),
                "episode_length": int(identity["episode_length"]),
                "instruction": identity["instruction"],
                "suite": annotation["suite"],
                "annotation_task_index": int(annotation["official_task_index"]),
                "annotation_episode_index": int(annotation["official_episode_index"]),
                "action_sha256": identity["action_sha256"],
                "parquet_relative_path": identity["parquet_relative_path"],
                "identity_read_source": identity["identity_read_source"],
            }
        )
    episode_mapping = pd.DataFrame(episode_records).sort_values("lerobot_episode_index")
    if (
        len(episode_mapping) != EXPECTED_EPISODES
        or not episode_mapping["lerobot_episode_index"].is_unique
        or not episode_mapping["action_sha256"].is_unique
        or episode_mapping[["suite", "annotation_episode_index"]].duplicated().any()
    ):
        raise ValueError("Episode mapping is not one-to-one")

    if args.cross_check_libero10_mapping is not None:
        existing = json.loads(args.cross_check_libero10_mapping.resolve(strict=True).read_text())
        existing_by_index = {int(row["lerobot_episode_index"]): row for row in existing["episodes"]}
        subset = episode_mapping.loc[episode_mapping["suite"].eq("libero_10")]
        if set(subset["lerobot_episode_index"]) != set(existing_by_index):
            raise ValueError("Four-suite mapping does not select the existing LIBERO-10 population")
        for row in subset.itertuples(index=False):
            old = existing_by_index[int(row.lerobot_episode_index)]
            if (
                row.action_sha256 != old["action_sha256"]
                or int(row.annotation_task_index) != int(old["annotation_task_index"])
                or int(row.annotation_episode_index) != int(old["annotation_episode_index"])
            ):
                raise ValueError("Four-suite mapping differs from validated LIBERO-10 mapping")

    episode_shards = release_episodes.rename(
        columns={"task_index": "annotation_task_index", "episode_index": "annotation_episode_index"}
    )
    episode_mapping = episode_mapping.merge(
        episode_shards[
            [
                "suite",
                "annotation_task_index",
                "annotation_episode_index",
                "shard_path",
                "shard_sha256",
            ]
        ],
        on=["suite", "annotation_task_index", "annotation_episode_index"],
        how="left",
        validate="one_to_one",
    )
    if episode_mapping["shard_path"].isna().any():
        raise ValueError("A selected LeRobot episode lacks its frozen mask shard")

    selected_keys = episode_mapping[
        [
            "lerobot_episode_index",
            "lerobot_task_index",
            "lerobot_dataset_from_index",
            "suite",
            "annotation_task_index",
            "annotation_episode_index",
            "action_sha256",
        ]
    ]
    frame_mapping = release_frames.rename(
        columns={
            "task_index": "annotation_task_index",
            "episode_index": "annotation_episode_index",
            "frame_index": "annotation_frame_index",
        }
    ).merge(
        selected_keys,
        on=["suite", "annotation_task_index", "annotation_episode_index"],
        how="inner",
        validate="many_to_one",
    )
    frame_mapping["lerobot_frame_index"] = frame_mapping["annotation_frame_index"].astype(np.int64)
    frame_mapping["lerobot_dataset_index"] = (
        frame_mapping["lerobot_dataset_from_index"] + frame_mapping["lerobot_frame_index"]
    )
    frame_mapping = frame_mapping[
        [
            "lerobot_dataset_index",
            "lerobot_episode_index",
            "lerobot_task_index",
            "lerobot_frame_index",
            "suite",
            "annotation_task_index",
            "annotation_episode_index",
            "annotation_frame_index",
            "raw_state_index",
            "mask_shard_path",
            "mask_shard_row",
            "action_sha256",
        ]
    ].sort_values("lerobot_dataset_index")
    if (
        len(frame_mapping) != EXPECTED_FRAMES
        or frame_mapping["lerobot_dataset_index"].tolist() != list(range(EXPECTED_FRAMES))
        or not frame_mapping["lerobot_dataset_index"].is_unique
    ):
        raise ValueError("Frame mapping does not cover the exact contiguous LeRobot population")

    selected_pairs = set(zip(episode_mapping["suite"], episode_mapping["annotation_episode_index"], strict=True))
    annotation_hash_by_pair = {
        (row["suite"], int(row["official_episode_index"])): row["action_sha256"] for row in annotation_rows
    }
    unmatched = release_episodes.loc[
        ~release_episodes.apply(lambda row: (row["suite"], int(row["episode_index"])) in selected_pairs, axis=1)
    ].copy()
    unmatched["action_sha256"] = unmatched.apply(
        lambda row: annotation_hash_by_pair[(row["suite"], int(row["episode_index"]))], axis=1
    )
    unmatched["reason"] = "not_present_in_frozen_official_lerobot_revision"
    if len(unmatched) != EXPECTED_ANNOTATION_EPISODES - EXPECTED_EPISODES:
        raise ValueError("Unexpected unmatched annotation episode count")

    episode_path = output_root / "lerobot_episode_mapping.parquet"
    frame_path = output_root / "lerobot_frame_mapping.parquet"
    unmatched_path = output_root / "annotation_episodes_not_in_lerobot.parquet"
    episode_mapping.to_parquet(episode_path, index=False, compression="zstd")
    frame_mapping.to_parquet(frame_path, index=False, compression="zstd")
    unmatched.to_parquet(unmatched_path, index=False, compression="zstd")

    per_suite: dict[str, dict[str, int]] = {}
    for suite, expected in EXPECTED_PER_SUITE.items():
        episodes_count = int(episode_mapping["suite"].eq(suite).sum())
        frames_count = int(frame_mapping["suite"].eq(suite).sum())
        per_suite[suite] = {"episodes": episodes_count, "frames": frames_count}
        if per_suite[suite] != expected:
            raise ValueError(f"Unexpected LeRobot selection counts for {suite}: {per_suite[suite]}")
    local_remote = Counter(row["identity_read_source"] for row in identities)
    summary = {
        "status": "PASS",
        "schema": "libero_stage_relevant_masks.lerobot_policy_selection.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_policy": (
            "Use exactly the official LeRobot population for future four-suite policy training; "
            "retain all 2000 annotated episodes as the larger annotation/teacher source."
        ),
        "join_key": "sha256(contiguous little-endian float32 full episode action array)",
        "lerobot": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "episodes": EXPECTED_EPISODES,
            "frames": EXPECTED_FRAMES,
            "tasks": 40,
        },
        "annotation_release": {
            "episodes": EXPECTED_ANNOTATION_EPISODES,
            "frames": EXPECTED_ANNOTATION_FRAMES,
            "selected_episodes": EXPECTED_EPISODES,
            "unselected_retained_episodes": EXPECTED_ANNOTATION_EPISODES - EXPECTED_EPISODES,
        },
        "per_suite": per_suite,
        "checks": {
            "all_lerobot_episode_action_hashes_matched": True,
            "episode_mapping_one_to_one": True,
            "selected_annotation_episodes_unique": True,
            "frame_mapping_contiguous_0_to_273464": True,
            "every_selected_episode_has_mask_shard": True,
            "libero10_cross_check_passed": args.cross_check_libero10_mapping is not None,
            "source_release_modified_or_pruned": False,
        },
        "identity_read_sources": dict(sorted(local_remote.items())),
        "sources": {
            "lerobot_info": {"path": "meta/info.json", "sha256": sha256_file(info_path)},
            "lerobot_episodes": {
                "path": "meta/episodes.jsonl",
                "sha256": sha256_file(episodes_path),
            },
            "lerobot_tasks": {"path": "meta/tasks.jsonl", "sha256": sha256_file(tasks_path)},
            "annotation_action_index_sha256": sha256_file(annotation_path),
            "release_episode_index_sha256": sha256_file(release_episode_path),
            "release_frame_manifest_sha256": sha256_file(release_frame_path),
            "release_task_catalog_sha256": sha256_file(release_task_path),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {
            episode_path.name: {
                "rows": len(episode_mapping),
                "sha256": sha256_file(episode_path),
            },
            frame_path.name: {"rows": len(frame_mapping), "sha256": sha256_file(frame_path)},
            unmatched_path.name: {"rows": len(unmatched), "sha256": sha256_file(unmatched_path)},
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    summary_path = output_root / "selection_summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
