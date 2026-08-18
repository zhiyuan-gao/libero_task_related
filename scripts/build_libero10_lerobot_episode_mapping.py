#!/usr/bin/env python3
"""Map official LeRobot LIBERO-10 episodes to frozen annotations by action hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def action_sha256(frame: pd.DataFrame) -> str:
    actions = np.stack(frame["actions"].to_numpy()).astype(np.float32, copy=False)
    return hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--annotation-episodes", type=Path, required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    snapshot = args.snapshot.resolve(strict=True)
    annotation_path = args.annotation_episodes.resolve(strict=True)

    annotation_rows = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    if len(annotation_rows) != 500:
        raise ValueError(f"Expected 500 annotation episodes, found {len(annotation_rows)}")
    annotation_by_hash = {row["action_sha256"]: row for row in annotation_rows}
    if len(annotation_by_hash) != 500:
        raise ValueError("Annotation action hashes are not unique")

    task_metadata = [json.loads(line) for line in (snapshot / "meta/tasks.jsonl").read_text().splitlines()]
    task_index_by_text = {row["task"]: int(row["task_index"]) for row in task_metadata}
    metadata = [json.loads(line) for line in (snapshot / "meta/episodes.jsonl").read_text().splitlines()]
    frame_cursor = 0
    episode_offsets = {}
    expected_libero10_episodes = set()
    for episode in metadata:
        index = int(episode["episode_index"])
        length = int(episode["length"])
        episode_offsets[index] = (frame_cursor, frame_cursor + length)
        frame_cursor += length
        episode_tasks = episode["tasks"]
        if len(episode_tasks) != 1:
            raise ValueError(f"Episode {index} does not have exactly one task")
        if task_index_by_text[episode_tasks[0]] < 10:
            expected_libero10_episodes.add(index)

    records = []
    for parquet_path in sorted(snapshot.glob("data/chunk-*/episode_*.parquet")):
        frame = pd.read_parquet(
            parquet_path,
            columns=["actions", "episode_index", "frame_index", "task_index"],
        )
        episode_indices = frame["episode_index"].unique()
        task_indices = frame["task_index"].unique()
        if len(episode_indices) != 1 or len(task_indices) != 1:
            raise ValueError(f"Mixed episode/task rows in {parquet_path}")
        lerobot_episode = int(episode_indices[0])
        task_index = int(task_indices[0])
        if task_index >= 10:
            continue
        observed_frames = frame["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed_frames, np.arange(len(frame))):
            raise ValueError(f"Non-contiguous frames in {parquet_path}")
        digest = action_sha256(frame)
        if digest not in annotation_by_hash:
            raise ValueError(f"Unknown LIBERO-10 action hash in {parquet_path}: {digest}")
        annotation = annotation_by_hash[digest]
        if int(annotation["episode_length"]) != len(frame):
            raise ValueError(f"Episode length mismatch in {parquet_path}")
        dataset_from, dataset_to = episode_offsets[lerobot_episode]
        if dataset_to - dataset_from != len(frame):
            raise ValueError(f"Metadata length mismatch in {parquet_path}")
        records.append(
            {
                "lerobot_episode_index": lerobot_episode,
                "lerobot_task_index": task_index,
                "dataset_from_index": dataset_from,
                "dataset_to_index_exclusive": dataset_to,
                "episode_length": len(frame),
                "action_sha256": digest,
                "annotation_task_index": int(annotation["official_task_index"]),
                "annotation_episode_index": int(annotation["official_episode_index"]),
                "annotation_episode_id": (
                    f"libero_10/task_{int(annotation['official_task_index']):02d}/"
                    f"episode_{int(annotation['official_episode_index']):06d}"
                ),
                "parquet_relative_path": str(parquet_path.relative_to(snapshot)),
            }
        )

    records.sort(key=lambda row: row["lerobot_episode_index"])
    if len({row["action_sha256"] for row in records}) != len(records):
        raise ValueError("Mapped action hashes are not unique")
    if len({row["lerobot_episode_index"] for row in records}) != len(records):
        raise ValueError("Mapped LeRobot episode indices are not unique")
    official_to_lerobot: dict[int, set[int]] = {}
    lerobot_to_official: dict[int, set[int]] = {}
    for row in records:
        official_to_lerobot.setdefault(row["annotation_task_index"], set()).add(row["lerobot_task_index"])
        lerobot_to_official.setdefault(row["lerobot_task_index"], set()).add(row["annotation_task_index"])
    if any(len(values) != 1 for values in official_to_lerobot.values()):
        raise ValueError(f"An official task maps to multiple LeRobot tasks: {official_to_lerobot}")
    if any(len(values) != 1 for values in lerobot_to_official.values()):
        raise ValueError(f"A LeRobot task maps to multiple official tasks: {lerobot_to_official}")
    task_index_mapping = {
        str(official): next(iter(lerobot)) for official, lerobot in sorted(official_to_lerobot.items())
    }
    mapped_episode_indices = {row["lerobot_episode_index"] for row in records}
    complete = mapped_episode_indices == expected_libero10_episodes
    if complete and (len(official_to_lerobot) != 10 or len(lerobot_to_official) != 10):
        raise ValueError("Complete mapping did not cover exactly ten tasks on both sides")
    if args.require_complete and not complete:
        raise ValueError(
            f"Incomplete mapping: mapped {len(records)} / {len(expected_libero10_episodes)} "
            "official LeRobot LIBERO-10 episodes"
        )

    mapped_hashes = {row["action_sha256"] for row in records}
    unmatched_annotations = [
        {
            "action_sha256": row["action_sha256"],
            "official_task_index": int(row["official_task_index"]),
            "official_episode_index": int(row["official_episode_index"]),
            "episode_length": int(row["episode_length"]),
        }
        for row in annotation_rows
        if row["action_sha256"] not in mapped_hashes
    ]

    payload = {
        "status": "PASS" if complete else "PARTIAL",
        "schema": "openpi.libero10_lerobot_episode_mapping.v1",
        "join_key": "sha256(contiguous little-endian float32 full episode action array)",
        "hf_repo_id": "physical-intelligence/libero",
        "hf_revision": args.hf_revision,
        "snapshot": str(snapshot),
        "annotation_episode_source": str(annotation_path),
        "mapped_episode_count": len(records),
        "expected_lerobot_libero10_episode_count": len(expected_libero10_episodes),
        "annotation_episode_count": len(annotation_rows),
        "annotation_episodes_absent_from_official_lerobot": len(unmatched_annotations),
        "mapped_frame_count": sum(row["episode_length"] for row in records),
        "task_index_mapping_official_to_lerobot": task_index_mapping,
        "unmatched_annotation_episodes": unmatched_annotations,
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in ("episodes", "unmatched_annotation_episodes")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
