#!/usr/bin/env python3
"""Build the exact official OpenPI LIBERO-10 policy-training sample manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--episode-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_manifest.resolve(strict=True)
    mapping_path = args.episode_mapping.resolve(strict=True)
    source = pd.read_parquet(source_path)
    mapping_payload = json.loads(mapping_path.read_text())
    if mapping_payload.get("status") != "PASS":
        raise ValueError("Episode mapping must have PASS status")
    episode_records = mapping_payload["episodes"]
    episode_by_annotation_id = {row["annotation_episode_id"]: row for row in episode_records}
    if len(episode_by_annotation_id) != len(episode_records):
        raise ValueError("Annotation episode IDs are not unique in mapping")

    policy = source.loc[source["suite"].eq("libero_10") & source["episode_id"].isin(episode_by_annotation_id)].copy()
    if policy["sample_id"].duplicated().any():
        raise ValueError("Policy sample IDs are not unique")
    policy["lerobot_episode_index"] = policy["episode_id"].map(
        lambda value: episode_by_annotation_id[value]["lerobot_episode_index"]
    )
    policy["lerobot_task_index"] = policy["episode_id"].map(
        lambda value: episode_by_annotation_id[value]["lerobot_task_index"]
    )
    policy["lerobot_dataset_index"] = policy.apply(
        lambda row: (episode_by_annotation_id[row["episode_id"]]["dataset_from_index"] + int(row["frame_idx"])),
        axis=1,
    )
    policy["policy_train"] = True
    policy["split"] = "train"
    policy["smoke_selected"] = False
    policy["formal_selected"] = False
    policy["geometry_policy_extract_selected"] = policy["geometry_valid"].astype(bool)
    policy = policy.sort_values("lerobot_dataset_index").reset_index(drop=True)

    expected_frames = sum(int(row["episode_length"]) for row in episode_records)
    if len(policy) != expected_frames:
        raise ValueError(f"Expected {expected_frames} policy frames, found {len(policy)}")
    if not policy["lerobot_dataset_index"].is_unique:
        raise ValueError("LeRobot dataset indices are not unique")
    if policy["lerobot_dataset_index"].tolist() != list(range(expected_frames)):
        raise ValueError("LIBERO-10 policy dataset indices are not contiguous from zero")
    for episode_id, group in policy.groupby("episode_id", sort=False):
        expected_length = int(episode_by_annotation_id[episode_id]["episode_length"])
        if len(group) != expected_length or group["frame_idx"].tolist() != list(range(expected_length)):
            raise ValueError(f"Frame coverage mismatch for {episode_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.to_parquet(args.output, index=False, compression="zstd")
    report = {
        "status": "PASS",
        "schema": "openpi.libero10_policy_aux_manifest.v1",
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "episode_mapping": str(mapping_path),
        "episode_mapping_sha256": sha256_file(mapping_path),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "official_lerobot_episodes": len(episode_records),
        "policy_samples": len(policy),
        "geometry_valid_samples": int(policy["geometry_valid"].sum()),
        "geometry_invalid_samples": int((~policy["geometry_valid"]).sum()),
        "agent_mask_valid_samples": int(policy["agent_mask_valid"].sum()),
        "wrist_mask_valid_samples": int(policy["wrist_mask_valid"].sum()),
        "all_dataset_indices_unique_contiguous": True,
        "all_episode_frames_complete": True,
        "policy_distribution_rule": (
            "all and only frames from official physical-intelligence/libero episodes whose HF task index is 0..9"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
