"""Read-only preflight for a prepared target bundle and experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .configs import blocked_action_groups
from .configs import build_train_config
from .configs import expected_target_scope
from .constants import FOUR_SUITE_EPISODES
from .constants import FOUR_SUITE_FRAMES
from .constants import FOUR_SUITE_GEOMETRY_VALID
from .constants import FOUR_SUITE_MOTION_VALID
from .constants import LIBERO_REVISION
from .data_overlay import FourSuiteGeometryTargetIndex
from .data_overlay import FourSuiteMotionTargetIndex
from .paths import ArtifactPaths
from .paths import SourcePaths
from .prepare_joint_artifacts import sha256_file


def validate_artifacts(artifacts: ArtifactPaths, *, target_scope: str) -> dict:
    missing = [str(path) for path in artifacts.all_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"artifact bundle is incomplete: {missing}")
    provenance = json.loads(artifacts.provenance.read_text())
    motion_storage = provenance.get("motion_storage", {})
    if (
        provenance.get("schema") != "four_suite_joint_artifact_provenance_v2"
        or motion_storage.get("schema") != "sample_id_to_readonly_memmap_v1"
        or motion_storage.get("bit_exact") is not True
        or motion_storage.get("read_only") is not True
        or int(motion_storage.get("sample_count", -1)) != FOUR_SUITE_MOTION_VALID
        or int(motion_storage.get("feature_dim", -1)) != 256
        or motion_storage.get("dtype") != "float32"
        or int(motion_storage.get("source_shard_count", 0)) <= 0
        or provenance.get("status") != "PASS"
        or provenance.get("target_scope") != target_scope
    ):
        raise ValueError(
            f"artifact target scope/status differs: expected={target_scope}, "
            f"observed={provenance.get('target_scope')}/{provenance.get('status')}"
        )
    expected_counts = {
        "episode_count": FOUR_SUITE_EPISODES,
        "frame_count": FOUR_SUITE_FRAMES,
        "geometry_valid_count": FOUR_SUITE_GEOMETRY_VALID,
        "motion_valid_count": FOUR_SUITE_MOTION_VALID,
    }
    for key, expected in expected_counts.items():
        if int(provenance.get(key, -1)) != expected:
            raise ValueError(f"provenance {key} differs: {provenance.get(key)} != {expected}")
    for name, expected_hash in provenance.get("artifact_sha256", {}).items():
        path = artifacts.root / name
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ValueError(f"artifact hash mismatch: {path}")

    mapping = json.loads(artifacts.episode_mapping.read_text())
    if len(mapping.get("episodes", [])) != FOUR_SUITE_EPISODES:
        raise ValueError("episode mapping count differs")
    geometry_frame = pd.read_parquet(
        artifacts.geometry_index,
        columns=[
            "sample_id",
            "lerobot_dataset_index",
            "geometry_valid",
            "target_memmap_path",
        ],
    )
    motion_frame = pd.read_parquet(
        artifacts.motion_index,
        columns=["sample_id", "target_memmap_path", "target_memmap_row"],
    )
    if (
        len(geometry_frame) != FOUR_SUITE_FRAMES
        or int(geometry_frame["geometry_valid"].sum()) != FOUR_SUITE_GEOMETRY_VALID
    ):
        raise ValueError("Geometry index counts differ")
    if len(motion_frame) != FOUR_SUITE_MOTION_VALID:
        raise ValueError("Motion index count differs")
    for column, frame in (
        ("target_memmap_path", geometry_frame),
        ("target_memmap_path", motion_frame),
    ):
        for value in frame[column].dropna().drop_duplicates():
            path = Path(str(value))
            if not path.is_absolute():
                path = artifacts.root / path
            if not path.is_file():
                raise FileNotFoundError(path)

    geometry = FourSuiteGeometryTargetIndex(artifacts.geometry_index, artifacts.geometry_normalization)
    for dataset_index in (0, 101_469, FOUR_SUITE_FRAMES - 1):
        target, valid, sample_id = geometry.target_by_dataset_index(dataset_index)
        if valid and target is None:
            raise ValueError(f"Geometry smoke lookup failed: {sample_id}")
    motion = FourSuiteMotionTargetIndex(
        artifacts.motion_index,
        artifacts.motion_normalization,
        expected_count=FOUR_SUITE_MOTION_VALID,
    )
    for sample_id in (
        str(motion_frame.iloc[0]["sample_id"]),
        str(motion_frame.iloc[-1]["sample_id"]),
    ):
        target, valid = motion.target_by_sample_id(sample_id)
        if not valid or target is None:
            raise ValueError(f"Motion smoke lookup failed: {sample_id}")
    return {
        "status": "PASS",
        "target_scope": target_scope,
        **expected_counts,
        "geometry_memmaps": int(geometry_frame["target_memmap_path"].dropna().nunique()),
        "motion_memmaps": int(motion_frame["target_memmap_path"].nunique()),
    }


def validate_lerobot_snapshot(root: Path, *, require_complete: bool) -> dict:
    """Separate complete metadata from the episode files needed for training."""

    root = root.resolve()
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata is incomplete under {root}")
    info = json.loads(info_path.read_text())
    if root.name != LIBERO_REVISION:
        raise ValueError(f"LeRobot snapshot revision differs: {root.name}")
    if int(info.get("total_episodes", -1)) != FOUR_SUITE_EPISODES:
        raise ValueError("LeRobot snapshot episode metadata differs")
    if int(info.get("total_frames", -1)) != FOUR_SUITE_FRAMES:
        raise ValueError("LeRobot snapshot frame metadata differs")
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    if len(episodes) != FOUR_SUITE_EPISODES:
        raise ValueError("LeRobot episodes.jsonl population differs")
    chunk_size = int(info.get("chunks_size", 1_000))
    missing = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        path = root / f"data/chunk-{episode_index // chunk_size:03d}/episode_{episode_index:06d}.parquet"
        if not path.is_file():
            missing.append(path)
    report = {
        "lerobot_root": str(root),
        "episode_metadata_count": len(episodes),
        "episode_files_present": len(episodes) - len(missing),
        "episode_files_missing": len(missing),
        "training_data_ready": not missing,
    }
    if missing and require_complete:
        raise FileNotFoundError(
            f"official four-suite snapshot is incomplete: {len(missing)} episode files missing; "
            f"first missing={missing[0]}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--variant",
        choices=("trqc", "whole_scene", "no_query_access"),
        required=True,
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=2,
        help="Config validation only; no optimizer runs.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    args = parser.parse_args()
    target_scope = expected_target_scope(args.variant)
    source_paths = SourcePaths.defaults(args.artifact_dir, target_scope=target_scope)
    artifacts = ArtifactPaths(source_paths.artifact_dir)
    report = validate_artifacts(artifacts, target_scope=target_scope)
    report.update(validate_lerobot_snapshot(source_paths.lerobot_root, require_complete=False))
    config = build_train_config(
        variant=args.variant,
        artifacts=artifacts,
        exp_name="preflight_only",
        num_train_steps=args.num_train_steps,
        warmup_steps=args.warmup_steps,
        checkpoint_base_dir=artifacts.root / "preflight_checkpoints_not_created",
        lerobot_root=source_paths.lerobot_root,
        num_workers=0,
        wandb_enabled=False,
    )
    report.update(
        {
            "variant": args.variant,
            "target_scope": target_scope,
            "blocked_action_groups": sorted(blocked_action_groups(args.variant)),
            "config_name": config.name,
            "optimizer_steps_executed": 0,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
