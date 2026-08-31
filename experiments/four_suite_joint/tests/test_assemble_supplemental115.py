from __future__ import annotations

import json
from pathlib import Path

from four_suite_experiments.assemble_supplemental115 import PopulationSpec
from four_suite_experiments.assemble_supplemental115 import assemble_supplemental115
import numpy as np
import pandas as pd
import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path, PopulationSpec, np.ndarray, np.ndarray]:
    spec = PopulationSpec(
        revision="test-revision",
        base_episodes=2,
        base_frames=3,
        supplemental_episodes=1,
        supplemental_frames=2,
        source_episodes=4,
        source_frames=7,
        base_geometry_valid=2,
        base_motion_valid=2,
        target_geometry_valid=4,
        target_geometry_invalid=1,
        target_motion_valid=4,
        source_geometry_valid=6,
        source_motion_valid=6,
        geometry_dim=2,
        motion_dim=3,
    )
    base = tmp_path / "base"
    bundle = tmp_path / "bundle"
    base_rows = [
        {"episode_index": 0, "length": 1, "tasks": ["task 0"]},
        {"episode_index": 1, "length": 2, "tasks": ["task 1"]},
    ]
    all_rows = [
        *base_rows,
        {"episode_index": 2, "length": 2, "tasks": ["task 2"]},
        {"episode_index": 3, "length": 2, "tasks": ["task 3"]},
    ]
    info = {
        "total_episodes": 2,
        "total_frames": 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:2"},
    }
    _write_json(base / "meta/info.json", info)
    _write_jsonl(base / "meta/episodes.jsonl", base_rows)
    _write_json(base / "meta/stats.json", {"frozen": True})
    _write_jsonl(base / "meta/tasks.jsonl", [{"task_index": index} for index in range(4)])
    for episode_index in range(2):
        path = base / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"base-{episode_index}".encode())

    source_info = dict(info)
    source_info.update(total_episodes=4, total_frames=7, splits={"train": "0:4"})
    _write_json(bundle / "dataset_delta/meta/info.json", source_info)
    _write_jsonl(bundle / "dataset_delta/meta/episodes.jsonl", all_rows)
    for episode_index in range(2, 4):
        path = bundle / "dataset_delta/data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"delta-{episode_index}".encode())

    artifacts = bundle / "training_artifacts/task_relevant"
    artifacts.mkdir(parents=True)
    sample_ids = [f"sample-{index}" for index in range(7)]
    policy = pd.DataFrame(
        {
            "lerobot_dataset_index": range(7),
            "sample_id": sample_ids,
            "semantic_subtask": [f"subtask-{index}" for index in range(7)],
        }
    )
    policy.to_parquet(artifacts / "policy_manifest.parquet", index=False)
    cursor = 0
    records = []
    for row in all_rows:
        length = int(row["length"])
        records.append(
            {
                "lerobot_episode_index": int(row["episode_index"]),
                "dataset_from_index": cursor,
                "dataset_to_index_exclusive": cursor + length,
                "episode_length": length,
            }
        )
        cursor += length
    _write_json(
        artifacts / "episode_mapping.json",
        {
            "status": "PASS",
            "hf_repo_id": "physical-intelligence/libero",
            "hf_revision": spec.revision,
            "mapped_episode_count": 4,
            "mapped_frame_count": 7,
            "episodes": records,
        },
    )

    geometry_valid = np.array([True, True, False, True, True, True, True])
    geometry_rows = np.full(7, -1, dtype=np.int64)
    geometry_rows[geometry_valid] = np.arange(6)
    geometry = pd.DataFrame(
        {
            "lerobot_dataset_index": range(7),
            "sample_id": sample_ids,
            "geometry_valid": geometry_valid,
            "target_memmap_path": ["geometry_targets_fp32.npy"] * 7,
            "target_memmap_row": geometry_rows,
            "target_dim": [2] * 7,
            "target_dtype": ["float32"] * 7,
        }
    )
    geometry.to_parquet(artifacts / "geometry_index.parquet", index=False)
    geometry_targets = np.arange(12, dtype=np.float32).reshape(6, 2)
    np.save(artifacts / "geometry_targets_fp32.npy", geometry_targets)
    _write_json(
        artifacts / "geometry_normalization.json",
        {
            "status": "PASS",
            "split": "train",
            "sample_count": 6,
            "statistics_source_sample_count": 2,
            "feature_dim": 2,
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
        },
    )

    # The valid sample order is a target-store prefix even though invalid
    # policy samples are absent from the Motion index.
    motion_sample_indices = [0, 1, 3, 4, 5, 6]
    motion = pd.DataFrame(
        {
            "sample_id": [sample_ids[index] for index in motion_sample_indices],
            "target_memmap_path": ["motion_targets_fp32.npy"] * 6,
            "target_memmap_row": range(6),
            "target_dim": [3] * 6,
            "target_dtype": ["float32"] * 6,
        }
    )
    motion.to_parquet(artifacts / "motion_index.parquet", index=False)
    motion_targets = np.arange(18, dtype=np.float32).reshape(6, 3)
    np.save(artifacts / "motion_targets_fp32.npy", motion_targets)
    _write_json(
        artifacts / "motion_normalization.json",
        {
            "count": 6,
            "statistics_source_count": 2,
            "dtype": "float32",
            "feature_dim": 3,
            "finite": True,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    )
    return base, bundle, spec, geometry_targets, motion_targets


def test_assemble_supplemental115_selects_exact_prefix_without_modifying_sources(tmp_path) -> None:
    base, bundle, spec, geometry_targets, motion_targets = _fixture(tmp_path)
    base_info_before = (base / "meta/info.json").read_bytes()
    source_geometry_before = (bundle / "training_artifacts/task_relevant/geometry_targets_fp32.npy").read_bytes()
    output = tmp_path / "assembled"

    report = assemble_supplemental115(
        base_root=base,
        bundle_root=bundle,
        output_root=output,
        spec=spec,
    )

    assert report["status"] == "PASS"
    assert report["episodes"] == 3
    assert report["frames"] == 5
    assert len(list((output / "lerobot/test-revision/data").rglob("episode_*.parquet"))) == 3
    assert np.array_equal(
        np.load(output / "artifacts/task_relevant/geometry_targets_fp32.npy"),
        geometry_targets[:4],
    )
    assert np.array_equal(
        np.load(output / "artifacts/task_relevant/motion_targets_fp32.npy"),
        motion_targets[:4],
    )
    geometry = pd.read_parquet(output / "artifacts/task_relevant/geometry_index.parquet")
    assert set(geometry.loc[geometry["geometry_valid"], "target_memmap_path"]) == {
        str(output / "artifacts/task_relevant/geometry_targets_fp32.npy")
    }
    assert json.loads((output / "artifacts/task_relevant/provenance.json").read_text())["selection"][
        "episode_indices"
    ] == [2, 2]
    assert (base / "meta/info.json").read_bytes() == base_info_before
    assert (
        bundle / "training_artifacts/task_relevant/geometry_targets_fp32.npy"
    ).read_bytes() == source_geometry_before

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assemble_supplemental115(
            base_root=base,
            bundle_root=bundle,
            output_root=output,
            spec=spec,
        )


def test_assemble_supplemental115_rejects_nonprefix_motion_targets(tmp_path) -> None:
    base, bundle, spec, _, _ = _fixture(tmp_path)
    path = bundle / "training_artifacts/task_relevant/motion_index.parquet"
    motion = pd.read_parquet(path)
    motion.loc[motion["sample_id"].eq("sample-4"), "target_memmap_row"] = 5
    motion.loc[motion["sample_id"].eq("sample-6"), "target_memmap_row"] = 3
    motion.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="exact source prefix"):
        assemble_supplemental115(
            base_root=base,
            bundle_root=bundle,
            output_root=tmp_path / "assembled",
            spec=spec,
        )
    assert not (tmp_path / "assembled").exists()
