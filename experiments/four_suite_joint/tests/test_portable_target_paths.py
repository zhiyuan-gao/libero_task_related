from __future__ import annotations

import json
from pathlib import Path

from four_suite_experiments.data_overlay import FourSuiteMotionTargetIndex
from four_suite_experiments.paths import SourcePaths
from four_suite_experiments.prepare_joint_artifacts import _absolute_path
from four_suite_experiments.prepare_joint_artifacts import _validate_source_scope
from four_suite_experiments.prepare_joint_artifacts import materialize_motion_memmap
from four_suite_experiments.prepare_joint_artifacts import verify_motion_memmap
import numpy as np
import pandas as pd
import pytest


def test_relative_target_path_resolves_below_transferred_index(tmp_path: Path) -> None:
    cache = tmp_path / "motion_libero10_full_v1"
    index = cache / "index.parquet"
    target = cache / "shards/motion_00000_00999.npz"
    target.parent.mkdir(parents=True)
    target.touch()

    assert _absolute_path("shards/motion_00000_00999.npz", index) == str(target.resolve())


def test_stale_absolute_target_path_rebases_to_hpc_cache(tmp_path: Path) -> None:
    cache = tmp_path / "motion_libero_goal_object_spatial_v1"
    index = cache / "index.parquet"
    target = cache / "shards/motion_158000_158853.npz"
    target.parent.mkdir(parents=True)
    target.touch()
    stale = "/workspace/vla/old/policy_aux_v1/motion_libero_goal_object_spatial_v1/shards/motion_158000_158853.npz"

    assert _absolute_path(stale, index) == str(target.resolve())


def test_unrelocatable_absolute_target_path_fails(tmp_path: Path) -> None:
    index = tmp_path / "motion_libero10_full_v1/index.parquet"

    with pytest.raises(FileNotFoundError, match="cannot be rebased"):
        _absolute_path("/unrelated/cache/missing.npz", index)


def test_whole_scene_source_requires_explicit_scope(tmp_path: Path) -> None:
    source = tmp_path / "index.parquet"
    frame = pd.DataFrame({"sample_id": ["a"]})
    with pytest.raises(ValueError, match="lacks an explicit target_scope"):
        _validate_source_scope(
            frame,
            expected_scope="whole_scene",
            source_index=source,
        )


def test_source_scope_mismatch_fails(tmp_path: Path) -> None:
    source = tmp_path / "index.parquet"
    frame = pd.DataFrame({"target_scope": ["task_relevant"]})
    with pytest.raises(ValueError, match="source target scope differs"):
        _validate_source_scope(
            frame,
            expected_scope="whole_scene",
            source_index=source,
        )


def test_whole_scene_sources_are_independent_from_task_relevant(monkeypatch, tmp_path: Path) -> None:
    annotation = tmp_path / "annotation"
    monkeypatch.setenv("FOUR_SUITE_ANNOTATION_ROOT", str(annotation))
    monkeypatch.delenv("FOUR_SUITE_ARTIFACT_DIR", raising=False)
    task = SourcePaths.defaults(tmp_path / "prepared/task", target_scope="task_relevant")
    whole = SourcePaths.defaults(tmp_path / "prepared/whole", target_scope="whole_scene")
    assert task.target_scope == "task_relevant"
    assert whole.target_scope == "whole_scene"
    assert len(task.geometry_indices) == 2
    assert len(task.motion_indices) == 2
    assert whole.geometry_indices == (
        annotation / "policy_aux_v1/geometry_whole_scene_four_suite_v1/target_index.parquet",
    )
    assert whole.motion_indices == (annotation / "policy_aux_v1/motion_whole_scene_four_suite_v1/index.parquet",)


def test_motion_shards_materialize_to_one_exact_portable_memmap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first_ids = np.asarray(["sample-c", "sample-a"])
    first_targets = np.arange(2 * 256, dtype=np.float32).reshape(2, 256)
    second_ids = np.asarray(["sample-b"])
    second_targets = (1_000 + np.arange(256, dtype=np.float32)).reshape(1, 256)
    first = source / "first.npz"
    second = source / "second.npz"
    np.savez(first, sample_id=first_ids, motion_target_fp32=first_targets)
    np.savez(second, sample_id=second_ids, motion_target_fp32=second_targets)
    frame = pd.DataFrame(
        {
            "sample_id": ["sample-a", "sample-b", "sample-c"],
            "target_shard_path": [str(first), str(second), str(first)],
            "target_shard_row": [1, 0, 0],
            "target_dim": [256, 256, 256],
            "target_dtype": ["float32", "float32", "float32"],
        }
    )
    output = tmp_path / "artifacts/motion_targets_fp32.npy"
    portable = materialize_motion_memmap(frame, output)
    verification = verify_motion_memmap(frame, output)
    portable.to_parquet(tmp_path / "artifacts/motion_index.parquet", index=False)
    normalization = {
        "count": 3,
        "dtype": "float32",
        "feature_dim": 256,
        "finite": True,
        "mean": np.zeros(256, dtype=np.float32).tolist(),
        "std": np.ones(256, dtype=np.float32).tolist(),
    }
    (tmp_path / "artifacts/motion_normalization.json").write_text(json.dumps(normalization))

    assert portable["target_memmap_path"].unique().tolist() == [output.name]
    assert portable["target_memmap_row"].tolist() == [0, 1, 2]
    assert "target_shard_path" not in portable
    assert verification == {
        "schema": "sample_id_to_readonly_memmap_v1",
        "dtype": "float32",
        "feature_dim": 256,
        "sample_count": 3,
        "source_shard_count": 2,
        "bit_exact": True,
        "read_only": True,
    }
    targets = np.load(output, mmap_mode="r")
    np.testing.assert_array_equal(targets[0], first_targets[1])
    np.testing.assert_array_equal(targets[1], second_targets[0])
    np.testing.assert_array_equal(targets[2], first_targets[0])

    index = FourSuiteMotionTargetIndex(
        tmp_path / "artifacts/motion_index.parquet",
        tmp_path / "artifacts/motion_normalization.json",
        expected_count=3,
    )
    target, valid = index.target_by_sample_id("sample-b")
    assert valid
    np.testing.assert_array_equal(target, second_targets[0])
    assert index.target_by_sample_id("missing") == (None, False)
