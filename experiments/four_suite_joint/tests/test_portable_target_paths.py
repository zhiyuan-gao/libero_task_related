from __future__ import annotations

from pathlib import Path

from four_suite_experiments.paths import SourcePaths
from four_suite_experiments.prepare_joint_artifacts import _absolute_path
from four_suite_experiments.prepare_joint_artifacts import _validate_source_scope
import pandas as pd
import pytest


def test_relative_target_path_resolves_below_transferred_index(tmp_path: Path) -> None:
    cache = tmp_path / "motion_libero10_full_v1"
    index = cache / "index.parquet"
    target = cache / "shards/motion_00000_00999.npz"
    target.parent.mkdir(parents=True)
    target.touch()

    assert _absolute_path("shards/motion_00000_00999.npz", index) == str(
        target.resolve()
    )


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


def test_whole_scene_sources_are_independent_from_task_relevant(
    monkeypatch, tmp_path: Path
) -> None:
    annotation = tmp_path / "annotation"
    monkeypatch.setenv("FOUR_SUITE_ANNOTATION_ROOT", str(annotation))
    monkeypatch.delenv("FOUR_SUITE_ARTIFACT_DIR", raising=False)
    task = SourcePaths.defaults(
        tmp_path / "prepared/task", target_scope="task_relevant"
    )
    whole = SourcePaths.defaults(
        tmp_path / "prepared/whole", target_scope="whole_scene"
    )
    assert task.target_scope == "task_relevant"
    assert whole.target_scope == "whole_scene"
    assert len(task.geometry_indices) == 2
    assert len(task.motion_indices) == 2
    assert whole.geometry_indices == (
        annotation
        / "policy_aux_v1/geometry_whole_scene_four_suite_v1/target_index.parquet",
    )
    assert whole.motion_indices == (
        annotation / "policy_aux_v1/motion_whole_scene_four_suite_v1/index.parquet",
    )
