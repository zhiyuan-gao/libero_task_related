from __future__ import annotations

from pathlib import Path

from four_suite_experiments.prepare_joint_artifacts import _absolute_path
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
    stale = (
        "/workspace/vla/old/policy_aux_v1/"
        "motion_libero_goal_object_spatial_v1/shards/motion_158000_158853.npz"
    )

    assert _absolute_path(stale, index) == str(target.resolve())


def test_unrelocatable_absolute_target_path_fails(tmp_path: Path) -> None:
    index = tmp_path / "motion_libero10_full_v1/index.parquet"

    with pytest.raises(FileNotFoundError, match="cannot be rebased"):
        _absolute_path("/unrelated/cache/missing.npz", index)
