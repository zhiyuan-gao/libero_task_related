from __future__ import annotations

from four_suite_experiments.constants import GEOMETRY_DIM
from four_suite_experiments.whole_scene_geometry import NUM_VIEWS
from four_suite_experiments.whole_scene_geometry import PATCH_COUNT
from four_suite_experiments.whole_scene_geometry import paired_difference
from four_suite_experiments.whole_scene_geometry import pool_whole_scene_geometry
import numpy as np
import pytest
import torch


def test_uniform_geometry_pool_and_valid_view_fusion() -> None:
    base = torch.arange(PATCH_COUNT, dtype=torch.float32).view(1, 1, PATCH_COUNT, 1)
    patches = base.expand(2, NUM_VIEWS, PATCH_COUNT, GEOMETRY_DIM).clone()
    patches[0, 1] += 100
    patches[1, 1] += 200
    valid = torch.tensor([[True, False], [True, True]])
    pooled = pool_whole_scene_geometry(patches, valid)
    expected_mean = (PATCH_COUNT - 1) / 2
    assert pooled.shape == (2, GEOMETRY_DIM)
    assert torch.equal(pooled[0], torch.full_like(pooled[0], expected_mean))
    assert torch.equal(pooled[1], torch.full_like(pooled[1], expected_mean + 100))


def test_uniform_geometry_pool_rejects_empty_views() -> None:
    patches = torch.zeros(1, NUM_VIEWS, PATCH_COUNT, GEOMETRY_DIM)
    with pytest.raises(ValueError, match="no valid view"):
        pool_whole_scene_geometry(patches, torch.zeros(1, NUM_VIEWS, dtype=torch.bool))


def test_geometry_paired_difference_detects_change() -> None:
    task = np.zeros((2, GEOMETRY_DIM), dtype=np.float32)
    whole = task.copy()
    whole[0, 4] = 3
    report = paired_difference(whole, task)
    assert report["samples"] == 2
    assert report["different_rows"] == 1
    assert report["l2"]["max"] == 3
