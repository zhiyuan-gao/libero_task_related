from __future__ import annotations

from four_suite_experiments.whole_scene_motion import CLIP_LENGTH
from four_suite_experiments.whole_scene_motion import FEATURE_DIM
from four_suite_experiments.whole_scene_motion import NUM_LEVELS
from four_suite_experiments.whole_scene_motion import SOURCE_PATCH_COUNT
from four_suite_experiments.whole_scene_motion import paired_difference
from four_suite_experiments.whole_scene_motion import pool_source_uniform
from four_suite_experiments.whole_scene_motion import pool_whole_scene_levels
import numpy as np
import pytest
import torch


def test_uniform_pool_uses_all_source_patches() -> None:
    source = torch.arange(SOURCE_PATCH_COUNT, dtype=torch.float32)
    level = source.view(1, 1, SOURCE_PATCH_COUNT, 1).expand(
        1, CLIP_LENGTH, SOURCE_PATCH_COUNT, FEATURE_DIM
    )
    pooled = pool_source_uniform(level)
    assert pooled.shape == (FEATURE_DIM,)
    assert torch.equal(pooled, torch.full_like(pooled, source.mean()))


def test_all_levels_keep_expected_shape_and_float32() -> None:
    base = torch.arange(SOURCE_PATCH_COUNT, dtype=torch.float32).view(
        1, 1, SOURCE_PATCH_COUNT, 1
    )
    levels = [
        (base + level_index).expand(1, CLIP_LENGTH, SOURCE_PATCH_COUNT, FEATURE_DIM)
        for level_index in range(NUM_LEVELS)
    ]
    pooled = pool_whole_scene_levels(levels)
    assert pooled.shape == (NUM_LEVELS, FEATURE_DIM)
    assert pooled.dtype == np.float32
    for level_index in range(NUM_LEVELS):
        np.testing.assert_array_equal(
            pooled[level_index],
            np.full(
                FEATURE_DIM, (SOURCE_PATCH_COUNT - 1) / 2 + level_index, np.float32
            ),
        )


def test_pool_rejects_wrong_patch_population() -> None:
    level = torch.zeros(1, CLIP_LENGTH, SOURCE_PATCH_COUNT - 1, FEATURE_DIM)
    with pytest.raises(ValueError, match="source patches"):
        pool_source_uniform(level)


def test_paired_difference_detects_pooling_change() -> None:
    task = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    whole = task.copy()
    whole[1, 0] = 2.0
    report = paired_difference(whole, task)
    assert report["samples"] == 2
    assert report["exact_equal_rows"] == 1
    assert report["different_rows"] == 1
    assert report["l2"]["max"] == 2.0
