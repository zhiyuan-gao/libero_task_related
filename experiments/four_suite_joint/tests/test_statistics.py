from __future__ import annotations

from four_suite_experiments.statistics import pooled_population_moments
import numpy as np


def test_pooled_population_moments_matches_direct_population() -> None:
    first = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    second = np.asarray([[10.0, -2.0], [14.0, 2.0]])
    direct = np.concatenate([first, second], axis=0)
    mean, std = pooled_population_moments(
        [len(first), len(second)],
        [first.mean(axis=0), second.mean(axis=0)],
        [first.std(axis=0), second.std(axis=0)],
    )
    np.testing.assert_allclose(mean, direct.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(std, direct.std(axis=0), rtol=0, atol=1e-12)


def test_between_population_mean_shift_is_included() -> None:
    mean, std = pooled_population_moments(
        [1, 1],
        [np.asarray([0.0]), np.asarray([2.0])],
        [np.asarray([0.0]), np.asarray([0.0])],
    )
    np.testing.assert_array_equal(mean, np.asarray([1.0]))
    np.testing.assert_array_equal(std, np.asarray([1.0]))
