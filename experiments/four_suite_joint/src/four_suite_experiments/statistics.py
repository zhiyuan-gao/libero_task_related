"""Exact pooling of population moments without reopening large target arrays."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np


def pooled_population_moments(
    counts: Sequence[int],
    means: Sequence[np.ndarray],
    stds: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine population means/stds, including between-group mean variance."""

    if not counts or len(counts) != len(means) or len(counts) != len(stds):
        raise ValueError("counts, means, and stds must be non-empty and equally sized")
    if any(int(count) <= 0 for count in counts):
        raise ValueError("population counts must be positive")
    arrays_mean = [np.asarray(value, dtype=np.float64) for value in means]
    arrays_std = [np.asarray(value, dtype=np.float64) for value in stds]
    shape = arrays_mean[0].shape
    if any(value.shape != shape for value in (*arrays_mean, *arrays_std)):
        raise ValueError("all population moment arrays must have the same shape")
    if any(not np.isfinite(value).all() for value in (*arrays_mean, *arrays_std)):
        raise ValueError("population moments must be finite")
    if any((value < 0).any() for value in arrays_std):
        raise ValueError("population standard deviations must be non-negative")

    total = int(sum(int(count) for count in counts))
    pooled_mean = (
        sum(int(count) * mean for count, mean in zip(counts, arrays_mean, strict=True))
        / total
    )
    pooled_variance = (
        sum(
            int(count) * (std**2 + (mean - pooled_mean) ** 2)
            for count, mean, std in zip(counts, arrays_mean, arrays_std, strict=True)
        )
        / total
    )
    # Guard only against floating-point roundoff; a materially negative value is invalid.
    if float(pooled_variance.min()) < -1e-12:
        raise ValueError("pooled population variance is negative")
    pooled_std = np.sqrt(np.maximum(pooled_variance, 0.0))
    return pooled_mean, pooled_std


def pool_geometry_normalizations(paths: Sequence[Path]) -> dict:
    records = [json.loads(path.read_text()) for path in paths]
    if any(
        record.get("status") != "PASS" or record.get("split") != "train"
        for record in records
    ):
        raise ValueError("Geometry source normalizations must be PASS train statistics")
    dimensions = {int(record["feature_dim"]) for record in records}
    floors = {float(record["sigma_floor"]) for record in records}
    if len(dimensions) != 1 or len(floors) != 1:
        raise ValueError("Geometry normalization dimensions or sigma floors differ")
    counts = [int(record["sample_count"]) for record in records]
    mean, raw_std = pooled_population_moments(
        counts,
        [np.asarray(record["mean"]) for record in records],
        [np.asarray(record["raw_std"]) for record in records],
    )
    sigma_floor = floors.pop()
    std = np.maximum(raw_std, sigma_floor)
    return {
        "schema": "four_suite_geometry_normalization_v1",
        "status": "PASS",
        "split": "train",
        "valid_samples_only": True,
        "sample_count": sum(counts),
        "feature_dim": dimensions.pop(),
        "sigma_floor": sigma_floor,
        "floored_dimensions": int((raw_std < sigma_floor).sum()),
        "mean": mean.tolist(),
        "raw_std": raw_std.tolist(),
        "std": std.tolist(),
        "source_normalizations": [str(path.resolve()) for path in paths],
    }


def pool_motion_normalizations(paths: Sequence[Path]) -> dict:
    records = [json.loads(path.read_text()) for path in paths]
    if any(
        record.get("finite") is not True
        or record.get("dtype") != "float32"
        or int(record.get("feature_dim", -1)) != 256
        for record in records
    ):
        raise ValueError("Motion source normalizations are incompatible")
    counts = [int(record["count"]) for record in records]
    mean, std = pooled_population_moments(
        counts,
        [np.asarray(record["mean"]) for record in records],
        [np.asarray(record["std"]) for record in records],
    )
    return {
        "schema": "four_suite_motion_normalization_v1",
        "count": sum(counts),
        "dtype": "float32",
        "feature_dim": 256,
        "finite": True,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "source_normalizations": [str(path.resolve()) for path in paths],
    }
