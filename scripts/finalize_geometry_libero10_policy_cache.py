#!/usr/bin/env python3
"""Finalize/validate the full official LIBERO-10 Geometry policy target cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sigma-floor", type=float, default=1e-6)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    if args.sigma_floor <= 0:
        raise ValueError("sigma floor must be positive")

    source = pd.read_parquet(manifest_path).sort_values("lerobot_dataset_index").reset_index(drop=True)
    if len(source) != 101_469 or not source["policy_train"].astype(bool).all():
        raise ValueError("Expected the exact 101469-frame official LIBERO-10 policy manifest")
    if source["lerobot_dataset_index"].tolist() != list(range(len(source))):
        raise ValueError("Policy manifest dataset indices are not contiguous")
    if source["sample_id"].duplicated().any():
        raise ValueError("Policy manifest sample IDs are not unique")

    upstream_validation_path = cache_root / "cache_validation.json"
    upstream_index_path = cache_root / "index.parquet"
    upstream_statistics_path = cache_root / "target_statistics_train.json"
    shard_index_path = cache_root / "shard_index.json"
    upstream_validation = json.loads(upstream_validation_path.read_text())
    upstream_statistics = json.loads(upstream_statistics_path.read_text())
    shard_records = json.loads(shard_index_path.read_text())["shards"]
    valid_index = pd.read_parquet(upstream_index_path)
    valid_source = source.loc[source["geometry_valid"].astype(bool)].copy()
    if upstream_validation.get("status") != "PASS":
        raise ValueError("Upstream frozen-VGGT extraction did not pass")
    if len(valid_index) != len(valid_source) or len(valid_index) != 101_381:
        raise ValueError("Geometry valid target count differs from the policy manifest")
    if valid_index["sample_id"].duplicated().any():
        raise ValueError("Geometry target index sample IDs are not unique")
    if set(valid_index["sample_id"]) != set(valid_source["sample_id"]):
        raise ValueError("Geometry target IDs do not exactly match valid policy samples")

    source_by_id = source.set_index("sample_id", verify_integrity=True)
    valid_source = valid_source.sort_values("lerobot_dataset_index").reset_index(drop=True)
    memmap_row_by_sample_id = {sample_id: row for row, sample_id in enumerate(valid_source["sample_id"])}
    memmap_path = cache_root / "targets_valid_fp32.npy"
    target_memmap = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(len(valid_source), 2048))
    observed_ids: set[str] = set()
    view_valid_counts = {"agent_only": 0, "wrist_only": 0, "both": 0}
    for record in shard_records:
        path = Path(record["path"]).resolve(strict=True)
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Geometry shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            sample_ids = arrays["sample_id"].astype(str)
            features = arrays["geometry_target_fp32"]
            view_valid = arrays["view_valid"].astype(bool)
            mask_mass = arrays["mask_mass"]
        if features.shape != (len(sample_ids), 2048) or features.dtype != np.float32:
            raise ValueError(f"Unexpected target shape/dtype in {path}")
        if not np.isfinite(features).all() or not np.isfinite(mask_mass).all():
            raise ValueError(f"Non-finite Geometry shard contents: {path}")
        if view_valid.shape != (len(sample_ids), 2) or not view_valid.any(axis=1).all():
            raise ValueError(f"Invalid per-view Geometry validity in {path}")
        memmap_rows = np.asarray([memmap_row_by_sample_id[sample_id] for sample_id in sample_ids], dtype=np.int64)
        target_memmap[memmap_rows] = features
        for row_index, sample_id in enumerate(sample_ids):
            if sample_id in observed_ids:
                raise ValueError(f"Duplicate Geometry shard sample: {sample_id}")
            observed_ids.add(sample_id)
            source_row = source_by_id.loc[sample_id]
            expected = np.asarray([source_row["agent_mask_valid"], source_row["wrist_mask_valid"]], dtype=bool)
            if not np.array_equal(view_valid[row_index], expected):
                raise ValueError(f"Per-view validity differs from manifest for {sample_id}")
            if bool(expected[0]) and bool(expected[1]):
                view_valid_counts["both"] += 1
            elif bool(expected[0]):
                view_valid_counts["agent_only"] += 1
            else:
                view_valid_counts["wrist_only"] += 1
    if observed_ids != set(valid_source["sample_id"]):
        raise ValueError("Shard contents do not cover exactly every valid policy sample")
    target_memmap.flush()
    del target_memmap

    teacher = upstream_validation["teacher"]
    target_columns = valid_index[
        [
            "sample_id",
            "target_shard_path",
            "target_shard_row",
            "target_dim",
            "target_dtype",
            "target_shard_sha256",
        ]
    ]
    full_index = source[
        [
            "sample_id",
            "task_id",
            "task_name",
            "episode_id",
            "frame_idx",
            "lerobot_episode_index",
            "lerobot_dataset_index",
            "hdf5_path",
            "mask_shard_path",
            "mask_shard_row",
            "geometry_valid",
            "split",
        ]
    ].merge(target_columns, on="sample_id", how="left", validate="one_to_one")
    valid_mask = full_index["geometry_valid"].astype(bool)
    if full_index.loc[valid_mask, "target_shard_path"].isna().any():
        raise ValueError("A valid Geometry sample is missing its target")
    if full_index.loc[~valid_mask, "target_shard_path"].notna().any():
        raise ValueError("An invalid Geometry sample was assigned a fake target")
    full_index["vggt_commit"] = teacher["commit"]
    full_index["vggt_checkpoint_sha256"] = teacher["checkpoint_sha256"]
    full_index["teacher_recipe_version"] = "vggt_geometry_final_mask_weighted_fused_v1"
    full_index["target_memmap_path"] = memmap_path.name
    full_index["target_memmap_row"] = full_index["sample_id"].map(memmap_row_by_sample_id).astype("Int64")
    full_index.loc[~valid_mask, "target_memmap_path"] = None
    if full_index.loc[valid_mask, "target_memmap_row"].isna().any():
        raise ValueError("A valid Geometry sample is missing its memory-mapped target row")
    if full_index.loc[~valid_mask, "target_memmap_row"].notna().any():
        raise ValueError("An invalid Geometry sample was assigned a memory-mapped target row")
    full_index["target_dtype"] = full_index["target_dtype"].fillna("none_invalid")
    full_index["target_dim"] = full_index["target_dim"].fillna(2048).astype(np.int32)

    final_index_path = cache_root / "target_index.parquet"
    full_index.to_parquet(final_index_path, index=False, compression="zstd")
    mean = np.asarray(upstream_statistics["mean"], dtype=np.float64)
    raw_std = np.asarray(upstream_statistics["std"], dtype=np.float64)
    if mean.shape != (2048,) or raw_std.shape != (2048,):
        raise ValueError("Unexpected Geometry normalization vector shape")
    std = np.maximum(raw_std, args.sigma_floor)
    normalization = {
        "status": "PASS",
        "schema": "openpi.libero10_geometry_train_standardization.v1",
        "split": "train",
        "valid_samples_only": True,
        "sample_count": int(valid_mask.sum()),
        "feature_dim": 2048,
        "mean": mean.tolist(),
        "raw_std": raw_std.tolist(),
        "std": std.tolist(),
        "sigma_floor": args.sigma_floor,
        "floored_dimensions": int((raw_std < args.sigma_floor).sum()),
        "policy_manifest": str(manifest_path),
        "policy_manifest_sha256": sha256_file(manifest_path),
        "target_index": str(final_index_path),
        "target_index_sha256": sha256_file(final_index_path),
        "target_memmap": str(memmap_path),
        "target_memmap_sha256": sha256_file(memmap_path),
    }
    normalization_path = cache_root / "normalization" / "train_mean_std.json"
    json_dump(normalization_path, normalization)

    recipe = {
        "status": "FROZEN",
        "schema": "openpi.libero10_geometry_teacher_recipe.v1",
        "teacher": "facebook/VGGT-1B",
        "vggt_repo": teacher["repo"],
        "vggt_commit": teacher["commit"],
        "checkpoint": teacher["checkpoint"],
        "checkpoint_sha256": teacher["checkpoint_sha256"],
        "research_layer_label": teacher["layer_label"],
        "python_aggregator_index": teacher["python_index"],
        "target_dim": 2048,
        "transform": {
            "raw_opengl_to_teacher": "vertical flip RGB and mask together",
            "size": [518, 518],
            "crop": "none",
            "padding": "none",
            "rgb_interpolation": "bicubic",
            "mask_interpolation": "bilinear float coverage",
        },
        "pooling": teacher["pooling"],
        "geometry_valid": "agent_mask_valid OR wrist_mask_valid",
        "invalid_policy": "no target stored; loss masked to zero",
        "source_assets_modified": False,
    }
    recipe_path = cache_root / "recipe.json"
    json_dump(recipe_path, recipe)

    final_validation = {
        "status": "PASS",
        "schema": "openpi.libero10_geometry_policy_cache.v1",
        "policy_samples": len(full_index),
        "geometry_valid_samples": int(valid_mask.sum()),
        "geometry_invalid_samples": int((~valid_mask).sum()),
        "valid_targets_finite": True,
        "sample_ids_unique": bool(full_index["sample_id"].is_unique),
        "exact_policy_alignment": True,
        "invalid_samples_have_no_fake_target": True,
        "view_valid_counts": view_valid_counts,
        "pilot_overlap": upstream_validation["pilot_overlap"],
        "teacher_recipe_modified": False,
        "source_assets_modified": False,
        "policy_manifest": str(manifest_path),
        "policy_manifest_sha256": sha256_file(manifest_path),
        "upstream_extraction_validation": str(upstream_validation_path),
        "upstream_extraction_validation_sha256": sha256_file(upstream_validation_path),
        "target_index": str(final_index_path),
        "target_index_sha256": sha256_file(final_index_path),
        "target_memmap": str(memmap_path),
        "target_memmap_sha256": sha256_file(memmap_path),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "recipe": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "runtime": upstream_validation["runtime"],
    }
    final_validation_path = cache_root / "policy_cache_validation.json"
    json_dump(final_validation_path, final_validation)
    print(json.dumps(final_validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
