#!/usr/bin/env python3
"""Materialize the matched Whole-scene VGGT Geometry target.

The frozen VGGT input, final layer, view-validity semantics, and fused-view
reduction match the task-relevant target.  The only target change is replacing
mask-coverage-weighted patch pooling with a uniform mean over all 37 x 37
patches in each valid view.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .constants import FOUR_SUITE_FRAMES
from .constants import FOUR_SUITE_GEOMETRY_INVALID
from .constants import GEOMETRY_DIM
from .constants import SUITES

FINAL_LAYER_LABEL = -1
FINAL_PYTHON_INDEX = 23
NUM_VIEWS = 2
PATCH_START_INDEX = 5
PATCH_GRID = 37
PATCH_COUNT = PATCH_GRID * PATCH_GRID
OUTPUT_TOKENS = PATCH_START_INDEX + PATCH_COUNT
SCHEMA = "libero40.whole_scene_geometry_targets.v1"
TARGET_SCOPE = "whole_scene"
EXPECTED_HELPER_SHA256 = (
    "fda8bdf564fbf81a4e55d157bbcfd6b1ef9432af42b5880de3d1424c36de56d3"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        pq.write_table(table, temporary_name, compression="zstd")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def pool_whole_scene_geometry(
    patch_tokens: torch.Tensor, view_valid: torch.Tensor
) -> torch.Tensor:
    """Uniformly pool each valid view, then fuse valid views with equal weight."""

    if patch_tokens.ndim != 4 or tuple(patch_tokens.shape[1:]) != (
        NUM_VIEWS,
        PATCH_COUNT,
        GEOMETRY_DIM,
    ):
        raise ValueError(
            f"unexpected VGGT patch-token shape: {tuple(patch_tokens.shape)}"
        )
    if view_valid.shape != patch_tokens.shape[:2] or view_valid.dtype != torch.bool:
        raise ValueError("view-valid tensor shape/dtype differs")
    if not view_valid.any(dim=1).all():
        raise ValueError("Whole-scene Geometry received a sample with no valid view")
    per_view = patch_tokens.mean(dim=2)
    per_view = per_view * view_valid[..., None]
    return per_view.sum(dim=1) / view_valid.sum(dim=1)[:, None]


def pool_task_related_reference(
    patch_tokens: torch.Tensor,
    coverage: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Recompute the frozen task-related target for same-forward diagnostics."""

    weights = coverage.reshape(patch_tokens.shape[0], NUM_VIEWS, -1)
    if weights.shape[2] != PATCH_COUNT:
        raise ValueError(f"unexpected task coverage patch count: {weights.shape[2]}")
    masses = weights.sum(dim=2)
    pooled = torch.einsum("bspd,bsp->bsd", patch_tokens, weights.to(patch_tokens.dtype))
    pooled = pooled / masses.clamp_min(1e-12).to(patch_tokens.dtype)[..., None]
    pooled = pooled * view_valid[..., None]
    return pooled.sum(dim=1) / view_valid.sum(dim=1).clamp_min(1)[:, None]


def selected_rows(
    manifest: Path,
    selection_column: str,
    diagnostic_samples_per_suite: int | None = None,
) -> list[dict[str, Any]]:
    table = pq.read_table(manifest)
    if selection_column not in table.column_names:
        raise KeyError(f"missing selection column: {selection_column}")
    selection = np.asarray(table[selection_column].to_numpy(), dtype=bool)
    rows = (
        table.filter(pa.array(selection))
        .sort_by([("sample_id", "ascending")])
        .to_pylist()
    )
    if not rows or not all(bool(row["geometry_valid"]) for row in rows):
        raise ValueError("Geometry selection is empty or includes invalid samples")
    if diagnostic_samples_per_suite is not None:
        if diagnostic_samples_per_suite <= 0:
            raise ValueError("diagnostic-samples-per-suite must be positive")
        by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_suite[str(row["suite"])].append(row)
        if set(by_suite) != set(SUITES):
            raise ValueError(f"unexpected suite population: {sorted(by_suite)}")
        rows = sorted(
            [
                row
                for suite in SUITES
                for row in by_suite[suite][:diagnostic_samples_per_suite]
            ],
            key=lambda row: str(row["sample_id"]),
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("selected Geometry sample IDs are not unique")
    return rows


def transform_row(
    row: dict[str, Any], geometry: Any
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw_rgbs, raw_masks = geometry.load_raw_inputs(row)
    rgb_views = []
    coverage_views = []
    masses = np.zeros(NUM_VIEWS, dtype=np.float32)
    for view_index, view in enumerate(("agent", "wrist")):
        rgb, _, coverage = geometry.raw_to_teacher(raw_rgbs[view], raw_masks[view])
        mass = float(coverage.sum().item())
        masses[view_index] = mass
        if (mass > 0.0) != bool(row[f"{view}_mask_valid"]):
            raise ValueError(
                f"{row['sample_id']}: {view} validity differs after transform"
            )
        rgb_views.append(rgb)
        coverage_views.append(coverage)
    return torch.stack(rgb_views), torch.stack(coverage_views), masses


def validate_existing_shard(
    path: Path,
    expected_indices: list[int],
    *,
    expect_task_reference: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as arrays:
            valid = bool(
                arrays["global_index"].astype(int).tolist() == expected_indices
                and arrays["geometry_target_fp32"].shape
                == (len(expected_indices), GEOMETRY_DIM)
                and arrays["geometry_target_fp32"].dtype == np.float32
                and np.isfinite(arrays["geometry_target_fp32"]).all()
                and arrays["view_valid"].shape == (len(expected_indices), NUM_VIEWS)
                and arrays["view_valid"].any(axis=1).all()
                and (arrays["whole_scene_patch_count"] == PATCH_COUNT).all()
                and (arrays["target_scope"].astype(str) == TARGET_SCOPE).all()
            )
            if expect_task_reference:
                valid = valid and bool(
                    "task_related_reference_fp32" in arrays
                    and arrays["task_related_reference_fp32"].shape
                    == (len(expected_indices), GEOMETRY_DIM)
                    and np.isfinite(arrays["task_related_reference_fp32"]).all()
                )
            return valid
    except (KeyError, OSError, ValueError):
        return False


def preflight(args: argparse.Namespace, geometry: Any) -> dict[str, Any]:
    hook = json.loads(
        (args.teacher_reference_root / "hook1/introspection.json").read_text()
    )
    smoke = json.loads(
        (args.teacher_reference_root / "smoke16/smoke_report.json").read_text()
    )
    if hook.get("status") != "PASS" or smoke.get("status") != "PASS":
        raise RuntimeError("frozen VGGT Geometry reference did not pass")
    helper_path = Path(inspect.getfile(geometry)).resolve()
    helper_sha = sha256_file(helper_path)
    if helper_sha != EXPECTED_HELPER_SHA256:
        raise ValueError(f"frozen Geometry helper drifted: {helper_path}")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != hook["teacher"]["checkpoint_sha256"]:
        raise ValueError("VGGT checkpoint differs from frozen provenance")
    commit = subprocess.run(
        ["git", "-C", str(args.vggt_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != hook["teacher"]["git_commit"]:
        raise ValueError("VGGT commit differs from frozen provenance")
    return {
        "vggt_commit": commit,
        "checkpoint_sha256": checkpoint_sha,
        "geometry_helper": {"path": str(helper_path), "sha256": helper_sha},
    }


def run_worker(args: argparse.Namespace) -> int:
    import run_vggt_geometry_hook1 as geometry

    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index is outside num-workers")
    provenance = preflight(args, geometry)
    rows = selected_rows(
        args.manifest, args.selection_column, args.diagnostic_samples_per_suite
    )
    assigned = [
        index
        for index in range(len(rows))
        if index % args.num_workers == args.worker_index
    ]
    worker_root = args.output_root / "workers" / f"worker_{args.worker_index:02d}"
    shard_root = worker_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    if str(args.vggt_repo) not in os.sys.path:
        os.sys.path.insert(0, str(args.vggt_repo))
    from vggt.models.aggregator import Aggregator

    construct_start = time.perf_counter()
    state = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    aggregator_state = {
        key.removeprefix("aggregator."): value
        for key, value in state.items()
        if key.startswith("aggregator.")
    }
    model = Aggregator()
    if model.depth - 1 != FINAL_PYTHON_INDEX:
        raise ValueError(f"unexpected VGGT final index: {model.depth - 1}")
    model.cached_layer_indices = {FINAL_PYTHON_INDEX}
    load_result = model.load_state_dict(aggregator_state, strict=True)
    del state, aggregator_state
    gc.collect()
    device = torch.device(args.device)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    construct_seconds = time.perf_counter() - construct_start

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats(device)
    extraction_start = time.perf_counter()
    processed = 0
    skipped = 0
    shard_records: list[dict[str, Any]] = []
    chunks = [
        assigned[offset : offset + args.shard_size]
        for offset in range(0, len(assigned), args.shard_size)
    ]
    with ThreadPoolExecutor(max_workers=args.loader_workers) as executor:
        for shard_index, global_indices in enumerate(chunks):
            target_path = shard_root / f"geometry_{shard_index:05d}.npz"
            if args.resume and validate_existing_shard(
                target_path,
                global_indices,
                expect_task_reference=args.diagnostic_compare_task_related,
            ):
                skipped += len(global_indices)
                shard_records.append(
                    {
                        "path": str(target_path),
                        "sha256": sha256_file(target_path),
                        "samples": len(global_indices),
                        "resumed": True,
                    }
                )
                continue

            features = np.zeros((len(global_indices), GEOMETRY_DIM), dtype=np.float32)
            view_valid = np.zeros((len(global_indices), NUM_VIEWS), dtype=bool)
            mask_mass = np.zeros((len(global_indices), NUM_VIEWS), dtype=np.float32)
            task_reference = (
                np.zeros_like(features)
                if args.diagnostic_compare_task_related
                else None
            )
            sample_ids: list[str] = []
            for batch_start in range(0, len(global_indices), args.batch_size):
                batch_indices = global_indices[
                    batch_start : batch_start + args.batch_size
                ]
                batch_rows = [rows[index] for index in batch_indices]
                loaded = list(
                    executor.map(lambda row: transform_row(row, geometry), batch_rows)
                )
                batch_rgb = torch.stack([item[0] for item in loaded]).to(device)
                coverage = torch.stack([item[1] for item in loaded]).to(device)
                masses = np.stack([item[2] for item in loaded])
                valid_np = masses > 0.0
                valid = torch.from_numpy(valid_np).to(device)
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    outputs, patch_start_index = model(batch_rgb)
                if patch_start_index != PATCH_START_INDEX:
                    raise ValueError(
                        f"unexpected VGGT patch start: {patch_start_index}"
                    )
                output = outputs[FINAL_PYTHON_INDEX]
                expected = (len(batch_rows), NUM_VIEWS, OUTPUT_TOKENS, GEOMETRY_DIM)
                if output is None or tuple(output.shape) != expected:
                    raise ValueError(
                        f"unexpected final VGGT output: {getattr(output, 'shape', None)}"
                    )
                patches = output[:, :, patch_start_index:, :]
                whole = pool_whole_scene_geometry(patches, valid)
                local_slice = slice(batch_start, batch_start + len(batch_rows))
                features[local_slice] = whole.detach().cpu().float().numpy()
                view_valid[local_slice] = valid_np
                mask_mass[local_slice] = masses
                if task_reference is not None:
                    task = pool_task_related_reference(patches, coverage, valid)
                    task_reference[local_slice] = task.detach().cpu().float().numpy()
                    del task
                sample_ids.extend(str(row["sample_id"]) for row in batch_rows)
                processed += len(batch_rows)
                torch.cuda.synchronize(device)
                del batch_rgb, coverage, valid, outputs, output, patches, whole
                peak_rss = max(peak_rss, process.memory_info().rss)
                print(
                    json.dumps(
                        {
                            "worker": args.worker_index,
                            "progress": processed + skipped,
                            "assigned": len(assigned),
                        }
                    ),
                    flush=True,
                )

            arrays: dict[str, np.ndarray] = {
                "global_index": np.asarray(global_indices, dtype=np.int64),
                "sample_id": np.asarray(sample_ids),
                "geometry_target_fp32": features,
                "view_valid": view_valid,
                "matched_mask_mass": mask_mass,
                "whole_scene_patch_count": np.full(
                    len(global_indices), PATCH_COUNT, dtype=np.int32
                ),
                "target_scope": np.asarray([TARGET_SCOPE] * len(global_indices)),
            }
            if task_reference is not None:
                arrays["task_related_reference_fp32"] = task_reference
            atomic_npz(target_path, **arrays)
            shard_records.append(
                {
                    "path": str(target_path),
                    "sha256": sha256_file(target_path),
                    "samples": len(global_indices),
                    "resumed": False,
                }
            )

    extraction_seconds = time.perf_counter() - extraction_start
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "target_scope": TARGET_SCOPE,
        "pooling": f"uniform mean over all {PATCH_COUNT} patches in each matched-valid view",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "assigned_count": len(assigned),
        "processed_count": processed,
        "resumed_count": skipped,
        "diagnostic_samples_per_suite": args.diagnostic_samples_per_suite,
        "diagnostic_compare_task_related": args.diagnostic_compare_task_related,
        "construct_seconds": construct_seconds,
        "extraction_seconds": extraction_seconds,
        "effective_samples_per_second": processed / extraction_seconds
        if processed
        else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_process_rss_bytes": peak_rss,
        "device": torch.cuda.get_device_name(device),
        "strict_load_result": str(load_result),
        "provenance": provenance,
        "shards": shard_records,
        "matched_control": {
            "same_valid_population": True,
            "same_agent_and_wrist_inputs": True,
            "same_view_validity": True,
            "same_teacher_forward": True,
            "same_final_layer": FINAL_PYTHON_INDEX,
            "only_patch_pooling_changed": True,
        },
    }
    atomic_json(worker_root / "worker_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def load_task_related_targets(index_path: Path, sample_ids: list[str]) -> np.ndarray:
    table = pq.read_table(
        index_path,
        columns=[
            "sample_id",
            "geometry_valid",
            "target_memmap_path",
            "target_memmap_row",
        ],
    )
    rows = table.to_pylist()
    by_id = {str(row["sample_id"]): row for row in rows if bool(row["geometry_valid"])}
    if set(sample_ids) - set(by_id):
        raise ValueError("Whole-scene Geometry IDs differ from task-related valid IDs")
    features = np.zeros((len(sample_ids), GEOMETRY_DIM), dtype=np.float32)
    groups: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for output_index, sample_id in enumerate(sample_ids):
        row = by_id[sample_id]
        groups[Path(row["target_memmap_path"])].append(
            (output_index, int(row["target_memmap_row"]))
        )
    for path, locations in groups.items():
        targets = np.load(path, mmap_mode="r")
        for output_index, row_index in locations:
            features[output_index] = targets[row_index]
    return features


def paired_difference(whole: np.ndarray, task: np.ndarray) -> dict[str, Any]:
    delta = whole.astype(np.float64) - task.astype(np.float64)
    l2 = np.linalg.norm(delta, axis=1)
    whole_norm = np.linalg.norm(whole.astype(np.float64), axis=1)
    task_norm = np.linalg.norm(task.astype(np.float64), axis=1)
    denominator = whole_norm * task_norm
    cosine = np.divide(
        np.sum(whole.astype(np.float64) * task.astype(np.float64), axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    exact = np.all(whole == task, axis=1)
    return {
        "samples": len(whole),
        "exact_equal_rows": int(exact.sum()),
        "different_rows": int((~exact).sum()),
        "l2": {
            "mean": float(l2.mean()),
            "min": float(l2.min()),
            "max": float(l2.max()),
        },
        "cosine": {
            "mean": float(cosine.mean()),
            "min": float(cosine.min()),
            "max": float(cosine.max()),
        },
    }


def run_finalize(args: argparse.Namespace) -> int:
    rows = selected_rows(
        args.manifest, args.selection_column, args.diagnostic_samples_per_suite
    )
    sample_ids = [str(row["sample_id"]) for row in rows]
    reports = []
    shard_paths: list[Path] = []
    for worker_index in range(args.num_workers):
        worker_root = args.output_root / "workers" / f"worker_{worker_index:02d}"
        report = json.loads((worker_root / "worker_report.json").read_text())
        if (
            report.get("status") != "PASS"
            or report.get("schema_version") != SCHEMA
            or report.get("target_scope") != TARGET_SCOPE
            or report.get("diagnostic_samples_per_suite")
            != args.diagnostic_samples_per_suite
        ):
            raise ValueError(f"worker report provenance differs: {worker_index}")
        reports.append(report)
        shard_paths.extend(sorted((worker_root / "shards").glob("geometry_*.npz")))

    memmap_path = args.output_root / "targets_valid_fp32.npy"
    temporary_memmap = memmap_path.with_suffix(".npy.tmp")
    targets = np.lib.format.open_memmap(
        temporary_memmap,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), GEOMETRY_DIM),
    )
    observed = np.zeros(len(rows), dtype=bool)
    observed_ids: list[str | None] = [None] * len(rows)
    task_diagnostic = (
        np.zeros((len(rows), GEOMETRY_DIM), dtype=np.float32)
        if args.diagnostic_samples_per_suite
        else None
    )
    feature_sum = np.zeros(GEOMETRY_DIM, dtype=np.float64)
    feature_square_sum = np.zeros(GEOMETRY_DIM, dtype=np.float64)
    shard_records = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as arrays:
            indices = arrays["global_index"].astype(int)
            features = arrays["geometry_target_fp32"].astype(np.float32, copy=False)
            ids = arrays["sample_id"].astype(str)
            if not np.isfinite(features).all():
                raise ValueError(f"non-finite Geometry targets in {path}")
            if observed[indices].any() or len(indices) != len(set(indices.tolist())):
                raise ValueError(f"duplicate Geometry indices in {path}")
            targets[indices] = features
            observed[indices] = True
            for index, sample_id in zip(indices, ids, strict=True):
                observed_ids[int(index)] = sample_id
            feature64 = features.astype(np.float64)
            feature_sum += feature64.sum(axis=0)
            feature_square_sum += np.square(feature64).sum(axis=0)
            if task_diagnostic is not None:
                if "task_related_reference_fp32" not in arrays:
                    raise ValueError("diagnostic task-related reference is missing")
                task_diagnostic[indices] = arrays["task_related_reference_fp32"]
        shard_records.append(
            {"path": str(path), "sha256": sha256_file(path), "samples": len(indices)}
        )
    if not observed.all() or observed_ids != sample_ids:
        raise ValueError(
            "Geometry workers do not exactly cover the selected population"
        )
    targets.flush()
    del targets
    os.replace(temporary_memmap, memmap_path)

    task_table = pq.read_table(
        args.task_related_index, columns=["sample_id", "geometry_valid"]
    )
    task_rows = task_table.to_pylist()
    task_valid_ids = {
        str(row["sample_id"]) for row in task_rows if bool(row["geometry_valid"])
    }
    population_exact = set(sample_ids) == task_valid_ids
    if args.diagnostic_samples_per_suite is None and not population_exact:
        raise ValueError(
            "full Whole-scene Geometry population differs from task-related cache"
        )

    comparison = None
    task_cache_check = None
    if task_diagnostic is not None:
        task_cache = load_task_related_targets(args.task_related_index, sample_ids)
        absolute = np.abs(task_diagnostic - task_cache)
        task_cache_check = {
            "samples": len(rows),
            "within_atol_1e-5": bool(
                np.allclose(task_diagnostic, task_cache, rtol=0.0, atol=1e-5)
            ),
            "max_abs": float(absolute.max(initial=0.0)),
            "mean_abs": float(absolute.mean()),
        }
        if not task_cache_check["within_atol_1e-5"]:
            raise ValueError(f"same-forward task reference differs: {task_cache_check}")
        whole = np.load(memmap_path, mmap_mode="r")
        comparison = paired_difference(np.asarray(whole), task_cache)
        if comparison["different_rows"] == 0:
            raise ValueError(
                "Whole-scene Geometry did not change from task-related targets"
            )

    count = len(rows)
    mean = feature_sum / count
    variance = np.maximum(feature_square_sum / count - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    std = np.maximum(raw_std, args.sigma_floor)
    statistics = {
        "count": count,
        "feature_dim": GEOMETRY_DIM,
        "dtype": "float32",
        "finite": True,
        "mean": mean.tolist(),
        "std": raw_std.tolist(),
    }
    atomic_json(args.output_root / "target_statistics_train.json", statistics)
    normalization = {
        "status": "PASS",
        "schema": "openpi.libero40_whole_scene_geometry_train_standardization.v1",
        "target_scope": TARGET_SCOPE,
        "split": "train",
        "valid_samples_only": True,
        "sample_count": count,
        "feature_dim": GEOMETRY_DIM,
        "mean": mean.tolist(),
        "raw_std": raw_std.tolist(),
        "std": std.tolist(),
        "sigma_floor": args.sigma_floor,
        "floored_dimensions": int((raw_std < args.sigma_floor).sum()),
    }
    normalization_path = args.output_root / "normalization/train_mean_std.json"
    atomic_json(normalization_path, normalization)

    selected_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    full_manifest = pq.read_table(args.manifest).sort_by(
        [("lerobot_dataset_index", "ascending")]
    )
    if full_manifest.num_rows != FOUR_SUITE_FRAMES:
        raise ValueError("policy manifest frame count differs")
    diagnostic = args.diagnostic_samples_per_suite is not None
    index_source_rows = rows if diagnostic else full_manifest.to_pylist()
    index_rows = []
    for row in index_source_rows:
        sample_id = str(row["sample_id"])
        valid = bool(row["geometry_valid"])
        target_row = selected_index.get(sample_id)
        if valid != (target_row is not None):
            raise ValueError(f"Geometry validity/target mismatch: {sample_id}")
        index_rows.append(
            {
                "sample_id": sample_id,
                "suite": row["suite"],
                "split": row["split"],
                "lerobot_dataset_index": int(row["lerobot_dataset_index"]),
                "geometry_valid": valid,
                "target_scope": TARGET_SCOPE,
                "pooling": "uniform_patch_mean_matched_valid_views",
                "target_memmap_path": str(memmap_path) if valid else None,
                "target_memmap_row": target_row,
                "target_dim": GEOMETRY_DIM,
                "target_dtype": "float32" if valid else "none_invalid",
            }
        )
    target_index = args.output_root / "target_index.parquet"
    atomic_parquet(target_index, pa.Table.from_pylist(index_rows))

    suite_counts = Counter(str(row["suite"]) for row in rows)
    validation = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "target_scope": TARGET_SCOPE,
        "selection_column": args.selection_column,
        "diagnostic_samples_per_suite": args.diagnostic_samples_per_suite,
        "policy_samples": len(index_source_rows),
        "geometry_valid_samples": count,
        "geometry_invalid_samples": 0 if diagnostic else FOUR_SUITE_GEOMETRY_INVALID,
        "selected_by_suite": dict(sorted(suite_counts.items())),
        "shape": [count, GEOMETRY_DIM],
        "dtype": "float32",
        "all_finite": True,
        "sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
        "no_missing_targets": bool(observed.all()),
        "task_related_population_exact": population_exact,
        "same_forward_task_related_cache_check": task_cache_check,
        "whole_scene_vs_task_related": comparison,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "target_index": str(target_index),
        "target_index_sha256": sha256_file(target_index),
        "target_memmap": str(memmap_path),
        "target_memmap_sha256": sha256_file(memmap_path),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "teacher": {
            "model": "facebook/VGGT-1B",
            "layer_label": FINAL_LAYER_LABEL,
            "python_index": FINAL_PYTHON_INDEX,
            "patch_grid": [PATCH_GRID, PATCH_GRID],
            "pooling": "uniform patch mean in each matched-valid view; equal valid-view fusion",
        },
        "matched_control": {
            "same_valid_population": True,
            "same_agent_and_wrist_inputs": True,
            "same_view_validity": True,
            "same_teacher_forward": True,
            "same_final_layer": FINAL_PYTHON_INDEX,
            "only_patch_pooling_changed": True,
        },
        "workers": reports,
        "shards": shard_records,
    }
    atomic_json(args.output_root / "cache_validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "finalize"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-column", default="geometry_valid")
    parser.add_argument("--vggt-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--teacher-reference-root", type=Path)
    parser.add_argument("--task-related-index", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--sigma-floor", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic-samples-per-suite", type=int)
    parser.add_argument("--diagnostic-compare-task-related", action="store_true")
    args = parser.parse_args()
    for name in (
        "manifest",
        "vggt_repo",
        "checkpoint",
        "teacher_reference_root",
        "task_related_index",
        "output_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if (
        args.diagnostic_compare_task_related
        and args.diagnostic_samples_per_suite is None
    ):
        raise ValueError("diagnostic comparison is restricted to a smoke selection")
    required = (
        ("vggt_repo", "checkpoint", "teacher_reference_root")
        if args.mode == "worker"
        else ("task_related_index",)
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing arguments for {args.mode}: {missing}")
    if min(args.batch_size, args.loader_workers, args.shard_size) <= 0:
        raise ValueError("batch, loader-worker, and shard sizes must be positive")
    if args.sigma_floor <= 0:
        raise ValueError("sigma floor must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "worker":
        raise SystemExit(run_worker(arguments))
    raise SystemExit(run_finalize(arguments))
