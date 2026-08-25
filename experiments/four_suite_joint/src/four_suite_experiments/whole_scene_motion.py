#!/usr/bin/env python3
"""Materialize the matched Whole-scene Motion ablation target.

The Track4World forward recipe is deliberately identical to the frozen
task-related Motion recipe.  The only target change is the source-token
pooling operation:

* task-related: task-mask coverage-weighted mean over 45 x 45 source patches;
* whole-scene: uniform mean over all 45 x 45 source patches.

The worker/finalizer interface is resumable and is suitable for the eventual
eight-GPU cache run.  ``--diagnostic-samples-per-suite`` is intentionally
available only for a small, explicitly labelled smoke run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from collections.abc import Sequence
import gc
import hashlib
import inspect
from itertools import pairwise
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

FEATURE_DIM = 256
FINAL_LEVEL = 3
NUM_LEVELS = 4
SOURCE_TIME_INDEX = 0
SOURCE_PATCH_GRID = 45
SOURCE_PATCH_COUNT = SOURCE_PATCH_GRID * SOURCE_PATCH_GRID
CLIP_LENGTH = 11
SCHEMA = "libero40.whole_scene_motion_targets.v1"
TARGET_SCOPE = "whole_scene"

# These are the exact helper files used to generate the existing task-related
# cache.  Refusing drift makes the pooling change auditable.
EXPECTED_MOTION_HELPER_SHA256 = (
    "3de7691f7425ff244e6bf24553db29387fd22144e379341f0bdd51e2e04427fe"
)
EXPECTED_SMOKE_HELPER_SHA256 = (
    "7ef5a85929350b6ea658d6998279f6b4d16bae8961dfa1de6814c86f72ec5fd9"
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


def pool_source_uniform(level: torch.Tensor) -> torch.Tensor:
    """Pool all spatial source tokens at time index zero with equal weight."""

    if level.ndim != 4:
        raise ValueError(
            f"Expected [batch,time,patch,feature], got {tuple(level.shape)}"
        )
    if level.shape[0] != 1 or level.shape[1] != CLIP_LENGTH:
        raise ValueError(f"Unexpected batch/time shape: {tuple(level.shape[:2])}")
    if level.shape[2] != SOURCE_PATCH_COUNT:
        raise ValueError(
            f"Expected {SOURCE_PATCH_COUNT} source patches, got {level.shape[2]}"
        )
    if level.shape[3] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {level.shape[3]}")
    return level[0, SOURCE_TIME_INDEX].float().mean(dim=0)


def pool_whole_scene_levels(levels: Sequence[torch.Tensor]) -> np.ndarray:
    """Return all four pooled levels as deterministic float32 CPU targets."""

    if len(levels) != NUM_LEVELS:
        raise ValueError(f"Expected {NUM_LEVELS} levels, got {len(levels)}")
    pooled = torch.stack([pool_source_uniform(level) for level in levels])
    result = pooled.cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (NUM_LEVELS, FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("Whole-scene pooled level shape/finite gate failed")
    return result


def selected_rows(
    manifest: Path,
    selection_column: str,
    diagnostic_samples_per_suite: int | None = None,
) -> list[dict[str, Any]]:
    table = pq.read_table(manifest)
    if selection_column not in table.column_names:
        raise KeyError(f"Missing selection column: {selection_column}")
    selection = np.asarray(table[selection_column].to_numpy(), dtype=bool)
    selected = table.filter(pa.array(selection)).sort_by([("sample_id", "ascending")])
    rows = selected.to_pylist()
    if not rows:
        raise ValueError("No Motion anchors selected")
    if not all(bool(row["motion_valid"]) for row in rows):
        raise ValueError("Motion selection includes invalid anchors")

    if diagnostic_samples_per_suite is not None:
        if diagnostic_samples_per_suite <= 0:
            raise ValueError("diagnostic-samples-per-suite must be positive")
        by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_suite[str(row["suite"])].append(row)
        expected_suites = {
            "libero_10",
            "libero_goal",
            "libero_object",
            "libero_spatial",
        }
        if set(by_suite) != expected_suites:
            raise ValueError(f"Unexpected suite population: {sorted(by_suite)}")
        rows = sorted(
            [
                row
                for suite in sorted(expected_suites)
                for row in by_suite[suite][:diagnostic_samples_per_suite]
            ],
            key=lambda row: str(row["sample_id"]),
        )

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Selected Motion sample IDs are not unique")
    return rows


def validate_existing_shard(
    path: Path,
    expected_indices: list[int],
    *,
    expect_task_reference: bool,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as arrays:
            valid = bool(
                arrays["global_index"].astype(int).tolist() == expected_indices
                and arrays["motion_target_fp32"].shape
                == (len(expected_indices), FEATURE_DIM)
                and arrays["motion_target_fp32"].dtype == np.float32
                and np.isfinite(arrays["motion_target_fp32"]).all()
                and (arrays["whole_scene_patch_count"] == SOURCE_PATCH_COUNT).all()
                and (arrays["target_scope"].astype(str) == TARGET_SCOPE).all()
            )
            if expect_task_reference:
                valid = valid and bool(
                    "task_related_reference_fp32" in arrays
                    and arrays["task_related_reference_fp32"].shape
                    == (len(expected_indices), FEATURE_DIM)
                    and np.isfinite(arrays["task_related_reference_fp32"]).all()
                )
            return valid
    except (KeyError, OSError, ValueError):
        return False


def repo_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_helper(module: Any, expected_sha256: str) -> dict[str, str]:
    path = Path(inspect.getfile(module)).resolve()
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Frozen Motion helper drifted: {path}; expected={expected_sha256}, actual={actual}"
        )
    return {"path": str(path), "sha256": actual}


def preflight(args: argparse.Namespace, motion: Any, smoke: Any) -> dict[str, Any]:
    hook = json.loads(
        (args.teacher_reference_root / "hook1" / "introspection.json").read_text(
            encoding="utf-8"
        )
    )
    smoke_report = json.loads(
        (args.teacher_reference_root / "smoke16" / "smoke_report.json").read_text(
            encoding="utf-8"
        )
    )
    if hook.get("status") != "PASS" or smoke_report.get("status") != "PASS":
        raise RuntimeError("Frozen Motion hook1/smoke16 reference did not pass")
    track_commit = repo_commit(args.track4world_repo)
    utils_commit = repo_commit(args.utils3d_repo)
    if track_commit != hook["teacher"]["git_commit"]:
        raise ValueError("Track4World commit differs from frozen hook provenance")
    if utils_commit != hook["teacher"]["utils3d_commit"]:
        raise ValueError("utils3d commit differs from frozen hook provenance")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != hook["teacher"]["checkpoint_sha256"]:
        raise ValueError("Track4World checkpoint differs from frozen hook provenance")
    if args.da3_snapshot.resolve() != Path(hook["teacher"]["da3_snapshot"]).resolve():
        raise ValueError("DA3 snapshot differs from frozen hook provenance")
    return {
        "track4world_commit": track_commit,
        "utils3d_commit": utils_commit,
        "checkpoint_sha256": checkpoint_sha,
        "da3_snapshot": str(args.da3_snapshot),
        "motion_helper": validate_helper(motion, EXPECTED_MOTION_HELPER_SHA256),
        "smoke_helper": validate_helper(smoke, EXPECTED_SMOKE_HELPER_SHA256),
    }


def run_worker(args: argparse.Namespace) -> int:
    # Imports are delayed so pure pooling/unit tests do not require the external
    # frozen teacher scripts on sys.path.
    import run_track4world_motion_hook1 as motion
    import run_track4world_motion_smoke16 as smoke

    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index is outside num-workers")
    provenance = preflight(args, motion, smoke)
    rows = selected_rows(
        args.manifest, args.selection_column, args.diagnostic_samples_per_suite
    )
    source_columns = pq.read_table(args.source_manifest).to_pydict()
    assigned = [
        index
        for index in range(len(rows))
        if index % args.num_workers == args.worker_index
    ]
    worker_root = args.output_root / "workers" / f"worker_{args.worker_index:02d}"
    shard_root = worker_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    failure_path = worker_root / "extraction_failures.jsonl"
    if failure_path.exists() and failure_path.stat().st_size > 0:
        archived = (
            worker_root / f"previous_extraction_failures_{int(time.time())}.jsonl"
        )
        os.replace(failure_path, archived)
    failure_path.touch(exist_ok=True)

    os.environ["TRACK4WORLD_DA3_MODEL_SOURCE"] = str(args.da3_snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"
    for path in (args.track4world_repo, args.utils3d_repo.parent):
        if str(path) not in os.sys.path:
            os.sys.path.insert(0, str(path))
    from track4world.nets.model import Track4World

    construct_start = time.perf_counter()
    config_path = args.track4world_repo / "track4world/config/eval/v1.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))["model"]
    model = Track4World(
        **model_config,
        seqlen=16,
        use_3d=True,
        use_model="depthanythingv3",
    )
    state = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    missing, unexpected = model.load_pretrained_with_remap(state)
    allowed_missing = all(
        key.startswith("backbone.model.da3_metric.") for key in missing
    )
    if unexpected or not allowed_missing:
        raise ValueError(
            f"Track4World remap mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    del state
    gc.collect()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    device = torch.device(args.device)
    model.to(device)
    construct_seconds = time.perf_counter() - construct_start

    captured: list[list[torch.Tensor]] = []

    def aggregator_hook(_module: Any, _hook_input: Any, hook_output: Any) -> None:
        captured.append([level.detach().cpu() for level in hook_output])

    handle = model.flow_aggregator3d.register_forward_hook(aggregator_hook)
    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    extraction_start = time.perf_counter()
    processed = 0
    skipped = 0
    shard_records: list[dict[str, Any]] = []

    chunks = [
        assigned[offset : offset + args.shard_size]
        for offset in range(0, len(assigned), args.shard_size)
    ]
    for shard_index, global_indices in enumerate(chunks):
        target_path = shard_root / f"motion_{shard_index:05d}.npz"
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
            print(
                json.dumps(
                    {
                        "worker": args.worker_index,
                        "resume_skip": len(global_indices),
                        "progress": processed + skipped,
                        "assigned": len(assigned),
                    }
                ),
                flush=True,
            )
            continue

        features = np.zeros((len(global_indices), FEATURE_DIM), dtype=np.float32)
        task_references = (
            np.zeros((len(global_indices), FEATURE_DIM), dtype=np.float32)
            if args.diagnostic_compare_task_related
            else None
        )
        source_mask_masses = np.zeros(len(global_indices), dtype=np.float32)
        cross = np.zeros(len(global_indices), dtype=bool)
        boundary_near = np.zeros(len(global_indices), dtype=bool)
        raw_contiguous = np.zeros(len(global_indices), dtype=bool)
        sample_ids: list[str] = []
        for local_index, global_index in enumerate(global_indices):
            anchor = rows[global_index]
            clip_rows = smoke.clip_rows_for_anchor(source_columns, anchor)
            try:
                raw_clip = motion.load_raw_clip(clip_rows)
                raw_mask = motion.decode_agent_mask(anchor)
                teacher_rgb, _, coverage = motion.raw_to_teacher(raw_clip, raw_mask)
                source_mask_mass = float(coverage.sum().item())
                if source_mask_mass <= 0.0 or not bool(anchor["agent_mask_valid"]):
                    raise ValueError(
                        "Matched task-related Motion source mask is invalid"
                    )
                inputs = teacher_rgb.unsqueeze(0).to(device)
                normalized = inputs / 255.0
                normalized = (normalized - model.image_mean) / model.image_std
                flat = normalized.reshape(CLIP_LENGTH, 3, 640, 640).contiguous()
                captured.clear()
                with torch.inference_mode():
                    direct_outputs = model.get_fmaps(
                        flat,
                        1,
                        CLIP_LENGTH,
                        None,
                        False,  # noqa: FBT003
                    )
                if len(captured) != 1:
                    raise ValueError(
                        f"Expected one aggregator call, found {len(captured)}"
                    )
                whole_levels = pool_whole_scene_levels(captured[0])
                final = np.asarray(whole_levels[FINAL_LEVEL], dtype=np.float32)
                if args.diagnostic_compare_task_related:
                    task_levels = smoke.pooled_levels(captured[0], coverage)
                    task_final = np.asarray(task_levels[FINAL_LEVEL], dtype=np.float32)
                    if (
                        task_final.shape != (FEATURE_DIM,)
                        or not np.isfinite(task_final).all()
                    ):
                        raise ValueError("Task-related diagnostic reference is invalid")
                    assert task_references is not None
                    task_references[local_index] = task_final

                actual_segments = list(
                    dict.fromkeys(row["semantic_segment_id"] for row in clip_rows)
                )
                actual_cross = len(actual_segments) > 1
                if actual_cross != bool(anchor["motion_crosses_subtask_boundary"]):
                    raise ValueError(
                        "Actual semantic segment crossing differs from manifest"
                    )
                raw_indices = [int(row["raw_state_index"]) for row in clip_rows]
                if not all(b > a for a, b in pairwise(raw_indices)):
                    raise ValueError("Raw state indices repeat or reverse")

                features[local_index] = final
                source_mask_masses[local_index] = source_mask_mass
                cross[local_index] = actual_cross
                boundary_near[local_index] = bool(anchor["boundary_near"])
                raw_contiguous[local_index] = bool(anchor["motion_raw_clip_contiguous"])
                sample_ids.append(str(anchor["sample_id"]))
                processed += 1
                del inputs, normalized, flat, direct_outputs, whole_levels, final
                if args.diagnostic_compare_task_related:
                    del task_levels, task_final
                captured.clear()
            except Exception as error:
                failure = {
                    "worker_index": args.worker_index,
                    "global_index": global_index,
                    "sample_id": anchor["sample_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                with failure_path.open("a", encoding="utf-8") as failure_handle:
                    failure_handle.write(json.dumps(failure, sort_keys=True) + "\n")
                handle.remove()
                raise

            if processed % 10 == 0:
                print(
                    json.dumps(
                        {
                            "worker": args.worker_index,
                            "progress": processed + skipped,
                            "assigned": len(assigned),
                            "elapsed_seconds": round(
                                time.perf_counter() - extraction_start, 2
                            ),
                        }
                    ),
                    flush=True,
                )
            peak_rss = max(peak_rss, process.memory_info().rss)

        arrays: dict[str, np.ndarray] = {
            "global_index": np.asarray(global_indices, dtype=np.int64),
            "sample_id": np.asarray(sample_ids),
            "motion_target_fp32": features,
            "whole_scene_patch_count": np.full(
                len(global_indices), SOURCE_PATCH_COUNT, dtype=np.int32
            ),
            "target_scope": np.asarray([TARGET_SCOPE] * len(global_indices)),
            "matched_source_mask_mass": source_mask_masses,
            "motion_crosses_subtask_boundary": cross,
            "boundary_near": boundary_near,
            "motion_raw_clip_contiguous": raw_contiguous,
        }
        if task_references is not None:
            arrays["task_related_reference_fp32"] = task_references
        atomic_npz(target_path, **arrays)
        shard_records.append(
            {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
                "samples": len(global_indices),
                "resumed": False,
            }
        )

    handle.remove()
    extraction_seconds = time.perf_counter() - extraction_start
    if failure_path.stat().st_size:
        raise RuntimeError(f"Motion worker failures recorded in {failure_path}")
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "target_scope": TARGET_SCOPE,
        "pooling": f"uniform mean over all {SOURCE_PATCH_COUNT} source patches",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "assigned_count": len(assigned),
        "processed_count": processed,
        "resumed_count": skipped,
        "diagnostic_samples_per_suite": args.diagnostic_samples_per_suite,
        "diagnostic_compare_task_related": args.diagnostic_compare_task_related,
        "construct_seconds": construct_seconds,
        "extraction_seconds": extraction_seconds,
        "effective_clips_per_second": processed / extraction_seconds
        if processed
        else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_process_rss_bytes": peak_rss,
        "device": torch.cuda.get_device_name(device),
        "checkpoint_missing_count_allowed_da3_prefix": len(missing),
        "checkpoint_unexpected_count": len(unexpected),
        "provenance": provenance,
        "shards": shard_records,
        "matched_control": {
            "same_valid_population": True,
            "same_agent_clip_t_to_t_plus_10": True,
            "same_teacher_forward": True,
            "same_final_level": FINAL_LEVEL,
            "same_source_time_index": SOURCE_TIME_INDEX,
            "only_pooling_changed": True,
        },
    }
    atomic_json(worker_root / "worker_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def finite_statistics(features: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    return {
        "count": len(features),
        "feature_dim": int(features.shape[1]),
        "dtype": str(features.dtype),
        "finite": bool(np.isfinite(features).all()),
        "mean": features.mean(axis=0, dtype=np.float64).tolist(),
        "std": features.std(axis=0, dtype=np.float64).tolist(),
        "norm": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
    }


def load_task_related_targets(
    index_path: Path, sample_ids: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    table = pq.read_table(index_path)
    required = {
        "sample_id",
        "target_shard_path",
        "target_shard_row",
        "target_shard_sha256",
    }
    if not required.issubset(table.column_names):
        raise ValueError(
            f"Task-related index is missing columns: {sorted(required - set(table.column_names))}"
        )
    rows = table.to_pylist()
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Task-related Motion index contains duplicate sample IDs")
    missing = sorted(set(sample_ids) - set(by_id))
    if missing:
        raise ValueError(
            f"Whole-scene population is absent from task-related index: {missing[:20]}"
        )

    locations: dict[Path, list[tuple[int, int, str]]] = defaultdict(list)
    for output_index, sample_id in enumerate(sample_ids):
        row = by_id[sample_id]
        locations[Path(row["target_shard_path"])].append(
            (
                output_index,
                int(row["target_shard_row"]),
                str(row["target_shard_sha256"]),
            )
        )
    features = np.zeros((len(sample_ids), FEATURE_DIM), dtype=np.float32)
    checked_shards = 0
    for path, entries in locations.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing task-related Motion shard: {path}")
        expected_shas = {entry[2] for entry in entries}
        if len(expected_shas) != 1 or sha256_file(path) not in expected_shas:
            raise ValueError(f"Task-related Motion shard checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            shard_features = arrays["motion_target_fp32"]
            for output_index, shard_row, _ in entries:
                features[output_index] = np.asarray(
                    shard_features[shard_row], dtype=np.float32
                )
        checked_shards += 1
    if not np.isfinite(features).all():
        raise ValueError("Task-related Motion reference contains non-finite values")
    return features, {
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "index_samples": len(rows),
        "selected_samples": len(sample_ids),
        "population_exact": len(rows) == len(sample_ids)
        and set(by_id) == set(sample_ids),
        "checked_shards": checked_shards,
    }


def paired_difference(whole: np.ndarray, task: np.ndarray) -> dict[str, Any]:
    if whole.shape != task.shape:
        raise ValueError(f"Paired target shapes differ: {whole.shape} vs {task.shape}")
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
    exact_rows = np.all(whole == task, axis=1)
    return {
        "samples": len(whole),
        "exact_equal_rows": int(exact_rows.sum()),
        "different_rows": int((~exact_rows).sum()),
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
    reports: list[dict[str, Any]] = []
    by_index: dict[int, dict[str, Any]] = {}
    for worker_index in range(args.num_workers):
        worker_root = args.output_root / "workers" / f"worker_{worker_index:02d}"
        report_path = worker_root / "worker_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing worker report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "PASS"
            or report.get("schema_version") != SCHEMA
            or report.get("target_scope") != TARGET_SCOPE
            or report.get("diagnostic_samples_per_suite")
            != args.diagnostic_samples_per_suite
        ):
            raise RuntimeError(f"Worker provenance did not pass: {report_path}")
        reports.append(report)
        for shard_path in sorted((worker_root / "shards").glob("motion_*.npz")):
            with np.load(shard_path, allow_pickle=False) as arrays:
                for local_index, global_index in enumerate(
                    arrays["global_index"].astype(int)
                ):
                    if int(global_index) in by_index:
                        raise ValueError(
                            f"Duplicate Motion global index: {global_index}"
                        )
                    record: dict[str, Any] = {
                        "sample_id": str(arrays["sample_id"][local_index]),
                        "feature": np.asarray(
                            arrays["motion_target_fp32"][local_index], dtype=np.float32
                        ),
                        "patch_count": int(
                            arrays["whole_scene_patch_count"][local_index]
                        ),
                        "scope": str(arrays["target_scope"][local_index]),
                    }
                    if "task_related_reference_fp32" in arrays:
                        record["task_reference"] = np.asarray(
                            arrays["task_related_reference_fp32"][local_index],
                            dtype=np.float32,
                        )
                    by_index[int(global_index)] = record

    expected_indices = set(range(len(rows)))
    if set(by_index) != expected_indices:
        missing = sorted(expected_indices - set(by_index))[:20]
        extra = sorted(set(by_index) - expected_indices)[:20]
        raise ValueError(
            f"Worker index coverage mismatch: missing={missing}, extra={extra}"
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    observed_ids = [by_index[index]["sample_id"] for index in range(len(rows))]
    if observed_ids != sample_ids:
        raise ValueError("Worker sample IDs do not align with selected manifest order")
    if any(
        by_index[index]["patch_count"] != SOURCE_PATCH_COUNT
        or by_index[index]["scope"] != TARGET_SCOPE
        for index in range(len(rows))
    ):
        raise ValueError("Whole-scene target scope/patch-count gate failed")
    features = np.stack([by_index[index]["feature"] for index in range(len(rows))])
    if features.shape != (len(rows), FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError("Final Whole-scene Motion shape/finite gate failed")

    task_features, task_population = load_task_related_targets(
        args.task_related_index, sample_ids
    )
    if (
        args.diagnostic_samples_per_suite is None
        and not task_population["population_exact"]
    ):
        raise ValueError(
            "Full Whole-scene population must exactly equal task-related population"
        )
    difference = paired_difference(features, task_features)
    if difference["different_rows"] == 0:
        raise ValueError("Whole-scene targets did not change from task-related targets")

    computed_task_records = [
        by_index[index].get("task_reference") for index in range(len(rows))
    ]
    computed_task_validation: dict[str, Any] | None = None
    if any(record is not None for record in computed_task_records):
        if not all(record is not None for record in computed_task_records):
            raise ValueError("Diagnostic task-related references are incomplete")
        computed_task = np.stack(computed_task_records)
        absolute = np.abs(computed_task - task_features)
        computed_task_validation = {
            "samples": len(rows),
            "exact_equal_to_existing_cache": bool(
                np.array_equal(computed_task, task_features)
            ),
            "max_abs": float(absolute.max(initial=0.0)),
            "mean_abs": float(absolute.mean()),
        }
        if not computed_task_validation["exact_equal_to_existing_cache"]:
            raise ValueError(
                f"Same-forward task-related targets differ from existing cache: {computed_task_validation}"
            )

    final_shard_root = args.output_root / "shards"
    final_shard_root.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for shard_start in range(0, len(rows), args.final_shard_size):
        shard_end = min(shard_start + args.final_shard_size, len(rows))
        path = final_shard_root / f"motion_{shard_start:06d}_{shard_end - 1:06d}.npz"
        atomic_npz(
            path,
            sample_id=np.asarray(sample_ids[shard_start:shard_end]),
            motion_target_fp32=features[shard_start:shard_end],
            whole_scene_patch_count=np.full(
                shard_end - shard_start, SOURCE_PATCH_COUNT, dtype=np.int32
            ),
            target_scope=np.asarray([TARGET_SCOPE] * (shard_end - shard_start)),
        )
        shard_sha = sha256_file(path)
        shard_records.append(
            {
                "path": str(path),
                "sha256": shard_sha,
                "bytes": path.stat().st_size,
                "samples": shard_end - shard_start,
            }
        )
        for local_index, row in enumerate(rows[shard_start:shard_end]):
            index_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "suite": row["suite"],
                    "split": row["split"],
                    "motion_crosses_subtask_boundary": bool(
                        row["motion_crosses_subtask_boundary"]
                    ),
                    "boundary_near": bool(row["boundary_near"]),
                    "target_scope": TARGET_SCOPE,
                    "pooling": "uniform_source_patch_mean",
                    "target_shard_path": str(path),
                    "target_shard_row": local_index,
                    "target_dim": FEATURE_DIM,
                    "target_dtype": "float32",
                    "target_shard_sha256": shard_sha,
                }
            )
    index_table = pa.Table.from_pylist(index_rows).sort_by([("sample_id", "ascending")])
    atomic_parquet(args.output_root / "index.parquet", index_table)
    atomic_json(args.output_root / "shard_index.json", {"shards": shard_records})
    train_mask = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    atomic_json(
        args.output_root / "target_statistics_train.json",
        finite_statistics(features[train_mask]),
    )

    wall_proxy = max(float(report["extraction_seconds"]) for report in reports)
    processed = sum(int(report["processed_count"]) for report in reports)
    suite_counts = Counter(str(row["suite"]) for row in rows)
    validation = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "target_scope": TARGET_SCOPE,
        "selection_column": args.selection_column,
        "diagnostic_samples_per_suite": args.diagnostic_samples_per_suite,
        "selected_samples": len(rows),
        "selected_by_suite": dict(sorted(suite_counts.items())),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "all_finite": bool(np.isfinite(features).all()),
        "sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
        "no_missing_targets": len(index_table) == len(rows),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "index_sha256": sha256_file(args.output_root / "index.parquet"),
        "task_related_population": task_population,
        "same_forward_task_related_cache_check": computed_task_validation,
        "whole_scene_vs_task_related": difference,
        "teacher": {
            "module": "flow_aggregator3d.global_blocks.3",
            "source_time_index": SOURCE_TIME_INDEX,
            "pooling": f"uniform mean over all {SOURCE_PATCH_COUNT} source patches",
            "clip": "11 real agent frames t:t+10",
            "camera": "agent only",
            "feature_dim": FEATURE_DIM,
        },
        "matched_control": {
            "same_valid_population": True,
            "same_agent_clip_t_to_t_plus_10": True,
            "same_teacher_forward": True,
            "same_final_level": FINAL_LEVEL,
            "same_source_time_index": SOURCE_TIME_INDEX,
            "only_pooling_changed": True,
        },
        "workers": reports,
        "runtime": {
            "worker_count": args.num_workers,
            "processed_this_run": processed,
            "wall_proxy_seconds": wall_proxy,
            "overall_clips_per_second": processed / wall_proxy if processed else None,
            "sum_peak_vram_bytes": sum(
                int(report["peak_vram_bytes"]) for report in reports
            ),
        },
        "shards": shard_records,
    }
    atomic_json(args.output_root / "cache_validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "finalize"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--selection-column", default="motion_valid")
    parser.add_argument("--track4world-repo", type=Path)
    parser.add_argument("--utils3d-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--da3-snapshot", type=Path)
    parser.add_argument("--teacher-reference-root", type=Path)
    parser.add_argument("--task-related-index", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--final-shard-size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic-samples-per-suite", type=int)
    parser.add_argument("--diagnostic-compare-task-related", action="store_true")
    args = parser.parse_args()
    for name in (
        "manifest",
        "source_manifest",
        "track4world_repo",
        "utils3d_repo",
        "checkpoint",
        "da3_snapshot",
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
        raise ValueError(
            "--diagnostic-compare-task-related is restricted to a diagnostic smoke"
        )
    if args.mode == "worker":
        required = (
            "track4world_repo",
            "utils3d_repo",
            "checkpoint",
            "da3_snapshot",
            "teacher_reference_root",
            "source_manifest",
        )
    else:
        required = ("task_related_index",)
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments for {args.mode}: {missing}")
    if args.shard_size <= 0 or args.final_shard_size <= 0:
        raise ValueError("Shard sizes must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "worker":
        raise SystemExit(run_worker(arguments))
    raise SystemExit(run_finalize(arguments))
