"""Consolidate immutable RoboCasa teacher caches into training memory maps.

This is a CPU-only, one-time conversion.  It never rewrites a source cache and
refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .auxiliary import PreparedAuxiliaryPaths
from .auxiliary import TargetScope
from .constants import GEOMETRY_DIM
from .constants import MOTION_DIM
from .constants import TASKS


def _read_ordered(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns).sort_values("task_frame_index")
    expected = np.arange(len(frame), dtype=np.int64)
    actual = frame["task_frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError(f"non-contiguous task_frame_index: {path}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate sample IDs: {path}")
    return frame.reset_index(drop=True)


def _relocated_path(raw: object, root: Path, task: str) -> Path:
    value = str(raw)
    direct = Path(value)
    candidates = [direct, root / value]
    parts = direct.parts
    if task in parts:
        candidates.append(root.joinpath(*parts[parts.index(task) :]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve cache path {value!r} below {root}")


class _Moments:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_sq = np.zeros(dim, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        if values64.ndim != 2 or values64.shape[1] != self.dim:
            raise ValueError(f"target matrix shape differs: {values64.shape}")
        if not np.isfinite(values64).all():
            raise ValueError("teacher target contains NaN or Inf")
        self.count += len(values64)
        self.total += values64.sum(axis=0, dtype=np.float64)
        self.total_sq += np.square(values64).sum(axis=0, dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.count <= 0:
            raise ValueError("cannot normalize an empty teacher population")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        raw_std = np.sqrt(variance)
        std = np.maximum(raw_std, 1e-6)
        payload = {
            "count": self.count,
            "feature_dim": self.dim,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "std_floor": 1e-6,
            "floored_features": int(np.count_nonzero(raw_std < 1e-6)),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _same_ids(actual: Iterable[object], expected: Iterable[object], label: str) -> None:
    actual_ids = tuple(map(str, actual))
    expected_ids = tuple(map(str, expected))
    if actual_ids != expected_ids:
        actual_set, expected_set = set(actual_ids), set(expected_ids)
        raise ValueError(
            f"{label} identity/order mismatch: "
            f"missing={len(expected_set - actual_set)}, extra={len(actual_set - expected_set)}"
        )


def _load_sharded_targets(
    *,
    index: pd.DataFrame,
    cache_root: Path,
    task: str,
    value_key: str,
    target_dim: int,
    destination: np.memmap,
    destination_rows: np.ndarray,
    moments: _Moments,
) -> None:
    if len(index) != len(destination_rows):
        raise ValueError("cache rows/destination rows differ")
    rows_by_path: dict[str, list[int]] = {}
    for row_position, raw_path in enumerate(index["target_shard_path"]):
        rows_by_path.setdefault(str(raw_path), []).append(row_position)
    for raw_path, positions in rows_by_path.items():
        shard_path = _relocated_path(raw_path, cache_root, task)
        positions_array = np.asarray(positions, dtype=np.int64)
        shard_rows = index.iloc[positions_array]["target_shard_row"].to_numpy(
            dtype=np.int64
        )
        with np.load(shard_path, allow_pickle=False) as shard:
            values = np.asarray(shard[value_key][shard_rows], dtype=np.float32)
            shard_ids = np.asarray(shard["sample_id"])[shard_rows].astype(str)
        expected_ids = index.iloc[positions_array]["sample_id"].astype(str).to_numpy()
        if not np.array_equal(shard_ids, expected_ids):
            raise ValueError(f"shard sample IDs differ: {shard_path}")
        if (
            values.shape != (len(positions), target_dim)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"invalid target values: {shard_path} {values.shape}")
        output_rows = destination_rows[positions_array]
        destination[output_rows] = values
        moments.add(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    *,
    scope: TargetScope,
    manifest_root: Path,
    semantic_root: Path,
    geometry_root: Path,
    motion_root: Path,
    output_dir: Path,
    tasks: tuple[str, ...] = TASKS,
    write_checksums: bool = True,
) -> dict:
    """Create one scope-specific aligned target artifact atomically."""

    if scope not in ("task_relevant", "whole_scene"):
        raise ValueError(scope)
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in TASKS for task in tasks)
    ):
        raise ValueError("invalid Atomic-24 task selection")
    roots = [manifest_root, semantic_root, geometry_root, motion_root]
    manifest_root, semantic_root, geometry_root, motion_root = [
        Path(root).resolve(strict=True) for root in roots
    ]
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prepared artifact: {output_dir}")

    sources: dict[str, pd.DataFrame] = {}
    total_frames = 0
    for task in tasks:
        source = _read_ordered(
            manifest_root / task / "source" / "source_manifest.parquet",
            [
                "sample_id",
                "task",
                "task_frame_index",
                "source_role",
                "review_accepted",
                "geometry_valid",
                "motion_valid",
            ],
        )
        if (
            not source["task"].eq(task).all()
            or not source["source_role"].eq("base50").all()
        ):
            raise ValueError(f"{task}: source population is not fixed base50")
        if not source["review_accepted"].to_numpy(dtype=bool).all():
            raise ValueError(f"{task}: source manifest contains unaccepted annotations")
        sources[task] = source
        total_frames += len(source)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    paths = PreparedAuxiliaryPaths(temporary)
    try:
        semantic_input = np.lib.format.open_memmap(
            paths.semantic_input_ids,
            mode="w+",
            dtype=np.int64,
            shape=(total_frames, 31),
        )
        semantic_labels = np.lib.format.open_memmap(
            paths.semantic_labels, mode="w+", dtype=np.int64, shape=(total_frames, 32)
        )
        semantic_mask = np.lib.format.open_memmap(
            paths.semantic_loss_mask,
            mode="w+",
            dtype=np.bool_,
            shape=(total_frames, 32),
        )
        geometry = np.lib.format.open_memmap(
            paths.geometry_targets,
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, GEOMETRY_DIM),
        )
        geometry_valid = np.lib.format.open_memmap(
            paths.geometry_valid, mode="w+", dtype=np.bool_, shape=(total_frames,)
        )
        motion = np.lib.format.open_memmap(
            paths.motion_targets,
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, MOTION_DIM),
        )
        motion_valid = np.lib.format.open_memmap(
            paths.motion_valid, mode="w+", dtype=np.bool_, shape=(total_frames,)
        )
        # Invalid supervision rows are explicitly zero, never uninitialized.
        geometry[:] = 0.0
        geometry_valid[:] = False
        motion[:] = 0.0
        motion_valid[:] = False

        geometry_moments = _Moments(GEOMETRY_DIM)
        motion_moments = _Moments(MOTION_DIM)
        semantic_review_true_count = 0
        index_frames: list[pd.DataFrame] = []
        offset = 0
        geometry_subdir = (
            "geometry" if scope == "task_relevant" else "geometry_whole_scene"
        )
        motion_subdir = "motion" if scope == "task_relevant" else "motion_whole_scene"

        for task in tasks:
            source = sources[task]
            count = len(source)
            output_rows = np.arange(offset, offset + count, dtype=np.int64)
            source_ids = source["sample_id"].astype(str)

            semantic_index = _read_ordered(
                semantic_root / task / "semantic" / "index.parquet",
                ["sample_id", "task_frame_index", "target_path", "target_row"],
            )
            _same_ids(semantic_index["sample_id"], source_ids, f"{task} Semantic")
            semantic_paths = semantic_index["target_path"].astype(str).unique()
            if len(semantic_paths) != 1:
                raise ValueError(
                    f"{task}: Semantic target must be one immutable task file"
                )
            semantic_path = _relocated_path(semantic_paths[0], semantic_root, task)
            semantic_rows = semantic_index["target_row"].to_numpy(dtype=np.int64)
            with np.load(semantic_path, allow_pickle=False) as payload:
                _same_ids(
                    payload["sample_id"][semantic_rows],
                    source_ids,
                    f"{task} Semantic NPZ",
                )
                valid = np.asarray(payload["valid"][semantic_rows], dtype=bool)
                reviewed = np.asarray(payload["review"][semantic_rows], dtype=bool)
                if not valid.all():
                    raise ValueError(f"{task}: Semantic contains invalid rows")
                # ``review`` in the released Semantic NPZ records whether the
                # cached text was manually re-reviewed during export; it is
                # uniformly False for this cache. Annotation acceptance is the
                # authoritative source-manifest ``review_accepted`` field,
                # checked above for every frame.
                semantic_review_true_count += int(reviewed.sum())
                semantic_input[output_rows] = payload["semantic_input_ids"][
                    semantic_rows
                ]
                semantic_labels[output_rows] = payload["semantic_labels"][semantic_rows]
                semantic_mask[output_rows] = payload["semantic_loss_mask"][
                    semantic_rows
                ]

            geometry_index = (
                pd.read_parquet(
                    geometry_root / task / geometry_subdir / "final" / "index.parquet",
                    columns=[
                        "sample_id",
                        "task_frame_index",
                        "geometry_available",
                        "target_shard_path",
                        "target_shard_row",
                        "target_dim",
                        "target_dtype",
                    ],
                )
                .sort_values("task_frame_index")
                .reset_index(drop=True)
            )
            if geometry_index["sample_id"].astype(str).duplicated().any():
                raise ValueError(f"{task}: duplicate Geometry IDs")
            expected_geometry_valid = source["geometry_valid"].to_numpy(dtype=bool)
            expected_geometry_ids = source_ids[expected_geometry_valid]
            actual_available = geometry_index["geometry_available"].to_numpy(dtype=bool)
            if len(geometry_index) == count:
                _same_ids(geometry_index["sample_id"], source_ids, f"{task} Geometry")
                if not np.array_equal(actual_available, expected_geometry_valid):
                    raise ValueError(
                        f"{task}: {scope} Geometry changed the task-relevant valid population"
                    )
                valid_geometry = geometry_index[actual_available].reset_index(drop=True)
            else:
                if not actual_available.all():
                    raise ValueError(
                        f"{task}: sparse Geometry index contains unavailable rows"
                    )
                _same_ids(
                    geometry_index["sample_id"],
                    expected_geometry_ids,
                    f"{task} Geometry",
                )
                valid_geometry = geometry_index
            geometry_valid[output_rows] = expected_geometry_valid
            if (
                not valid_geometry["target_dim"].eq(GEOMETRY_DIM).all()
                or not valid_geometry["target_dtype"].eq("float32").all()
            ):
                raise ValueError(f"{task}: Geometry cache schema differs")
            _load_sharded_targets(
                index=valid_geometry,
                cache_root=geometry_root,
                task=task,
                value_key="geometry_target_fp32",
                target_dim=GEOMETRY_DIM,
                destination=geometry,
                destination_rows=output_rows[expected_geometry_valid],
                moments=geometry_moments,
            )

            expected_motion_ids = source_ids[
                source["motion_valid"].to_numpy(dtype=bool)
            ]
            motion_index_path = (
                motion_root / task / motion_subdir / "final" / "index.parquet"
            )
            motion_index = (
                pd.read_parquet(
                    motion_index_path,
                    columns=[
                        "sample_id",
                        "task_frame_index",
                        "target_shard_path",
                        "target_shard_row",
                        "target_dim",
                        "target_dtype",
                    ],
                )
                .sort_values("task_frame_index")
                .reset_index(drop=True)
            )
            if motion_index["sample_id"].astype(str).duplicated().any():
                raise ValueError(f"{task}: duplicate Motion IDs")
            _same_ids(motion_index["sample_id"], expected_motion_ids, f"{task} Motion")
            if (
                not motion_index["target_dim"].eq(MOTION_DIM).all()
                or not motion_index["target_dtype"].eq("float32").all()
            ):
                raise ValueError(f"{task}: Motion cache schema differs")
            task_motion_valid = source["motion_valid"].to_numpy(dtype=bool)
            motion_valid[output_rows] = task_motion_valid
            _load_sharded_targets(
                index=motion_index,
                cache_root=motion_root,
                task=task,
                value_key="motion_target_fp32",
                target_dim=MOTION_DIM,
                destination=motion,
                destination_rows=output_rows[task_motion_valid],
                moments=motion_moments,
            )

            index_frames.append(
                pd.DataFrame(
                    {
                        "dataset_index": output_rows,
                        "sample_id": source_ids.to_numpy(),
                        "task": task,
                        "task_frame_index": source["task_frame_index"].to_numpy(
                            dtype=np.int64
                        ),
                        "target_scope": scope,
                    }
                )
            )
            offset += count

        if offset != total_frames:
            raise AssertionError("prepared frame count differs")
        for array in (
            semantic_input,
            semantic_labels,
            semantic_mask,
            geometry,
            geometry_valid,
            motion,
            motion_valid,
        ):
            array.flush()
        pd.concat(index_frames, ignore_index=True).to_parquet(paths.index, index=False)
        geometry_moments.save(paths.geometry_normalization)
        motion_moments.save(paths.motion_normalization)

        artifact_files = sorted(path for path in temporary.iterdir() if path.is_file())
        report = {
            "status": "PASS",
            "target_scope": scope,
            "tasks": list(tasks),
            "task_count": len(tasks),
            "frame_count": total_frames,
            "geometry_valid_count": geometry_moments.count,
            "motion_valid_count": motion_moments.count,
            "semantic_valid_count": total_frames,
            "semantic_cache_review_true_count": semantic_review_true_count,
            "source_review_accepted_count": total_frames,
            "files": {
                path.name: {
                    "bytes": path.stat().st_size,
                    **({"sha256": _sha256(path)} if write_checksums else {}),
                }
                for path in artifact_files
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output_dir)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=("task_relevant", "whole_scene"), required=True
    )
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    report = prepare(
        scope=args.scope,
        manifest_root=args.manifest_root,
        semantic_root=args.semantic_root,
        geometry_root=args.geometry_root,
        motion_root=args.motion_root,
        output_dir=args.output_dir,
        write_checksums=not args.skip_checksums,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
