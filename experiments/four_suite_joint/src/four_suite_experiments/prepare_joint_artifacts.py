"""Build small additive metadata artifacts over immutable four-suite caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FOUR_SUITE_EPISODES
from .constants import FOUR_SUITE_FRAMES
from .constants import FOUR_SUITE_GEOMETRY_INVALID
from .constants import FOUR_SUITE_GEOMETRY_VALID
from .constants import FOUR_SUITE_MOTION_VALID
from .constants import FOUR_SUITE_TASKS
from .constants import GEOMETRY_DIM
from .constants import LIBERO_REPO_ID
from .constants import LIBERO_REVISION
from .constants import MOTION_DIM
from .constants import SUITES
from .paths import ArtifactPaths
from .paths import SourcePaths
from .statistics import pool_geometry_normalizations
from .statistics import pool_motion_normalizations


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "suite",
        "episode_id",
        "annotation_episode_index",
        "annotation_task_index",
        "frame_idx",
        "episode_length",
        "action_sha256",
        "semantic_subtask",
        "geometry_valid",
        "motion_valid",
        "lerobot_episode_index",
        "lerobot_task_index",
        "lerobot_frame_index",
        "lerobot_dataset_index",
        "instruction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"joint manifest is missing columns: {missing}")
    frame = frame.sort_values("lerobot_dataset_index").reset_index(drop=True)
    if len(frame) != FOUR_SUITE_FRAMES or not frame["sample_id"].is_unique:
        raise ValueError("joint manifest does not contain the frozen unique frame population")
    if frame["lerobot_dataset_index"].astype(int).tolist() != list(range(FOUR_SUITE_FRAMES)):
        raise ValueError("joint manifest is not in exact contiguous LeRobot frame order")
    if set(frame["suite"].astype(str)) != set(SUITES):
        raise ValueError("joint manifest suite population differs from the frozen four suites")
    if frame["semantic_subtask"].isna().any() or frame["semantic_subtask"].astype(str).str.len().eq(0).any():
        raise ValueError("joint manifest has missing Semantic targets")
    if int(frame["geometry_valid"].astype(bool).sum()) != FOUR_SUITE_GEOMETRY_VALID:
        raise ValueError("joint manifest Geometry valid count differs")
    if int(frame["motion_valid"].astype(bool).sum()) != FOUR_SUITE_MOTION_VALID:
        raise ValueError("joint manifest Motion valid count differs")
    if frame["lerobot_episode_index"].nunique() != FOUR_SUITE_EPISODES:
        raise ValueError("joint manifest episode count differs")
    if frame["lerobot_task_index"].nunique() != FOUR_SUITE_TASKS:
        raise ValueError("joint manifest task count differs")
    return frame


def build_episode_mapping(frame: pd.DataFrame, lerobot_root: Path, manifest_sha256: str) -> dict:
    episode_meta = {int(row["episode_index"]): row for row in _read_jsonl(lerobot_root / "meta/episodes.jsonl")}
    task_meta = {int(row["task_index"]): str(row["task"]) for row in _read_jsonl(lerobot_root / "meta/tasks.jsonl")}
    records = []
    for raw_episode_index, raw_group in frame.groupby("lerobot_episode_index", sort=True):
        episode_index = int(raw_episode_index)
        group = raw_group.sort_values("lerobot_dataset_index")
        length = len(group)
        invariant_columns = (
            "episode_id",
            "annotation_episode_index",
            "annotation_task_index",
            "episode_length",
            "action_sha256",
            "lerobot_task_index",
            "instruction",
            "suite",
        )
        for column in invariant_columns:
            if group[column].nunique(dropna=False) != 1:  # noqa: PD101
                raise ValueError(f"episode {episode_index} has non-invariant {column}")
        if group["lerobot_frame_index"].astype(int).tolist() != list(range(length)):
            raise ValueError(f"episode {episode_index} has non-contiguous frame indices")
        if group["frame_idx"].astype(int).tolist() != list(range(length)):
            raise ValueError(f"episode {episode_index} annotation frames are not contiguous")
        meta = episode_meta.get(episode_index)
        task_index = int(group["lerobot_task_index"].iloc[0])
        instruction = str(group["instruction"].iloc[0])
        if meta is None or int(meta["length"]) != length or meta["tasks"] != [instruction]:
            raise ValueError(f"episode {episode_index} disagrees with official episode metadata")
        if task_meta.get(task_index) != instruction:
            raise ValueError(f"episode {episode_index} disagrees with official task metadata")
        start = int(group["lerobot_dataset_index"].iloc[0])
        stop = int(group["lerobot_dataset_index"].iloc[-1]) + 1
        relative_path = Path(f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet")
        records.append(
            {
                "action_sha256": str(group["action_sha256"].iloc[0]),
                "annotation_episode_id": str(group["episode_id"].iloc[0]),
                "annotation_episode_index": int(group["annotation_episode_index"].iloc[0]),
                "annotation_task_index": int(group["annotation_task_index"].iloc[0]),
                "dataset_from_index": start,
                "dataset_to_index_exclusive": stop,
                "episode_length": length,
                "instruction": instruction,
                "lerobot_episode_index": episode_index,
                "lerobot_task_index": task_index,
                "parquet_relative_path": str(relative_path),
                "suite": str(group["suite"].iloc[0]),
            }
        )
    if len(records) != FOUR_SUITE_EPISODES:
        raise ValueError("episode mapping count differs")
    if sum(record["episode_length"] for record in records) != FOUR_SUITE_FRAMES:
        raise ValueError("episode mapping frame count differs")
    if [record["lerobot_episode_index"] for record in records] != list(range(FOUR_SUITE_EPISODES)):
        raise ValueError("episode mapping is not contiguous")
    return {
        "schema": "four_suite_lerobot_episode_mapping_v1",
        "status": "PASS",
        "hf_repo_id": LIBERO_REPO_ID,
        "hf_revision": LIBERO_REVISION,
        "mapped_episode_count": FOUR_SUITE_EPISODES,
        "mapped_frame_count": FOUR_SUITE_FRAMES,
        "mapped_task_count": FOUR_SUITE_TASKS,
        "source_manifest_sha256": manifest_sha256,
        "episodes": records,
    }


def _absolute_path(value: object, source_index: Path) -> object:
    if pd.isna(value):
        return value
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return str((source_index.parent / path).resolve(strict=True))
    if path.is_file():
        return str(path.resolve(strict=True))

    # Teacher indices generated on another machine may contain an absolute
    # shard path.  The cache directory itself is transferred intact, so rebase
    # the suffix below the source-index directory onto its new HPC location.
    parent_name = source_index.parent.name
    matching_positions = [index for index, part in enumerate(path.parts) if part == parent_name]
    for position in reversed(matching_positions):
        candidate = source_index.parent.joinpath(*path.parts[position + 1 :])
        if candidate.is_file():
            return str(candidate.resolve(strict=True))
    raise FileNotFoundError(f"target path is unavailable and cannot be rebased below {source_index.parent}: {path}")


def _validate_source_scope(
    frame: pd.DataFrame,
    *,
    expected_scope: str,
    source_index: Path,
) -> None:
    if expected_scope not in ("task_relevant", "whole_scene"):
        raise ValueError(f"unsupported target scope: {expected_scope}")
    if "target_scope" not in frame:
        if expected_scope == "whole_scene":
            raise ValueError(f"Whole-scene source index lacks an explicit target_scope: {source_index}")
        return
    observed = set(frame["target_scope"].dropna().astype(str))
    if observed != {expected_scope}:
        raise ValueError(
            f"source target scope differs: expected={expected_scope}, observed={sorted(observed)}, index={source_index}"
        )


def build_geometry_index(paths: tuple[Path, ...], *, target_scope: str = "task_relevant") -> pd.DataFrame:
    parts = []
    for path in paths:
        part = pd.read_parquet(path).copy()
        _validate_source_scope(
            part,
            expected_scope=target_scope,
            source_index=path,
        )
        if "target_memmap_path" not in part:
            raise ValueError(f"Geometry index lacks target_memmap_path: {path}")
        part["target_memmap_path"] = part["target_memmap_path"].map(
            lambda value, source=path: _absolute_path(value, source)
        )
        part["source_index_path"] = str(path.resolve())
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True, sort=False).sort_values("lerobot_dataset_index").reset_index(drop=True)
    if len(frame) != FOUR_SUITE_FRAMES or not frame["sample_id"].is_unique:
        raise ValueError("combined Geometry index has the wrong population")
    if frame["lerobot_dataset_index"].astype(int).tolist() != list(range(FOUR_SUITE_FRAMES)):
        raise ValueError("combined Geometry index is not in exact LeRobot order")
    valid = frame["geometry_valid"].astype(bool)
    if int(valid.sum()) != FOUR_SUITE_GEOMETRY_VALID or int((~valid).sum()) != FOUR_SUITE_GEOMETRY_INVALID:
        raise ValueError("combined Geometry validity counts differ")
    if frame.loc[valid, "target_memmap_path"].isna().any() or frame.loc[valid, "target_memmap_row"].isna().any():
        raise ValueError("valid Geometry rows lack target locations")
    if not frame.loc[valid, "target_dim"].eq(GEOMETRY_DIM).all():
        raise ValueError("Geometry dimensions differ")
    if not frame.loc[valid, "target_dtype"].eq("float32").all():
        raise ValueError("Geometry dtypes differ")
    return frame


def build_motion_index(paths: tuple[Path, ...], *, target_scope: str = "task_relevant") -> pd.DataFrame:
    parts = []
    for path in paths:
        part = pd.read_parquet(path).copy()
        _validate_source_scope(
            part,
            expected_scope=target_scope,
            source_index=path,
        )
        part["target_shard_path"] = part["target_shard_path"].map(
            lambda value, source=path: _absolute_path(value, source)
        )
        part["source_index_path"] = str(path.resolve())
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True, sort=False)
    if len(frame) != FOUR_SUITE_MOTION_VALID or not frame["sample_id"].is_unique:
        raise ValueError("combined Motion index has the wrong valid population")
    if not frame["target_dim"].eq(MOTION_DIM).all() or not frame["target_dtype"].eq("float32").all():
        raise ValueError("Motion shape/dtype differs")
    return frame.sort_values("sample_id").reset_index(drop=True)


def materialize_motion_memmap(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Copy immutable Motion shards into one portable read-only target array.

    The returned index is deliberately independent of the source shard layout:
    training resolves every stable sample ID to one row in ``output_path``.
    Source shards remain immutable inputs and are read exactly once here.
    """

    required = {
        "sample_id",
        "target_shard_path",
        "target_shard_row",
        "target_dim",
        "target_dtype",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Motion source index is missing columns: {missing}")
    frame = frame.reset_index(drop=True).copy()
    if not frame["sample_id"].is_unique:
        raise ValueError("Motion source sample IDs are not unique")
    if not frame["target_dim"].eq(MOTION_DIM).all() or not frame["target_dtype"].eq("float32").all():
        raise ValueError("Motion source targets are not float32[256]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    targets = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), MOTION_DIM),
    )
    try:
        for raw_shard_path, group in frame.groupby("target_shard_path", sort=False):
            shard_path = Path(str(raw_shard_path)).resolve(strict=True)
            source_rows = group["target_shard_row"].astype(np.int64).to_numpy()
            output_rows = group.index.to_numpy(dtype=np.int64)
            with np.load(shard_path, allow_pickle=False) as shard:
                if "sample_id" not in shard or "motion_target_fp32" not in shard:
                    raise ValueError(f"Motion shard schema differs: {shard_path}")
                shard_sample_ids = np.asarray(shard["sample_id"])
                shard_targets = np.asarray(shard["motion_target_fp32"])
                if source_rows.min(initial=0) < 0 or source_rows.max(initial=-1) >= len(shard_sample_ids):
                    raise ValueError(f"Motion shard row is out of bounds: {shard_path}")
                observed_ids = shard_sample_ids[source_rows].astype(str)
                expected_ids = group["sample_id"].astype(str).to_numpy()
                if not np.array_equal(observed_ids, expected_ids):
                    raise ValueError(f"Motion shard identity mismatch: {shard_path}")
                selected = shard_targets[source_rows]
                if selected.shape != (len(group), MOTION_DIM) or selected.dtype != np.float32:
                    raise ValueError(f"Motion shard shape/dtype differs: {shard_path}")
                if not np.isfinite(selected).all():
                    raise ValueError(f"Motion shard contains non-finite targets: {shard_path}")
                targets[output_rows] = selected
        targets.flush()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    del targets
    temporary_path.replace(output_path)

    portable_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "target_shard_path",
            "target_shard_row",
            "target_shard_sha256",
            "source_index_path",
        }
    ]
    portable = frame[portable_columns].copy()
    portable["target_memmap_path"] = output_path.name
    portable["target_memmap_row"] = np.arange(len(portable), dtype=np.int64)
    return portable


def verify_motion_memmap(frame: pd.DataFrame, output_path: Path) -> dict:
    """Re-read every source target and require bit-exact materialization."""

    targets = np.load(output_path.resolve(strict=True), mmap_mode="r")
    if targets.shape != (len(frame), MOTION_DIM) or targets.dtype != np.float32:
        raise ValueError("Materialized Motion memmap shape/dtype differs")
    if targets.flags.writeable:
        raise ValueError("Materialized Motion memmap is unexpectedly writeable")
    checked = 0
    for raw_shard_path, group in frame.reset_index(drop=True).groupby("target_shard_path", sort=False):
        shard_path = Path(str(raw_shard_path)).resolve(strict=True)
        if "target_shard_sha256" in group:
            expected_hashes = group["target_shard_sha256"].dropna().unique().tolist()
            if len(expected_hashes) != 1 or sha256_file(shard_path) != expected_hashes[0]:
                raise ValueError(f"Motion source shard hash mismatch: {shard_path}")
        source_rows = group["target_shard_row"].astype(np.int64).to_numpy()
        output_rows = group.index.to_numpy(dtype=np.int64)
        with np.load(shard_path, allow_pickle=False) as shard:
            expected = np.asarray(shard["motion_target_fp32"])[source_rows]
            observed = np.asarray(targets[output_rows])
            if not np.array_equal(observed, expected):
                raise ValueError(f"Materialized Motion values differ: {shard_path}")
        checked += len(group)
    if checked != len(frame):
        raise ValueError(f"Materialized Motion verification count differs: {checked}")
    return {
        "schema": "sample_id_to_readonly_memmap_v1",
        "dtype": "float32",
        "feature_dim": MOTION_DIM,
        "sample_count": checked,
        "source_shard_count": int(frame["target_shard_path"].nunique()),
        "bit_exact": True,
        "read_only": True,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(paths: SourcePaths, *, force: bool = False) -> ArtifactPaths:
    for source in paths.required_sources():
        if not source.exists():
            raise FileNotFoundError(source)
    output = ArtifactPaths(paths.artifact_dir)
    output.root.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output.all_files() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"artifact files already exist; validate them or pass --force: {existing}")

    manifest_sha = sha256_file(paths.joint_manifest)
    manifest = _validate_manifest(pd.read_parquet(paths.joint_manifest))
    mapping = build_episode_mapping(manifest, paths.lerobot_root, manifest_sha)
    geometry = build_geometry_index(
        paths.geometry_indices,
        target_scope=paths.target_scope,
    )
    motion = build_motion_index(
        paths.motion_indices,
        target_scope=paths.target_scope,
    )

    manifest_identity = manifest[["lerobot_dataset_index", "sample_id"]]
    if not geometry[["lerobot_dataset_index", "sample_id"]].equals(manifest_identity):
        raise ValueError("Geometry and policy manifest frame identities differ")
    manifest_motion_ids = set(manifest.loc[manifest["motion_valid"].astype(bool), "sample_id"].astype(str))
    if set(motion["sample_id"].astype(str)) != manifest_motion_ids:
        raise ValueError("Motion and policy manifest valid identities differ")

    motion_source = motion
    motion = materialize_motion_memmap(motion_source, output.motion_targets)
    motion_storage = verify_motion_memmap(motion_source, output.motion_targets)

    geometry_normalization = pool_geometry_normalizations(paths.geometry_normalizations)
    motion_normalization = pool_motion_normalizations(paths.motion_normalizations)
    if geometry_normalization["sample_count"] != FOUR_SUITE_GEOMETRY_VALID:
        raise ValueError("pooled Geometry normalization count differs")
    if motion_normalization["count"] != FOUR_SUITE_MOTION_VALID:
        raise ValueError("pooled Motion normalization count differs")

    manifest[["lerobot_dataset_index", "sample_id", "semantic_subtask"]].to_parquet(
        output.policy_manifest,
        index=False,
    )
    _write_json(output.episode_mapping, mapping)
    geometry.to_parquet(output.geometry_index, index=False)
    _write_json(output.geometry_normalization, geometry_normalization)
    motion.to_parquet(output.motion_index, index=False)
    _write_json(output.motion_normalization, motion_normalization)
    source_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in (
            paths.joint_manifest,
            *paths.geometry_indices,
            *paths.geometry_normalizations,
            *paths.motion_indices,
            *paths.motion_normalizations,
        )
    }
    provenance = {
        "schema": "four_suite_joint_artifact_provenance_v2",
        "status": "PASS",
        "target_scope": paths.target_scope,
        "hf_repo_id": LIBERO_REPO_ID,
        "hf_revision": LIBERO_REVISION,
        "episode_count": FOUR_SUITE_EPISODES,
        "frame_count": FOUR_SUITE_FRAMES,
        "task_count": FOUR_SUITE_TASKS,
        "geometry_valid_count": FOUR_SUITE_GEOMETRY_VALID,
        "motion_valid_count": FOUR_SUITE_MOTION_VALID,
        "motion_storage": motion_storage,
        "source_sha256": source_hashes,
        "artifact_sha256": {path.name: sha256_file(path) for path in output.all_files() if path != output.provenance},
    }
    _write_json(output.provenance, provenance)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--target-scope",
        choices=("task_relevant", "whole_scene"),
        default="task_relevant",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = SourcePaths.defaults(args.artifact_dir, target_scope=args.target_scope)
    output = prepare(paths, force=args.force)
    print(json.dumps({"status": "PASS", "artifact_dir": str(output.root)}, indent=2))


if __name__ == "__main__":
    main()
