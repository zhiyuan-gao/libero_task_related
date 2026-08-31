"""Assemble the exact 1,808-episode supplemental-115 training population.

The immutable Hugging Face completion bundle contains 239 additive episodes
and portable targets for the final 1,932-episode population.  The adopted
97.85% checkpoint used only the first 115 additions.  This module selects that
prefix, combines it with the public 1,693-episode LeRobot snapshot, and writes
an exact portable dataset and exact trimmed Geometry/Motion target stores.

Neither source tree is modified.  The output is built atomically and existing
output paths are never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .constants import AUGMENTED_FRAMES
from .constants import AUGMENTED_GEOMETRY_INVALID
from .constants import AUGMENTED_GEOMETRY_VALID
from .constants import AUGMENTED_MOTION_VALID
from .constants import COMPLETED_EPISODES
from .constants import COMPLETED_FRAMES
from .constants import COMPLETED_GEOMETRY_VALID
from .constants import COMPLETED_MOTION_VALID
from .constants import FOUR_SUITE_EPISODES
from .constants import FOUR_SUITE_FRAMES
from .constants import FOUR_SUITE_GEOMETRY_VALID
from .constants import FOUR_SUITE_MOTION_VALID
from .constants import GEOMETRY_DIM
from .constants import LIBERO_REPO_ID
from .constants import LIBERO_REVISION
from .constants import MOTION_DIM
from .constants import SUPPLEMENTAL_EPISODES

ASSET_REPO_ID = "Zhiyuan17/libero40-trqc-assets"
ASSET_BUNDLE_TAG = "official-completion-1932-v1"
ASSET_BUNDLE_COMMIT = "476ed61df46b58aa14b363e4be20b6152581791f"


@dataclass(frozen=True)
class PopulationSpec:
    revision: str
    base_episodes: int
    base_frames: int
    supplemental_episodes: int
    supplemental_frames: int
    source_episodes: int
    source_frames: int
    base_geometry_valid: int
    base_motion_valid: int
    target_geometry_valid: int
    target_geometry_invalid: int
    target_motion_valid: int
    source_geometry_valid: int
    source_motion_valid: int
    geometry_dim: int
    motion_dim: int

    @property
    def target_episodes(self) -> int:
        return self.base_episodes + self.supplemental_episodes

    @property
    def target_frames(self) -> int:
        return self.base_frames + self.supplemental_frames


FROZEN_SPEC = PopulationSpec(
    revision=LIBERO_REVISION,
    base_episodes=FOUR_SUITE_EPISODES,
    base_frames=FOUR_SUITE_FRAMES,
    supplemental_episodes=SUPPLEMENTAL_EPISODES,
    supplemental_frames=AUGMENTED_FRAMES - FOUR_SUITE_FRAMES,
    source_episodes=COMPLETED_EPISODES,
    source_frames=COMPLETED_FRAMES,
    base_geometry_valid=FOUR_SUITE_GEOMETRY_VALID,
    base_motion_valid=FOUR_SUITE_MOTION_VALID,
    target_geometry_valid=AUGMENTED_GEOMETRY_VALID,
    target_geometry_invalid=AUGMENTED_GEOMETRY_INVALID,
    target_motion_valid=AUGMENTED_MOTION_VALID,
    source_geometry_valid=COMPLETED_GEOMETRY_VALID,
    source_motion_valid=COMPLETED_MOTION_VALID,
    geometry_dim=GEOMETRY_DIM,
    motion_dim=MOTION_DIM,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = source.resolve(strict=True)
    try:
        os.link(resolved, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(resolved, destination)


def _episode_files(data_root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(data_root.rglob("episode_*.parquet")):
        try:
            episode_index = int(path.stem.removeprefix("episode_"))
        except ValueError as error:
            raise ValueError(f"invalid LeRobot episode filename: {path}") from error
        if episode_index in result:
            raise ValueError(f"duplicate LeRobot episode index {episode_index}")
        result[episode_index] = path
    return result


def _validate_episode_rows(rows: list[dict[str, Any]], *, episodes: int, frames: int) -> None:
    if len(rows) != episodes:
        raise ValueError(f"episode metadata count differs: expected {episodes}, observed {len(rows)}")
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(episodes)):
        raise ValueError("episode metadata is not contiguous")
    if sum(int(row["length"]) for row in rows) != frames:
        raise ValueError("episode metadata frame total differs")


def _copy_npy_prefix(
    source: Path,
    destination: Path,
    *,
    source_rows: int,
    target_rows: int,
    feature_dim: int,
) -> None:
    source_array = np.load(source.resolve(strict=True), mmap_mode="r")
    if source_array.shape != (source_rows, feature_dim) or source_array.dtype != np.float32:
        raise ValueError(f"unexpected source target store: {source} {source_array.shape} {source_array.dtype}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(target_rows, feature_dim),
    )
    chunk_rows = max(1, min(4_096, 64 * 1024 * 1024 // (feature_dim * 4)))
    for start in range(0, target_rows, chunk_rows):
        stop = min(target_rows, start + chunk_rows)
        block = np.asarray(source_array[start:stop], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"non-finite target values in {source} rows {start}:{stop}")
        target[start:stop] = block
    target.flush()
    del target


def _assemble_lerobot(
    *,
    base: Path,
    bundle: Path,
    output: Path,
    spec: PopulationSpec,
) -> None:
    base_info = _read_json(base / "meta/info.json")
    source_meta = bundle / "dataset_delta/meta"
    source_info = _read_json(source_meta / "info.json")
    if (
        int(base_info.get("total_episodes", -1)) != spec.base_episodes
        or int(base_info.get("total_frames", -1)) != spec.base_frames
    ):
        raise ValueError("public base is not the frozen base population")
    if (
        int(source_info.get("total_episodes", -1)) != spec.source_episodes
        or int(source_info.get("total_frames", -1)) != spec.source_frames
    ):
        raise ValueError("completion bundle is not the frozen source population")

    base_rows = _read_jsonl(base / "meta/episodes.jsonl")
    source_rows = _read_jsonl(source_meta / "episodes.jsonl")
    _validate_episode_rows(base_rows, episodes=spec.base_episodes, frames=spec.base_frames)
    _validate_episode_rows(source_rows, episodes=spec.source_episodes, frames=spec.source_frames)
    if source_rows[: spec.base_episodes] != base_rows:
        raise ValueError("completion metadata does not preserve the public base prefix")
    selected_rows = source_rows[: spec.target_episodes]
    _validate_episode_rows(selected_rows, episodes=spec.target_episodes, frames=spec.target_frames)

    base_files = _episode_files(base / "data")
    delta_files = _episode_files(bundle / "dataset_delta/data")
    if set(base_files) != set(range(spec.base_episodes)):
        raise ValueError("public base parquet population differs")
    expected_delta = set(range(spec.base_episodes, spec.source_episodes))
    if set(delta_files) != expected_delta:
        raise ValueError("completion bundle parquet population differs")

    for episode_index in range(spec.base_episodes):
        source = base_files[episode_index]
        _link_or_copy(source, output / "data" / source.relative_to(base / "data"))
    for episode_index in range(spec.base_episodes, spec.target_episodes):
        source = delta_files[episode_index]
        _link_or_copy(
            source,
            output / "data" / source.relative_to(bundle / "dataset_delta/data"),
        )

    shutil.copytree(base / "meta", output / "meta")
    _write_jsonl(output / "meta/episodes.jsonl", selected_rows)
    target_info = dict(base_info)
    chunk_size = int(target_info["chunks_size"])
    target_info.update(
        {
            "total_episodes": spec.target_episodes,
            "total_frames": spec.target_frames,
            "total_chunks": (spec.target_episodes + chunk_size - 1) // chunk_size,
            "splits": {"train": f"0:{spec.target_episodes}"},
        }
    )
    _write_json(output / "meta/info.json", target_info)


def _assemble_artifacts(
    *,
    bundle: Path,
    temporary_output: Path,
    spec: PopulationSpec,
) -> None:
    source = bundle / "training_artifacts/task_relevant"
    policy = pd.read_parquet(source / "policy_manifest.parquet").sort_values("lerobot_dataset_index")
    if (
        len(policy) != spec.source_frames
        or not policy["sample_id"].is_unique
        or policy["lerobot_dataset_index"].astype(int).tolist() != list(range(spec.source_frames))
    ):
        raise ValueError("completion policy manifest differs")
    selected_policy = policy.iloc[: spec.target_frames].copy()
    selected_policy.to_parquet(temporary_output / "policy_manifest.parquet", index=False, compression="zstd")
    selected_sample_ids = set(selected_policy["sample_id"].astype(str))

    mapping = _read_json(source / "episode_mapping.json")
    records = sorted(mapping.get("episodes", []), key=lambda row: int(row["lerobot_episode_index"]))
    if (
        mapping.get("status") != "PASS"
        or mapping.get("hf_repo_id") != LIBERO_REPO_ID
        or mapping.get("hf_revision") != spec.revision
        or len(records) != spec.source_episodes
    ):
        raise ValueError("completion episode mapping differs")
    selected_records = records[: spec.target_episodes]
    if [int(row["lerobot_episode_index"]) for row in selected_records] != list(range(spec.target_episodes)) or sum(
        int(row["episode_length"]) for row in selected_records
    ) != spec.target_frames:
        raise ValueError("supplemental-115 episode mapping differs")
    target_mapping = {
        "status": "PASS",
        "schema_version": "libero_four_suite_supplemental115_mapping.v1",
        "hf_repo_id": LIBERO_REPO_ID,
        "hf_revision": spec.revision,
        "mapped_episode_count": spec.target_episodes,
        "mapped_frame_count": spec.target_frames,
        "base_episode_count": spec.base_episodes,
        "supplemental_episode_count": spec.supplemental_episodes,
        "episodes": selected_records,
    }
    _write_json(temporary_output / "episode_mapping.json", target_mapping)

    geometry = pd.read_parquet(source / "geometry_index.parquet").sort_values("lerobot_dataset_index")
    valid = geometry["geometry_valid"].astype(bool)
    if (
        len(geometry) != spec.source_frames
        or not geometry["sample_id"].is_unique
        or int(valid.sum()) != spec.source_geometry_valid
        or geometry["lerobot_dataset_index"].astype(int).tolist() != list(range(spec.source_frames))
        or geometry["sample_id"].astype(str).tolist() != policy["sample_id"].astype(str).tolist()
    ):
        raise ValueError("completion Geometry index differs")
    source_geometry_rows = geometry.loc[valid, "target_memmap_row"].astype(np.int64).to_numpy()
    if not np.array_equal(source_geometry_rows, np.arange(spec.source_geometry_valid)):
        raise ValueError("completion Geometry rows are not the frozen contiguous order")
    selected_geometry = geometry.iloc[: spec.target_frames].copy()
    selected_valid = selected_geometry["geometry_valid"].astype(bool)
    if (
        int(selected_valid.sum()) != spec.target_geometry_valid
        or int((~selected_valid).sum()) != spec.target_geometry_invalid
    ):
        raise ValueError("supplemental-115 Geometry validity population differs")
    selected_rows = selected_geometry.loc[selected_valid, "target_memmap_row"].astype(np.int64).to_numpy()
    if not np.array_equal(selected_rows, np.arange(spec.target_geometry_valid)):
        raise ValueError("supplemental-115 Geometry targets are not a source prefix")
    selected_geometry.loc[selected_valid, "target_memmap_path"] = "geometry_targets_fp32.npy"
    selected_geometry.to_parquet(temporary_output / "geometry_index.parquet", index=False, compression="zstd")
    _copy_npy_prefix(
        source / "geometry_targets_fp32.npy",
        temporary_output / "geometry_targets_fp32.npy",
        source_rows=spec.source_geometry_valid,
        target_rows=spec.target_geometry_valid,
        feature_dim=spec.geometry_dim,
    )
    geometry_norm = _read_json(source / "geometry_normalization.json")
    geometry_norm["sample_count"] = spec.target_geometry_valid
    geometry_norm.setdefault("statistics_source_sample_count", spec.base_geometry_valid)
    geometry_norm["statistics_policy"] = "frozen from the step-30000 base population"
    _write_json(temporary_output / "geometry_normalization.json", geometry_norm)

    motion = pd.read_parquet(source / "motion_index.parquet")
    if len(motion) != spec.source_motion_valid or not motion["sample_id"].is_unique:
        raise ValueError("completion Motion index differs")
    source_motion_rows = motion["target_memmap_row"].astype(np.int64).to_numpy()
    if not np.array_equal(np.sort(source_motion_rows), np.arange(spec.source_motion_valid)):
        raise ValueError("completion Motion rows are not a permutation of the source population")
    selected_motion = motion.loc[motion["sample_id"].astype(str).isin(selected_sample_ids)].copy()
    selected_motion = selected_motion.sort_values("target_memmap_row")
    selected_motion_rows = selected_motion["target_memmap_row"].astype(np.int64).to_numpy()
    if len(selected_motion) != spec.target_motion_valid or not np.array_equal(
        selected_motion_rows, np.arange(spec.target_motion_valid)
    ):
        raise ValueError("supplemental-115 Motion targets are not the exact source prefix")
    selected_motion["target_memmap_path"] = "motion_targets_fp32.npy"
    selected_motion.to_parquet(temporary_output / "motion_index.parquet", index=False, compression="zstd")
    _copy_npy_prefix(
        source / "motion_targets_fp32.npy",
        temporary_output / "motion_targets_fp32.npy",
        source_rows=spec.source_motion_valid,
        target_rows=spec.target_motion_valid,
        feature_dim=spec.motion_dim,
    )
    motion_norm = _read_json(source / "motion_normalization.json")
    motion_norm["count"] = spec.target_motion_valid
    motion_norm.setdefault("statistics_source_count", spec.base_motion_valid)
    motion_norm["statistics_policy"] = "frozen from the step-30000 base population"
    _write_json(temporary_output / "motion_normalization.json", motion_norm)

    provenance = {
        "status": "PASS",
        "schema_version": "libero_four_suite_supplemental115_portable_artifacts.v1",
        "source_asset_repo": ASSET_REPO_ID,
        "source_asset_tag": ASSET_BUNDLE_TAG,
        "source_asset_commit": ASSET_BUNDLE_COMMIT,
        "selection": {
            "episode_indices": [spec.base_episodes, spec.target_episodes - 1],
            "rule": "first 115 additions from official-completion-1932-v1",
        },
        "population": {
            "base_episodes": spec.base_episodes,
            "supplemental_episodes": spec.supplemental_episodes,
            "episodes": spec.target_episodes,
            "frames": spec.target_frames,
            "geometry_valid": spec.target_geometry_valid,
            "motion_valid": spec.target_motion_valid,
        },
        "normalization": "frozen from the step-30000 base training population",
        "supplemental_observation_recipe": (
            "official HDF5 same-frame RGB/state/action; RGB rotate180 then bicubic 128->256"
        ),
        "original_population_modified": False,
    }
    _write_json(temporary_output / "provenance.json", provenance)


def _validate_output(*, root: Path, spec: PopulationSpec) -> dict[str, Any]:
    lerobot = root / "lerobot" / spec.revision
    artifacts = root / "artifacts/task_relevant"
    info = _read_json(lerobot / "meta/info.json")
    episodes = _read_jsonl(lerobot / "meta/episodes.jsonl")
    policy = pd.read_parquet(artifacts / "policy_manifest.parquet")
    geometry = pd.read_parquet(artifacts / "geometry_index.parquet")
    motion = pd.read_parquet(artifacts / "motion_index.parquet")
    geometry_targets = np.load(artifacts / "geometry_targets_fp32.npy", mmap_mode="r")
    motion_targets = np.load(artifacts / "motion_targets_fp32.npy", mmap_mode="r")
    checks = {
        "info_population": (
            int(info.get("total_episodes", -1)) == spec.target_episodes
            and int(info.get("total_frames", -1)) == spec.target_frames
        ),
        "episode_metadata_population": len(episodes) == spec.target_episodes,
        "episode_parquet_population": len(_episode_files(lerobot / "data")) == spec.target_episodes,
        "policy_population": len(policy) == spec.target_frames,
        "geometry_population": len(geometry) == spec.target_frames,
        "geometry_valid_population": int(geometry["geometry_valid"].astype(bool).sum()) == spec.target_geometry_valid,
        "geometry_target_shape": geometry_targets.shape == (spec.target_geometry_valid, spec.geometry_dim),
        "motion_population": len(motion) == spec.target_motion_valid,
        "motion_target_shape": motion_targets.shape == (spec.target_motion_valid, spec.motion_dim),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "lerobot_root": str(lerobot),
        "artifact_root": str(artifacts),
        "episodes": spec.target_episodes,
        "frames": spec.target_frames,
        "geometry_valid": spec.target_geometry_valid,
        "motion_valid": spec.target_motion_valid,
        "source_asset_repo": ASSET_REPO_ID,
        "source_asset_commit": ASSET_BUNDLE_COMMIT,
    }
    if report["status"] != "PASS":
        raise ValueError(f"supplemental-115 assembly failed: {checks}")
    return report


def assemble_supplemental115(
    *,
    base_root: Path,
    bundle_root: Path,
    output_root: Path,
    spec: PopulationSpec = FROZEN_SPEC,
) -> dict[str, Any]:
    base = base_root.resolve(strict=True)
    bundle = bundle_root.resolve(strict=True)
    output = output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if spec is FROZEN_SPEC and base.name != spec.revision:
        raise ValueError(f"base root must end with frozen revision {spec.revision}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        temporary_lerobot = temporary / "lerobot" / spec.revision
        temporary_artifacts = temporary / "artifacts/task_relevant"
        temporary_lerobot.mkdir(parents=True)
        temporary_artifacts.mkdir(parents=True)
        _assemble_lerobot(base=base, bundle=bundle, output=temporary_lerobot, spec=spec)
        _assemble_artifacts(
            bundle=bundle,
            temporary_output=temporary_artifacts,
            spec=spec,
        )
        report = _validate_output(root=temporary, spec=spec)
        report["lerobot_root"] = str(output / "lerobot" / spec.revision)
        report["artifact_root"] = str(output / "artifacts/task_relevant")
        _write_json(temporary_artifacts / "validation.json", report)
        _write_json(temporary / "assembly_report.json", report)
        os.replace(temporary, output)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help="Downloaded official_completion_1932_v1 directory from the frozen HF asset tag.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory that will receive lerobot/<revision> and artifacts/task_relevant.",
    )
    args = parser.parse_args()
    report = assemble_supplemental115(
        base_root=args.base_root,
        bundle_root=args.bundle_root,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
