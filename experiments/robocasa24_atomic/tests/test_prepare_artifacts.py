from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from robocasa24_finetune.auxiliary import PreparedAuxiliaryPaths
from robocasa24_finetune.auxiliary import PreparedAuxiliaryStore
from robocasa24_finetune.auxiliary import RoboCasaPolicyAuxTransformedDataset
from robocasa24_finetune.auxiliary import require_prepared_target_scope
from robocasa24_finetune.configs import _aux_config
from robocasa24_finetune.constants import GEOMETRY_DIM
from robocasa24_finetune.constants import MOTION_DIM
from robocasa24_finetune.constants import TASKS
from robocasa24_finetune.prepare_artifacts import prepare


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_prepare_aligns_sparse_targets_and_zeroes_invalid_rows(tmp_path: Path) -> None:
    task = TASKS[0]
    roots = {
        name: _mkdir(tmp_path / name)
        for name in ("manifest", "semantic", "geometry", "motion")
    }
    sample_ids = [f"{task}/base50__demo_0/frame_{i:06d}" for i in range(3)]
    source = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "task": task,
            "task_frame_index": [0, 1, 2],
            "source_role": "base50",
            "review_accepted": [True, True, True],
            "geometry_valid": [True, False, True],
            "motion_valid": [True, False, True],
        }
    )
    source_path = roots["manifest"] / task / "source/source_manifest.parquet"
    _mkdir(source_path.parent)
    source.to_parquet(source_path, index=False)

    semantic_dir = _mkdir(roots["semantic"] / task / "semantic")
    semantic_rel = f"{task}/semantic/semantic_targets.npz"
    semantic_index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "task_frame_index": [0, 1, 2],
            "target_path": semantic_rel,
            "target_row": [0, 1, 2],
        }
    )
    semantic_index.to_parquet(semantic_dir / "index.parquet", index=False)
    np.savez_compressed(
        semantic_dir / "semantic_targets.npz",
        sample_id=np.asarray(sample_ids),
        semantic_input_ids=np.ones((3, 31), dtype=np.int64),
        semantic_labels=np.ones((3, 32), dtype=np.int64),
        semantic_loss_mask=np.ones((3, 32), dtype=bool),
        valid=np.ones(3, dtype=bool),
        review=np.zeros(3, dtype=bool),
    )

    geometry_final = _mkdir(roots["geometry"] / task / "geometry/final")
    geometry_shard = _mkdir(geometry_final / "shards") / "geometry.npz"
    geometry_values = np.stack(
        [
            np.ones(GEOMETRY_DIM, dtype=np.float32),
            np.full(GEOMETRY_DIM, 3, dtype=np.float32),
        ]
    )
    np.savez(
        geometry_shard,
        sample_id=np.asarray([sample_ids[0], sample_ids[2]]),
        geometry_target_fp32=geometry_values,
    )
    stale_geometry_path = (
        f"/relocated/machine/{task}/geometry/final/shards/geometry.npz"
    )
    pd.DataFrame(
        {
            "sample_id": [sample_ids[0], sample_ids[2]],
            "task_frame_index": [0, 2],
            "geometry_available": [True, True],
            "target_shard_path": [stale_geometry_path, stale_geometry_path],
            "target_shard_row": [0, 1],
            "target_dim": [GEOMETRY_DIM] * 2,
            "target_dtype": ["float32"] * 2,
        }
    ).to_parquet(geometry_final / "index.parquet", index=False)

    motion_final = _mkdir(roots["motion"] / task / "motion/final")
    motion_shard = _mkdir(motion_final / "shards") / "motion.npz"
    motion_values = np.stack(
        [
            np.full(MOTION_DIM, 2, dtype=np.float32),
            np.full(MOTION_DIM, 4, dtype=np.float32),
        ]
    )
    np.savez(
        motion_shard,
        sample_id=np.asarray([sample_ids[0], sample_ids[2]]),
        motion_target_fp32=motion_values,
    )
    stale_motion_path = f"/relocated/machine/{task}/motion/final/shards/motion.npz"
    pd.DataFrame(
        {
            "sample_id": [sample_ids[0], sample_ids[2]],
            "task_frame_index": [0, 2],
            "target_shard_path": [stale_motion_path, stale_motion_path],
            "target_shard_row": [0, 1],
            "target_dim": [MOTION_DIM] * 2,
            "target_dtype": ["float32"] * 2,
        }
    ).to_parquet(motion_final / "index.parquet", index=False)

    output = tmp_path / "prepared"
    report = prepare(
        scope="task_relevant",
        manifest_root=roots["manifest"],
        semantic_root=roots["semantic"],
        geometry_root=roots["geometry"],
        motion_root=roots["motion"],
        output_dir=output,
        tasks=(task,),
        write_checksums=False,
    )
    assert report["status"] == "PASS"
    assert report["geometry_valid_count"] == 2
    assert report["motion_valid_count"] == 2
    assert report["semantic_cache_review_true_count"] == 0
    assert report["source_review_accepted_count"] == 3
    store = PreparedAuxiliaryStore(output, tuple(sample_ids))
    np.testing.assert_array_equal(
        store.item(1)["geometry"], np.zeros(GEOMETRY_DIM, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        store.item(1)["motion"], np.zeros(MOTION_DIM, dtype=np.float32)
    )
    assert not bool(store.item(1)["geometry_valid"])
    assert not bool(store.item(1)["motion_valid"])
    require_prepared_target_scope(PreparedAuxiliaryPaths(output), "task_relevant")
    with pytest.raises(ValueError, match="target scope differs"):
        require_prepared_target_scope(PreparedAuxiliaryPaths(output), "whole_scene")
    assert (
        json.loads((output / "geometry_normalization.json").read_text())["count"] == 2
    )
    aux = _aux_config("task_relevant", output)
    assert aux is not None
    assert aux.target_scope == "task_relevant"
    assert aux.motion_target_count == 2
    assert (aux.lambda_geo, aux.lambda_sem, aux.lambda_motion) == (0.05, 0.01, 0.05)

    expected_ids = tuple(sample_ids)

    class Source:
        sample_ids = expected_ids

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, index):
            return {"row": np.asarray(index)}

        def resolve_sample_index(self, index):
            return int(index)

    class UpstreamTransformWrapper:
        def __init__(self, source):
            self._dataset = source

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, index):
            return self._dataset[index]

    joined = RoboCasaPolicyAuxTransformedDataset(
        UpstreamTransformWrapper(Source()), aux
    )
    assert joined[1]["policy_aux"]["geometry"].shape == (GEOMETRY_DIM,)

    subset_sample_ids = (sample_ids[2], sample_ids[0])

    class SubsetSource:
        # Deliberately reverse two valid rows to verify sample-ID resolution,
        # rather than accidental positional indexing into the full artifact.
        sample_ids = subset_sample_ids

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, index):
            return {"sample_id": self.sample_ids[index]}

        def resolve_sample_index(self, index):
            return int(index)

    subset = RoboCasaPolicyAuxTransformedDataset(SubsetSource(), aux)
    np.testing.assert_array_equal(
        subset[0]["policy_aux"]["geometry"], geometry_values[1]
    )
    np.testing.assert_array_equal(
        subset[1]["policy_aux"]["motion"], motion_values[0]
    )

    class RemappedSource:
        sample_ids = expected_ids
        sampled_rows = (2, 0, 1)

        def __len__(self):
            return len(self.sample_ids)

        def resolve_sample_index(self, index):
            return self.sampled_rows[index]

        def __getitem__(self, index):
            return {"raw_row": np.asarray(self.resolve_sample_index(index))}

    remapped = RoboCasaPolicyAuxTransformedDataset(RemappedSource(), aux)
    assert int(remapped[0]["raw_row"]) == 2
    np.testing.assert_array_equal(
        remapped[0]["policy_aux"]["geometry"], geometry_values[1]
    )
    np.testing.assert_array_equal(
        remapped[1]["policy_aux"]["motion"], motion_values[0]
    )
