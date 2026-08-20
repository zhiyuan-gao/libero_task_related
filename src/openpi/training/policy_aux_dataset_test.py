from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training import config as _config
from openpi.training import policy_aux_dataset as _policy_aux_dataset
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig
from openpi.training.policy_aux_dataset import PolicySemanticTokenizer


def test_fixed_semantic_teacher_uses_native_lm_shift_and_eos() -> None:
    tokenizer = PolicySemanticTokenizer(max_target_len=16)
    encoded = tokenizer.encode_target("pick up the white mug")
    fixed = tokenizer.fixed("pick up the white mug")

    assert fixed.labels.shape == (16,)
    assert fixed.input_ids.shape == (15,)
    assert fixed.loss_mask.shape == (16,)
    assert fixed.labels[: len(encoded)].tolist() == encoded
    assert fixed.input_ids[: len(encoded) - 1].tolist() == encoded[:-1]
    assert encoded[-1] == 1
    assert int(fixed.loss_mask.sum()) == len(encoded)
    assert np.all(fixed.labels[len(encoded) :] == 0)


def test_p2_allows_parameterized_lambdas_but_approved_config_requires_values() -> None:
    common = {
        "mode": "ground_geometry_semantic_lm",
        "policy_manifest_path": "manifest.parquet",
        "episode_mapping_path": "mapping.json",
        "geometry_target_index_path": "target_index.parquet",
        "geometry_normalization_path": "train_mean_std.json",
        "lambda_geo": 1.0,
    }
    template = PolicyAuxTrainConfig(**common)
    assert template.loss_coefficients_approved is False
    assert template.lambda_sem is None
    assert template.lambda_ground is None

    with pytest.raises(ValueError, match="required loss coefficient"):
        PolicyAuxTrainConfig(**common, loss_coefficients_approved=True)

    config = PolicyAuxTrainConfig(
        **common,
        lambda_sem=0.2,
        lambda_ground=0.3,
        loss_coefficients_approved=True,
    )
    assert config.num_ground_queries == 8
    assert config.num_geometry_queries == 8


def test_p1_approved_config_requires_geometry_lambda() -> None:
    common = {
        "mode": "geometry",
        "policy_manifest_path": "manifest.parquet",
        "episode_mapping_path": "mapping.json",
        "geometry_target_index_path": "target_index.parquet",
        "geometry_normalization_path": "train_mean_std.json",
    }
    template = PolicyAuxTrainConfig(**common)
    assert template.lambda_geo is None
    with pytest.raises(ValueError, match="required loss coefficient"):
        PolicyAuxTrainConfig(**common, loss_coefficients_approved=True)
    approved = PolicyAuxTrainConfig(
        **common,
        lambda_geo=0.4,
        loss_coefficients_approved=True,
    )
    assert approved.lambda_geo == 0.4
    with pytest.raises(ValueError, match="strictly positive"):
        PolicyAuxTrainConfig(
            **common,
            lambda_geo=0.0,
            loss_coefficients_approved=True,
        )


def test_semantic_geometry_requires_semantic_lambda_and_zero_ground_queries() -> None:
    common = {
        "mode": "semantic_geometry",
        "policy_manifest_path": "manifest.parquet",
        "episode_mapping_path": "mapping.json",
        "geometry_target_index_path": "target_index.parquet",
        "geometry_normalization_path": "train_mean_std.json",
        "lambda_geo": 0.15,
        "num_ground_queries": 0,
    }
    with pytest.raises(ValueError, match="required loss coefficient"):
        PolicyAuxTrainConfig(**common, loss_coefficients_approved=True)
    config = PolicyAuxTrainConfig(
        **common,
        lambda_sem=0.01,
        loss_coefficients_approved=True,
    )
    assert config.lambda_ground is None
    assert config.num_geometry_queries == 8


def test_semantic_geometry_target_join_never_loads_ground_masks() -> None:
    class FakeAnnotations:
        def row(self, episode_index: int, frame_index: int):
            assert (episode_index, frame_index) == (7, 3)
            return {"sample_id": "sample-7-3", "semantic_subtask": "pick up the white mug"}

        def load_upright_ground_masks(self, _row):
            raise AssertionError("Semantic+Geometry must not load Ground masks")

    class FakeGeometry:
        mean = np.zeros((2048,), dtype=np.float32)
        std = np.ones((2048,), dtype=np.float32)

        def target_by_dataset_index(self, dataset_index: int):
            assert dataset_index == 11
            return np.ones((2048,), dtype=np.float32), True, "sample-7-3"

    target_index = _policy_aux_dataset.PolicyAuxTargetIndex.__new__(
        _policy_aux_dataset.PolicyAuxTargetIndex
    )
    target_index.config = SimpleNamespace(mode="semantic_geometry")
    target_index.annotations = FakeAnnotations()
    target_index.geometry = FakeGeometry()
    target_index.semantic_tokenizer = PolicySemanticTokenizer(max_target_len=16)
    target_index._dataset_identity = {11: (7, 3)}  # noqa: SLF001

    item = target_index.item(11)
    assert "ground_masks" not in item
    assert "ground_valid_views" not in item
    assert set(item) == {
        "geometry",
        "geometry_valid",
        "geometry_mean",
        "geometry_std",
        "semantic_input_ids",
        "semantic_labels",
        "semantic_loss_mask",
    }


def test_primary_train_configs_reject_lambda_or_architecture_override() -> None:
    p1 = _config.get_config("pi05_libero_p1_aux")
    p2 = _config.get_config("pi05_libero_p2_aux")

    assert p1.policy_aux.lambda_geo == 0.15
    assert p2.policy_aux.lambda_geo == 0.15
    assert p2.policy_aux.lambda_ground == 0.50
    assert p2.policy_aux.lambda_sem == 0.01

    with pytest.raises(ValueError, match="architecture/lambdas are frozen"):
        dataclasses.replace(
            p1,
            policy_aux=dataclasses.replace(p1.policy_aux, lambda_geo=0.16),
        )
    with pytest.raises(ValueError, match="architecture/lambdas are frozen"):
        dataclasses.replace(
            p2,
            policy_aux=dataclasses.replace(p2.policy_aux, lambda_ground=0.49),
        )


def test_libero3_pilot_configs_are_frozen_and_matched() -> None:
    p1 = _config.get_config("pi05_libero3_p1_aux")
    p2 = _config.get_config("pi05_libero3_p2_aux")
    semantic_geometry = _config.get_config("pi05_libero3_semantic_geometry_aux")

    assert p1.policy_aux.lerobot_task_indices == (0, 3, 8)
    assert p2.policy_aux.lerobot_task_indices == (0, 3, 8)
    assert p1.num_train_steps == p2.num_train_steps == 3_209
    assert p1.lr_schedule.warmup_steps == p2.lr_schedule.warmup_steps == 1_069
    assert p1.batch_size == p2.batch_size == 256
    assert p1.gradient_accumulation_steps == p2.gradient_accumulation_steps == 1
    assert p1.pytorch_weight_path == p2.pytorch_weight_path
    assert semantic_geometry.policy_aux.mode == "semantic_geometry"
    assert semantic_geometry.policy_aux.lerobot_task_indices == (0, 3, 8)
    assert semantic_geometry.policy_aux.num_ground_queries == 0
    assert semantic_geometry.policy_aux.lambda_geo == 0.15
    assert semantic_geometry.policy_aux.lambda_sem == 0.01
    assert semantic_geometry.policy_aux.lambda_ground is None
    assert semantic_geometry.ema_decay is None
    assert semantic_geometry.num_train_steps == 3_209
    assert semantic_geometry.lr_schedule.warmup_steps == 1_069
    assert semantic_geometry.batch_size == 256
    assert semantic_geometry.gradient_accumulation_steps == 1
    assert semantic_geometry.pytorch_weight_path == p1.pytorch_weight_path

    with pytest.raises(ValueError, match="only approved reduced pilot population"):
        dataclasses.replace(p1.policy_aux, lerobot_task_indices=(0, 1, 2))


def test_libero3_episode_and_global_frame_selection(tmp_path) -> None:
    def lengths(count: int, total: int) -> list[int]:
        values = [total // count] * count
        for index in range(total % count):
            values[index] += 1
        return values

    selected = (
        [(0, length) for length in lengths(38, 9_807)]
        + [(3, length) for length in lengths(41, 10_866)]
        + [(8, length) for length in lengths(35, 8_577)]
    )
    remaining = [(1, length) for length in lengths(265, 72_219)]
    population = []
    while selected or remaining:
        if selected:
            population.append(selected.pop(0))
        if remaining:
            population.append(remaining.pop(0))

    records = []
    dataset_index = 0
    for episode_index, (task_index, episode_length) in enumerate(population):
        records.append(
            {
                "lerobot_episode_index": episode_index,
                "lerobot_task_index": task_index,
                "episode_length": episode_length,
                "dataset_from_index": dataset_index,
                "dataset_to_index_exclusive": dataset_index + episode_length,
            }
        )
        dataset_index += episode_length
    mapping = {
        "status": "PASS",
        "hf_repo_id": "physical-intelligence/libero",
        "hf_revision": _policy_aux_dataset.CANONICAL_LIBERO_REVISION,
        "mapped_episode_count": 379,
        "mapped_frame_count": 101_469,
        "episodes": records,
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping))
    config = PolicyAuxTrainConfig(
        mode="geometry",
        policy_manifest_path="manifest.parquet",
        episode_mapping_path=str(mapping_path),
        geometry_target_index_path="target_index.parquet",
        geometry_normalization_path="train_mean_std.json",
        lambda_geo=0.15,
        lerobot_task_indices=(0, 3, 8),
    )

    selected_records = [row for row in records if row["lerobot_task_index"] in {0, 3, 8}]
    expected_episodes = [row["lerobot_episode_index"] for row in selected_records]
    expected_frames = [
        frame
        for row in selected_records
        for frame in range(row["dataset_from_index"], row["dataset_to_index_exclusive"])
    ]
    assert config.lerobot_episode_indices() == expected_episodes
    assert config.lerobot_dataset_indices() == expected_frames
    assert len(expected_episodes) == 114
    assert len(expected_frames) == 29_250
