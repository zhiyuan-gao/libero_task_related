from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpi.training import config as _config
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
