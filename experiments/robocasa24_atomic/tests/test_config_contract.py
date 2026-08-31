from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from robocasa24_finetune.configs import _validate_fp32_base
from robocasa24_finetune.configs import build_train_config
from robocasa24_finetune.constants import DATASET_REPO_ID
from robocasa24_finetune.integration import RoboCasaRuntimeDataConfig
from robocasa24_finetune.train import _configure_performance_environment
from safetensors.torch import save_file
import torch

from openpi.shared import normalize


def _base(path: Path, dtype: torch.dtype) -> Path:
    path.mkdir()
    save_file({"test": torch.ones(1, dtype=dtype)}, path / "model.safetensors")
    return path


def test_base_checkpoint_must_be_fp32(tmp_path: Path) -> None:
    _validate_fp32_base(_base(tmp_path / "fp32", torch.float32))
    with pytest.raises(ValueError, match="FP32-converted"):
        _validate_fp32_base(_base(tmp_path / "bf16", torch.bfloat16))


def test_performance_environment_matches_validated_libero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENPI_USE_DEFAULT_CUDA_ALLOCATOR",
        "OPENPI_LOG_MEMORY_STATS",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(name, raising=False)
    _configure_performance_environment()
    assert os.environ["OPENPI_USE_DEFAULT_CUDA_ALLOCATOR"] == "1"
    assert os.environ["OPENPI_LOG_MEMORY_STATS"] == "0"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_formal_config_contract(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manifests = tmp_path / "manifests"
    assets = tmp_path / "assets"
    checkpoints = tmp_path / "checkpoints"
    for path in (data, manifests, assets, checkpoints):
        path.mkdir()
    normalize.save(
        assets / DATASET_REPO_ID,
        {
            "state": normalize.NormStats(
                mean=np.zeros(16),
                std=np.ones(16),
                q01=-np.ones(16),
                q99=np.ones(16),
            ),
            "actions": normalize.NormStats(
                mean=np.zeros(12),
                std=np.ones(12),
                q01=-np.ones(12),
                q99=np.ones(12),
            ),
        },
    )
    config = build_train_config(
        variant="baseline",
        exp_name="contract",
        data_root=data,
        manifest_root=manifests,
        policy_assets_root=assets,
        artifact_dir=None,
        base_weight_dir=_base(tmp_path / "base", torch.float32),
        checkpoint_base_dir=checkpoints,
        wandb_enabled=False,
    )
    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.model.action_dim == 32
    assert config.batch_size * config.gradient_accumulation_steps == 128
    assert config.num_train_steps == 30_000
    assert config.lr_schedule.warmup_steps == 10_000
    assert config.ema_decay is None
    assert config.fsdp_devices == 1
    assert config.max_checkpoints_to_keep == 4
    assert config.policy_metadata["execution_horizon"] == 25
    runtime = config.data.create(config.assets_dirs, config.model)
    assert isinstance(runtime, RoboCasaRuntimeDataConfig)
    assert runtime.repo_id == DATASET_REPO_ID
    # Match the official RoboCasa OpenPI data config. This must stay explicit
    # because the shared local checkout uses quantiles for LIBERO pi0.5.
    assert runtime.use_quantile_norm is False
    assert runtime.norm_stats is not None
