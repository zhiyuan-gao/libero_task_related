from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from openpi.training import config as _config
from openpi.training import pytorch_ema

from . import train_pytorch


class _ResumeLoader:
    def __init__(self, token: int):
        self.token = token
        self.loaded = None

    def state_dict(self) -> dict:
        return {"token": self.token}

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


def _config_for_checkpoint(tmp_path, **changes) -> _config.TrainConfig:
    values = {
        "exp_name": "ema_roundtrip",
        "checkpoint_base_dir": str(tmp_path),
        "ema_decay": 0.9,
        "num_train_steps": 3,
        "save_interval": 3,
        "keep_period": 2,
        "overwrite": False,
        "resume": False,
        "wandb_enabled": False,
    }
    values.update(changes)
    config = dataclasses.replace(
        _config.get_config("debug"),
        **values,
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return config


def test_trajectory_config_ignores_runtime_controls_but_not_recipe(tmp_path) -> None:
    base = _config_for_checkpoint(tmp_path)
    runtime_change = dataclasses.replace(
        base,
        num_train_steps=4,
        save_interval=1000,
        save_final_checkpoint=False,
        keep_period=None,
        checkpoint_keep_steps=(1, 4),
        max_checkpoints_to_keep=8,
        max_resume_checkpoints_to_keep=2,
        log_interval=1,
        resume=True,
        wandb_enabled=True,
    )
    recipe_change = dataclasses.replace(base, seed=base.seed + 1)

    assert train_pytorch.trajectory_config(runtime_change) == train_pytorch.trajectory_config(base)
    assert train_pytorch.trajectory_config(recipe_change) != train_pytorch.trajectory_config(base)


def test_prune_checkpoints_preserves_periodic_and_latest(tmp_path) -> None:
    for step in (1000, 2000, 3000, 4000, 5000):
        (tmp_path / str(step)).mkdir()
    (tmp_path / "tmp_6000").mkdir()
    (tmp_path / "notes").mkdir()

    removed = train_pytorch.prune_checkpoints(tmp_path, keep_period=2000)

    assert removed == [1000, 3000]
    assert {path.name for path in tmp_path.iterdir()} == {"2000", "4000", "5000", "tmp_6000", "notes"}


def test_prune_checkpoints_preserves_exact_p3_milestones(tmp_path) -> None:
    for step in (500, 1000, 1500, 2000, 2500, 3000, 3209):
        (tmp_path / str(step)).mkdir()

    removed = train_pytorch.prune_checkpoints(
        tmp_path,
        keep_period=None,
        keep_steps=(500, 1000, 2000, 3209),
    )

    assert removed == [1500, 2500, 3000]
    assert {path.name for path in tmp_path.iterdir()} == {"500", "1000", "2000", "3209"}


def test_prune_checkpoints_can_keep_last_eight_for_libero10(tmp_path) -> None:
    for step in (*range(1000, 12_000, 1000), 11_132):
        (tmp_path / str(step)).mkdir()

    removed = train_pytorch.prune_checkpoints(tmp_path, keep_period=None, max_to_keep=8)

    assert removed == [1000, 2000, 3000, 4000]
    assert {int(path.name) for path in tmp_path.iterdir()} == {
        5000,
        6000,
        7000,
        8000,
        9000,
        10_000,
        11_000,
        11_132,
    }


def test_demote_old_resume_checkpoints_preserves_evaluation_weights(tmp_path) -> None:
    steps = tuple(range(1000, 30_001, 1000))
    for step in steps:
        checkpoint = tmp_path / str(step)
        checkpoint.mkdir()
        for name in (
            "model.safetensors",
            "train_model.safetensors",
            "optimizer.pt",
            "training_state.pt",
            "metadata.pt",
        ):
            (checkpoint / name).write_bytes(b"test")

    demoted = train_pytorch.demote_old_resume_checkpoints(tmp_path, max_to_keep=2)

    assert demoted == list(steps[:-2])
    assert all((tmp_path / str(step) / "model.safetensors").is_file() for step in steps)
    old = tmp_path / "1000"
    assert (old / "model.safetensors").is_file()
    assert (old / "metadata.pt").is_file()
    assert (old / "EVALUATION_ONLY").is_file()
    assert not (old / "train_model.safetensors").exists()
    assert not (old / "optimizer.pt").exists()
    assert not (old / "training_state.pt").exists()
    assert not train_pytorch.checkpoint_is_resumable(old)
    assert not any(train_pytorch.checkpoint_is_resumable(tmp_path / str(step)) for step in steps[:-2])
    assert train_pytorch.checkpoint_is_resumable(tmp_path / "29000")
    assert train_pytorch.checkpoint_is_resumable(tmp_path / "30000")


def test_save_checkpoint_keeps_all_models_and_latest_two_resume_states(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_pytorch, "log_memory_usage", lambda *args, **kwargs: None)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = _config_for_checkpoint(
        tmp_path,
        ema_decay=None,
        num_train_steps=3,
        save_interval=1,
        keep_period=1,
        max_checkpoints_to_keep=None,
        max_resume_checkpoints_to_keep=2,
    )
    for step in (1, 2, 3):
        train_pytorch.save_checkpoint(
            model,
            optimizer,
            step,
            config,
            is_main=True,
            data_config=SimpleNamespace(norm_stats=None, asset_id=None),
            data_loader=_ResumeLoader(step),
            ema=None,
            micro_step_in_update=0,
        )

    assert {path.name for path in config.checkpoint_dir.iterdir()} == {"1", "2", "3"}
    assert all((config.checkpoint_dir / str(step) / "model.safetensors").is_file() for step in (1, 2, 3))
    assert not train_pytorch.checkpoint_is_resumable(config.checkpoint_dir / "1")
    assert train_pytorch.checkpoint_is_resumable(config.checkpoint_dir / "2")
    assert train_pytorch.checkpoint_is_resumable(config.checkpoint_dir / "3")
    assert train_pytorch.get_latest_checkpoint_step(config.checkpoint_dir) == 3
    assert train_pytorch.get_latest_checkpoint_step(config.checkpoint_dir, resumable_only=True) == 3

    evaluation_only = config.checkpoint_dir / "4"
    evaluation_only.mkdir()
    (evaluation_only / "model.safetensors").write_bytes(b"test")
    assert train_pytorch.get_latest_checkpoint_step(config.checkpoint_dir) == 4
    assert train_pytorch.get_latest_checkpoint_step(config.checkpoint_dir, resumable_only=True) == 3


def test_checkpoint_roundtrip_restores_raw_ema_and_allows_runtime_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_pytorch, "log_memory_usage", lambda *args, **kwargs: None)
    torch.manual_seed(31)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.9)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        ema.update(model)

    raw_reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with ema.average_parameters(model):
        ema_reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    config = _config_for_checkpoint(tmp_path)
    data_config = SimpleNamespace(norm_stats=None, asset_id=None)
    train_pytorch.save_checkpoint(
        model,
        optimizer,
        3,
        config,
        is_main=True,
        data_config=data_config,
        data_loader=_ResumeLoader(73),
        ema=ema,
        micro_step_in_update=0,
    )
    checkpoint = config.checkpoint_dir / "3"
    assert (checkpoint / "train_model.safetensors").is_file()
    assert (checkpoint / "model.safetensors").is_file()

    resumed_config = dataclasses.replace(
        config,
        num_train_steps=4,
        save_interval=1000,
        save_final_checkpoint=False,
        log_interval=1,
        keep_period=None,
        resume=True,
    )
    resumed_model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_ema = pytorch_ema.ExponentialMovingAverage(resumed_model, 0.9)
    resumed_loader = _ResumeLoader(999)
    step = train_pytorch.load_checkpoint(
        resumed_model,
        resumed_optimizer,
        config.checkpoint_dir,
        torch.device("cpu"),
        resumed_loader,
        resumed_config,
        resumed_ema,
    )

    assert step == 3
    assert resumed_ema.num_updates == 3
    assert resumed_loader.loaded == {"token": 73}
    for name, parameter in resumed_model.named_parameters():
        assert torch.equal(parameter, raw_reference[name])
    with resumed_ema.average_parameters(resumed_model):
        for name, parameter in resumed_model.named_parameters():
            assert torch.equal(parameter, ema_reference[name])


def test_checkpoint_roundtrip_without_ema_uses_standard_model_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_pytorch, "log_memory_usage", lambda *args, **kwargs: None)
    torch.manual_seed(41)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    raw_reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    config = _config_for_checkpoint(tmp_path, ema_decay=None)
    train_pytorch.save_checkpoint(
        model,
        optimizer,
        3,
        config,
        is_main=True,
        data_config=SimpleNamespace(norm_stats=None, asset_id=None),
        data_loader=_ResumeLoader(83),
        ema=None,
        micro_step_in_update=0,
    )
    checkpoint = config.checkpoint_dir / "3"
    assert (checkpoint / "model.safetensors").is_file()
    assert not (checkpoint / "train_model.safetensors").exists()
    metadata = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=False)
    assert metadata["ema"] is None

    resumed_model = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_loader = _ResumeLoader(999)
    step = train_pytorch.load_checkpoint(
        resumed_model,
        resumed_optimizer,
        config.checkpoint_dir,
        torch.device("cpu"),
        resumed_loader,
        dataclasses.replace(config, num_train_steps=4, resume=True),
        None,
    )

    assert step == 3
    assert resumed_loader.loaded == {"token": 83}
    for name, parameter in resumed_model.named_parameters():
        assert torch.equal(parameter, raw_reference[name])


def test_checkpoint_refuses_trajectory_change_before_loading_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train_pytorch, "log_memory_usage", lambda *args, **kwargs: None)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.9)
    for _ in range(3):
        ema.update(model)
    config = _config_for_checkpoint(tmp_path)
    train_pytorch.save_checkpoint(
        model,
        optimizer,
        3,
        config,
        is_main=True,
        data_config=SimpleNamespace(norm_stats=None, asset_id=None),
        data_loader=_ResumeLoader(4),
        ema=ema,
        micro_step_in_update=0,
    )
    changed = dataclasses.replace(config, seed=config.seed + 1, num_train_steps=4, resume=True)

    with pytest.raises(RuntimeError, match="trajectory-affecting"):
        train_pytorch.load_checkpoint(
            torch.nn.Linear(2, 1),
            torch.optim.AdamW(torch.nn.Linear(2, 1).parameters()),
            config.checkpoint_dir,
            torch.device("cpu"),
            _ResumeLoader(0),
            changed,
            pytorch_ema.ExponentialMovingAverage(torch.nn.Linear(2, 1), 0.9),
        )
