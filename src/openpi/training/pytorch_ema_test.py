from __future__ import annotations

import safetensors.torch
import torch

from openpi.training import pytorch_ema


class _TinyAuxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 2)
        self.geometry_queries = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        self.ground_head = torch.nn.Linear(2, 1)


def test_ema_initializes_updates_and_covers_auxiliary_parameters() -> None:
    torch.manual_seed(7)
    model = _TinyAuxModel()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.75)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(4.0)
    raw = {name: value.detach().clone() for name, value in model.named_parameters()}
    ema.update(model)

    assert ema.num_updates == 1
    assert set(ema.metadata()["parameter_names"]) == set(initial)
    with ema.average_parameters(model):
        averaged = dict(model.named_parameters())
        for name, initial_parameter in initial.items():
            assert torch.equal(averaged[name], initial_parameter * 0.75 + raw[name] * 0.25)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, raw[name])


def test_raw_and_ema_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(11)
    model = _TinyAuxModel()
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    ema.update(model)
    raw_reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with ema.average_parameters(model):
        ema_reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    raw_path = tmp_path / "train_model.safetensors"
    ema_path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(model, raw_path)
    ema.save_model(model, ema_path)

    resumed = _TinyAuxModel()
    resumed_ema = pytorch_ema.ExponentialMovingAverage(resumed, 0.5)
    safetensors.torch.load_model(resumed, raw_path, strict=True, device="cpu")
    resumed_ema.load_model(resumed, ema_path, device="cpu")
    resumed_ema.load_metadata(ema.metadata(), resumed)

    for name, parameter in resumed.named_parameters():
        assert torch.equal(parameter, raw_reference[name])
    with resumed_ema.average_parameters(resumed):
        for name, parameter in resumed.named_parameters():
            assert torch.equal(parameter, ema_reference[name])
    assert resumed_ema.num_updates == 1
