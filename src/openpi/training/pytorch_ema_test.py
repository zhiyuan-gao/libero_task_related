from __future__ import annotations

import pytest
import safetensors.torch
import torch

from openpi.training import pytorch_ema


class _TinyAuxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 2)
        self.geometry_queries = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        self.ground_head = torch.nn.Linear(2, 1)


class _MixedDtypeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.float32 = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        self.float64 = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        self.frozen_integer = torch.nn.Parameter(torch.tensor([5, 6], dtype=torch.int64), requires_grad=False)


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


def test_ema_foreach_update_matches_scalar_formula_for_mixed_dtypes() -> None:
    model = _MixedDtypeModel()
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.75)

    with torch.no_grad():
        model.float32.add_(4.0)
        model.float64.sub_(2.0)
        model.frozen_integer.add_(3)
    raw = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    ema.update(model)

    with ema.average_parameters(model):
        averaged = dict(model.named_parameters())
        assert torch.equal(averaged["float32"], initial["float32"] * 0.75 + raw["float32"] * 0.25)
        assert torch.equal(averaged["float64"], initial["float64"] * 0.75 + raw["float64"] * 0.25)
        assert torch.equal(averaged["frozen_integer"], raw["frozen_integer"])


def test_bfloat16_parameters_accumulate_ema_in_float32() -> None:
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.999)

    with torch.no_grad():
        model.weight.fill_(1.0078125)  # The next BF16 value above 1.0.
    raw = model.weight.detach().clone()
    legacy_bfloat16_shadow = torch.ones_like(raw)
    for _ in range(1000):
        ema.update(model)
        legacy_bfloat16_shadow.mul_(0.999).add_(raw, alpha=0.001)

    expected = 1.0 * 0.999**1000 + float(raw.item()) * (1.0 - 0.999**1000)
    assert legacy_bfloat16_shadow.item() == 1.0
    with ema.average_parameters(model):
        assert model.weight.dtype == torch.float32
        assert model.weight.item() == pytest.approx(expected, abs=2e-5)
        assert model.weight.item() > 1.003

    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model.weight, raw)
    assert ema.metadata()["shadow_dtypes"] == (("weight", "torch.float32"),)


def test_bfloat16_raw_and_float32_ema_checkpoint_roundtrip(tmp_path) -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.999)
    with torch.no_grad():
        model.weight.add_(0.125)
    ema.update(model)

    raw_reference = model.weight.detach().clone()
    with ema.average_parameters(model):
        ema_reference = model.weight.detach().clone()

    raw_path = tmp_path / "train_model.safetensors"
    ema_path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(model, raw_path)
    ema.save_model(model, ema_path)

    assert safetensors.torch.load_file(raw_path)["weight"].dtype == torch.bfloat16
    assert safetensors.torch.load_file(ema_path)["weight"].dtype == torch.float32

    resumed = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    resumed_ema = pytorch_ema.ExponentialMovingAverage(resumed, 0.999)
    safetensors.torch.load_model(resumed, raw_path, strict=True, device="cpu")
    resumed_ema.load_model(resumed, ema_path, device="cpu")
    resumed_ema.load_metadata(ema.metadata(), resumed)

    assert resumed.weight.dtype == torch.bfloat16
    assert torch.equal(resumed.weight, raw_reference)
    with resumed_ema.average_parameters(resumed):
        assert resumed.weight.dtype == torch.float32
        assert torch.equal(resumed.weight, ema_reference)


def test_rejects_legacy_same_dtype_ema_metadata() -> None:
    model = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
    ema = pytorch_ema.ExponentialMovingAverage(model, 0.999)
    legacy_metadata = ema.metadata()
    legacy_metadata["schema"] = "openpi.pytorch_ema.v1"
    legacy_metadata.pop("shadow_dtypes")

    with pytest.raises(RuntimeError, match="Unsupported EMA metadata"):
        ema.load_metadata(legacy_metadata, model)
