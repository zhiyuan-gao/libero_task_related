"""Exact exponential-moving-average state for PyTorch training and serving."""

from __future__ import annotations

import contextlib
from pathlib import Path

import safetensors.torch
import torch


class ExponentialMovingAverage:
    """Track every model parameter and temporarily expose EMA weights on a module."""

    _SCHEMA = "openpi.pytorch_ema.v1"

    def __init__(self, module: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), found {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        self._shadow = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
        if not self._shadow:
            raise ValueError("EMA requires a model with at least one parameter")

    def _parameters(self, module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        parameters = dict(module.named_parameters())
        if parameters.keys() != self._shadow.keys():
            missing = sorted(self._shadow.keys() - parameters.keys())
            unexpected = sorted(parameters.keys() - self._shadow.keys())
            raise RuntimeError(f"EMA parameter topology changed: missing={missing}, unexpected={unexpected}")
        for name, parameter in parameters.items():
            shadow = self._shadow[name]
            if shadow.shape != parameter.shape or shadow.dtype != parameter.dtype or shadow.device != parameter.device:
                raise RuntimeError(
                    f"EMA parameter mismatch for {name}: "
                    f"shadow={shadow.shape}/{shadow.dtype}/{shadow.device}, "
                    f"model={parameter.shape}/{parameter.dtype}/{parameter.device}"
                )
        return parameters

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        parameters = self._parameters(module)
        for name, parameter in parameters.items():
            shadow = self._shadow[name]
            if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(parameter.detach())
        self.num_updates += 1

    def metadata(self) -> dict:
        return {
            "schema": self._SCHEMA,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "parameter_names": tuple(self._shadow),
        }

    def load_metadata(self, metadata: dict, module: torch.nn.Module) -> None:
        if metadata.get("schema") != self._SCHEMA:
            raise RuntimeError(f"Unsupported EMA metadata: {metadata.get('schema')}")
        if float(metadata["decay"]) != self.decay:
            raise RuntimeError(f"EMA decay mismatch: saved={metadata['decay']}, current={self.decay}")
        if tuple(metadata["parameter_names"]) != tuple(self._shadow):
            raise RuntimeError("EMA checkpoint parameter names do not match the current model")
        num_updates = int(metadata["num_updates"])
        if num_updates < 0:
            raise RuntimeError(f"EMA update count must be non-negative, found {num_updates}")
        self._parameters(module)
        self.num_updates = num_updates

    @contextlib.contextmanager
    def average_parameters(self, module: torch.nn.Module):
        """Swap EMA tensors into ``module`` without allocating a second model."""

        parameters = self._parameters(module)
        with torch.no_grad():
            for name, parameter in parameters.items():
                raw = parameter.data
                parameter.data = self._shadow[name]
                self._shadow[name] = raw
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    average = parameter.data
                    parameter.data = self._shadow[name]
                    self._shadow[name] = average
            self._parameters(module)

    def save_model(self, module: torch.nn.Module, path: str | Path) -> None:
        with self.average_parameters(module):
            safetensors.torch.save_model(module, path)

    def load_model(self, module: torch.nn.Module, path: str | Path, *, device: str) -> None:
        with self.average_parameters(module):
            missing, unexpected = safetensors.torch.load_model(module, path, strict=True, device=device)
        if missing or unexpected:
            raise RuntimeError(f"Strict EMA load failed: missing={missing}, unexpected={unexpected}")
