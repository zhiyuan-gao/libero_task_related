"""Exact exponential-moving-average state for PyTorch training and serving."""

from __future__ import annotations

import contextlib
from pathlib import Path

import safetensors.torch
import torch


class ExponentialMovingAverage:
    """Track model parameters with full-precision accumulation for low-precision training."""

    _SCHEMA = "openpi.pytorch_ema.v2"

    def __init__(self, module: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), found {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        # A decay close to one produces updates much smaller than one BF16/FP16
        # ULP. Accumulating those updates in the model dtype can permanently
        # freeze the EMA shadow. Keep low-precision floating-point parameters in
        # FP32 and cast only when the checkpoint is loaded into a BF16/FP16
        # serving model.
        self._shadow = {
            name: parameter.detach().to(dtype=self._shadow_dtype(parameter), copy=True)
            for name, parameter in module.named_parameters()
        }
        if not self._shadow:
            raise ValueError("EMA requires a model with at least one parameter")

    @staticmethod
    def _shadow_dtype(parameter: torch.Tensor) -> torch.dtype:
        if parameter.dtype in (torch.bfloat16, torch.float16):
            return torch.float32
        return parameter.dtype

    def _parameters(self, module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        parameters = dict(module.named_parameters())
        if parameters.keys() != self._shadow.keys():
            missing = sorted(self._shadow.keys() - parameters.keys())
            unexpected = sorted(parameters.keys() - self._shadow.keys())
            raise RuntimeError(f"EMA parameter topology changed: missing={missing}, unexpected={unexpected}")
        for name, parameter in parameters.items():
            shadow = self._shadow[name]
            expected_shadow_dtype = self._shadow_dtype(parameter)
            if (
                shadow.shape != parameter.shape
                or shadow.dtype != expected_shadow_dtype
                or shadow.device != parameter.device
            ):
                raise RuntimeError(
                    f"EMA parameter mismatch for {name}: "
                    f"shadow={shadow.shape}/{shadow.dtype}/{shadow.device}, "
                    f"model={parameter.shape}/{parameter.dtype}/{parameter.device}, "
                    f"expected_shadow_dtype={expected_shadow_dtype}"
                )
        return parameters

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        parameters = self._parameters(module)
        floating_shadows = []
        floating_parameters = []
        copied_shadows = []
        copied_parameters = []
        for name, parameter in parameters.items():
            shadow = self._shadow[name]
            if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                floating_shadows.append(shadow)
                floating_parameters.append(parameter.detach())
            else:
                copied_shadows.append(shadow)
                copied_parameters.append(parameter.detach())

        # Launch batched foreach kernels instead of two kernels per parameter.
        # This preserves the original update exactly: shadow = decay * shadow + (1 - decay) * parameter.
        if floating_shadows:
            torch._foreach_mul_(floating_shadows, self.decay)
            torch._foreach_add_(floating_shadows, floating_parameters, alpha=1.0 - self.decay)
        if copied_shadows:
            torch._foreach_copy_(copied_shadows, copied_parameters)
        self.num_updates += 1

    def metadata(self) -> dict:
        return {
            "schema": self._SCHEMA,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "parameter_names": tuple(self._shadow),
            "shadow_dtypes": tuple((name, str(shadow.dtype)) for name, shadow in self._shadow.items()),
        }

    def load_metadata(self, metadata: dict, module: torch.nn.Module) -> None:
        if metadata.get("schema") != self._SCHEMA:
            raise RuntimeError(f"Unsupported EMA metadata: {metadata.get('schema')}")
        if float(metadata["decay"]) != self.decay:
            raise RuntimeError(f"EMA decay mismatch: saved={metadata['decay']}, current={self.decay}")
        if tuple(metadata["parameter_names"]) != tuple(self._shadow):
            raise RuntimeError("EMA checkpoint parameter names do not match the current model")
        expected_shadow_dtypes = tuple((name, str(shadow.dtype)) for name, shadow in self._shadow.items())
        if tuple(tuple(item) for item in metadata.get("shadow_dtypes", ())) != expected_shadow_dtypes:
            raise RuntimeError(
                "EMA checkpoint shadow dtypes do not match full-precision accumulation: "
                f"saved={metadata.get('shadow_dtypes')}, expected={expected_shadow_dtypes}"
            )
        num_updates = int(metadata["num_updates"])
        if num_updates < 0:
            raise RuntimeError(f"EMA update count must be non-negative, found {num_updates}")
        self._parameters(module)
        self.num_updates = num_updates

    @contextlib.contextmanager
    def average_parameters(self, module: torch.nn.Module):
        """Swap EMA tensors into ``module`` without allocating a second model.

        Low-precision module parameters temporarily become FP32 inside the
        context. This is intended for checkpoint save/load, not model forward.
        Loading the resulting FP32 checkpoint into a BF16/FP16 serving model
        performs the single final cast after EMA accumulation.
        """

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
