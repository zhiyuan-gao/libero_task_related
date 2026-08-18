#!/usr/bin/env python3
"""Prove policy_aux_mode=none is bitwise equivalent to official PI0Pytorch."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

from policy_aux_gate_utils import load_real_libero_item
from policy_aux_gate_utils import move_observation
import safetensors.torch
import torch

from openpi.models import pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig


def fixed_forward(model, observation, actions, noise, diffusion_time) -> torch.Tensor:
    torch.manual_seed(20260818)
    torch.cuda.manual_seed_all(20260818)
    model.eval()
    with torch.no_grad():
        return (
            model(
                observation,
                actions,
                noise=noise,
                time=diffusion_time,
            )
            .detach()
            .cpu()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    observation, actions, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.annotation_manifest,
    )
    observation = move_observation(observation, device)
    actions = actions.to(device)
    generator = torch.Generator(device=device).manual_seed(20260818)
    noise = torch.randn(actions.shape, generator=generator, device=device)
    diffusion_time = torch.full((1,), 0.5, dtype=torch.float32, device=device)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )

    official = PI0Pytorch(config)
    official_missing, official_unexpected = safetensors.torch.load_model(
        official, str(args.checkpoint), strict=True, device="cpu"
    )
    official.to(device)
    official_output = fixed_forward(official, observation, actions, noise, diffusion_time)
    del official
    torch.cuda.empty_cache()

    disabled = PI05AuxPolicy(config, PolicyAuxConfig(mode="none"))
    disabled_load = disabled.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
    disabled.to(device)
    disabled_output = fixed_forward(disabled, observation, actions, noise, diffusion_time)
    max_abs_difference = float((official_output - disabled_output).abs().max())
    checks = {
        "official_checkpoint_strict_load": not official_missing and not official_unexpected,
        "disabled_checkpoint_strict_load": disabled_load == {"missing": [], "unexpected": []},
        "disabled_has_no_auxiliary_parameters": disabled.expected_auxiliary_state_keys() == set(),
        "fixed_real_batch_output_bitwise_equal": bool(torch.equal(official_output, disabled_output)),
        "maximum_absolute_difference_is_zero": max_abs_difference == 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Disabled-equivalence gate failed: {checks}")
    payload = {
        "status": "PASS",
        "gate": "pi05_policy_aux_disabled_official_bitwise_equivalence_v1",
        "sample_id": auxiliary["sample_id"],
        "model_config": dataclasses.asdict(config),
        "checkpoint": str(args.checkpoint.resolve()),
        "official_strict_load": {
            "missing": sorted(official_missing),
            "unexpected": sorted(official_unexpected),
        },
        "disabled_strict_load": disabled_load,
        "fixed_inputs": {
            "seed": 20260818,
            "diffusion_time": 0.5,
            "action_shape": list(actions.shape),
            "noise_shape": list(noise.shape),
        },
        "output_shape": list(disabled_output.shape),
        "max_abs_difference": max_abs_difference,
        "elapsed_seconds": time.monotonic() - started,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
