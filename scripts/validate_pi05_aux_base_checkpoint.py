#!/usr/bin/env python3
"""Validate strict official pi0.5-base loading for P0/P1/P2 PyTorch models."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess

from openpi.models import pi0_config
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    modes = ("none", "geometry", "ground_geometry_semantic_lm")
    results = {}
    for mode in modes:
        model = PI05AuxPolicy(model_config, PolicyAuxConfig(mode=mode))
        results[mode] = model.load_official_base_checkpoint(str(checkpoint), device="cpu")
        del model

    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    payload = {
        "status": "PASS",
        "gate": "pi05_aux_official_base_strict_load_v1",
        "openpi_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": dataclasses.asdict(model_config),
        "loads": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
