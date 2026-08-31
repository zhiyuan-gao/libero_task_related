#!/usr/bin/env python3
"""Validate P1/P2 model/optimizer/metadata checkpoint round trips."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time

from policy_aux_gate_utils import load_real_libero_item
from policy_aux_gate_utils import move_observation
import safetensors.torch
import torch

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
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--geometry-normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    observation, _, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.policy_manifest,
        include_ground_masks=False,
    )
    observation = move_observation(observation, device)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    reports = {}
    for mode in ("geometry", "semantic_geometry", "ground_geometry_semantic_lm"):
        aux_config = PolicyAuxConfig(
            mode=mode,
            num_ground_queries=0 if mode == "semantic_geometry" else 8,
            lambda_geo=0.0,
            lambda_sem=0.0 if mode in ("semantic_geometry", "ground_geometry_semantic_lm") else None,
            lambda_ground=0.0 if mode == "ground_geometry_semantic_lm" else None,
            geometry_normalization_path=str(args.geometry_normalization),
        )
        model = PI05AuxPolicy(config, aux_config)
        base_load = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
        model.to(device).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # Create non-empty optimizer state for every new query/head parameter
        # without claiming this synthetic update is a training smoke.
        synthetic = sum(
            parameter.float().square().mean()
            for name, parameter in model.named_parameters()
            if name in model.expected_auxiliary_state_keys()
        )
        synthetic.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device=device).manual_seed(20260818)
        noise = torch.randn((1, config.action_horizon, config.action_dim), generator=generator, device=device)
        reference = model.sample_actions(device, observation, noise=noise.clone(), num_steps=2).cpu()

        with tempfile.TemporaryDirectory(prefix=f"openpi_{mode}_roundtrip_") as temporary:
            checkpoint_dir = Path(temporary)
            model_path = checkpoint_dir / "model.safetensors"
            optimizer_path = checkpoint_dir / "optimizer.pt"
            metadata_path = checkpoint_dir / "metadata.json"
            safetensors.torch.save_model(model, model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            metadata = {
                "global_step": 1,
                "scheduler": {"type": "stateless_cosine", "step": 1},
                "model_config": dataclasses.asdict(config),
                "policy_aux_config": dataclasses.asdict(aux_config),
                "geometry_normalization": str(args.geometry_normalization.resolve(strict=True)),
                "engineering_synthetic_optimizer_state": True,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            file_records = {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (model_path, optimizer_path, metadata_path)
            }

            auxiliary_state_keys = sorted(model.expected_auxiliary_state_keys())
            del model, optimizer, synthetic
            gc.collect()
            torch.cuda.empty_cache()

            # Strict-load on CPU so the round-trip never requires two complete
            # policy copies (plus safetensors staging buffers) in GPU memory.
            reloaded = PI05AuxPolicy(config, aux_config).eval()
            missing, unexpected = safetensors.torch.load_model(reloaded, model_path, strict=True, device="cpu")
            reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-4)
            reloaded_optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
            reloaded_metadata = json.loads(metadata_path.read_text())
            reloaded.to(device)
            candidate = reloaded.sample_actions(device, observation, noise=noise.clone(), num_steps=2).cpu()
            checks = {
                "base_initialization_exact": not base_load["unexpected"],
                "roundtrip_model_strict": not missing and not unexpected,
                "fixed_inference_bitwise_equal": bool(torch.equal(reference, candidate)),
                "optimizer_state_nonempty": bool(reloaded_optimizer.state_dict()["state"]),
                "global_step_preserved": reloaded_metadata["global_step"] == 1,
                "scheduler_step_preserved": reloaded_metadata["scheduler"]["step"] == 1,
                "aux_config_preserved": reloaded_metadata["policy_aux_config"] == dataclasses.asdict(aux_config),
                "geometry_normalization_reference_preserved": (
                    reloaded_metadata["geometry_normalization"] == str(args.geometry_normalization.resolve(strict=True))
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"{mode} checkpoint roundtrip failed: {checks}")
            reports[mode] = {
                "checks": checks,
                "base_load": base_load,
                "temporary_file_records_before_cleanup": file_records,
                "auxiliary_state_keys": auxiliary_state_keys,
            }
            del reloaded, reloaded_optimizer
        del reference, candidate
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "status": "PASS",
        "gate": "pi05_p1_p2_checkpoint_roundtrip_v1",
        "sample_id": auxiliary["sample_id"],
        "large_roundtrip_files_retained": False,
        "reports": reports,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
