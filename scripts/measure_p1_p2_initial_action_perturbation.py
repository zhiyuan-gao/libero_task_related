#!/usr/bin/env python3
"""Measure initial action-loss perturbation from random P1/P2 auxiliary queries.

This is a fixed-input, no-gradient diagnostic. It deliberately creates no
optimizer and performs no parameter update.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import jax
import pandas as pd
import safetensors.torch
import torch

from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxTargets
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.policy_aux_dataset import PolicyAuxTargetIndex
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig
from openpi.training.policy_aux_dataset import PolicyAuxTransformedDataset

MODEL_INITIALIZATION_SEED = 20260818
SAMPLE_SEED_BASE = 20260818
DIFFUSION_TIMESTEP = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def to_targets(batch: dict) -> PolicyAuxTargets:
    return PolicyAuxTargets(
        geometry=batch["geometry"],
        geometry_valid=batch["geometry_valid"],
        geometry_mean=batch["geometry_mean"],
        geometry_std=batch["geometry_std"],
        ground_masks=batch["ground_masks"],
        ground_valid_views=batch["ground_valid_views"],
        semantic_input_ids=batch["semantic_input_ids"],
        semantic_labels=batch["semantic_labels"],
        semantic_loss_mask=batch["semantic_loss_mask"],
    )


def build_model(
    mode: str,
    *,
    model_config,
    checkpoint: Path,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(MODEL_INITIALIZATION_SEED)
    if mode == "aux_disabled":
        model = PI0Pytorch(model_config)
        missing, unexpected = safetensors.torch.load_model(model, checkpoint, strict=True, device="cpu")
        strict_load = {"missing": list(missing), "unexpected": list(unexpected)}
    else:
        aux_config = PolicyAuxConfig(
            mode=mode,
            lambda_geo=1.0,
            lambda_ground=1.0 if mode == "ground_geometry_semantic_lm" else None,
            lambda_sem=1.0 if mode == "ground_geometry_semantic_lm" else None,
        )
        model = PI05AuxPolicy(model_config, aux_config)
        strict_load = model.load_official_base_checkpoint(str(checkpoint), device="cpu")
    return model, strict_load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--libero-assets-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--geometry-index", type=Path, required=True)
    parser.add_argument("--geometry-normalization", type=Path, required=True)
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.monotonic()
    device = torch.device(args.device)
    source_calibration = json.loads(args.source_calibration.read_text())
    if source_calibration.get("status") != "PASS":
        raise ValueError("Source pre-lambda calibration is not PASS")
    selected = [int(value) for value in source_calibration["dataset_indices"]]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("Source calibration does not contain the frozen 16-sample subset")

    base = _config.get_config("pi05_libero")
    if not (base.model.pi05 and base.model.action_horizon == 10 and not base.model.discrete_state_input):
        raise ValueError("Official pi05_libero input semantics changed")
    data_factory = dataclasses.replace(
        base.data,
        assets=_config.AssetsConfig(assets_dir=str(args.libero_assets_root.resolve(strict=True))),
    )
    data_config = data_factory.create(Path("/nonexistent/assets_not_used"), base.model)

    # Recompute the original deterministic selection and require exact equality
    # with the source calibration rather than silently accepting a stale list.
    target_frame = pd.read_parquet(args.geometry_index)
    valid = target_frame.loc[target_frame["geometry_valid"].astype(bool)]
    groups = [group for _, group in valid.groupby("task_id", sort=True)]
    recomputed = [int(group.iloc[0]["lerobot_dataset_index"]) for group in groups]
    recomputed.extend(int(group.iloc[len(group) // 2]["lerobot_dataset_index"]) for group in groups[:6])
    if recomputed != selected:
        raise ValueError("Fixed 16-sample selection differs from source calibration")

    # Use one P2-capable dataset instance for all three modes so every mode sees
    # exactly the same decoded observation, action, and auxiliary target batch.
    target_config = PolicyAuxTrainConfig(
        mode="ground_geometry_semantic_lm",
        policy_manifest_path=str(args.policy_manifest.resolve(strict=True)),
        episode_mapping_path=str(args.mapping.resolve(strict=True)),
        geometry_target_index_path=str(args.geometry_index.resolve(strict=True)),
        geometry_normalization_path=str(args.geometry_normalization.resolve(strict=True)),
        lambda_geo=1.0,
        lambda_ground=1.0,
        lambda_sem=1.0,
        lerobot_root=str(args.lerobot_root.resolve(strict=True)),
    )
    raw_dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=10,
        model_config=base.model,
        policy_aux_config=target_config,
    )
    transformed = _data_loader.transform_dataset(raw_dataset, data_config)
    dataset = PolicyAuxTransformedDataset(transformed, target_config)
    target_index = PolicyAuxTargetIndex(target_config)
    del target_index
    fixed_batches = []
    for dataset_index in selected:
        item = dataset[dataset_index]
        batch = _data_loader._collate_fn([item])  # noqa: SLF001
        fixed_batches.append(jax.tree.map(torch.as_tensor, batch))

    reports = {}
    noise_hashes_by_mode = {}
    for mode in ("aux_disabled", "geometry", "ground_geometry_semantic_lm"):
        model, strict_load = build_model(
            mode,
            model_config=base.model,
            checkpoint=args.checkpoint,
        )
        model.to(device).eval()
        losses = []
        noise_digest = hashlib.sha256()
        torch.cuda.reset_peak_memory_stats(device)
        for ordinal, batch in enumerate(fixed_batches):
            observation = _model.Observation.from_dict(batch)
            observation = jax.tree.map(lambda value: value.to(device), observation)
            actions = batch["actions"].to(device=device, dtype=torch.float32)
            target_batch = jax.tree.map(lambda value: value.to(device), batch["policy_aux"])
            targets = to_targets(target_batch)
            sample_seed = SAMPLE_SEED_BASE + ordinal
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            noise = torch.randn(actions.shape, generator=generator, device=device)
            noise_digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
            diffusion_time = torch.full((1,), DIFFUSION_TIMESTEP, dtype=torch.float32, device=device)
            # This reset makes the stochastic training-time image geometry
            # identical across aux-disabled, P1, and P2 for the same sample.
            torch.manual_seed(sample_seed)
            with torch.no_grad():
                if mode == "aux_disabled":
                    action_loss = model(
                        observation,
                        actions,
                        noise=noise,
                        time=diffusion_time,
                    ).mean()
                else:
                    action_loss = model.forward_with_aux(
                        observation,
                        actions,
                        targets,
                        noise=noise,
                        time=diffusion_time,
                    )["losses"]["action"]
            if not bool(torch.isfinite(action_loss)):
                raise ValueError(f"Non-finite action loss for mode={mode}, ordinal={ordinal}")
            losses.append(float(action_loss))
        noise_hashes_by_mode[mode] = noise_digest.hexdigest()
        reports[mode] = {
            "action_loss": stats(losses),
            "per_sample_action_loss": losses,
            "strict_load": strict_load,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(set(noise_hashes_by_mode.values())) != 1:
        raise RuntimeError("Noise tensors were not identical across the three modes")
    baseline = reports["aux_disabled"]["action_loss"]["mean"]
    if baseline <= 0.0:
        raise ValueError("Aux-disabled mean action loss must be positive")
    deltas = {}
    baseline_samples = reports["aux_disabled"]["per_sample_action_loss"]
    for mode in ("geometry", "ground_geometry_semantic_lm"):
        mean = reports[mode]["action_loss"]["mean"]
        signed = mean - baseline
        paired = [
            current - reference
            for current, reference in zip(reports[mode]["per_sample_action_loss"], baseline_samples, strict=True)
        ]
        deltas[mode] = {
            "signed_mean_delta": signed,
            "absolute_mean_delta": abs(signed),
            "signed_relative_mean_delta": signed / baseline,
            "absolute_relative_mean_delta": abs(signed) / baseline,
            "mean_absolute_paired_delta": statistics.fmean(abs(value) for value in paired),
            "max_absolute_paired_delta": max(abs(value) for value in paired),
        }

    payload = {
        "status": "PASS_AWAITING_HUMAN_LAMBDA_FREEZE",
        "schema": "openpi.p1_p2_initial_action_perturbation.v1",
        "scope": "fixed-input no-gradient diagnostic; no optimizer constructed or step performed",
        "optimizer_constructed": False,
        "optimizer_steps_run": 0,
        "architecture_modified": False,
        "motion_integrated": False,
        "checkpoint": str(args.checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_calibration": str(args.source_calibration.resolve(strict=True)),
        "source_calibration_sha256": sha256_file(args.source_calibration),
        "sample_count": len(selected),
        "dataset_indices": selected,
        "model_initialization_seed": MODEL_INITIALIZATION_SEED,
        "sample_seed_rule": f"{SAMPLE_SEED_BASE} + ordinal",
        "diffusion_timestep": DIFFUSION_TIMESTEP,
        "noise_sha256": next(iter(noise_hashes_by_mode.values())),
        "same_noise_all_modes": True,
        "same_decoded_batches_all_modes": True,
        "same_training_image_seed_all_modes": True,
        "reports": reports,
        "deltas_from_aux_disabled": deltas,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
