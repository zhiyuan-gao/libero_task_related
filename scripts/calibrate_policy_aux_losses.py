#!/usr/bin/env python3
"""Measure raw P1/P2 loss scales on a fixed diverse 16-sample engineering subset."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import statistics
import time

import jax
import pandas as pd
import torch

from openpi.models import model as _model
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxTargets
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.policy_aux_dataset import PolicyAuxTargetIndex
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig
from openpi.training.policy_aux_dataset import PolicyAuxTransformedDataset


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
        ground_masks=batch.get("ground_masks"),
        ground_valid_views=batch.get("ground_valid_views"),
        semantic_input_ids=batch.get("semantic_input_ids"),
        semantic_labels=batch.get("semantic_labels"),
        semantic_loss_mask=batch.get("semantic_loss_mask"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--libero-assets-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--geometry-index", type=Path, required=True)
    parser.add_argument("--geometry-normalization", type=Path, required=True)
    parser.add_argument("--component-gradient-gate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)
    base = _config.get_config("pi05_libero")
    data_factory = dataclasses.replace(
        base.data,
        assets=_config.AssetsConfig(assets_dir=str(args.libero_assets_root.resolve(strict=True))),
    )
    data_config = data_factory.create(Path("/nonexistent/assets_not_used"), base.model)
    target_frame = pd.read_parquet(args.geometry_index)
    valid = target_frame.loc[target_frame["geometry_valid"].astype(bool)]
    groups = [group for _, group in valid.groupby("task_id", sort=True)]
    selected = [int(group.iloc[0]["lerobot_dataset_index"]) for group in groups]
    selected.extend(int(group.iloc[len(group) // 2]["lerobot_dataset_index"]) for group in groups[:6])
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("Could not construct the fixed diverse 16-sample calibration subset")

    reports = {}
    for mode in ("geometry", "ground_geometry_semantic_lm"):
        aux_train_config = PolicyAuxTrainConfig(
            mode=mode,
            policy_manifest_path=str(args.policy_manifest.resolve(strict=True)),
            episode_mapping_path=str(args.mapping.resolve(strict=True)),
            geometry_target_index_path=str(args.geometry_index.resolve(strict=True)),
            geometry_normalization_path=str(args.geometry_normalization.resolve(strict=True)),
            lambda_geo=1.0,
            lambda_sem=1.0 if mode == "ground_geometry_semantic_lm" else None,
            lambda_ground=1.0 if mode == "ground_geometry_semantic_lm" else None,
            lerobot_root=str(args.lerobot_root.resolve(strict=True)),
        )
        raw_dataset = _data_loader.create_torch_dataset(
            data_config,
            action_horizon=10,
            model_config=base.model,
            policy_aux_config=aux_train_config,
        )
        transformed = _data_loader.transform_dataset(raw_dataset, data_config)
        dataset = PolicyAuxTransformedDataset(transformed, aux_train_config)
        # Force index validation before loading the model so data errors fail cheaply.
        target_index = PolicyAuxTargetIndex(aux_train_config)
        del target_index

        torch.manual_seed(20260818)
        aux_model_config = PolicyAuxConfig(
            mode=mode,
            lambda_geo=1.0,
            lambda_sem=1.0 if mode == "ground_geometry_semantic_lm" else None,
            lambda_ground=1.0 if mode == "ground_geometry_semantic_lm" else None,
        )
        model = PI05AuxPolicy(base.model, aux_model_config)
        strict_load = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
        model.to(device).eval()
        values: dict[str, list[float]] = {}
        torch.cuda.reset_peak_memory_stats(device)
        for ordinal, dataset_index in enumerate(selected):
            item = dataset[dataset_index]
            batch = _data_loader._collate_fn([item])  # noqa: SLF001
            batch = jax.tree.map(torch.as_tensor, batch)
            observation = _model.Observation.from_dict(batch)
            observation = jax.tree.map(lambda value: value.to(device), observation)
            actions = batch["actions"].to(device=device, dtype=torch.float32)
            target_batch = jax.tree.map(lambda value: value.to(device), batch["policy_aux"])
            targets = to_targets(target_batch)
            generator = torch.Generator(device=device).manual_seed(20260818 + ordinal)
            noise = torch.randn(actions.shape, generator=generator, device=device)
            diffusion_time = torch.full((1,), 0.5, dtype=torch.float32, device=device)
            torch.manual_seed(20260818 + ordinal)
            with torch.no_grad():
                result = model.forward_with_aux(
                    observation,
                    actions,
                    targets,
                    noise=noise,
                    time=diffusion_time,
                )
            for name, loss in result["losses"].items():
                if name != "total":
                    values.setdefault(name, []).append(float(loss))
        reports[mode] = {
            "strict_load": strict_load,
            "raw_loss_statistics": {name: stats(losses) for name, losses in values.items()},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        del model, dataset, raw_dataset, transformed
        torch.cuda.empty_cache()

    gradient_gate = json.loads(args.component_gradient_gate.read_text())
    if gradient_gate.get("status") != "PASS":
        raise ValueError("Component-gradient gate must pass before calibration reporting")
    payload = {
        "status": "PASS",
        "schema": "openpi.p1_p2_loss_scale_calibration.v1",
        "scope": "engineering raw-scale diagnostic; not a scientific hyperparameter search",
        "sample_count": len(selected),
        "dataset_indices": selected,
        "reports": reports,
        "component_gradient_gate": str(args.component_gradient_gate.resolve(strict=True)),
        "component_gradient_norms": {mode: report["gradient_norms"] for mode, report in gradient_gate["modes"].items()},
        "candidate_weighted_contribution": None,
        "lambda_values_approved": False,
        "required_human_review": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# P1/P2 raw loss-scale calibration",
        "",
        "Status: **PASS — human lambda review required**",
        "",
        "This is a 16-sample engineering diagnostic, not a hyperparameter search.",
        "No lambda value is approved or written into a launch config.",
        "",
    ]
    for mode, report in reports.items():
        lines.extend([f"## {mode}", "", "| Loss | Mean | Std | Median |", "|---|---:|---:|---:|"])
        for name, summary in report["raw_loss_statistics"].items():
            lines.append(f"| {name} | {summary['mean']:.6f} | {summary['std']:.6f} | {summary['median']:.6f} |")
        lines.append("")
    lines.extend(
        [
            "## Decision gate",
            "",
            "Full P1/P2 training remains hard-blocked until the researcher explicitly freezes",
            "`lambda_sem`, `lambda_ground`, and `lambda_geo` after reviewing this report.",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
