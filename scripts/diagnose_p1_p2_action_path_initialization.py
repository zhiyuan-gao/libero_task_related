#!/usr/bin/env python3
"""No-gradient causal decomposition of initial P1/P2 action-path perturbation.

The diagnostic temporarily masks only action-suffix reads into selected query
columns. It does not modify the model source, query initialization, prefix
connectivity, position IDs, optimizer state, or parameters.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
import openpi.models_pytorch.pi05_aux_queries as _aux_module
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


def span_dict(span) -> dict[str, int] | None:
    if span is None:
        return None
    return {"start": int(span.start), "end": int(span.end), "length": int(span.length)}


@contextmanager
def temporary_action_query_block(blocked_groups: frozenset[str]):
    """Mask action rows to requested query groups for one diagnostic call."""

    original = _aux_module.build_explicit_aux_train_attention

    def diagnostic_attention(prefix_pad_mask, suffix_pad_mask, suffix_ar_mask, layout):
        full = original(prefix_pad_mask, suffix_pad_mask, suffix_ar_mask, layout)
        prefix_length = prefix_pad_mask.shape[1]
        for name in blocked_groups:
            span = layout.query_groups.get(name)
            if span is None:
                raise ValueError(f"Cannot block absent query group: {name}")
            full[:, prefix_length:, span.start : span.end] = False
        return full

    _aux_module.build_explicit_aux_train_attention = diagnostic_attention
    try:
        yield
    finally:
        _aux_module.build_explicit_aux_train_attention = original


@contextmanager
def capture_joint_position_ids(model):
    """Capture the position IDs supplied to the action joint transformer call."""

    module = model.paligemma_with_expert
    original = module.forward
    captured: list[torch.Tensor] = []

    def wrapped(*args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            raise ValueError("Joint action forward did not receive position_ids")
        captured.append(position_ids.detach().to(device="cpu", dtype=torch.int64).clone())
        return original(*args, **kwargs)

    module.forward = wrapped
    try:
        yield captured
    finally:
        module.forward = original


def build_model(mode: str, *, model_config, checkpoint: Path):
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


def evaluate_configuration(
    *,
    name: str,
    model,
    fixed_batches: list[dict],
    selected: list[int],
    device: torch.device,
    blocked_groups: frozenset[str],
) -> dict:
    losses = []
    positions = []
    layout_record = None
    noise_digest = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats(device)
    with temporary_action_query_block(blocked_groups), capture_joint_position_ids(model) as captured:
        for ordinal, (_dataset_index, batch) in enumerate(zip(selected, fixed_batches, strict=True)):
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
            torch.manual_seed(sample_seed)
            captured_before = len(captured)
            with torch.no_grad():
                if name == "B0":
                    action_loss = model(
                        observation,
                        actions,
                        noise=noise,
                        time=diffusion_time,
                    ).mean()
                    result_layout = None
                else:
                    result = model.forward_with_aux(
                        observation,
                        actions,
                        targets,
                        noise=noise,
                        time=diffusion_time,
                    )
                    action_loss = result["losses"]["action"]
                    result_layout = result["layout"]
            if len(captured) != captured_before + 1:
                raise RuntimeError(f"{name} did not produce exactly one joint position-ID record")
            if not bool(torch.isfinite(action_loss)):
                raise ValueError(f"Non-finite action loss for {name}, ordinal={ordinal}")
            position_row = captured[-1]
            if position_row.shape[0] != 1:
                raise ValueError("Diagnostic expects batch size one")
            positions.append(position_row[0].tolist())
            losses.append(float(action_loss))
            if result_layout is not None:
                current_layout = {
                    "context": span_dict(result_layout.context),
                    "ground": span_dict(result_layout.ground),
                    "geometry": span_dict(result_layout.geometry),
                    "action_suffix": span_dict(result_layout.action_suffix),
                    "language": span_dict(result_layout.language),
                    "view_spans": {key: span_dict(value) for key, value in result_layout.view_spans.items()},
                    "real_view_names": list(result_layout.real_view_names),
                    "padded_view_names": list(result_layout.padded_view_names),
                }
                if layout_record is None:
                    layout_record = current_layout
                elif layout_record != current_layout:
                    raise RuntimeError(f"{name} token layout changed across fixed samples")

    if name == "B0":
        total_length = len(positions[0])
        layout_record = {
            "context": {"start": 0, "end": total_length - 10, "length": total_length - 10},
            "ground": None,
            "geometry": None,
            "action_suffix": {"start": total_length - 10, "end": total_length, "length": 10},
        }
    return {
        "configuration": name,
        "blocked_action_query_groups": sorted(blocked_groups),
        "action_loss": stats(losses),
        "per_sample_action_loss": losses,
        "position_ids": positions,
        "layout": layout_record,
        "noise_sha256": noise_digest.hexdigest(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def position_report(
    *,
    selected: list[int],
    fixed_batches: list[dict],
    b0: dict,
    p1: dict,
    p2: dict,
) -> dict:
    reports = {"aux_disabled": [], "p1": [], "p2": []}
    for ordinal, (dataset_index, batch) in enumerate(zip(selected, fixed_batches, strict=True)):
        valid_prompt_tokens = int(batch["tokenized_prompt_mask"].sum())
        mode_inputs = {
            "aux_disabled": b0,
            "p1": p1,
            "p2": p2,
        }
        for mode_name, source in mode_inputs.items():
            layout = source["layout"]
            all_ids = source["position_ids"][ordinal]
            context = layout["context"]
            action = layout["action_suffix"]
            ground = layout.get("ground")
            geometry = layout.get("geometry")
            row = {
                "ordinal": ordinal,
                "lerobot_dataset_index": dataset_index,
                "valid_prompt_tokens": valid_prompt_tokens,
                "context_valid_tokens": int(all_ids[action["start"]])
                - (0 if mode_name == "aux_disabled" else 8 if mode_name == "p1" else 16),
                "context_position_ids": all_ids[context["start"] : context["end"]],
                "prefix_position_ids": all_ids[: action["start"]],
                "ground_query_position_ids": (all_ids[ground["start"] : ground["end"]] if ground is not None else []),
                "geometry_query_position_ids": (
                    all_ids[geometry["start"] : geometry["end"]] if geometry is not None else []
                ),
                "action_position_ids": all_ids[action["start"] : action["end"]],
            }
            reports[mode_name].append(row)

    for ordinal in range(len(selected)):
        disabled = reports["aux_disabled"][ordinal]
        p1_row = reports["p1"][ordinal]
        p2_row = reports["p2"][ordinal]
        if not (disabled["context_position_ids"] == p1_row["context_position_ids"] == p2_row["context_position_ids"]):
            raise RuntimeError("Context position IDs changed after query insertion")
        if any(
            current - reference != 8
            for current, reference in zip(
                p1_row["action_position_ids"],
                disabled["action_position_ids"],
                strict=True,
            )
        ):
            raise RuntimeError("P1 action positions are not an exact +8 shift")
        if any(
            current - reference != 16
            for current, reference in zip(
                p2_row["action_position_ids"],
                disabled["action_position_ids"],
                strict=True,
            )
        ):
            raise RuntimeError("P2 action positions are not an exact +16 shift")
    return reports


def delta_report(current: list[float], baseline: list[float]) -> dict[str, float]:
    paired = [value - reference for value, reference in zip(current, baseline, strict=True)]
    baseline_mean = statistics.fmean(baseline)
    signed_mean_delta = statistics.fmean(current) - baseline_mean
    return {
        "signed_mean_delta": signed_mean_delta,
        "absolute_mean_delta": abs(signed_mean_delta),
        "signed_relative_mean_delta": signed_mean_delta / baseline_mean,
        "absolute_relative_mean_delta": abs(signed_mean_delta) / baseline_mean,
        "mean_absolute_paired_delta": statistics.fmean(abs(value) for value in paired),
        "max_absolute_paired_delta": max(abs(value) for value in paired),
    }


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
    parser.add_argument("--source-perturbation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)

    source_calibration = json.loads(args.source_calibration.read_text())
    source_perturbation = json.loads(args.source_perturbation.read_text())
    if source_calibration.get("status") != "PASS":
        raise ValueError("Source calibration is not PASS")
    if source_perturbation.get("status") != "PASS_AWAITING_HUMAN_LAMBDA_FREEZE":
        raise ValueError("Source perturbation diagnostic is not complete")
    selected = [int(value) for value in source_calibration["dataset_indices"]]
    if selected != source_perturbation["dataset_indices"] or len(selected) != 16:
        raise ValueError("Frozen 16-sample identities differ between source diagnostics")

    base = _config.get_config("pi05_libero")
    if not (base.model.pi05 and base.model.action_horizon == 10 and not base.model.discrete_state_input):
        raise ValueError("Official pi05_libero input semantics changed")
    data_factory = dataclasses.replace(
        base.data,
        assets=_config.AssetsConfig(assets_dir=str(args.libero_assets_root.resolve(strict=True))),
    )
    data_config = data_factory.create(Path("/nonexistent/assets_not_used"), base.model)

    target_frame = pd.read_parquet(args.geometry_index)
    valid = target_frame.loc[target_frame["geometry_valid"].astype(bool)]
    groups = [group for _, group in valid.groupby("task_id", sort=True)]
    recomputed = [int(group.iloc[0]["lerobot_dataset_index"]) for group in groups]
    recomputed.extend(int(group.iloc[len(group) // 2]["lerobot_dataset_index"]) for group in groups[:6])
    if recomputed != selected:
        raise ValueError("Fixed 16-sample selection differs from source calibration")

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
        fixed_batches.append(jax.tree.map(torch.as_tensor, _data_loader._collate_fn([item])))  # noqa: SLF001

    configurations = {}
    strict_loads = {}

    base_model, strict_loads["aux_disabled"] = build_model(
        "aux_disabled", model_config=base.model, checkpoint=args.checkpoint
    )
    base_model.to(device).eval()
    configurations["B0"] = evaluate_configuration(
        name="B0",
        model=base_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset(),
    )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    p1_model, strict_loads["p1"] = build_model("geometry", model_config=base.model, checkpoint=args.checkpoint)
    p1_model.to(device).eval()
    configurations["B1"] = evaluate_configuration(
        name="B1",
        model=p1_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset({"geometry"}),
    )
    configurations["B2"] = evaluate_configuration(
        name="B2",
        model=p1_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset(),
    )
    del p1_model
    gc.collect()
    torch.cuda.empty_cache()

    p2_model, strict_loads["p2"] = build_model(
        "ground_geometry_semantic_lm", model_config=base.model, checkpoint=args.checkpoint
    )
    p2_model.to(device).eval()
    configurations["B3"] = evaluate_configuration(
        name="B3",
        model=p2_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset({"ground", "geometry"}),
    )
    configurations["B4"] = evaluate_configuration(
        name="B4",
        model=p2_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset({"ground"}),
    )
    configurations["B5"] = evaluate_configuration(
        name="B5",
        model=p2_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset(),
    )
    del p2_model
    gc.collect()
    torch.cuda.empty_cache()

    if len({report["noise_sha256"] for report in configurations.values()}) != 1:
        raise RuntimeError("Noise tensors differ across B0-B5")
    if configurations["B1"]["position_ids"] != configurations["B2"]["position_ids"]:
        raise RuntimeError("P1 position IDs changed when only action-query attention changed")
    if not (
        configurations["B3"]["position_ids"]
        == configurations["B4"]["position_ids"]
        == configurations["B5"]["position_ids"]
    ):
        raise RuntimeError("P2 position IDs changed when only action-query attention changed")

    source_mode_map = {
        "B0": "aux_disabled",
        "B2": "geometry",
        "B5": "ground_geometry_semantic_lm",
    }
    source_reproduction = {}
    for config_name, source_name in source_mode_map.items():
        current = configurations[config_name]["per_sample_action_loss"]
        reference = source_perturbation["reports"][source_name]["per_sample_action_loss"]
        exact = current == reference
        source_reproduction[config_name] = exact
        if not exact:
            raise RuntimeError(f"{config_name} does not exactly reproduce source perturbation")

    positions = position_report(
        selected=selected,
        fixed_batches=fixed_batches,
        b0=configurations["B0"],
        p1=configurations["B2"],
        p2=configurations["B5"],
    )
    baseline = configurations["B0"]["per_sample_action_loss"]
    deltas = {name: delta_report(report["per_sample_action_loss"], baseline) for name, report in configurations.items()}
    incremental = {
        "p1_query_attention_B2_minus_B1": delta_report(
            configurations["B2"]["per_sample_action_loss"],
            configurations["B1"]["per_sample_action_loss"],
        ),
        "p2_geometry_attention_B4_minus_B3": delta_report(
            configurations["B4"]["per_sample_action_loss"],
            configurations["B3"]["per_sample_action_loss"],
        ),
        "p2_ground_attention_B5_minus_B4": delta_report(
            configurations["B5"]["per_sample_action_loss"],
            configurations["B4"]["per_sample_action_loss"],
        ),
    }

    payload = {
        "status": "DIAGNOSTIC_COMPLETE_AWAITING_HUMAN_REVIEW",
        "schema": "openpi.p1_p2_action_path_initialization_decomposition.v1",
        "scope": "fixed-input no-gradient causal decomposition; diagnostic masking only",
        "optimizer_constructed": False,
        "optimizer_steps_run": 0,
        "backward_calls": 0,
        "architecture_modified": False,
        "position_ids_modified": False,
        "query_initialization_modified": False,
        "persistent_attention_topology_modified": False,
        "motion_integrated": False,
        "lambda_freeze_blocked": True,
        "tiny_overfit_blocked": True,
        "checkpoint": str(args.checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_calibration": str(args.source_calibration.resolve(strict=True)),
        "source_calibration_sha256": sha256_file(args.source_calibration),
        "source_perturbation": str(args.source_perturbation.resolve(strict=True)),
        "source_perturbation_sha256": sha256_file(args.source_perturbation),
        "sample_count": len(selected),
        "dataset_indices": selected,
        "model_initialization_seed": MODEL_INITIALIZATION_SEED,
        "sample_seed_rule": f"{SAMPLE_SEED_BASE} + ordinal",
        "diffusion_timestep": DIFFUSION_TIMESTEP,
        "noise_sha256": configurations["B0"]["noise_sha256"],
        "strict_loads": strict_loads,
        "configuration_definitions": {
            "B0": "official aux-disabled",
            "B1": "P1 layout; Action->Geometry blocked",
            "B2": "normal P1; Action->Geometry enabled",
            "B3": "P2 layout; Action->Ground and Action->Geometry blocked",
            "B4": "P2 layout; Action->Geometry enabled; Action->Ground blocked",
            "B5": "normal P2; Action->Ground and Action->Geometry enabled",
        },
        "source_reproduction_bitwise_float_equal": source_reproduction,
        "configurations": configurations,
        "deltas_from_B0": deltas,
        "incremental_effects": incremental,
        "position_id_report": positions,
        "position_checks": {
            "context_ids_identical_across_B0_P1_P2": True,
            "B1_B2_ids_identical": True,
            "B3_B4_B5_ids_identical": True,
            "P1_action_ids_are_B0_plus_8": True,
            "P2_action_ids_are_B0_plus_16": True,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
