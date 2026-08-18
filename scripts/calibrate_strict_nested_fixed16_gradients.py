#!/usr/bin/env python3
"""Refresh P1/P2 component gradient scales on the strict-nested fixed-16 set.

For each component, this script accumulates gradients of ``loss / 16`` over
sixteen one-sample microbatches. It does not construct an optimizer, update any
parameter, approve a lambda, or run tiny-overfit.
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

from diagnose_p1_p2_action_path_initialization import DIFFUSION_TIMESTEP
from diagnose_p1_p2_action_path_initialization import MODEL_INITIALIZATION_SEED
from diagnose_p1_p2_action_path_initialization import SAMPLE_SEED_BASE
from diagnose_p1_p2_action_path_initialization import sha256_file
from diagnose_p1_p2_nested_initialization import compare_geometry_initialization
from diagnose_p1_p2_nested_initialization import geometry_snapshot
import jax
import pandas as pd
import torch
from validate_p1_p2_component_gradients import selected_parameters

from openpi.models import model as _model
from openpi.models_pytorch.pi05_aux_queries import GEOMETRY_HEAD_INIT_SEED
from openpi.models_pytorch.pi05_aux_queries import GEOMETRY_QUERY_INIT_SEED
from openpi.models_pytorch.pi05_aux_queries import GROUND_HEAD_INIT_SEED
from openpi.models_pytorch.pi05_aux_queries import GROUND_QUERY_INIT_SEED
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxTargets
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig
from openpi.training.policy_aux_dataset import PolicyAuxTransformedDataset

CONDITIONAL_CANDIDATE = {
    "lambda_geo": 0.5,
    "lambda_ground": 0.5,
    "lambda_sem": 0.05,
}


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


def noise_hash_for_fixed_samples(fixed_batches: list[dict], device: torch.device) -> str:
    digest = hashlib.sha256()
    for ordinal, batch in enumerate(fixed_batches):
        actions = batch["actions"]
        generator = torch.Generator(device=device).manual_seed(SAMPLE_SEED_BASE + ordinal)
        noise = torch.randn(actions.shape, generator=generator, device=device)
        digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_sample(batch: dict, *, ordinal: int, device: torch.device) -> tuple:
    observation = _model.Observation.from_dict(batch)
    observation = jax.tree.map(lambda value: value.to(device), observation)
    actions = batch["actions"].to(device=device, dtype=torch.float32)
    target_batch = jax.tree.map(lambda value: value.to(device), batch["policy_aux"])
    targets = to_targets(target_batch)
    generator = torch.Generator(device=device).manual_seed(SAMPLE_SEED_BASE + ordinal)
    noise = torch.randn(actions.shape, generator=generator, device=device)
    diffusion_time = torch.full((1,), DIFFUSION_TIMESTEP, dtype=torch.float32, device=device)
    return observation, actions, targets, noise, diffusion_time


def assert_layout(mode: str, layout) -> dict[str, dict[str, int] | None]:
    record = {
        "context": span_dict(layout.context),
        "geometry": span_dict(layout.geometry),
        "ground": span_dict(layout.ground),
        "action_suffix": span_dict(layout.action_suffix),
    }
    if mode == "geometry":
        passed = (
            layout.context.end == layout.geometry.start
            and layout.geometry.end == layout.action_suffix.start
            and layout.ground is None
        )
    else:
        passed = (
            layout.context.end == layout.geometry.start
            and layout.geometry.end == layout.ground.start
            and layout.ground.end == layout.action_suffix.start
        )
    if not passed:
        raise RuntimeError(f"Strict-nested layout changed for {mode}: {record}")
    return record


def calibrate_mode(
    *,
    mode: str,
    model,
    fixed_batches: list[dict],
    device: torch.device,
) -> dict:
    parameters = selected_parameters(model)
    loss_order = ["action", "geometry"]
    if mode == "ground_geometry_semantic_lm":
        loss_order.extend(("ground", "semantic"))
    accumulated = {
        loss_name: {
            label: torch.zeros_like(parameter, dtype=torch.float32, device=device)
            for label, parameter in parameters.items()
        }
        for loss_name in loss_order
    }
    raw_losses = {loss_name: [] for loss_name in loss_order}
    layout_record = None
    parameter_versions = {label: parameter._version for label, parameter in parameters.items()}  # noqa: SLF001
    torch.cuda.reset_peak_memory_stats(device)

    for ordinal, batch in enumerate(fixed_batches):
        observation, actions, targets, noise, diffusion_time = prepare_sample(batch, ordinal=ordinal, device=device)
        torch.manual_seed(SAMPLE_SEED_BASE + ordinal)
        result = model.forward_with_aux(
            observation,
            actions,
            targets,
            noise=noise,
            time=diffusion_time,
        )
        if layout_record is None:
            layout_record = assert_layout(mode, result["layout"])
        for loss_name in loss_order:
            loss = result["losses"][loss_name]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite {mode}/{loss_name} loss at ordinal {ordinal}")
            raw_losses[loss_name].append(float(loss.detach()))

        for loss_index, loss_name in enumerate(loss_order):
            gradients = torch.autograd.grad(
                result["losses"][loss_name] / len(fixed_batches),
                tuple(parameters.values()),
                retain_graph=loss_index < len(loss_order) - 1,
                allow_unused=True,
            )
            for (label, _parameter), gradient in zip(parameters.items(), gradients, strict=True):
                if gradient is not None:
                    accumulated[loss_name][label].add_(gradient.detach().float())

        print(
            f"{mode}: accumulated microbatch {ordinal + 1}/{len(fixed_batches)}",
            flush=True,
        )
        del result, observation, actions, targets, noise, diffusion_time, gradients

    current_versions = {label: parameter._version for label, parameter in parameters.items()}  # noqa: SLF001
    if current_versions != parameter_versions:
        raise RuntimeError("A representative parameter changed during no-optimizer calibration")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("autograd.grad unexpectedly populated model .grad fields")

    norms = {
        loss_name: {label: float(gradient.norm()) for label, gradient in component_gradients.items()}
        for loss_name, component_gradients in accumulated.items()
    }
    return {
        "gradient_definition": "L2 norm of FP32 accumulation of grad(loss_i / 16)",
        "microbatch_size": 1,
        "microbatch_count": len(fixed_batches),
        "raw_loss_statistics": {loss_name: stats(values) for loss_name, values in raw_losses.items()},
        "gradient_norms": norms,
        "layout": layout_record,
        "all_representative_parameter_versions_unchanged": True,
        "all_model_grad_fields_none_after_calibration": True,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def render_markdown(payload: dict) -> str:
    reports = payload["reports"]
    lines = [
        "# P1/P2 strict-nested fixed-16 gradient calibration",
        "",
        "Status: **PASS — AWAITING HUMAN LAMBDA APPROVAL**",
        "",
        "No optimizer was constructed, no parameter was updated, and tiny-overfit was not started.",
        "Each reported norm is the L2 norm after FP32 accumulation of `grad(loss_i / 16)`",
        "over the same sixteen one-sample microbatches.",
        "",
        "## Raw fixed-16 mean losses",
        "",
        "| Variant | Action | Geometry | Grounding | Semantic |",
        "|---|---:|---:|---:|---:|",
    ]
    p1_losses = reports["geometry"]["raw_loss_statistics"]
    p2_losses = reports["ground_geometry_semantic_lm"]["raw_loss_statistics"]
    lines.extend(
        (
            f"| P1 | {p1_losses['action']['mean']:.15g} | {p1_losses['geometry']['mean']:.15g} | -- | -- |",
            f"| P2 | {p2_losses['action']['mean']:.15g} | "
            f"{p2_losses['geometry']['mean']:.15g} | "
            f"{p2_losses['ground']['mean']:.15g} | "
            f"{p2_losses['semantic']['mean']:.15g} |",
            "",
            "## Component gradient norms",
            "",
            "| Variant | Loss | Shared vision | Shared language | Action expert | Action output | Geometry queries | Geometry head | Ground queries | Ground head |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    labels = (
        "shared_vision",
        "shared_language",
        "action_expert",
        "action_output",
        "geometry_queries",
        "geometry_head",
        "ground_queries",
        "ground_head",
    )
    for variant, mode in (
        ("P1", "geometry"),
        ("P2", "ground_geometry_semantic_lm"),
    ):
        for loss_name, values in reports[mode]["gradient_norms"].items():
            cells = [variant, loss_name.capitalize()]
            cells.extend("--" if label not in values else repr(values[label]) for label in labels)
            lines.append("| " + " | ".join(cells) + " |")
    candidate = payload["conditional_candidate_for_comparison_only"]
    lines.extend(
        (
            "",
            "## Human decision gate",
            "",
            "Conditional candidate recorded for comparison only; it is **not approved or applied**:",
            "",
            "```text",
            f"lambda_geo    = {candidate['lambda_geo']}",
            f"lambda_ground = {candidate['lambda_ground']}",
            f"lambda_sem    = {candidate['lambda_sem']}",
            "```",
            "",
            "Execution is stopped before any optimizer step or fixed-16 tiny-overfit.",
        )
    )
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--source-strict-nested", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)

    source_calibration = json.loads(args.source_calibration.read_text())
    source_strict_nested = json.loads(args.source_strict_nested.read_text())
    if source_calibration.get("status") != "PASS":
        raise ValueError("Source calibration is not PASS")
    if source_strict_nested.get("status") != "DIAGNOSTIC_COMPLETE_AWAITING_HUMAN_REVIEW":
        raise ValueError("Strict-nested diagnostic is not complete")
    selected = [int(value) for value in source_calibration["dataset_indices"]]
    if selected != source_strict_nested["dataset_indices"] or len(selected) != 16:
        raise ValueError("The fixed-16 identities differ between source artifacts")

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
        raise ValueError("Fixed-16 selection differs from the source calibration")

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
    fixed_batches = []
    for dataset_index in selected:
        item = dataset[dataset_index]
        fixed_batches.append(
            jax.tree.map(torch.as_tensor, _data_loader._collate_fn([item]))  # noqa: SLF001
        )

    reports = {}
    strict_loads = {}
    p1_geometry = None
    initialization_checks = None
    for mode in ("geometry", "ground_geometry_semantic_lm"):
        torch.manual_seed(MODEL_INITIALIZATION_SEED)
        model = PI05AuxPolicy(
            base.model,
            PolicyAuxConfig(
                mode=mode,
                lambda_geo=1.0,
                lambda_ground=1.0 if mode == "ground_geometry_semantic_lm" else None,
                lambda_sem=1.0 if mode == "ground_geometry_semantic_lm" else None,
            ),
        )
        strict_loads[mode] = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
        if mode == "geometry":
            p1_geometry = geometry_snapshot(model)
        else:
            initialization_checks = compare_geometry_initialization(p1_geometry, model)
            del p1_geometry
        model.to(device).train()
        model.gradient_checkpointing_enable()
        reports[mode] = calibrate_mode(
            mode=mode,
            model=model,
            fixed_batches=fixed_batches,
            device=device,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    p1_layout = reports["geometry"]["layout"]
    p2_layout = reports["ground_geometry_semantic_lm"]["layout"]
    if p1_layout["geometry"] != p2_layout["geometry"]:
        raise RuntimeError("P1/P2 Geometry spans differ in refreshed calibration")

    payload = {
        "status": "PASS_AWAITING_HUMAN_LAMBDA_APPROVAL",
        "schema": "openpi.p1_p2_strict_nested_fixed16_gradient_calibration.v1",
        "scope": "unweighted component gradients of fixed-16 mean losses",
        "optimizer_imported_or_constructed": False,
        "optimizer_steps_run": 0,
        "parameters_updated": False,
        "tiny_overfit_started": False,
        "lambda_values_approved": False,
        "lambda_values_written_to_training_config": False,
        "action_query_gate_added": False,
        "query_warmup_added": False,
        "auxiliary_pretraining_used": False,
        "external_data_used": False,
        "motion_integrated": False,
        "component_losses_differentiated_unweighted": True,
        "conditional_candidate_for_comparison_only": CONDITIONAL_CANDIDATE,
        "checkpoint": str(args.checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_calibration": str(args.source_calibration.resolve(strict=True)),
        "source_calibration_sha256": sha256_file(args.source_calibration),
        "source_strict_nested": str(args.source_strict_nested.resolve(strict=True)),
        "source_strict_nested_sha256": sha256_file(args.source_strict_nested),
        "sample_count": len(selected),
        "dataset_indices": selected,
        "microbatch_size": 1,
        "microbatch_count": len(selected),
        "gradient_accumulation": "for each loss: sum_i grad(loss_i / 16) in FP32",
        "model_initialization_seed": MODEL_INITIALIZATION_SEED,
        "branch_initialization_seeds": {
            "geometry_queries": GEOMETRY_QUERY_INIT_SEED,
            "geometry_head": GEOMETRY_HEAD_INIT_SEED,
            "ground_queries": GROUND_QUERY_INIT_SEED,
            "ground_head": GROUND_HEAD_INIT_SEED,
        },
        "sample_seed_rule": f"{SAMPLE_SEED_BASE} + ordinal",
        "diffusion_timestep": DIFFUSION_TIMESTEP,
        "noise_sha256": noise_hash_for_fixed_samples(fixed_batches, device),
        "strict_loads": strict_loads,
        "initialization_checks": initialization_checks,
        "geometry_spans_identical_between_p1_p2": True,
        "representative_parameter_groups": {
            "shared_vision": ("paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"),
            "shared_language": ("paligemma.model.language_model.layers.0.self_attn.q_proj.weight"),
            "action_expert": "gemma_expert.model.layers.0.self_attn.q_proj.weight",
            "action_output": "action_out_proj.weight",
            "geometry_queries": "geometry_queries",
            "geometry_head": "geometry_head.output_projection.weight",
            "ground_queries": "ground_queries",
            "ground_head": "ground_head.query_projection.weight",
        },
        "reports": reports,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
