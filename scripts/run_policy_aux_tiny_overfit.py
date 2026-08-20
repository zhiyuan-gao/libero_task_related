#!/usr/bin/env python3
"""Run an explicitly approved fixed-16-sample P1 or P2 engineering overfit gate."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import statistics
import time

from diagnose_p1_p2_action_path_initialization import sha256_file
from diagnose_p1_p2_action_path_initialization import temporary_action_query_block
import jax
import pandas as pd
from PIL import Image
import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models import model as _model
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxTargets
from openpi.models_pytorch.policy_aux_preprocessing import preprocess_observation_and_ground_masks_pytorch
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig
from openpi.training.policy_aux_dataset import PolicyAuxTransformedDataset

FROZEN_LAMBDA_GEO = 0.15
FROZEN_LAMBDA_GROUND = 0.50
FROZEN_LAMBDA_SEM = 0.01
FIXED_SEED = 20260818


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


def resolve_parameter(model: torch.nn.Module, suffix: str) -> torch.nn.Parameter:
    matches = [parameter for name, parameter in model.named_parameters() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Could not resolve parameter suffix {suffix}: {len(matches)} matches")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("geometry", "semantic_geometry", "ground_geometry_semantic_lm"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--libero-assets-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--geometry-index", type=Path, required=True)
    parser.add_argument("--geometry-normalization", type=Path, required=True)
    parser.add_argument("--source-gradient-calibration", type=Path, required=True)
    parser.add_argument("--lambda-geo", type=float, required=True)
    parser.add_argument("--lambda-sem", type=float)
    parser.add_argument("--lambda-ground", type=float)
    parser.add_argument("--lambda-values-approved", action="store_true")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.lambda_values_approved:
        raise RuntimeError("Tiny overfit performs optimizer steps and requires explicit lambda approval")
    if args.lambda_geo <= 0 or args.learning_rate <= 0 or args.updates <= 0:
        raise ValueError("Geometry lambda, learning rate, and update count are invalid")
    if args.mode in ("semantic_geometry", "ground_geometry_semantic_lm") and args.lambda_sem is None:
        raise ValueError("Semantic tiny overfit requires lambda-sem")
    if args.mode == "ground_geometry_semantic_lm" and args.lambda_ground is None:
        raise ValueError("P2 tiny overfit requires lambda-sem and lambda-ground")
    if any(value is not None and value <= 0 for value in (args.lambda_sem, args.lambda_ground)):
        raise ValueError("P2 lambdas must be non-negative")
    expected_lambdas = {
        "lambda_geo": FROZEN_LAMBDA_GEO,
        "lambda_ground": (FROZEN_LAMBDA_GROUND if args.mode == "ground_geometry_semantic_lm" else None),
        "lambda_sem": (
            FROZEN_LAMBDA_SEM if args.mode in ("semantic_geometry", "ground_geometry_semantic_lm") else None
        ),
    }
    supplied_lambdas = {
        "lambda_geo": args.lambda_geo,
        "lambda_ground": args.lambda_ground,
        "lambda_sem": args.lambda_sem,
    }
    if supplied_lambdas != expected_lambdas:
        raise RuntimeError(
            f"Tiny-overfit lambdas differ from human-frozen values: "
            f"supplied={supplied_lambdas}, expected={expected_lambdas}"
        )

    started = time.monotonic()
    device = torch.device(args.device)
    base = _config.get_config("pi05_libero")
    data_factory = dataclasses.replace(
        base.data,
        assets=_config.AssetsConfig(assets_dir=str(args.libero_assets_root.resolve(strict=True))),
    )
    data_config = data_factory.create(Path("/nonexistent/assets_not_used"), base.model)
    source_gradient_calibration = json.loads(args.source_gradient_calibration.read_text())
    if source_gradient_calibration.get("status") != "PASS_AWAITING_HUMAN_LAMBDA_APPROVAL":
        raise RuntimeError("Strict-nested fixed-16 gradient calibration is not PASS")
    manifest = pd.read_parquet(args.policy_manifest).set_index("lerobot_dataset_index")
    if args.mode == "semantic_geometry":
        # Deterministic, task-balanced coverage of the exact A population.
        selected = []
        for task_index, count in ((0, 6), (3, 5), (8, 5)):
            candidates = manifest.loc[
                manifest["lerobot_task_index"].eq(task_index) & manifest["geometry_valid"].astype(bool)
            ].sort_index()
            positions = torch.linspace(0, len(candidates) - 1, steps=count).round().to(torch.int64).tolist()
            selected.extend(int(candidates.index[position]) for position in positions)
    else:
        selected = [int(value) for value in source_gradient_calibration["dataset_indices"]]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise RuntimeError("Tiny-overfit selection does not contain 16 unique samples")
    selected_rows = manifest.loc[selected]
    required_validity = selected_rows["geometry_valid"].astype(bool)
    if args.mode == "ground_geometry_semantic_lm":
        # Grounding is explicitly masked per view. A sample remains valid when
        # at least one real policy view has a target; invalid views contribute
        # neither loss nor metrics.
        required_validity &= selected_rows["agent_mask_valid"].astype(bool) | selected_rows["wrist_mask_valid"].astype(
            bool
        )
    if not bool(required_validity.all()):
        raise RuntimeError("A frozen tiny-overfit sample lacks a required auxiliary target")

    aux_train_config = PolicyAuxTrainConfig(
        mode=args.mode,
        policy_manifest_path=str(args.policy_manifest.resolve(strict=True)),
        episode_mapping_path=str(args.mapping.resolve(strict=True)),
        geometry_target_index_path=str(args.geometry_index.resolve(strict=True)),
        geometry_normalization_path=str(args.geometry_normalization.resolve(strict=True)),
        lambda_geo=args.lambda_geo,
        lambda_sem=args.lambda_sem,
        lambda_ground=args.lambda_ground,
        num_ground_queries=0 if args.mode == "semantic_geometry" else 8,
        lerobot_root=str(args.lerobot_root.resolve(strict=True)),
        lerobot_task_indices=(0, 3, 8) if args.mode == "semantic_geometry" else None,
        loss_coefficients_approved=True,
    )
    raw_dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=10,
        model_config=base.model,
        policy_aux_config=aux_train_config,
    )
    transformed = _data_loader.transform_dataset(raw_dataset, data_config)
    dataset = PolicyAuxTransformedDataset(transformed, aux_train_config)

    aux_model_config = PolicyAuxConfig(
        mode=args.mode,
        num_ground_queries=0 if args.mode == "semantic_geometry" else 8,
        lambda_geo=args.lambda_geo,
        lambda_sem=args.lambda_sem,
        lambda_ground=args.lambda_ground,
    )
    torch.manual_seed(FIXED_SEED)
    model = PI05AuxPolicy(base.model, aux_model_config)
    strict_load = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
    model.to(device)
    model.gradient_checkpointing_enable()

    subset_dataset_indices = aux_train_config.lerobot_dataset_indices()
    full_to_local = {dataset_index: local_index for local_index, dataset_index in enumerate(subset_dataset_indices)}
    cached_batches = []
    for dataset_index in selected:
        batch = _data_loader._collate_fn([dataset[full_to_local[dataset_index]]])  # noqa: SLF001
        cached_batches.append(jax.tree.map(torch.as_tensor, batch))

    def prepared(ordinal: int):
        batch = cached_batches[ordinal]
        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda value: value.to(device), observation)
        actions = batch["actions"].to(device=device, dtype=torch.float32)
        target_batch = jax.tree.map(lambda value: value.to(device), batch["policy_aux"])
        targets = to_targets(target_batch)
        generator = torch.Generator(device=device).manual_seed(FIXED_SEED + ordinal)
        noise = torch.randn(actions.shape, generator=generator, device=device)
        diffusion_time = torch.full((1,), 0.5, dtype=torch.float32, device=device)
        return observation, actions, targets, noise, diffusion_time

    def evaluate(*, blocked_groups: frozenset[str] = frozenset()) -> dict[str, float]:
        was_training = model.training
        model.eval()
        values: dict[str, list[float]] = {}
        with temporary_action_query_block(blocked_groups), torch.no_grad():
            for ordinal in range(len(selected)):
                observation, actions, targets, noise, diffusion_time = prepared(ordinal)
                torch.manual_seed(FIXED_SEED + ordinal)
                result = model.forward_with_aux(
                    observation,
                    actions,
                    targets,
                    noise=noise,
                    time=diffusion_time,
                )
                for name, value in result["losses"].items():
                    values.setdefault(f"loss_{name}", []).append(float(value))
                for name, value in result["diagnostics"].items():
                    if value.ndim == 0:
                        values.setdefault(name, []).append(float(value))
        model.train(was_training)
        means = {name: statistics.fmean(items) for name, items in values.items()}
        if not all(torch.isfinite(torch.tensor(value)) for value in means.values()):
            raise RuntimeError(f"Non-finite fixed-16 evaluation metrics: {means}")
        return means

    tracked_gradient_parameters = {
        "geometry_queries": resolve_parameter(model, "geometry_queries"),
        "geometry_head": resolve_parameter(model, "geometry_head.output_projection.weight"),
    }
    if args.mode == "ground_geometry_semantic_lm":
        tracked_gradient_parameters.update(
            {
                "ground_queries": resolve_parameter(model, "ground_queries"),
                "ground_head": resolve_parameter(model, "ground_head.query_projection.weight"),
            }
        )

    def current_tracked_gradient_norms() -> dict[str, float]:
        return {
            name: (0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm()))
            for name, parameter in tracked_gradient_parameters.items()
        }

    def fixed16_total_gradient_norms() -> dict[str, float]:
        was_training = model.training
        model.train()
        accumulated = {
            name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
            for name, parameter in tracked_gradient_parameters.items()
        }
        model.zero_grad(set_to_none=True)
        for ordinal in range(len(selected)):
            observation, actions, targets, noise, diffusion_time = prepared(ordinal)
            torch.manual_seed(FIXED_SEED + ordinal)
            result = model.forward_with_aux(
                observation,
                actions,
                targets,
                noise=noise,
                time=diffusion_time,
            )
            gradients = torch.autograd.grad(
                result["losses"]["total"] / len(selected),
                tuple(tracked_gradient_parameters.values()),
                allow_unused=True,
            )
            for (name, _parameter), gradient in zip(tracked_gradient_parameters.items(), gradients, strict=True):
                if gradient is not None:
                    accumulated[name].add_(gradient.detach().float())
        model.train(was_training)
        model.zero_grad(set_to_none=True)
        norms = {name: float(gradient.norm()) for name, gradient in accumulated.items()}
        if not all(torch.isfinite(torch.tensor(value)) for value in norms.values()):
            raise RuntimeError(f"Non-finite final fixed-16 query/head gradient norms: {norms}")
        return norms

    torch.cuda.reset_peak_memory_stats(device)
    initial = evaluate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    action_expert_parameter = resolve_parameter(model, "gemma_expert.model.layers.0.self_attn.q_proj.weight")
    first_action_expert_grad_norm = None
    query_head_gradient_trace = []
    trace = []
    fixed16_evaluation_trajectory = [
        {
            "update": 0,
            "loss_action": initial["loss_action"],
            "loss_total": initial["loss_total"],
        }
    ]
    evaluation_updates = {update for update in (1, 2, 3, 5, 10, 20, 50, args.updates) if update <= args.updates}
    trace_updates = {
        update for update in range(1, args.updates + 1) if update <= 20 or update % 10 == 0 or update == args.updates
    }
    model.train()
    for update in range(args.updates):
        ordinal = update % len(selected)
        dataset_index = selected[ordinal]
        observation, actions, targets, noise, diffusion_time = prepared(ordinal)
        torch.manual_seed(FIXED_SEED + ordinal)
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_with_aux(
            observation,
            actions,
            targets,
            noise=noise,
            time=diffusion_time,
        )
        total = result["losses"]["total"]
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"Non-finite tiny-overfit loss at update {update}")
        total.backward()
        if update == 0:
            if action_expert_parameter.grad is None:
                raise RuntimeError("Action expert has no gradient on the first tiny-overfit update")
            first_action_expert_grad_norm = float(action_expert_parameter.grad.float().norm())
        tracked_norms = current_tracked_gradient_norms()
        if not all(torch.isfinite(torch.tensor(value)) for value in tracked_norms.values()):
            raise RuntimeError(f"Non-finite query/head gradient at update {update + 1}: {tracked_norms}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"Non-finite global gradient norm at update {update + 1}")
        optimizer.step()
        step = update + 1
        if step in trace_updates:
            trace.append(
                {
                    "update": step,
                    "sample_ordinal": ordinal,
                    "lerobot_dataset_index": dataset_index,
                    "loss_total": float(total.detach()),
                    "loss_action": float(result["losses"]["action"].detach()),
                    "loss_geometry": float(result["losses"]["geometry"].detach()),
                    "loss_ground": (
                        float(result["losses"]["ground"].detach()) if "ground" in result["losses"] else None
                    ),
                    "loss_semantic": (
                        float(result["losses"]["semantic"].detach()) if "semantic" in result["losses"] else None
                    ),
                    "grad_norm": float(grad_norm),
                }
            )
            query_head_gradient_trace.append({"update": step, "sample_ordinal": ordinal, **tracked_norms})
            print(
                f"{args.mode}: update {step}/{args.updates} "
                f"action={float(result['losses']['action'].detach()):.6f} "
                f"total={float(total.detach()):.6f}",
                flush=True,
            )
        if step in evaluation_updates:
            checkpoint_evaluation = evaluate()
            fixed16_evaluation_trajectory.append(
                {
                    "update": step,
                    "loss_action": checkpoint_evaluation["loss_action"],
                    "loss_total": checkpoint_evaluation["loss_total"],
                }
            )
            print(
                f"{args.mode}: fixed16@{step} "
                f"action={checkpoint_evaluation['loss_action']:.6f} "
                f"total={checkpoint_evaluation['loss_total']:.6f}",
                flush=True,
            )

    model_output = (
        args.model_output
        if args.model_output is not None
        else args.output.with_name(f"{args.output.stem}_model.safetensors")
    )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, model_output)
    print(f"{args.mode}: saved trained model to {model_output}", flush=True)

    final = evaluate()
    blocked_groups = (
        frozenset({"geometry"})
        if args.mode in ("geometry", "semantic_geometry")
        else frozenset({"geometry", "ground"})
    )
    trained_state_versions = {
        name: parameter._version  # noqa: SLF001
        for name, parameter in tracked_gradient_parameters.items()
    }
    final_blocked = evaluate(blocked_groups=blocked_groups)
    if trained_state_versions != {
        name: parameter._version  # noqa: SLF001
        for name, parameter in tracked_gradient_parameters.items()
    }:
        raise RuntimeError("Normal/blocked terminal evaluation changed trained parameters")
    final_fixed16_query_head_gradient_norms = fixed16_total_gradient_norms()
    action_block_delta = final_blocked["loss_action"] - final["loss_action"]
    action_read_comparison = {
        "normal_action_loss": final["loss_action"],
        "blocked_action_loss": final_blocked["loss_action"],
        "blocked_minus_normal_signed_delta": action_block_delta,
        "absolute_delta": abs(action_block_delta),
        "blocked_minus_normal_relative_delta": action_block_delta / final["loss_action"],
        "same_trained_state_no_retraining": True,
        "temporarily_blocked_groups": sorted(blocked_groups),
    }
    visual_output = None
    if args.mode == "ground_geometry_semantic_lm":
        visual_output = args.output.with_name(f"{args.output.stem}_grounding_prediction_audit.png")
        observation, actions, targets, noise, diffusion_time = prepared(0)
        visual_seed = FIXED_SEED
        torch.manual_seed(visual_seed)
        synchronized = preprocess_observation_and_ground_masks_pytorch(
            observation,
            targets.ground_masks,
            train=True,
        )
        torch.manual_seed(visual_seed)
        with torch.no_grad():
            visual_result = model.forward_with_aux(
                observation,
                actions,
                targets,
                noise=noise,
                time=diffusion_time,
            )
        probabilities = visual_result["diagnostics"]["ground_logits"].float().sigmoid()
        rows = []
        for view_index, view_name in enumerate(("base_0_rgb", "left_wrist_0_rgb")):
            rgb_tensor = synchronized.observation.images[view_name][0].detach().float().cpu()
            if rgb_tensor.shape[0] == 3:
                rgb_tensor = rgb_tensor.permute(1, 2, 0)
            rgb = ((rgb_tensor.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).numpy()
            gt = synchronized.ground_masks[view_name][0].detach().float().cpu().clamp(0, 1)
            prediction = F.interpolate(
                probabilities[0, view_index].reshape(1, 1, 16, 16),
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu()
            gt_overlay = torch.from_numpy(rgb).float()
            gt_overlay[..., 1] = gt_overlay[..., 1] * (1 - 0.55 * gt) + 255 * 0.55 * gt
            prediction_overlay = torch.from_numpy(rgb).float()
            prediction_overlay[..., 0] = prediction_overlay[..., 0] * (1 - 0.55 * prediction) + 255 * 0.55 * prediction
            rows.append(
                torch.cat(
                    (
                        torch.from_numpy(rgb),
                        gt_overlay.clamp(0, 255).to(torch.uint8),
                        prediction_overlay.clamp(0, 255).to(torch.uint8),
                    ),
                    dim=1,
                )
            )
        canvas = torch.cat(rows, dim=0).numpy()
        visual_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(visual_output)
    required_losses = ["loss_geometry"]
    if args.mode == "semantic_geometry":
        required_losses.append("loss_semantic")
    elif args.mode == "ground_geometry_semantic_lm":
        required_losses.extend(("loss_ground", "loss_semantic"))
    all_evaluation_values = (
        *initial.values(),
        *final.values(),
        *final_blocked.values(),
        *action_read_comparison.values(),
    )
    numeric_evaluation_values = [
        value for value in all_evaluation_values if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    checks = {
        "initial_and_final_action_finite": all(
            torch.isfinite(torch.tensor(value)) for value in (initial["loss_action"], final["loss_action"])
        ),
        "action_expert_ran_and_received_gradient": (
            first_action_expert_grad_norm is not None and first_action_expert_grad_norm > 0
        ),
        "enabled_auxiliary_losses_drop_at_least_20_percent": all(
            final[name] <= 0.8 * initial[name] for name in required_losses
        ),
        "all_reported_metrics_finite": all(torch.isfinite(torch.tensor(value)) for value in numeric_evaluation_values),
        "all_training_trace_losses_and_grad_norms_finite": all(
            torch.isfinite(torch.tensor(value))
            for row in trace
            for key, value in row.items()
            if value is not None and (key.startswith("loss_") or key == "grad_norm")
        ),
        "all_query_head_gradient_norms_finite": all(
            torch.isfinite(torch.tensor(value))
            for row in query_head_gradient_trace
            for key, value in row.items()
            if key not in ("update", "sample_ordinal")
        )
        and all(torch.isfinite(torch.tensor(value)) for value in final_fixed16_query_head_gradient_norms.values()),
        "normal_blocked_same_trained_state": action_read_comparison["same_trained_state_no_retraining"],
    }
    status = "PASS" if all(checks.values()) else "FAIL_AWAITING_HUMAN_REVIEW"
    payload = {
        "status": status,
        "gate": f"pi05_{args.mode}_fixed16_tiny_overfit_v1",
        "scope": "engineering gate only; no policy-quality claim",
        "mode": args.mode,
        "architecture": {
            "geometry": "Context|Geometryx8|Action",
            "semantic_geometry": (
                "P2 joint-masked Context|Geometryx8|SemanticTeacher + Action suffix, with Ground removed"
            ),
            "ground_geometry_semantic_lm": (
                "Context|Geometryx8|Groundx8|Action + native semantic LM"
            ),
        }[args.mode],
        "strict_load": strict_load,
        "source_gradient_calibration": str(args.source_gradient_calibration.resolve(strict=True)),
        "source_gradient_calibration_sha256": sha256_file(args.source_gradient_calibration),
        "trained_model": str(model_output.resolve(strict=True)),
        "trained_model_bytes": model_output.stat().st_size,
        "trained_model_sha256": sha256_file(model_output),
        "dataset_indices": selected,
        "fixed_sample_noise_seed_rule": f"{FIXED_SEED} + sample ordinal",
        "same_fixed_samples_and_noise_for_all_evaluations": True,
        "updates": args.updates,
        "optimizer_steps_run": args.updates,
        "learning_rate": args.learning_rate,
        "loss_coefficients_approved": True,
        "approved_lambdas": {
            "lambda_geo": args.lambda_geo,
            "lambda_sem": args.lambda_sem,
            "lambda_ground": args.lambda_ground,
        },
        "initial": initial,
        "final_normal": final,
        "final_blocked": final_blocked,
        "final_action_query_read_comparison": action_read_comparison,
        "relative_changes": {
            name: (final[name] - initial[name]) / initial[name]
            for name in ("loss_total", "loss_action", *required_losses)
        },
        "first_action_expert_grad_norm": first_action_expert_grad_norm,
        "query_head_gradient_norm_trace": query_head_gradient_trace,
        "final_fixed16_mean_total_loss_query_head_gradient_norms": (final_fixed16_query_head_gradient_norms),
        "fixed16_evaluation_trajectory": fixed16_evaluation_trajectory,
        "checks": checks,
        "trace": trace,
        "grounding_prediction_audit": str(visual_output) if visual_output is not None else None,
        "grounding_prediction_audit_layout": (
            "rows=agent,wrist; columns=policy RGB, green GT coverage, red predicted probability"
            if visual_output is not None
            else None
        ),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "action_query_gate_added": False,
        "query_warmup_added": False,
        "query_pretraining_used": False,
        "auxiliary_pretraining_used": False,
        "external_data_used": False,
        "motion_integrated": False,
        "full_8gpu_training_launched": False,
        "stop_for_human_review": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
