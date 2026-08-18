#!/usr/bin/env python3
"""No-gradient sanity diagnostic for strict P1 -> P2 nesting.

This diagnostic verifies bitwise-shared Geometry initialization and identical
Geometry positions, then measures C0--C3 action losses. It never constructs an
optimizer, calls backward, or changes lambda coefficients.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import time

from diagnose_p1_p2_action_path_initialization import DIFFUSION_TIMESTEP
from diagnose_p1_p2_action_path_initialization import MODEL_INITIALIZATION_SEED
from diagnose_p1_p2_action_path_initialization import SAMPLE_SEED_BASE
from diagnose_p1_p2_action_path_initialization import build_model
from diagnose_p1_p2_action_path_initialization import delta_report
from diagnose_p1_p2_action_path_initialization import evaluate_configuration
from diagnose_p1_p2_action_path_initialization import sha256_file
import jax
import pandas as pd
import torch

from openpi.models_pytorch.pi05_aux_queries import GEOMETRY_HEAD_INIT_SEED
from openpi.models_pytorch.pi05_aux_queries import GEOMETRY_QUERY_INIT_SEED
from openpi.models_pytorch.pi05_aux_queries import GROUND_HEAD_INIT_SEED
from openpi.models_pytorch.pi05_aux_queries import GROUND_QUERY_INIT_SEED
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig
from openpi.training.policy_aux_dataset import PolicyAuxTransformedDataset


def tensor_mapping_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def geometry_snapshot(model) -> dict[str, torch.Tensor]:
    snapshot = {"geometry_queries": model.geometry_queries.detach().cpu().clone()}
    snapshot.update(
        {
            f"geometry_head.{name}": value.detach().cpu().clone()
            for name, value in model.geometry_head.state_dict().items()
        }
    )
    return snapshot


def compare_geometry_initialization(p1_snapshot: dict[str, torch.Tensor], p2_model) -> dict:
    p2_snapshot = geometry_snapshot(p2_model)
    if p1_snapshot.keys() != p2_snapshot.keys():
        raise RuntimeError("P1/P2 Geometry state keys differ")
    per_tensor = {name: bool(torch.equal(p1_snapshot[name], p2_snapshot[name])) for name in p1_snapshot}
    queries_equal = per_tensor["geometry_queries"]
    head_equal = all(equal for name, equal in per_tensor.items() if name.startswith("geometry_head."))
    if not queries_equal or not head_equal:
        raise RuntimeError(
            "P1/P2 Geometry initialization is not bitwise identical: "
            f"queries={queries_equal}, head={head_equal}, per_tensor={per_tensor}"
        )
    p1_hash = tensor_mapping_sha256(p1_snapshot)
    p2_hash = tensor_mapping_sha256(p2_snapshot)
    if p1_hash != p2_hash:
        raise RuntimeError("P1/P2 Geometry hashes differ despite tensor comparison")
    return {
        "p1_p2_geometry_queries_bitwise_identical": queries_equal,
        "p1_p2_geometry_head_bitwise_identical": head_equal,
        "all_geometry_tensors_bitwise_identical": all(per_tensor.values()),
        "per_tensor_bitwise_identical": per_tensor,
        "p1_geometry_state_sha256": p1_hash,
        "p2_geometry_state_sha256": p2_hash,
    }


def selected_position_ids(report: dict, group: str, ordinal: int) -> list[int]:
    span = report["layout"].get(group)
    if span is None:
        return []
    return report["position_ids"][ordinal][span["start"] : span["end"]]


def build_position_report(
    *, selected: list[int], fixed_batches: list[dict], configurations: dict[str, dict]
) -> tuple[list[dict], dict[str, bool]]:
    c0 = configurations["C0"]
    c1 = configurations["C1"]
    c2 = configurations["C2"]
    c3 = configurations["C3"]
    if c2["position_ids"] != c3["position_ids"]:
        raise RuntimeError("C2/C3 position IDs differ when only Ground attention changes")
    if c1["layout"]["geometry"] != c2["layout"]["geometry"]:
        raise RuntimeError("P1/P2 Geometry token-index spans differ")

    rows = []
    for ordinal, (dataset_index, batch) in enumerate(zip(selected, fixed_batches, strict=True)):
        c0_action = selected_position_ids(c0, "action_suffix", ordinal)
        c1_geometry = selected_position_ids(c1, "geometry", ordinal)
        c1_action = selected_position_ids(c1, "action_suffix", ordinal)
        c2_geometry = selected_position_ids(c2, "geometry", ordinal)
        c2_ground = selected_position_ids(c2, "ground", ordinal)
        c2_action = selected_position_ids(c2, "action_suffix", ordinal)
        if c1_geometry != c2_geometry:
            raise RuntimeError(f"P1/P2 Geometry position IDs differ for sample ordinal {ordinal}")
        if any(current - reference != 8 for current, reference in zip(c1_action, c0_action, strict=True)):
            raise RuntimeError("P1 action position IDs are not B0 + 8")
        if any(current - reference != 16 for current, reference in zip(c2_action, c0_action, strict=True)):
            raise RuntimeError("P2 action position IDs are not B0 + 16")
        rows.append(
            {
                "ordinal": ordinal,
                "lerobot_dataset_index": dataset_index,
                "valid_prompt_tokens": int(batch["tokenized_prompt_mask"].sum()),
                "C0_action_position_ids": c0_action,
                "C1_geometry_position_ids": c1_geometry,
                "C1_action_position_ids": c1_action,
                "C2_C3_geometry_position_ids": c2_geometry,
                "C2_C3_ground_position_ids": c2_ground,
                "C2_C3_action_position_ids": c2_action,
            }
        )

    checks = {
        "p1_p2_geometry_token_index_spans_identical": True,
        "p1_p2_geometry_position_ids_identical_all_16": True,
        "C2_C3_all_position_ids_identical": True,
        "P1_action_position_ids_are_C0_plus_8": True,
        "P2_action_position_ids_are_C0_plus_16": True,
        "p2_canonical_order_context_geometry_ground_action": bool(
            c2["layout"]["context"]["end"] == c2["layout"]["geometry"]["start"]
            and c2["layout"]["geometry"]["end"] == c2["layout"]["ground"]["start"]
            and c2["layout"]["ground"]["end"] == c2["layout"]["action_suffix"]["start"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Position/layout checks failed: {checks}")
    return rows, checks


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
    parser.add_argument("--source-decomposition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device(args.device)

    source_calibration = json.loads(args.source_calibration.read_text())
    source_decomposition = json.loads(args.source_decomposition.read_text())
    if source_calibration.get("status") != "PASS":
        raise ValueError("Source calibration is not PASS")
    if source_decomposition.get("status") != "DIAGNOSTIC_COMPLETE_AWAITING_HUMAN_REVIEW":
        raise ValueError("Source causal decomposition is not complete")
    selected = [int(value) for value in source_calibration["dataset_indices"]]
    if selected != source_decomposition["dataset_indices"] or len(selected) != 16:
        raise ValueError("Frozen 16-sample identities differ from source diagnostics")

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
    fixed_batches = []
    for dataset_index in selected:
        item = dataset[dataset_index]
        fixed_batches.append(
            jax.tree.map(torch.as_tensor, _data_loader._collate_fn([item]))  # noqa: SLF001
        )

    configurations = {}
    strict_loads = {}

    base_model, strict_loads["aux_disabled"] = build_model(
        "aux_disabled", model_config=base.model, checkpoint=args.checkpoint
    )
    base_model.to(device).eval()
    c0 = evaluate_configuration(
        name="B0",
        model=base_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset(),
    )
    c0["configuration"] = "C0"
    configurations["C0"] = c0
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    p1_model, strict_loads["p1"] = build_model("geometry", model_config=base.model, checkpoint=args.checkpoint)
    p1_geometry = geometry_snapshot(p1_model)
    p1_model.to(device).eval()
    configurations["C1"] = evaluate_configuration(
        name="C1",
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
    initialization_checks = compare_geometry_initialization(p1_geometry, p2_model)
    del p1_geometry
    p2_model.to(device).eval()
    configurations["C2"] = evaluate_configuration(
        name="C2",
        model=p2_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset({"ground"}),
    )
    configurations["C3"] = evaluate_configuration(
        name="C3",
        model=p2_model,
        fixed_batches=fixed_batches,
        selected=selected,
        device=device,
        blocked_groups=frozenset(),
    )
    del p2_model
    gc.collect()
    torch.cuda.empty_cache()

    noise_hashes = {report["noise_sha256"] for report in configurations.values()}
    if len(noise_hashes) != 1:
        raise RuntimeError("Noise tensors differ across C0--C3")
    source_c0 = source_decomposition["configurations"]["B0"]["per_sample_action_loss"]
    c0_reproduced = configurations["C0"]["per_sample_action_loss"] == source_c0
    if not c0_reproduced:
        raise RuntimeError("C0 does not exactly reproduce the prior B0 result")

    position_rows, position_checks = build_position_report(
        selected=selected,
        fixed_batches=fixed_batches,
        configurations=configurations,
    )
    baseline = configurations["C0"]["per_sample_action_loss"]
    deltas = {name: delta_report(report["per_sample_action_loss"], baseline) for name, report in configurations.items()}
    paired_deltas = {
        name: [value - reference for value, reference in zip(report["per_sample_action_loss"], baseline, strict=True)]
        for name, report in configurations.items()
    }
    incremental = {
        "nested_ground_token_insertion_C2_minus_C1": delta_report(
            configurations["C2"]["per_sample_action_loss"],
            configurations["C1"]["per_sample_action_loss"],
        ),
        "ground_attention_C3_minus_C2": delta_report(
            configurations["C3"]["per_sample_action_loss"],
            configurations["C2"]["per_sample_action_loss"],
        ),
    }

    payload = {
        "status": "DIAGNOSTIC_COMPLETE_AWAITING_HUMAN_REVIEW",
        "schema": "openpi.p1_p2_strict_nested_initialization.v1",
        "scope": "fixed-input no-gradient strict-nesting sanity diagnostic",
        "optimizer_constructed": False,
        "optimizer_steps_run": 0,
        "backward_calls": 0,
        "lambda_values_changed": False,
        "action_query_gate_added": False,
        "warmup_schedule_added": False,
        "geometry_probe_checkpoint_used": False,
        "motion_integrated": False,
        "lambda_freeze_blocked": True,
        "tiny_overfit_blocked": True,
        "checkpoint": str(args.checkpoint.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_calibration": str(args.source_calibration.resolve(strict=True)),
        "source_calibration_sha256": sha256_file(args.source_calibration),
        "source_decomposition": str(args.source_decomposition.resolve(strict=True)),
        "source_decomposition_sha256": sha256_file(args.source_decomposition),
        "sample_count": len(selected),
        "dataset_indices": selected,
        "model_initialization_seed": MODEL_INITIALIZATION_SEED,
        "branch_initialization_seeds": {
            "geometry_queries": GEOMETRY_QUERY_INIT_SEED,
            "geometry_head": GEOMETRY_HEAD_INIT_SEED,
            "ground_queries": GROUND_QUERY_INIT_SEED,
            "ground_head": GROUND_HEAD_INIT_SEED,
        },
        "sample_seed_rule": f"{SAMPLE_SEED_BASE} + ordinal",
        "diffusion_timestep": DIFFUSION_TIMESTEP,
        "noise_sha256": configurations["C0"]["noise_sha256"],
        "strict_loads": strict_loads,
        "configuration_definitions": {
            "C0": "official aux-disabled",
            "C1": "normal P1: Context|Geometry; Action->Geometry enabled",
            "C2": ("P2 Context|Geometry|Ground; Action->Geometry enabled; Action->Ground blocked"),
            "C3": ("normal P2 Context|Geometry|Ground; Action->Geometry and Action->Ground enabled"),
        },
        "source_C0_bitwise_float_equal": c0_reproduced,
        "initialization_checks": initialization_checks,
        "position_checks": position_checks,
        "position_id_report": position_rows,
        "configurations": configurations,
        "deltas_from_C0": deltas,
        "paired_deltas_from_C0": paired_deltas,
        "incremental_effects": incremental,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
