#!/usr/bin/env python3
"""Summarize completed P1/P2 development gates before human lambda freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pass(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise ValueError(f"Required artifact is not PASS: {path}: {payload.get('status')}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    root = args.policy_root.resolve(strict=True)
    paths = {
        "geometry_cache": root / "geometry_libero10/policy_cache_validation.json",
        "component_gradients": root / "unit_gates/p1_p2_component_gradient_gate.json",
        "checkpoint_roundtrip": root / "unit_gates/checkpoint_roundtrip_gate.json",
        "dataloader_join": root / "unit_gates/dataloader_join_gate.json",
        "loss_calibration": root / "calibration/p1_p2_loss_scale_report.json",
    }
    artifacts = {name: load_pass(path) for name, path in paths.items()}
    cache = artifacts["geometry_cache"]
    calibration = artifacts["loss_calibration"]
    payload = {
        "status": "PASS_AWAITING_HUMAN_LAMBDA_FREEZE",
        "schema": "openpi.policy_aux_pre_lambda_status.v1",
        "architecture_revision": "2026-08-18-native-semantic-lm-no-semantic-query",
        "geometry_cache": {
            "policy_samples": cache["policy_samples"],
            "geometry_valid_samples": cache["geometry_valid_samples"],
            "geometry_invalid_samples": cache["geometry_invalid_samples"],
            "target_index_sha256": cache["target_index_sha256"],
            "target_memmap_sha256": cache["target_memmap_sha256"],
            "normalization_sha256": cache["normalization_sha256"],
            "pilot_overlap": cache["pilot_overlap"],
        },
        "gates": {
            name: {"status": artifact["status"], "path": str(paths[name])}
            for name, artifact in artifacts.items()
            if name != "loss_calibration"
        },
        "raw_loss_calibration": {
            mode: report["raw_loss_statistics"] for mode, report in calibration["reports"].items()
        },
        "raw_component_gradient_norms": calibration["component_gradient_norms"],
        "optimizer_steps_run": 0,
        "required_human_decision": {
            "lambda_geo": None,
            "lambda_ground": None,
            "lambda_sem": None,
            "next_after_approval": "run fixed-16 P1 and P2 tiny-overfit gates only",
        },
        "full_training_authorized": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# P1/P2 pre-lambda gate status",
        "",
        "Status: **PASS — awaiting human lambda freeze**",
        "",
        f"Geometry cache: {cache['geometry_valid_samples']:,} valid / "
        f"{cache['policy_samples']:,} policy frames; {cache['geometry_invalid_samples']:,} invalid.",
        "",
    ]
    for mode, report in calibration["reports"].items():
        lines.extend(
            [
                f"## {mode}",
                "",
                "| Raw loss | Mean | Std | Median |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in report["raw_loss_statistics"].items():
            lines.append(f"| {name} | {values['mean']:.6f} | {values['std']:.6f} | {values['median']:.6f} |")
        lines.append("")
    lines.extend(
        [
            "No optimizer step has been run. Full policy training remains blocked.",
            "After lambda approval, run only the fixed-16 P1/P2 tiny-overfit gates.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
