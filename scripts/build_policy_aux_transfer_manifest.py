#!/usr/bin/env python3
"""Build a content-addressed P1/P2 transfer manifest without copying large assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

WORKSPACE = Path("/workspace/vla")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_path(path: Path) -> tuple[int, str, list[dict]]:
    """Return physical size, deterministic tree digest, and file records."""

    path = path.resolve(strict=True)
    if path.is_file():
        digest = sha256_file(path)
        size = path.stat().st_size
        return size, digest, [{"relative_path": path.name, "type": "file", "size": size, "sha256": digest}]

    records = []
    physical_size = 0
    tree_digest = hashlib.sha256()
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix()
        mode = child.lstat().st_mode
        if stat.S_ISLNK(mode):
            target = os.readlink(child)
            digest = hashlib.sha256(f"symlink:{target}".encode()).hexdigest()
            record = {
                "relative_path": relative,
                "type": "symlink",
                "size": len(target.encode()),
                "sha256": digest,
                "target": target,
            }
        elif stat.S_ISREG(mode):
            size = child.stat().st_size
            digest = sha256_file(child)
            physical_size += size
            record = {
                "relative_path": relative,
                "type": "file",
                "size": size,
                "sha256": digest,
            }
        else:
            continue
        records.append(record)
        tree_digest.update(
            f"{record['relative_path']}\t{record['type']}\t{record['size']}\t{record['sha256']}\n".encode()
        )
    return physical_size, tree_digest.hexdigest(), records


def destination(path: Path) -> str:
    return os.path.relpath(path.resolve(), WORKSPACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-bundle", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-sha256", type=Path, required=True)
    args = parser.parse_args()

    openpi_root = WORKSPACE / "third_party/openpi"
    implementation_commit = subprocess.check_output(
        ["git", "-C", str(openpi_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.implementation_commit != implementation_commit:
        raise RuntimeError(
            "Implementation commit does not match the checked-out OpenPI HEAD: "
            f"requested={args.implementation_commit}, HEAD={implementation_commit}"
        )
    subprocess.run(
        ["git", "-C", str(openpi_root), "bundle", "verify", str(args.openpi_bundle.resolve())],
        check=True,
    )

    root = WORKSPACE / "data/libero_four_suite_annotation/policy_aux_v1"
    items = [
        ("openpi_local_git_bundle", args.openpi_bundle, True, False),
        ("official_pi05_base_pytorch", WORKSPACE / "models/openpi/pi05_base_pytorch", True, False),
        ("libero_normalization_assets", WORKSPACE / "models/openpi/pi05_libero_pytorch/assets", True, False),
        (
            "official_lerobot_libero10_hf_repo_exact_revision",
            WORKSPACE / "cache/huggingface/hub/datasets--physical-intelligence--libero",
            True,
            True,
        ),
        (
            "frozen_libero10_grounding_mask_shards",
            WORKSPACE / "data/libero_four_suite_annotation/stage_relevant_masks_v1/full_v1/shards/libero_10",
            True,
            False,
        ),
        (
            "frozen_grounding_mask_validation_report",
            WORKSPACE
            / "data/libero_four_suite_annotation/stage_relevant_masks_v1/full_v1/full_mask_validation_report.json",
            True,
            False,
        ),
        (
            "frozen_grounding_mask_recipe",
            WORKSPACE / "data/libero_four_suite_annotation/stage_relevant_masks_v1/full_v1/run_config.json",
            True,
            False,
        ),
        (
            "frozen_grounding_mask_checksums",
            WORKSPACE
            / "data/libero_four_suite_annotation/stage_relevant_masks_v1/full_v1/derived_artifact_checksums.json",
            True,
            False,
        ),
        (
            "frozen_libero10_semantic_annotations",
            WORKSPACE / "data/libero_four_suite_annotation/semantic_subtask_outputs/libero_10",
            True,
            False,
        ),
        ("full_libero10_geometry_policy_cache", root / "geometry_libero10", True, True),
        ("policy_aux_manifests", root / "manifests", True, True),
        ("policy_aux_debug_provenance", root / "debug", True, True),
        ("policy_aux_gate_evidence", root / "unit_gates", True, True),
        ("policy_aux_loss_calibration", root / "calibration", True, True),
        ("policy_aux_frozen_configs", root / "configs", True, True),
        ("policy_aux_bootstrap_provenance", root / "provenance", True, True),
        (
            "tiny_overfit_human_report",
            root / "tiny_overfit/P1_P2_STRICT_NESTED_FIXED16_TINY_OVERFIT_REPORT.md",
            True,
            True,
        ),
        ("tiny_overfit_p1_metrics", root / "tiny_overfit/p1_tiny_overfit.json", True, True),
        ("tiny_overfit_p2_metrics", root / "tiny_overfit/p2_tiny_overfit.json", True, True),
        (
            "tiny_overfit_grounding_audit",
            root / "tiny_overfit/p2_tiny_overfit_grounding_prediction_audit.png",
            True,
            True,
        ),
        ("target_machine_preflight", root / "handoff/TARGET_MACHINE_PREFLIGHT.md", True, True),
        ("implementation_report", root / "handoff/CODEX_IMPLEMENTATION_REPORT.md", True, True),
        (
            "optional_regenerable_arrow_cache",
            Path("/workspace/.hf_home/datasets/parquet/default-643f0e0d963845cf"),
            False,
            True,
        ),
    ]

    started = time.monotonic()
    entries = []
    checksum_lines = []
    for logical_name, path, required, regenerable in items:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required transfer item is missing: {path}")
            continue
        size, digest, records = inspect_path(path)
        entry = {
            "logical_name": logical_name,
            "source_path": str(path.resolve()),
            "relative_destination": destination(path),
            "kind": "directory" if path.is_dir() else "file",
            "physical_size_bytes": size,
            "sha256": digest,
            "required": required,
            "regenerable": regenerable,
            "file_count": sum(record["type"] == "file" for record in records),
            "symlink_count": sum(record["type"] == "symlink" for record in records),
        }
        entries.append(entry)
        checksum_lines.append(f"# TREE {digest}  {path.resolve()}")
        if path.is_dir():
            checksum_lines.extend(
                f"{record['sha256']}  {path.resolve() / record['relative_path']}"
                for record in records
                if record["type"] == "file"
            )
        else:
            checksum_lines.append(f"{digest}  {path.resolve()}")

    payload = {
        "status": "PASS",
        "schema": "openpi.policy_aux_transfer_manifest.v1",
        "workspace_root": str(WORKSPACE),
        "openpi_upstream_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "openpi_implementation_commit": implementation_commit,
        "architecture_revision": "2026-08-18-native-semantic-lm-no-semantic-query",
        "required_total_physical_bytes": sum(entry["physical_size_bytes"] for entry in entries if entry["required"]),
        "optional_total_physical_bytes": sum(
            entry["physical_size_bytes"] for entry in entries if not entry["required"]
        ),
        "entries": entries,
        "explicit_exclusions": [
            "official_hdf5 (Geometry cache already materialized)",
            "VGGT teacher checkpoint/repository (not used during policy training)",
            "Motion/Track4World artifacts",
            "P1/P2 tiny-overfit trained checkpoints (engineering-gate outputs; not full-training inputs)",
            "non-LIBERO-10 Grounding mask shards",
            "RoboTwin/LingBot/other suites",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_sha = sha256_file(args.output_manifest)
    checksum_lines.insert(0, f"{manifest_sha}  {args.output_manifest.resolve()}")
    args.output_sha256.write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
