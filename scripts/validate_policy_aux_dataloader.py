#!/usr/bin/env python3
"""Validate the real official LIBERO-10 P1/P2 DataLoader target join."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.policy_aux_dataset import PolicyAuxTrainConfig

CANONICAL_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def validate_canonical_population(lerobot_root: Path, mapping_path: Path, manifest_path: Path) -> dict[str, bool]:
    root = lerobot_root.resolve(strict=True)
    mapping = json.loads(mapping_path.resolve(strict=True).read_text())
    manifest = pd.read_parquet(manifest_path.resolve(strict=True)).sort_values("lerobot_dataset_index")
    records = sorted(mapping["episodes"], key=lambda row: int(row["lerobot_episode_index"]))
    checks = {
        "exact_snapshot_revision": root.name == CANONICAL_REVISION,
        "mapping_pass_and_exact_revision": (
            mapping.get("status") == "PASS"
            and mapping.get("hf_repo_id") == "physical-intelligence/libero"
            and mapping.get("hf_revision") == CANONICAL_REVISION
        ),
        "exact_episode_population": (
            int(mapping.get("mapped_episode_count", -1)) == 379
            and len(records) == 379
            and [int(row["lerobot_episode_index"]) for row in records] == list(range(379))
        ),
        "exact_frame_population": (
            int(mapping.get("mapped_frame_count", -1)) == 101_469
            and sum(int(row["episode_length"]) for row in records) == 101_469
            and len(manifest) == 101_469
        ),
        "manifest_exact_policy_only": (
            bool(manifest["policy_train"].astype(bool).all())
            and bool(manifest["sample_id"].is_unique)
            and manifest["lerobot_dataset_index"].tolist() == list(range(101_469))
        ),
    }
    episode_identity_valid = True
    next_dataset_index = 0
    for record in records:
        episode_index = int(record["lerobot_episode_index"])
        episode_length = int(record["episode_length"])
        dataset_from = int(record["dataset_from_index"])
        dataset_to = int(record["dataset_to_index_exclusive"])
        rows = manifest.loc[manifest["lerobot_episode_index"].eq(episode_index)]
        episode_identity_valid &= (
            dataset_from == next_dataset_index
            and dataset_to == dataset_from + episode_length
            and len(rows) == episode_length
            and rows["lerobot_dataset_index"].tolist() == list(range(dataset_from, dataset_to))
            and bool(rows["action_sha256"].astype(str).eq(str(record["action_sha256"])).all())
            and bool(rows["episode_id"].astype(str).eq(str(record["annotation_episode_id"])).all())
        )
        next_dataset_index = dataset_to
    checks["per_episode_identity_join"] = episode_identity_valid and next_dataset_index == 101_469
    if not all(checks.values()):
        raise RuntimeError(f"Canonical LIBERO-10 population gate failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--libero-assets-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--geometry-index", type=Path, required=True)
    parser.add_argument("--geometry-normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    population_checks = validate_canonical_population(args.lerobot_root, args.mapping, args.policy_manifest)
    base = _config.get_config("pi05_libero")
    data_factory = dataclasses.replace(
        base.data,
        assets=_config.AssetsConfig(assets_dir=str(args.libero_assets_root.resolve(strict=True))),
    )
    data_config = data_factory.create(Path("/nonexistent/assets_not_used"), base.model)
    if data_config.repo_id != "physical-intelligence/libero" or data_config.norm_stats is None:
        raise ValueError("Official pi05_libero data config/norm assets were not resolved")

    reports = {}
    for mode in ("geometry", "semantic_geometry", "ground_geometry_semantic_lm"):
        aux_config = PolicyAuxTrainConfig(
            mode=mode,
            policy_manifest_path=str(args.policy_manifest.resolve(strict=True)),
            episode_mapping_path=str(args.mapping.resolve(strict=True)),
            geometry_target_index_path=str(args.geometry_index.resolve(strict=True)),
            geometry_normalization_path=str(args.geometry_normalization.resolve(strict=True)),
            lambda_geo=1.0,
            lambda_sem=1.0 if mode in ("semantic_geometry", "ground_geometry_semantic_lm") else None,
            lambda_ground=1.0 if mode == "ground_geometry_semantic_lm" else None,
            num_ground_queries=0 if mode == "semantic_geometry" else 8,
            lerobot_root=str(args.lerobot_root.resolve(strict=True)),
            lerobot_task_indices=(0, 3, 8) if mode == "semantic_geometry" else None,
        )
        loader = _data_loader.create_torch_data_loader(
            data_config,
            model_config=base.model,
            action_horizon=10,
            batch_size=2,
            shuffle=False,
            num_batches=1,
            num_workers=args.num_workers,
            seed=20260818,
            framework="pytorch",
            policy_aux_config=aux_config,
        )
        observation, actions, targets = next(iter(loader))
        expected_keys = {"geometry", "geometry_valid", "geometry_mean", "geometry_std"}
        if mode == "ground_geometry_semantic_lm":
            expected_keys.update(
                {
                    "ground_masks",
                    "ground_valid_views",
                }
            )
        if mode in ("semantic_geometry", "ground_geometry_semantic_lm"):
            expected_keys.update(
                {
                    "semantic_input_ids",
                    "semantic_labels",
                    "semantic_loss_mask",
                }
            )
        checks = {
            "exact_target_keys": set(targets) == expected_keys,
            "official_action_horizon_and_dim": list(actions.shape) == [2, 10, 32],
            "official_two_real_one_padded_views": (
                list(observation.images) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
                and bool(observation.image_masks["base_0_rgb"].all())
                and bool(observation.image_masks["left_wrist_0_rgb"].all())
                and not bool(observation.image_masks["right_wrist_0_rgb"].any())
            ),
            "geometry_shape_finite_and_valid": (
                list(targets["geometry"].shape) == [2, 2048]
                and bool(torch.isfinite(targets["geometry"]).all())
                and bool(targets["geometry_valid"].all())
            ),
            "normalization_is_train_only_shared_vector": (
                list(targets["geometry_mean"].shape) == [2, 2048]
                and list(targets["geometry_std"].shape) == [2, 2048]
                and bool(torch.equal(targets["geometry_mean"][0], targets["geometry_mean"][1]))
                and bool(torch.equal(targets["geometry_std"][0], targets["geometry_std"][1]))
                and bool((targets["geometry_std"] > 0).all())
            ),
        }
        if mode == "ground_geometry_semantic_lm":
            checks.update(
                {
                    "ground_agent_wrist_shapes": all(
                        list(mask.shape) == [2, 128, 128] for mask in targets["ground_masks"].values()
                    ),
                    "ground_valid_view_shape": list(targets["ground_valid_views"].shape) == [2, 2],
                }
            )
        if mode in ("semantic_geometry", "ground_geometry_semantic_lm"):
            checks.update(
                {
                    "semantic_fixed_teacher_shapes": (
                        list(targets["semantic_input_ids"].shape) == [2, 31]
                        and list(targets["semantic_labels"].shape) == [2, 32]
                        and list(targets["semantic_loss_mask"].shape) == [2, 32]
                    ),
                    "semantic_has_eos_and_no_empty_target": bool(
                        (
                            targets["semantic_labels"]
                            * targets["semantic_loss_mask"].to(targets["semantic_labels"].dtype)
                        )
                        .eq(1)
                        .any(dim=1)
                        .all()
                    ),
                }
            )
        if mode == "semantic_geometry":
            selected_records = [
                row
                for row in json.loads(args.mapping.read_text())["episodes"]
                if int(row["lerobot_task_index"]) in {0, 3, 8}
            ]
            checks.update(
                {
                    "no_ground_targets_loaded": (
                        "ground_masks" not in targets and "ground_valid_views" not in targets
                    ),
                    "exact_three_task_episode_population": len(selected_records) == 114,
                    "exact_three_task_frame_population": (
                        sum(int(row["episode_length"]) for row in selected_records) == 29_250
                    ),
                    "exact_three_task_indices": (
                        {int(row["lerobot_task_index"]) for row in selected_records} == {0, 3, 8}
                    ),
                }
            )
        if not all(checks.values()):
            raise RuntimeError(f"{mode} DataLoader gate failed: {checks}")
        reports[mode] = {
            "checks": checks,
            "target_keys": sorted(targets),
            "geometry_norm_mean": float(targets["geometry"].float().norm(dim=1).mean()),
            "actions_finite": bool(np.isfinite(actions.numpy()).all()),
            "num_workers": args.num_workers,
        }

    payload = {
        "status": "PASS",
        "gate": "pi05_p1_p2_real_libero10_dataloader_join_v1",
        "official_lerobot_episode_filter": list(range(379)),
        "official_lerobot_revision": CANONICAL_REVISION,
        "policy_samples": 101_469,
        "canonical_population_checks": population_checks,
        "reports": reports,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
