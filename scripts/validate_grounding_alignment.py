#!/usr/bin/env python3
"""Validate real LIBERO agent/wrist mask alignment in official policy geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from policy_aux_gate_utils import load_real_libero_item
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import preprocessing_pytorch
from openpi.models_pytorch.policy_aux_preprocessing import patch_foreground_coverage
from openpi.models_pytorch.policy_aux_preprocessing import preprocess_observation_and_ground_masks_pytorch
from openpi.shared import image_tools


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.astype(np.float64).ravel(), right.astype(np.float64).ravel())[0, 1])


def to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    return ((image.numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)


def overlay(rgb: np.ndarray, coverage: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    alpha = np.clip(coverage[..., None], 0.0, 1.0) * 0.55
    colored = np.broadcast_to(np.asarray(color, dtype=np.float32), rgb.shape)
    return (rgb * (1.0 - alpha) + colored * alpha).round().clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--prefix-layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    args = parser.parse_args()
    observation, _, auxiliary, row = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.annotation_manifest,
    )
    layout = json.loads(args.prefix_layout.read_text())["modes"]["ground_geometry_semantic_lm"]
    token_counts = layout["tokens_per_view"]
    real_views = layout["real_view_names"]
    if real_views != ["base_0_rgb", "left_wrist_0_rgb"]:
        raise ValueError(f"Unexpected real views: {real_views}")

    synchronized = preprocess_observation_and_ground_masks_pytorch(observation, auxiliary["ground_masks"], train=False)
    torch.manual_seed(20260818)
    official_train = preprocessing_pytorch.preprocess_observation_pytorch(observation, train=True)
    torch.manual_seed(20260818)
    synchronized_train = preprocess_observation_and_ground_masks_pytorch(
        observation, auxiliary["ground_masks"], train=True
    )
    rgb_path_equal = {
        name: bool(torch.equal(official_train.images[name], synchronized_train.observation.images[name]))
        for name in observation.images
    }

    with h5py.File(row["hdf5_path"], "r") as handle:
        episode = handle["data"][row["source_demo_key"]]["obs"]
        raw_images = {
            "base_0_rgb": episode["agentview_rgb"][int(row["raw_state_index"])],
            "left_wrist_0_rgb": episode["eye_in_hand_rgb"][int(row["raw_state_index"])],
        }

    view_reports = {}
    audit_rows = []
    for name in real_views:
        policy_rgb = to_uint8(synchronized.observation.images[name][0])
        raw = raw_images[name]
        orientation_correlations = {}
        for orientation, candidate in (
            ("raw", raw),
            ("vertical_flip", raw[::-1]),
            ("rot180", raw[::-1, ::-1]),
        ):
            resized = np.asarray(image_tools.resize_with_pad(candidate, 224, 224))
            orientation_correlations[orientation] = correlation(resized, policy_rgb)

        transformed_mask = synchronized.ground_masks[name]
        patch_count = int(token_counts[name])
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise ValueError(f"Non-square runtime image token grid for {name}: {patch_count}")
        patches = patch_foreground_coverage(transformed_mask, grid_height=grid_size, grid_width=grid_size)
        patch_map = patches.reshape(1, 1, grid_size, grid_size)
        patch_up = F.interpolate(patch_map, size=(224, 224), mode="nearest")[0, 0].numpy()
        mask_np = transformed_mask[0].numpy()
        mass_error = abs(float(mask_np.mean()) - float(patches.mean()))
        view_reports[name] = {
            "orientation_correlations": orientation_correlations,
            "selected_orientation": "rot180",
            "transformed_mask_shape": list(mask_np.shape),
            "transformed_mask_mass": float(mask_np.sum()),
            "patch_grid": [grid_size, grid_size],
            "patch_count": patch_count,
            "patch_mass_mean_error": mass_error,
            "ground_valid": bool(auxiliary["ground_valid_views"][0, real_views.index(name)].item()),
        }
        audit_rows.append(
            np.concatenate(
                (
                    policy_rgb,
                    overlay(policy_rgb, mask_np, (255, 20, 20)),
                    overlay(policy_rgb, patch_up, (20, 255, 20)),
                ),
                axis=1,
            )
        )

    checks = {
        "official_and_synchronized_rgb_paths_bitwise_equal": all(rgb_path_equal.values()),
        "both_real_views_valid": bool(auxiliary["ground_valid_views"].all()),
        "both_views_rot180_cross_render_correlation_gt_0_90": all(
            report["orientation_correlations"]["rot180"] > 0.90 for report in view_reports.values()
        ),
        "rot180_is_best_orientation_both_views": all(
            report["orientation_correlations"]["rot180"] == max(report["orientation_correlations"].values())
            for report in view_reports.values()
        ),
        "rot180_correlation_margin_gt_0_50_both_views": all(
            report["orientation_correlations"]["rot180"]
            - max(value for key, value in report["orientation_correlations"].items() if key != "rot180")
            > 0.50
            for report in view_reports.values()
        ),
        "runtime_patch_grid_is_16x16_both_views": all(
            report["patch_grid"] == [16, 16] for report in view_reports.values()
        ),
        "patch_coverage_preserves_mean_mass": all(
            report["patch_mass_mean_error"] < 1e-6 for report in view_reports.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Grounding alignment gate failed: {checks}")
    args.visualization.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(audit_rows, axis=0)).save(args.visualization)
    payload = {
        "status": "PASS",
        "gate": "pi05_policy_grounding_alignment_real_agent_wrist_v1",
        "sample_id": auxiliary["sample_id"],
        "canonical_mask_storage": "raw HDF5/MuJoCo OpenGL row order; source untouched",
        "raw_to_policy_orientation": "deterministic rot180 applied to RGB/mask coordinates",
        "policy_training_geometry": (
            "official ResizeImages(224,224), then synchronized PyTorch crop/resize/rotation; color jitter RGB-only"
        ),
        "rgb_path_equal_by_view": rgb_path_equal,
        "view_reports": view_reports,
        "checks": checks,
        "visualization": str(args.visualization.resolve()),
        "visualization_columns": ["policy RGB", "continuous mask overlay", "16x16 patch coverage overlay"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
