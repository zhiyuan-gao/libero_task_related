from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from four_suite_experiments.configs import blocked_action_groups
from four_suite_experiments.configs import build_train_config
from four_suite_experiments.constants import FOUR_SUITE_EPISODES
from four_suite_experiments.constants import FOUR_SUITE_FRAMES
from four_suite_experiments.data_overlay import FourSuitePolicyAuxTrainConfig
from four_suite_experiments.paths import ArtifactPaths


def test_four_suite_mapping_population(tmp_path) -> None:
    episodes = []
    cursor = 0
    # Minimal lengths preserve the exact total while exercising all episode IDs.
    lengths = [1] * FOUR_SUITE_EPISODES
    lengths[-1] += FOUR_SUITE_FRAMES - FOUR_SUITE_EPISODES
    for episode_index, length in enumerate(lengths):
        episodes.append(
            {
                "lerobot_episode_index": episode_index,
                "dataset_from_index": cursor,
                "dataset_to_index_exclusive": cursor + length,
                "episode_length": length,
            }
        )
        cursor += length
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "status": "PASS",
                "hf_repo_id": "physical-intelligence/libero",
                "hf_revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
                "mapped_episode_count": FOUR_SUITE_EPISODES,
                "mapped_frame_count": FOUR_SUITE_FRAMES,
                "episodes": episodes,
            }
        )
    )
    config = FourSuitePolicyAuxTrainConfig(
        mode="semantic_geometry_motion",
        policy_manifest_path="manifest",
        episode_mapping_path=str(mapping),
        geometry_target_index_path="geometry",
        geometry_normalization_path="geometry_norm",
        motion_target_index_path="motion",
        motion_normalization_path="motion_norm",
        motion_target_count=256_401,
        lambda_geo=0.05,
        lambda_sem=0.01,
        lambda_motion=0.05,
        num_ground_queries=0,
        num_motion_queries=8,
        loss_coefficients_approved=True,
    )
    assert config.lerobot_episode_indices() == list(range(FOUR_SUITE_EPISODES))
    assert config.lerobot_dataset_indices() == list(range(FOUR_SUITE_FRAMES))


def test_main_and_supervision_only_configs_differ_only_by_identity(tmp_path) -> None:
    common = {
        "artifacts": ArtifactPaths(tmp_path / "artifacts"),
        "num_train_steps": 30_000,
        "warmup_steps": 10_000,
        "checkpoint_base_dir": tmp_path / "checkpoints",
        "lerobot_root": tmp_path / "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        "base_weight_path": tmp_path / "pi05_base_pytorch_fp32",
        "libero_assets_dir": tmp_path / "assets",
        "wandb_enabled": False,
    }
    main = build_train_config(variant="main", exp_name="main", **common)
    control = build_train_config(
        variant="supervision_only", exp_name="supervision_only", **common
    )
    differing = []
    for field in dataclasses.fields(main):
        left = getattr(main, field.name)
        right = getattr(control, field.name)
        try:
            equal = left == right
        except (TypeError, ValueError):
            equal = False
        if not isinstance(equal, bool) or not equal:
            differing.append(field.name)
    assert differing == ["name", "exp_name"]
    assert blocked_action_groups("main") == frozenset()
    assert blocked_action_groups("supervision_only") == frozenset(
        {"geometry", "motion"}
    )
    assert main.policy_aux.lambda_geo == 0.05
    assert main.policy_aux.lambda_sem == 0.01
    assert main.policy_aux.lambda_motion == 0.05
    assert main.policy_aux.motion_target_count == 256_401
    assert Path(main.pytorch_weight_path).name == "pi05_base_pytorch_fp32"
