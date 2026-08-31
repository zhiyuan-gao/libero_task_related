from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from four_suite_experiments.configs import blocked_action_groups
from four_suite_experiments.configs import build_train_config
from four_suite_experiments.configs import expected_target_scope
from four_suite_experiments.constants import FOUR_SUITE_EPISODES
from four_suite_experiments.constants import FOUR_SUITE_FRAMES
from four_suite_experiments.data_overlay import FourSuitePolicyAuxTrainConfig
from four_suite_experiments.data_overlay import FourSuitePolicyAuxTransformedDataset
from four_suite_experiments.data_overlay import install_data_overlay
from four_suite_experiments.paths import ArtifactPaths

from openpi.training import policy_aux_dataset as upstream_policy_aux_dataset


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


def test_four_suite_dataset_overlay_is_spawn_safe(monkeypatch) -> None:
    sentinel = object()
    config = object()
    dataset = FourSuitePolicyAuxTransformedDataset.__new__(FourSuitePolicyAuxTransformedDataset)
    dataset.config = config
    monkeypatch.setattr(
        "four_suite_experiments.data_overlay.FourSuitePolicyAuxTargetIndex",
        lambda received: sentinel if received is config else None,
    )
    assert dataset._make_target_index() is sentinel  # noqa: SLF001

    install_data_overlay()
    assert upstream_policy_aux_dataset.PolicyAuxTransformedDataset is FourSuitePolicyAuxTransformedDataset


def test_trqc_and_no_query_access_configs_differ_only_by_identity(tmp_path) -> None:
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
    main = build_train_config(variant="trqc", exp_name="trqc", **common)
    control = build_train_config(variant="no_query_access", exp_name="no_query_access", **common)
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
    assert blocked_action_groups("trqc") == frozenset()
    assert blocked_action_groups("no_query_access") == frozenset({"geometry", "motion"})
    assert expected_target_scope("trqc") == "task_relevant"
    assert expected_target_scope("no_query_access") == "task_relevant"
    assert main.policy_aux.lambda_geo == 0.05
    assert main.policy_aux.lambda_sem == 0.01
    assert main.policy_aux.lambda_motion == 0.05
    assert main.policy_aux.motion_target_count == 256_401
    assert main.policy_aux.target_scope == "task_relevant"
    assert Path(main.pytorch_weight_path).name == "pi05_base_pytorch_fp32"
    assert main.save_interval == 1_000
    assert main.late_save_interval == 500
    assert main.late_save_start_step == 20_000
    assert main.keep_period is None
    assert main.max_checkpoints_to_keep == 30
    assert main.max_resume_checkpoints_to_keep == 2


def test_whole_scene_variant_changes_only_scope_and_artifact_paths(tmp_path) -> None:
    common = {
        "num_train_steps": 30_000,
        "warmup_steps": 10_000,
        "checkpoint_base_dir": tmp_path / "checkpoints",
        "lerobot_root": tmp_path / "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        "base_weight_path": tmp_path / "pi05_base_pytorch_fp32",
        "libero_assets_dir": tmp_path / "assets",
        "wandb_enabled": False,
    }
    task = build_train_config(
        variant="trqc",
        artifacts=ArtifactPaths(tmp_path / "task_relevant"),
        exp_name="trqc",
        **common,
    )
    whole = build_train_config(
        variant="whole_scene",
        artifacts=ArtifactPaths(tmp_path / "whole_scene"),
        exp_name="whole_scene",
        **common,
    )
    assert blocked_action_groups("whole_scene") == frozenset()
    assert expected_target_scope("whole_scene") == "whole_scene"
    assert whole.policy_aux.target_scope == "whole_scene"
    assert task.policy_aux.lambda_sem == whole.policy_aux.lambda_sem == 0.01
    assert task.policy_aux.lambda_geo == whole.policy_aux.lambda_geo == 0.05
    assert task.policy_aux.lambda_motion == whole.policy_aux.lambda_motion == 0.05
    assert task.policy_aux.num_geometry_queries == whole.policy_aux.num_geometry_queries == 8
    assert task.policy_aux.num_motion_queries == whole.policy_aux.num_motion_queries == 8
    assert task.policy_aux.geometry_target_index_path != whole.policy_aux.geometry_target_index_path
    assert task.policy_aux.motion_target_index_path != whole.policy_aux.motion_target_index_path


def test_8gpu_launcher_uses_validated_allocator_settings() -> None:
    launcher = Path(__file__).resolve().parents[1] / "jobs/run_8gpu.sh"
    text = launcher.read_text()
    assert "export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=1" in text
    assert "export OPENPI_LOG_MEMORY_STATS=0" in text
    assert "export TOKENIZERS_PARALLELISM=false" in text
