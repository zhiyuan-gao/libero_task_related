from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

from four_suite_experiments.configs import blocked_action_groups
from four_suite_experiments.configs import build_train_config
from four_suite_experiments.configs import expected_target_scope
from four_suite_experiments.constants import COMPLETED_EPISODES
from four_suite_experiments.constants import COMPLETED_FRAMES
from four_suite_experiments.constants import COMPLETED_MOTION_VALID
from four_suite_experiments.constants import FOUR_SUITE_EPISODES
from four_suite_experiments.constants import FOUR_SUITE_FRAMES
from four_suite_experiments.data_overlay import FourSuitePolicyAuxTrainConfig
from four_suite_experiments.data_overlay import FourSuitePolicyAuxTransformedDataset
from four_suite_experiments.data_overlay import install_data_overlay
from four_suite_experiments.paths import ArtifactPaths
import pytest

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


def test_official_completion_population_is_explicit_and_mutually_exclusive(tmp_path) -> None:
    common = {
        "variant": "trqc",
        "artifacts": ArtifactPaths(tmp_path / "artifacts"),
        "exp_name": "official_completion",
        "num_train_steps": 3_000,
        "warmup_steps": 200,
        "checkpoint_base_dir": tmp_path / "checkpoints",
        "lerobot_root": tmp_path / "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        "base_weight_path": tmp_path / "step30000",
        "libero_assets_dir": tmp_path / "assets",
        "wandb_enabled": False,
    }
    completed = build_train_config(official_completion=True, **common)
    assert completed.policy_aux.expected_episodes == COMPLETED_EPISODES == 1_932
    assert completed.policy_aux.expected_frames == COMPLETED_FRAMES == 328_636
    assert completed.policy_aux.motion_target_count == COMPLETED_MOTION_VALID == 309_147
    assert completed.policy_aux.official_completion is True
    assert completed.policy_aux.supplemental_augmentation is False

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_config(
            supplemental_augmentation=True,
            official_completion=True,
            **common,
        )


def test_8gpu_launcher_uses_validated_allocator_settings() -> None:
    launcher = Path(__file__).resolve().parents[1] / "jobs/run_8gpu.sh"
    text = launcher.read_text()
    assert "export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=1" in text
    assert "export OPENPI_LOG_MEMORY_STATS=0" in text
    assert "export TOKENIZERS_PARALLELISM=false" in text


def test_eval_launcher_routes_24_simulators_to_eight_policy_servers(tmp_path) -> None:
    project = Path(__file__).resolve().parents[1]
    launcher = project / "jobs/eval_checkpoint_8gpu.sh"
    checkpoint = tmp_path / "30000"
    (checkpoint / "assets/physical-intelligence/libero").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "assets/physical-intelligence/libero/norm_stats.json").write_text("{}")
    libero = tmp_path / "libero_source/libero/libero"
    for directory in ("assets", "bddl_files", "init_files"):
        (libero / directory).mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT": str(checkpoint),
            "RUN_ROOT": str(tmp_path / "run"),
            "FOUR_SUITE_PYTHON": sys.executable,
            "FOUR_SUITE_BASE_WEIGHTS": str(tmp_path / "base"),
            "FOUR_SUITE_LIBERO_ASSETS": str(tmp_path / "assets"),
            "LIBERO_EVAL_SOURCE_ROOT": str(tmp_path / "libero_source"),
            "NUM_SHARDS": "24",
            "ALLOW_EXPERIMENTAL_WORKER_COUNT": "1",
            "DRY_RUN": "1",
        }
    )
    formal_env = env.copy()
    formal_env.pop("ALLOW_EXPERIMENTAL_WORKER_COUNT")
    rejected = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env=formal_env,
    )
    assert rejected.returncode == 2
    assert "Formal LIBERO-40 evaluation is fixed at NUM_SHARDS=16" in rejected.stderr
    result = subprocess.run(
        ["bash", str(launcher)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "8 policy servers and 24 simulator workers" in result.stdout
    text = launcher.read_text()
    assert "for gpu in $(seq 0 $((NUM_POLICY_SERVERS - 1)))" in text
    assert '--args.port "$((PORT_BASE + gpu))"' in text


def test_formal_eval_launchers_are_fixed_to_16_workers() -> None:
    jobs = Path(__file__).resolve().parents[1] / "jobs"
    for name in ("eval_checkpoints_reverse_8gpu.sh", "eval_selected_multiseed_8gpu.sh"):
        assert "readonly NUM_SHARDS=16" in (jobs / name).read_text()


def test_official_completion_continuation_is_one_continuous_6k_schedule(tmp_path) -> None:
    config = build_train_config(
        variant="trqc",
        artifacts=ArtifactPaths(tmp_path / "artifacts"),
        exp_name="continuous_6k",
        num_train_steps=6_000,
        warmup_steps=200,
        checkpoint_base_dir=tmp_path / "checkpoints",
        lerobot_root=tmp_path / "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        base_weight_path=tmp_path / "parent" / "30000",
        libero_assets_dir=tmp_path / "assets",
        wandb_enabled=False,
        peak_lr=1e-5,
        decay_steps=6_000,
        decay_lr=1e-6,
        save_interval=500,
        late_save_interval=None,
        late_save_start_step=None,
        max_checkpoints_to_keep=12,
        max_resume_checkpoints_to_keep=2,
        official_completion=True,
    )
    assert config.num_train_steps == 6_000
    assert config.lr_schedule.warmup_steps == 200
    assert config.lr_schedule.decay_steps == 6_000
    assert config.save_interval == 500
    assert config.max_checkpoints_to_keep == 12
    assert config.max_resume_checkpoints_to_keep == 2


def test_dual_continuation_plan_is_ordered_and_formal_launch_is_gated(tmp_path) -> None:
    jobs = Path(__file__).resolve().parents[1] / "jobs"
    controller = jobs / "run_dual_continuation_8gpu.sh"
    background = jobs / "launch_dual_continuation_background.sh"
    env = os.environ.copy()
    env["FOUR_SUITE_CHECKPOINT_BASE_DIR"] = str(tmp_path / "checkpoints")
    result = subprocess.run(
        ["bash", str(controller), "plan"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    assert "1932_from_main" in lines[0]
    assert "continuous_updates=6000" in lines[0]
    assert "final=6000" in lines[0]
    assert "old115_exact_continue" in lines[1]
    assert "population=1808" in lines[1]
    assert "additional_updates=3000" in lines[1]
    assert "final=6000" in lines[1]
    assert "LIBERO_DUAL_CONTINUATION_APPROVED" in background.read_text()
