#!/usr/bin/env python3
"""Validate frozen P1/P2 configs and matching 4/8-GPU launch guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    reports = {}
    loaded_configs = {}
    for variant, config_name, mode in (
        ("p1", "pi05_libero_p1_aux", "geometry"),
        ("p2", "pi05_libero_p2_aux", "ground_geometry_semantic_lm"),
    ):
        config = _config.get_config(config_name)
        loaded_configs[variant] = config
        checks = {
            "official_pi05_semantics": (
                config.model.pi05 is True
                and config.model.action_horizon == 10
                and config.model.discrete_state_input is False
            ),
            "expected_mode": config.policy_aux is not None and config.policy_aux.mode == mode,
            "loss_coefficients_approved": (
                config.policy_aux is not None and config.policy_aux.loss_coefficients_approved is True
            ),
            "geometry_lambda_frozen": config.policy_aux.lambda_geo == 0.15,
            "query_counts_frozen": (
                config.policy_aux.num_ground_queries == 8 and config.policy_aux.num_geometry_queries == 8
            ),
            "effective_batch_256": (config.batch_size * config.gradient_accumulation_steps == 256),
            "official_8gpu_local_micro_batch_32": (
                config.batch_size == 256 and config.gradient_accumulation_steps == 1
            ),
            "data_scaled_training_steps": config.num_train_steps == 11_132,
            "data_scaled_warmup_steps": config.lr_schedule.warmup_steps == 3_710,
            "official_base_not_libero_checkpoint": config.pytorch_weight_path.endswith("/pi05_base_pytorch"),
            "canonical_dataset_revision": config.policy_aux.lerobot_root.endswith(
                "/a4336d589d589045d1c56423ffdf3b88a0e19b1f"
            ),
            "official_ema_decay": config.ema_decay == 0.999,
        }
        if variant == "p2":
            checks.update(
                {
                    "semantic_and_ground_lambdas_frozen": (
                        config.policy_aux.lambda_sem == 0.01 and config.policy_aux.lambda_ground == 0.50
                    ),
                    "no_semantic_query_config_field": not hasattr(config.policy_aux, "num_semantic_queries"),
                }
            )
        reports[variant] = {"config_name": config_name, "checks": checks}

    p1 = loaded_configs["p1"]
    p2 = loaded_configs["p2"]
    fairness_checks = {
        "same_official_pi05_base_initialization": (p1.pytorch_weight_path == p2.pytorch_weight_path),
        "same_model_and_official_input_semantics": p1.model == p2.model,
        "same_official_policy_data_config": p1.data == p2.data,
        "same_optimizer": p1.optimizer == p2.optimizer,
        "same_lr_schedule": p1.lr_schedule == p2.lr_schedule,
        "same_effective_batch": (
            p1.batch_size * p1.gradient_accumulation_steps == p2.batch_size * p2.gradient_accumulation_steps == 256
        ),
        "same_data_scaled_training_steps": p1.num_train_steps == p2.num_train_steps == 11_132,
        "same_data_scaled_warmup": p1.lr_schedule.warmup_steps == p2.lr_schedule.warmup_steps == 3_710,
        "same_official_ema_policy": p1.ema_decay == p2.ema_decay == 0.999,
        "same_precision": p1.pytorch_training_precision == p2.pytorch_training_precision,
        "same_seed": p1.seed == p2.seed,
        "same_checkpoint_cadence": (
            p1.save_interval == p2.save_interval
            and p1.keep_period == p2.keep_period
            and p1.save_final_checkpoint == p2.save_final_checkpoint
        ),
        "same_trainable_original_parameters": p1.freeze_filter == p2.freeze_filter,
        "same_policy_manifest_and_episode_mapping": (
            p1.policy_aux.policy_manifest_path == p2.policy_aux.policy_manifest_path
            and p1.policy_aux.episode_mapping_path == p2.policy_aux.episode_mapping_path
            and p1.policy_aux.lerobot_revision == p2.policy_aux.lerobot_revision
        ),
    }

    common_script = repo / "scripts/policy_aux_gpu_common.sh"
    common_text = common_script.read_text()
    wrapper_checks = {
        "common_script_executable": common_script.stat().st_mode & 0o111 != 0,
        "requires_exact_profile_gpu_count": "Expected exactly ${gpu_profile} visible CUDA devices" in common_text,
        "four_gpu_profile_is_local32_accum2": (
            "profile_global_micro_batch=128" in common_text and "profile_accumulation_steps=2" in common_text
        ),
        "eight_gpu_profile_is_local32_accum1": (
            "profile_global_micro_batch=256" in common_text and "profile_accumulation_steps=1" in common_text
        ),
        "requires_local_micro_batch_32": "local/per-GPU micro-batch of 32" in common_text,
        "requires_effective_batch_256": (
            "GLOBAL_MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS must equal 256" in common_text
        ),
        "frozen_data_scaled_steps": "readonly frozen_num_train_steps=11132" in common_text,
        "frozen_data_scaled_warmup_documented": "readonly frozen_warmup_steps=3710" in common_text,
        "frozen_official_ema": "readonly frozen_ema_decay=0.999" in common_text,
        "passes_lambda_approval": "loss-coefficients-approved" in common_text,
        "frozen_lambda_geo": "readonly frozen_lambda_geo=0.15" in common_text,
        "frozen_lambda_ground": "readonly frozen_lambda_ground=0.50" in common_text,
        "frozen_lambda_sem": "readonly frozen_lambda_sem=0.01" in common_text,
        "no_lambda_environment_override": all(
            token not in common_text for token in ("$LAMBDA_GEO", "$LAMBDA_GROUND", "$LAMBDA_SEM")
        ),
        "full_requires_training_approval": "FULL_TRAINING_APPROVED" in common_text,
        "historical_p0_not_a_launch_guard": "P0_PARITY_APPROVED" not in common_text,
        "preflight_tests_resume": "--resume" in common_text,
        "no_full_step_or_cadence_override": (
            "NUM_TRAIN_STEPS" not in common_text and "SAVE_INTERVAL" not in common_text
        ),
    }
    for profile in (4, 8):
        profile_script = repo / f"scripts/policy_aux_{profile}gpu_common.sh"
        wrapper_checks[f"policy_aux_{profile}gpu_common.sh_executable"] = (
            profile_script.exists() and profile_script.stat().st_mode & 0o111 != 0
        )
    for name in (
        *(f"preflight_{variant}_libero10_{profile}gpu.sh" for profile in (4, 8) for variant in ("p1", "p2")),
        *(f"launch_{variant}_libero10_{profile}gpu.sh" for profile in (4, 8) for variant in ("p1", "p2")),
    ):
        path = repo / f"scripts/{name}"
        wrapper_checks[f"{name}_executable"] = path.exists() and path.stat().st_mode & 0o111 != 0

    all_checks = (
        [value for report in reports.values() for value in report["checks"].values()]
        + list(fairness_checks.values())
        + list(wrapper_checks.values())
    )
    if not all(all_checks):
        raise RuntimeError(f"Launch-template gate failed: reports={reports}, wrappers={wrapper_checks}")
    payload = {
        "status": "PASS",
        "gate": "pi05_p1_p2_data_scaled_4gpu_8gpu_launch_templates_v3",
        "development_machine_gpu_count_required": False,
        "preflight_executed": False,
        "reports": reports,
        "p1_p2_fairness_checks": fairness_checks,
        "wrapper_checks": wrapper_checks,
        "known_blockers_before_full_launch": [
            "matching 4-GPU or 8-GPU preflight",
            "explicit full-training approval",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
