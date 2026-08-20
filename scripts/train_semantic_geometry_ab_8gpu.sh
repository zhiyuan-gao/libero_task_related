#!/usr/bin/env bash
set -euo pipefail

openpi_root=${OPENPI_ROOT:-/workspace/vla/libero_task_related}
openpi_python=${OPENPI_PYTHON:-${openpi_root}/.venv/bin/python}
a_launcher=${A_LAUNCHER:-${openpi_root}/scripts/semantic_geometry_libero3_8gpu.sh}
b_launcher=${B_LAUNCHER:-${openpi_root}/scripts/semantic_geometry_motion_libero3_8gpu.sh}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/p3/checkpoints/semantic_geometry_motion}
dry_run=${DRY_RUN:-NO}

if [[ $dry_run != YES && ${FORMAL_AB_TRAINING_APPROVED:-NO} != YES ]]; then
  echo "Formal A/B training is blocked: set FORMAL_AB_TRAINING_APPROVED=YES only after researcher confirmation" >&2
  exit 2
fi
if [[ ! -x $a_launcher || ! -x $b_launcher ]]; then
  echo "A and B launchers must both exist and be executable" >&2
  exit 2
fi
if [[ $dry_run != YES && ( -z ${A_EXP_NAME:-} || -z ${B_EXP_NAME:-} ) ]]; then
  echo "A_EXP_NAME and B_EXP_NAME are required for formal sequential training" >&2
  exit 2
fi
if [[ -n ${A_EXP_NAME:-} && ${A_EXP_NAME:-} == ${B_EXP_NAME:-} ]]; then
  echo "A_EXP_NAME and B_EXP_NAME must differ" >&2
  exit 2
fi

"$openpi_python" - <<'PY'
from pathlib import Path

from openpi.training import config

a = config.get_config("pi05_libero3_semantic_geometry_aux")
b = config.get_config("pi05_libero3_semantic_geometry_motion_aux")
assert a.policy_aux.mode == "semantic_geometry"
assert b.policy_aux.mode == "semantic_geometry_motion"
assert a.policy_aux.num_ground_queries == b.policy_aux.num_ground_queries == 0
assert b.policy_aux.num_motion_queries == 8
assert a.policy_aux.lambda_geo == b.policy_aux.lambda_geo == 0.15
assert a.policy_aux.lambda_sem == b.policy_aux.lambda_sem == 0.01
assert b.policy_aux.lambda_motion == 0.05
assert a.policy_aux.lerobot_task_indices == b.policy_aux.lerobot_task_indices == (0, 3, 8)
assert a.pytorch_weight_path == b.pytorch_weight_path
assert Path(a.pytorch_weight_path).resolve() == Path("/workspace/vla/models/openpi/pi05_base_pytorch_fp32").resolve()
assert a.ema_decay is None and b.ema_decay is None
assert a.batch_size == b.batch_size == 256
assert a.num_train_steps == b.num_train_steps == 3209
assert a.lr_schedule.warmup_steps == b.lr_schedule.warmup_steps == 1069
print("AB_INDEPENDENT_SHARED_BASE_GATE=PASS")
print("A_THEN_B_ORDER_GATE=PASS")
PY

a_exp_name=${A_EXP_NAME:-sg_A_dry_run}
b_exp_name=${B_EXP_NAME:-sgm_B_dry_run}
a_resume=${A_RESUME:-NO}
b_resume=${B_RESUME:-NO}

if [[ $dry_run == YES ]]; then
  echo "DRY_RUN A: EXP_NAME=$a_exp_name RESUME=$a_resume $a_launcher full"
  echo "DRY_RUN B_AFTER_A_SUCCESS: EXP_NAME=$b_exp_name RESUME=$b_resume $b_launcher full"
  exit 0
fi

echo "Starting A: $a_exp_name"
FULL_TRAINING_APPROVED=YES \
EXP_NAME="$a_exp_name" \
RESUME="$a_resume" \
CHECKPOINT_BASE_DIR="$checkpoint_base_dir" \
"$a_launcher" full

echo "A completed successfully; starting independent B from the strict FP32 base: $b_exp_name"
FULL_TRAINING_APPROVED=YES \
EXP_NAME="$b_exp_name" \
RESUME="$b_resume" \
CHECKPOINT_BASE_DIR="$checkpoint_base_dir" \
"$b_launcher" full

echo "Sequential A/B training completed successfully"
