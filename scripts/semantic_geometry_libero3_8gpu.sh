#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != preflight && $1 != full ]]; then
  echo "usage: $0 <preflight|full>" >&2
  exit 2
fi

phase=$1
openpi_root=${OPENPI_ROOT:-/workspace/vla/libero_task_related}
openpi_python=${OPENPI_PYTHON:-${openpi_root}/.venv/bin/python}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/p3/checkpoints/semantic_geometry_motion}
num_workers=${NUM_WORKERS:-8}

readonly config_name=pi05_libero3_semantic_geometry_aux
readonly nproc_per_node=8
readonly global_batch=256
readonly accumulation_steps=1
readonly local_batch=32
readonly total_updates=3209
readonly warmup_updates=1069
readonly lambda_geo=0.15
readonly lambda_sem=0.01

export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=1
export OPENPI_LOG_MEMORY_STATS=0
export TOKENIZERS_PARALLELISM=false

if [[ ! $num_workers =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be a non-negative integer, found '$num_workers'" >&2
  exit 2
fi
if (( global_batch / nproc_per_node != local_batch )); then
  echo "Semantic+Geometry requires local/per-GPU batch 32" >&2
  exit 2
fi
if (( global_batch * accumulation_steps != 256 )); then
  echo "Semantic+Geometry requires effective global batch 256" >&2
  exit 2
fi

visible_gpus=$("$openpi_python" -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $nproc_per_node ]]; then
  echo "Expected exactly 8 visible CUDA devices, found $visible_gpus" >&2
  exit 2
fi

"$openpi_python" - <<'PY'
from openpi.training import config

cfg = config.get_config("pi05_libero3_semantic_geometry_aux")
assert cfg.ema_decay is None
assert cfg.policy_aux.mode == "semantic_geometry"
assert cfg.policy_aux.num_ground_queries == 0
assert cfg.policy_aux.lambda_geo == 0.15
assert cfg.policy_aux.lambda_sem == 0.01
assert cfg.policy_aux.lambda_ground is None
assert cfg.policy_aux.lerobot_task_indices == (0, 3, 8)
assert cfg.batch_size == 256
assert cfg.gradient_accumulation_steps == 1
assert cfg.num_train_steps == 3209
assert cfg.lr_schedule.warmup_steps == 1069
print("SEMANTIC_GEOMETRY_CONFIG_GATE=PASS")
PY

common_args=(
  "$config_name"
  --batch-size "$global_batch"
  --gradient-accumulation-steps "$accumulation_steps"
  --num-workers "$num_workers"
  --checkpoint-base-dir "$checkpoint_base_dir"
  --policy-aux.loss-coefficients-approved
  --policy-aux.lambda-geo "$lambda_geo"
  --policy-aux.lambda-sem "$lambda_sem"
  --no-wandb-enabled
)

run_train() {
  "$openpi_python" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$nproc_per_node" \
    "$openpi_root/scripts/train_pytorch.py" \
    "${common_args[@]}" \
    "$@"
}

case "$phase" in
  preflight)
    exp_name=${EXP_NAME:-sg_libero3_8gpu_preflight}
    run_train \
      --exp-name "$exp_name" \
      --num-train-steps 25 \
      --save-interval 25 \
      --log-interval 1 \
      --overwrite
    run_train \
      --exp-name "$exp_name" \
      --num-train-steps 26 \
      --save-interval 1000 \
      --no-save-final-checkpoint \
      --log-interval 1 \
      --resume
    ;;
  full)
    if [[ ${FULL_TRAINING_APPROVED:-NO} != YES ]]; then
      echo "Set FULL_TRAINING_APPROVED=YES only after the 25+1 preflight passes" >&2
      exit 2
    fi
    if [[ -z ${EXP_NAME:-} ]]; then
      echo "EXP_NAME is required for a full run" >&2
      exit 2
    fi
    resume_args=(--overwrite)
    if [[ ${RESUME:-NO} == YES ]]; then
      resume_args=(--resume)
    fi
    run_train \
      --exp-name "$EXP_NAME" \
      --num-train-steps "$total_updates" \
      --save-interval 1000 \
      --log-interval 1 \
      "${resume_args[@]}"
    ;;
esac

# Frozen protocol: warmup=${warmup_updates}, no EMA, RAW model.safetensors only.
