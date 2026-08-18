#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <preflight|full> <p1|p2>" >&2
  exit 2
fi

phase=$1
variant=$2
openpi_root=${OPENPI_ROOT:-/workspace/vla/third_party/openpi}
openpi_python=${OPENPI_PYTHON:-${openpi_root}/.venv/bin/python}
nproc_per_node=${NPROC_PER_NODE:-8}
global_micro_batch=${GLOBAL_MICRO_BATCH:-8}
accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-32}
expected_effective_batch=${EXPECTED_EFFECTIVE_BATCH:-256}
num_workers=${NUM_WORKERS:-8}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/checkpoints/openpi_policy_aux}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

require_positive_integer() {
  local name=$1
  local value=${!name:-}
  if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, found '$value'" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name=$1
  local value=${!name:-}
  if [[ ! $value =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, found '$value'" >&2
    exit 2
  fi
}

for variable in nproc_per_node global_micro_batch accumulation_steps expected_effective_batch; do
  require_positive_integer "$variable"
done
require_nonnegative_integer num_workers
if [[ $nproc_per_node -ne 8 ]]; then
  echo "P1/P2 target preflight requires exactly 8 DDP processes" >&2
  exit 2
fi
if (( global_micro_batch % nproc_per_node != 0 )); then
  echo "GLOBAL_MICRO_BATCH must be divisible by NPROC_PER_NODE" >&2
  exit 2
fi
if (( global_micro_batch * accumulation_steps != expected_effective_batch )); then
  echo "GLOBAL_MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS must equal EXPECTED_EFFECTIVE_BATCH" >&2
  exit 2
fi

visible_gpus=$(
  "$openpi_python" -c 'import torch; print(torch.cuda.device_count())'
)
if [[ $visible_gpus -ne 8 ]]; then
  echo "Expected exactly 8 visible CUDA devices, found $visible_gpus" >&2
  exit 2
fi

# Human-approved and frozen on 2026-08-18. These launchers intentionally expose
# no environment-variable override or sweep surface for the primary protocol.
readonly frozen_lambda_geo=0.15
readonly frozen_lambda_ground=0.50
readonly frozen_lambda_sem=0.01
lambda_args=(--policy-aux.lambda-geo "$frozen_lambda_geo")
case "$variant" in
  p1)
    config_name=pi05_libero_p1_aux
    ;;
  p2)
    config_name=pi05_libero_p2_aux
    lambda_args+=(
      --policy-aux.lambda-sem "$frozen_lambda_sem"
      --policy-aux.lambda-ground "$frozen_lambda_ground"
    )
    ;;
  *)
    echo "variant must be p1 or p2" >&2
    exit 2
    ;;
esac

common_args=(
  "$config_name"
  --batch-size "$global_micro_batch"
  --gradient-accumulation-steps "$accumulation_steps"
  --num-workers "$num_workers"
  --checkpoint-base-dir "$checkpoint_base_dir"
  --policy-aux.loss-coefficients-approved
  "${lambda_args[@]}"
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
    exp_name=${EXP_NAME:-${variant}_8gpu_preflight}
    run_train \
      --exp-name "$exp_name" \
      --num-train-steps 25 \
      --save-interval 25 \
      --log-interval 1 \
      --no-wandb-enabled \
      --overwrite
    # One additional optimizer update proves that the just-written checkpoint,
    # optimizer state, global step, and stateless LR schedule resume cleanly.
    run_train \
      --exp-name "$exp_name" \
      --num-train-steps 26 \
      --save-interval 1000 \
      --no-save-final-checkpoint \
      --log-interval 1 \
      --no-wandb-enabled \
      --resume
    ;;
  full)
    if [[ ${FULL_TRAINING_APPROVED:-NO} != YES ]]; then
      echo "Set FULL_TRAINING_APPROVED=YES only after the 8-GPU preflight passes" >&2
      exit 2
    fi
    if [[ -z ${EXP_NAME:-} ]]; then
      echo "EXP_NAME is required for a full run" >&2
      exit 2
    fi
    num_train_steps=${NUM_TRAIN_STEPS:-30000}
    save_interval=${SAVE_INTERVAL:-1000}
    require_positive_integer num_train_steps
    require_positive_integer save_interval
    resume_args=()
    if [[ ${RESUME:-NO} == YES ]]; then
      resume_args+=(--resume)
    fi
    run_train \
      --exp-name "$EXP_NAME" \
      --num-train-steps "$num_train_steps" \
      --save-interval "$save_interval" \
      "${resume_args[@]}"
    ;;
  *)
    echo "phase must be preflight or full" >&2
    exit 2
    ;;
esac
