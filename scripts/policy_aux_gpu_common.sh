#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <preflight|full> <p1|p2> <4|8>" >&2
  exit 2
fi

phase=$1
variant=$2
gpu_profile=$3
openpi_root=${OPENPI_ROOT:-/workspace/vla/third_party/openpi}
openpi_python=${OPENPI_PYTHON:-${openpi_root}/.venv/bin/python}
num_workers=${NUM_WORKERS:-8}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/checkpoints/openpi_policy_aux}

case "$gpu_profile" in
  4)
    readonly profile_global_micro_batch=128
    readonly profile_accumulation_steps=2
    ;;
  8)
    readonly profile_global_micro_batch=256
    readonly profile_accumulation_steps=1
    ;;
  *)
    echo "GPU profile must be 4 or 8" >&2
    exit 2
    ;;
esac

nproc_per_node=${NPROC_PER_NODE:-$gpu_profile}
global_micro_batch=${GLOBAL_MICRO_BATCH:-$profile_global_micro_batch}
accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-$profile_accumulation_steps}
readonly expected_effective_batch=256
readonly expected_local_micro_batch=32
readonly frozen_num_train_steps=11132
readonly frozen_warmup_steps=3710
readonly frozen_ema_decay=0.999

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

for variable in nproc_per_node global_micro_batch accumulation_steps; do
  require_positive_integer "$variable"
done
require_nonnegative_integer num_workers
if [[ $nproc_per_node -ne $gpu_profile ]]; then
  echo "The ${gpu_profile}-GPU profile requires exactly ${gpu_profile} DDP processes" >&2
  exit 2
fi
if [[ $global_micro_batch -ne $profile_global_micro_batch ]]; then
  echo "The ${gpu_profile}-GPU profile requires GLOBAL_MICRO_BATCH=${profile_global_micro_batch}" >&2
  exit 2
fi
if [[ $accumulation_steps -ne $profile_accumulation_steps ]]; then
  echo "The ${gpu_profile}-GPU profile requires GRADIENT_ACCUMULATION_STEPS=${profile_accumulation_steps}" >&2
  exit 2
fi
if (( global_micro_batch % nproc_per_node != 0 )); then
  echo "GLOBAL_MICRO_BATCH must be divisible by NPROC_PER_NODE" >&2
  exit 2
fi
if (( global_micro_batch / nproc_per_node != expected_local_micro_batch )); then
  echo "The official recipe requires a local/per-GPU micro-batch of 32" >&2
  exit 2
fi
if (( global_micro_batch * accumulation_steps != expected_effective_batch )); then
  echo "GLOBAL_MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS must equal 256" >&2
  exit 2
fi

visible_gpus=$("$openpi_python" -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $gpu_profile ]]; then
  echo "Expected exactly ${gpu_profile} visible CUDA devices, found $visible_gpus" >&2
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
  --ema-decay "$frozen_ema_decay"
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
    exp_name=${EXP_NAME:-${variant}_${gpu_profile}gpu_preflight}
    run_train \
      --exp-name "$exp_name" \
      --num-train-steps 25 \
      --save-interval 25 \
      --log-interval 1 \
      --no-wandb-enabled \
      --overwrite
    # One additional optimizer update proves raw/EMA weights, optimizer state,
    # data position, global step, and the stateless LR schedule resume exactly.
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
      echo "Set FULL_TRAINING_APPROVED=YES only after the ${gpu_profile}-GPU preflight passes" >&2
      exit 2
    fi
    if [[ -z ${EXP_NAME:-} ]]; then
      echo "EXP_NAME is required for a full run" >&2
      exit 2
    fi
    resume_args=()
    if [[ ${RESUME:-NO} == YES ]]; then
      resume_args+=(--resume)
    fi
    run_train \
      --exp-name "$EXP_NAME" \
      --num-train-steps "$frozen_num_train_steps" \
      --save-interval 1000 \
      "${resume_args[@]}"
    ;;
  *)
    echo "phase must be preflight or full" >&2
    exit 2
    ;;
esac

# The matching config freezes warmup at $frozen_warmup_steps updates.
