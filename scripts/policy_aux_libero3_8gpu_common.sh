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
num_workers=${NUM_WORKERS:-8}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/checkpoints/openpi_policy_aux}

readonly nproc_per_node=8
readonly global_micro_batch=256
readonly accumulation_steps=1
readonly local_micro_batch=32
readonly effective_batch=256
readonly frozen_num_train_steps=3209
readonly frozen_warmup_steps=1069
readonly frozen_ema_decay=0.999
readonly frozen_lambda_geo=0.15
readonly frozen_lambda_ground=0.50
readonly frozen_lambda_sem=0.01

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [[ ! $num_workers =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be a non-negative integer, found '$num_workers'" >&2
  exit 2
fi
if (( global_micro_batch / nproc_per_node != local_micro_batch )); then
  echo "The pilot requires local/per-GPU micro-batch 32" >&2
  exit 2
fi
if (( global_micro_batch * accumulation_steps != effective_batch )); then
  echo "The pilot requires effective/global batch 256" >&2
  exit 2
fi

visible_gpus=$("$openpi_python" -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $nproc_per_node ]]; then
  echo "Expected exactly 8 visible CUDA devices, found $visible_gpus" >&2
  exit 2
fi

lambda_args=(--policy-aux.lambda-geo "$frozen_lambda_geo")
case "$variant" in
  p1)
    config_name=pi05_libero3_p1_aux
    ;;
  p2)
    config_name=pi05_libero3_p2_aux
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
    exp_name=${EXP_NAME:-${variant}_libero3_8gpu_preflight}
    run_train \
      --exp-name "$exp_name" \
      --num-train-steps 25 \
      --save-interval 25 \
      --log-interval 1 \
      --no-wandb-enabled \
      --overwrite
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
      echo "Set FULL_TRAINING_APPROVED=YES only after both three-task preflights pass" >&2
      exit 2
    fi
    if [[ -z ${EXP_NAME:-} ]]; then
      echo "EXP_NAME is required for a full run" >&2
      exit 2
    fi
    resume_args=()
    if [[ ${RESUME:-NO} == YES ]]; then
      resume_args+=(--resume)
    else
      # Rank 0 owns new-directory creation; --overwrite provides the DDP barrier
      # that prevents nonzero ranks from mistaking the new directory for a stale run.
      resume_args+=(--overwrite)
    fi
    run_train \
      --exp-name "$EXP_NAME" \
      --num-train-steps "$frozen_num_train_steps" \
      --save-interval 1000 \
      --no-wandb-enabled \
      "${resume_args[@]}"
    ;;
  *)
    echo "phase must be preflight or full" >&2
    exit 2
    ;;
esac

# The matching configs freeze warmup at ${frozen_warmup_steps} updates.
