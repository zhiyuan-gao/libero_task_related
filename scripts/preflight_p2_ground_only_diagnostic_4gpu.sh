#!/usr/bin/env bash
set -euo pipefail

openpi_root=${OPENPI_ROOT:-/workspace/vla/third_party/openpi}
openpi_python=${OPENPI_PYTHON:-${openpi_root}/.venv/bin/python}
exp_name=${EXP_NAME:-p2_ground_only_diagnostic_4gpu_25step}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/checkpoints/openpi_diagnostic_benchmarks}
num_workers=${NUM_WORKERS:-8}

readonly nproc_per_node=4
readonly global_micro_batch=128
readonly accumulation_steps=2

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

visible_gpus=$("$openpi_python" -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $nproc_per_node ]]; then
  echo "Expected exactly 4 visible CUDA devices, found $visible_gpus" >&2
  exit 2
fi

# This is not a training recipe. It is an isolated timing ablation that keeps
# the full P2 Ground path and skips only the native semantic-LM pass.
exec "$openpi_python" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$nproc_per_node" \
  "$openpi_root/scripts/train_pytorch.py" \
  pi05_libero_p2_ground_only_diagnostic \
  --exp-name "$exp_name" \
  --batch-size "$global_micro_batch" \
  --gradient-accumulation-steps "$accumulation_steps" \
  --num-workers "$num_workers" \
  --checkpoint-base-dir "$checkpoint_base_dir" \
  --num-train-steps 25 \
  --save-interval 1000 \
  --no-save-final-checkpoint \
  --log-interval 1 \
  --no-wandb-enabled \
  --overwrite
