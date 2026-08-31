#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
repository_root=$(cd "$project_root/../.." && pwd)
openpi_root=${OPENPI_ROOT:-$repository_root}
python_bin=${ROBOCASA24_PYTHON:-$openpi_root/.venv/bin/python}
torchrun_bin=${ROBOCASA24_TORCHRUN:-$openpi_root/.venv/bin/torchrun}

export PYTHONPATH="$project_root/src:$openpi_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=${OPENPI_USE_DEFAULT_CUDA_ALLOCATOR:-1}
export OPENPI_LOG_MEMORY_STATS=${OPENPI_LOG_MEMORY_STATS:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

[[ -x "$python_bin" && -x "$torchrun_bin" ]] || {
  echo "RoboCasa/OpenPI Python environment is unavailable" >&2
  exit 2
}

command=${1:-}
shift || true
case "$command" in
  dry-run)
    exec "$python_bin" -m robocasa24_finetune.ablations.train \
      --openpi-root "$openpi_root" --dry-run "$@"
    ;;
  train)
    if [[ ${ROBOCASA24_ABLATION_TRAINING_APPROVED:-} != YES ]]; then
      echo "Ablation training is gated; dedicated smoke and approval are required" >&2
      exit 3
    fi
    exec "$torchrun_bin" --standalone --nnodes=1 --nproc_per_node=8 \
      -m robocasa24_finetune.ablations.train \
      --openpi-root "$openpi_root" "$@"
    ;;
  smoke)
    if [[ ${ROBOCASA24_ABLATION_SMOKE_APPROVED:-} != YES ]]; then
      echo "Ablation smoke is gated; all eight GPUs must be available" >&2
      exit 3
    fi
    exec "$torchrun_bin" --standalone --nnodes=1 --nproc_per_node=8 \
      -m robocasa24_finetune.ablations.smoke \
      --openpi-root "$openpi_root" "$@"
    ;;
  *)
    echo "Usage: $0 {dry-run|smoke|train} <arguments>" >&2
    exit 2
    ;;
esac
