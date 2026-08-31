#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${REPOSITORY_ROOT}}"
ROBOCASA24_PYTHON="${ROBOCASA24_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"
ROBOCASA24_TORCHRUN="${ROBOCASA24_TORCHRUN:-${OPENPI_ROOT}/.venv/bin/torchrun}"
export PYTHONPATH="${PROJECT_ROOT}/src:${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Match the allocator/logging profile used by the validated LIBERO 8-GPU
# launchers. Python entrypoints set the same defaults for direct module use.
export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR="${OPENPI_USE_DEFAULT_CUDA_ALLOCATOR:-1}"
export OPENPI_LOG_MEMORY_STATS="${OPENPI_LOG_MEMORY_STATS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -x "${ROBOCASA24_PYTHON}" || ! -x "${ROBOCASA24_TORCHRUN}" ]]; then
  echo "RoboCasa/OpenPI Python environment is unavailable" >&2
  exit 2
fi

command="${1:-}"
shift || true
case "${command}" in
  dry-run)
    exec "${ROBOCASA24_PYTHON}" -m robocasa24_finetune.train --openpi-root "${OPENPI_ROOT}" --dry-run "$@"
    ;;
  train)
    if [[ "${ROBOCASA24_FULL_TRAINING_APPROVED:-}" != "YES" ]]; then
      echo "Training is gated; complete the real 8-GPU smoke and obtain approval first" >&2
      exit 3
    fi
    exec "${ROBOCASA24_TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
      -m robocasa24_finetune.train --openpi-root "${OPENPI_ROOT}" "$@"
    ;;
  smoke)
    if [[ "${ROBOCASA24_SMOKE_APPROVED:-}" != "YES" ]]; then
      echo "Smoke is gated; wait until all eight GPUs are available" >&2
      exit 3
    fi
    exec "${ROBOCASA24_TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
      -m robocasa24_finetune.smoke --openpi-root "${OPENPI_ROOT}" "$@"
    ;;
  pilot)
    if [[ "${ROBOCASA24_PILOT_APPROVED:-}" != "YES" ]]; then
      echo "Three-task pilot is gated; explicit approval is required" >&2
      exit 3
    fi
    exec "${ROBOCASA24_TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
      -m robocasa24_finetune.pilot --openpi-root "${OPENPI_ROOT}" "$@"
    ;;
  pilot-dry-run)
    exec "${ROBOCASA24_PYTHON}" -m robocasa24_finetune.pilot \
      --openpi-root "${OPENPI_ROOT}" --dry-run "$@"
    ;;
  *)
    echo "Usage: $0 {dry-run|smoke|pilot-dry-run|pilot|train} <arguments>" >&2
    exit 2
    ;;
esac
