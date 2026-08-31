#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}}"
FOUR_SUITE_PYTHON="${FOUR_SUITE_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"
FOUR_SUITE_TORCHRUN="${FOUR_SUITE_TORCHRUN:-${OPENPI_ROOT}/.venv/bin/torchrun}"
CHECKPOINT_BASE_DIR="${FOUR_SUITE_CHECKPOINT_BASE_DIR:-${PROJECT_ROOT}/checkpoints}"
export OPENPI_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src:${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Match the validated LIBERO-10 B performance environment. The trainer would
# otherwise enable expandable CUDA segments automatically for an 8-GPU job;
# this fixed-shape workload is materially faster with PyTorch's default native
# caching allocator. Per-update allocator statistics are diagnostic-only and
# add synchronization, so keep them off for smoke and formal training.
export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=1
export OPENPI_LOG_MEMORY_STATS=0
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${FOUR_SUITE_PYTHON}" ]]; then
  echo "Python environment is unavailable: ${FOUR_SUITE_PYTHON}" >&2
  exit 2
fi

command="${1:-}"
if [[ -z "${command}" ]]; then
  echo "Usage: $0 {download-data|prepare|assemble-supplemental115|preflight|train|supplemental-finetune|test} [arguments...]" >&2
  exit 2
fi
shift

case "${command}" in
  download-data)
    exec "${FOUR_SUITE_PYTHON}" -m four_suite_experiments.download_lerobot "$@"
    ;;
  prepare)
    exec "${FOUR_SUITE_PYTHON}" -m four_suite_experiments.prepare_joint_artifacts "$@"
    ;;
  assemble-supplemental115)
    exec "${FOUR_SUITE_PYTHON}" -m four_suite_experiments.assemble_supplemental115 "$@"
    ;;
  preflight)
    exec "${FOUR_SUITE_PYTHON}" -m four_suite_experiments.validate "$@"
    ;;
  train)
    if [[ "${FOUR_SUITE_FULL_TRAINING_APPROVED:-}" != "YES" ]]; then
      echo "Training is gated. Set FOUR_SUITE_FULL_TRAINING_APPROVED=YES after freezing the budget." >&2
      exit 3
    fi
    exec "${FOUR_SUITE_TORCHRUN}" \
      --standalone \
      --nnodes=1 \
      --nproc_per_node=8 \
      -m four_suite_experiments.train \
      --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
      "$@"
    ;;
  supplemental-finetune)
    if [[ "${FOUR_SUITE_SUPPLEMENTAL_FINETUNE_APPROVED:-}" != "YES" ]]; then
      echo "Supplemental fine-tuning is gated. Set FOUR_SUITE_SUPPLEMENTAL_FINETUNE_APPROVED=YES." >&2
      exit 3
    fi
    exec "${FOUR_SUITE_TORCHRUN}" \
      --standalone \
      --nnodes=1 \
      --nproc_per_node=8 \
      -m four_suite_experiments.supplemental_finetune \
      --openpi-root "${OPENPI_ROOT}" \
      --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
      "$@"
    ;;
  test)
    exec "${FOUR_SUITE_PYTHON}" -m pytest -q "${PROJECT_ROOT}/tests" "$@"
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    exit 2
    ;;
esac
