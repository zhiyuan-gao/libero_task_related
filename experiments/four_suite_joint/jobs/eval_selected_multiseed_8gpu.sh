#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the completed LIBERO-40 experiment directory}"
BATCH_ROOT="${BATCH_ROOT:?Set BATCH_ROOT to a new or resumable evaluation directory}"
RESUME="${RESUME:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
readonly NUM_SHARDS=16
COMPILE_CACHE_ROOT="${TORCH_COMPILE_CACHE_ROOT:-${BATCH_ROOT}/torch_compile_cache}"

# Requested late checkpoints and the two established alternate LIBERO seeds.
steps=(30000 29000)
seeds=(17 27)

for step in "${steps[@]}"; do
  if [[ ! -s "${CHECKPOINT_ROOT}/${step}/model.safetensors" ]]; then
    echo "Required evaluation checkpoint is missing: ${CHECKPOINT_ROOT}/${step}" >&2
    exit 2
  fi
done

mkdir -p "${BATCH_ROOT}"
printf 'checkpoint_step\tevaluation_seed\n' >"${BATCH_ROOT}/evaluation_order.tsv"
for step in "${steps[@]}"; do
  for seed in "${seeds[@]}"; do
    printf '%s\t%s\n' "${step}" "${seed}" >>"${BATCH_ROOT}/evaluation_order.tsv"
  done
done

total=$((${#steps[@]} * ${#seeds[@]}))
index=0
for step in "${steps[@]}"; do
  for seed in "${seeds[@]}"; do
    index=$((index + 1))
    checkpoint="${CHECKPOINT_ROOT}/${step}"
    run_root="${BATCH_ROOT}/step_${step}_seed${seed}"
    echo "[${index}/${total}] checkpoint ${step}, evaluation seed ${seed}"
    if [[ -f "${run_root}/.complete" ]]; then
      echo "Already complete; skipping step ${step}, seed ${seed}"
      continue
    fi
    CHECKPOINT="${checkpoint}" \
    RUN_ROOT="${run_root}" \
    VARIANT=trqc \
    PORT_BASE=8400 \
    EVAL_SEED="${seed}" \
    RESUME="${RESUME}" \
    SAVE_VIDEO="${SAVE_VIDEO}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    TORCH_COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT}" \
      "${PROJECT_ROOT}/jobs/eval_checkpoint_8gpu.sh"
  done
done

touch "${BATCH_ROOT}/.complete"
echo "Selected multi-seed evaluation complete: ${BATCH_ROOT}"
