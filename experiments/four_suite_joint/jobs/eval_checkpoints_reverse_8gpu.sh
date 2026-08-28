#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN="${FOUR_SUITE_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the completed LIBERO-40 experiment directory}"
BATCH_ROOT="${BATCH_ROOT:?Set BATCH_ROOT to a new or resumable evaluation directory}"
RESUME="${RESUME:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
NUM_SHARDS="${NUM_SHARDS:-16}"
COMPILE_CACHE_ROOT="${TORCH_COMPILE_CACHE_ROOT:-${BATCH_ROOT}/torch_compile_cache}"

# The reviewed sweep intentionally covers only the newest ten checkpoints.
steps=(30000 29500 29000 28500 28000 27500 27000 26500 26000 25500)
for step in "${steps[@]}"; do
  if [[ ! -s "${CHECKPOINT_ROOT}/${step}/model.safetensors" ]]; then
    echo "Required evaluation checkpoint is missing: ${CHECKPOINT_ROOT}/${step}" >&2
    exit 2
  fi
done

mkdir -p "${BATCH_ROOT}"
printf '%s\n' "${steps[@]}" >"${BATCH_ROOT}/checkpoint_order_descending.txt"

for index in "${!steps[@]}"; do
  step="${steps[index]}"
  checkpoint="${CHECKPOINT_ROOT}/${step}"
  run_root="${BATCH_ROOT}/step_${step}"
  echo "[$((index + 1))/${#steps[@]}] formal four-suite checkpoint ${step}"
  if [[ -f "${run_root}/.complete" ]]; then
    echo "Already complete; skipping ${step}"
    continue
  fi
  CHECKPOINT="${checkpoint}" \
  RUN_ROOT="${run_root}" \
  VARIANT=trqc \
  PORT_BASE=8400 \
  RESUME="${RESUME}" \
  SAVE_VIDEO="${SAVE_VIDEO}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  TORCH_COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT}" \
    "${PROJECT_ROOT}/jobs/eval_checkpoint_8gpu.sh"
done

summary_inputs=()
for step in "${steps[@]}"; do
  summary_inputs+=("${BATCH_ROOT}/step_${step}/summary.json")
done
export PYTHONPATH="${PROJECT_ROOT}/src:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m four_suite_experiments.summarize_eval batch \
  --output "${BATCH_ROOT}/all_checkpoints.json" \
  "${summary_inputs[@]}" \
  >"${BATCH_ROOT}/all_checkpoints.txt"
touch "${BATCH_ROOT}/.complete"
echo "Reverse four-suite checkpoint sweep complete: ${BATCH_ROOT}/all_checkpoints.json"
