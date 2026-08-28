#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}}"
PYTHON_BIN="${FOUR_SUITE_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"
LIBERO_SOURCE_ROOT="${LIBERO_EVAL_SOURCE_ROOT:-${OPENPI_ROOT}/third_party/libero}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a numeric LIBERO-40 checkpoint directory}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the checkpoint evaluation output directory}"
VARIANT="${VARIANT:-trqc}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROJECT_ROOT}/artifacts/task_relevant}"
BASE_WEIGHT_PATH="${FOUR_SUITE_BASE_WEIGHTS:?Set FOUR_SUITE_BASE_WEIGHTS}"
LIBERO_ASSETS_DIR="${FOUR_SUITE_LIBERO_ASSETS:?Set FOUR_SUITE_LIBERO_ASSETS}"
PORT_BASE="${PORT_BASE:-8400}"
RESUME="${RESUME:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
DRY_RUN="${DRY_RUN:-0}"
TORCH_COMPILE_CACHE_ROOT="${TORCH_COMPILE_CACHE_ROOT:-${RUN_ROOT}/torch_compile_cache}"
NUM_SHARDS="${NUM_SHARDS:-16}"

readonly NUM_GPUS=8
readonly NUM_SHARDS
readonly CHECKPOINT_STEP="$(basename "${CHECKPOINT}")"
task_ids=(0 1 2 3 4 5 6 7 8 9)
suites=(libero_spatial libero_object libero_goal libero_10)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! "${CHECKPOINT_STEP}" =~ ^[0-9]+$ || ! -s "${CHECKPOINT}/model.safetensors" ]]; then
  echo "Checkpoint is not a complete numeric serving checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -s "${CHECKPOINT}/assets/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "Checkpoint is missing LIBERO norm stats: ${CHECKPOINT}" >&2
  exit 2
fi
for required in assets bddl_files init_files; do
  if [[ ! -d "${LIBERO_SOURCE_ROOT}/libero/libero/${required}" ]]; then
    echo "LIBERO benchmark asset is missing: ${LIBERO_SOURCE_ROOT}/libero/libero/${required}" >&2
    exit 2
  fi
done
if [[ ! "${NUM_SHARDS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SHARDS must be an integer; observed ${NUM_SHARDS}" >&2
  exit 2
fi
if ((NUM_SHARDS < NUM_GPUS || NUM_SHARDS % NUM_GPUS != 0)); then
  echo "NUM_SHARDS must be a positive multiple of ${NUM_GPUS}; observed ${NUM_SHARDS}" >&2
  exit 2
fi
if [[ "${PORT_BASE}" -lt 1024 || $((PORT_BASE + NUM_SHARDS - 1)) -gt 65535 ]]; then
  echo "Invalid PORT_BASE: ${PORT_BASE}" >&2
  exit 2
fi
if [[ "${RESUME}" != 1 && -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists; choose a new path or set RESUME=1: ${RUN_ROOT}" >&2
  exit 2
fi

manifest="checkpoint=${CHECKPOINT}
checkpoint_step=${CHECKPOINT_STEP}
variant=${VARIANT}
suites=libero_spatial,libero_object,libero_goal,libero_10
task_ids=0,1,2,3,4,5,6,7,8,9
trials_per_task=50
seed=7
num_shards=${NUM_SHARDS}
resize=224
replan_steps=5
policy_flow_steps=10
protocol=formal_four_suite"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY RUN OK: checkpoint=${CHECKPOINT_STEP} variant=${VARIANT}"
  echo "Would launch ${NUM_SHARDS} policy workers across ${NUM_GPUS} GPUs and 2,000 formal rollouts across four suites."
  exit 0
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/libero_config" "${TORCH_COMPILE_CACHE_ROOT}"
if [[ -f "${RUN_ROOT}/manifest.txt" ]]; then
  existing_manifest="$(<"${RUN_ROOT}/manifest.txt")"
  if [[ "${existing_manifest}" != "${manifest}" ]]; then
    echo "Existing evaluation manifest differs: ${RUN_ROOT}/manifest.txt" >&2
    exit 2
  fi
fi
printf '%s\n' "${manifest}" >"${RUN_ROOT}/manifest.txt"

libero_benchmark_root="${LIBERO_SOURCE_ROOT}/libero/libero"
printf '%s\n' \
  "benchmark_root: ${libero_benchmark_root}" \
  "bddl_files: ${libero_benchmark_root}/bddl_files" \
  "init_states: ${libero_benchmark_root}/init_files" \
  "datasets: ${LIBERO_SOURCE_ROOT}/libero/datasets" \
  "assets: ${libero_benchmark_root}/assets" \
  >"${RUN_ROOT}/libero_config/config.yaml"

server_pids=()
client_pids=()
cleanup() {
  local pid
  for pid in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  wait "${server_pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cpu_count="$(nproc)"
cores_per_pair=$((cpu_count / NUM_SHARDS))
((cores_per_pair >= 1)) || cores_per_pair=1
threads_per_process=$((cores_per_pair / 2))
((threads_per_process >= 1)) || threads_per_process=1

cpu_set_for_worker() {
  local worker="$1"
  local gpu=$((worker % NUM_GPUS))
  local replica=$((worker / NUM_GPUS))
  local replicas_per_gpu=$((NUM_SHARDS / NUM_GPUS))
  if ((cpu_count == 128 && 8 % replicas_per_gpu == 0)); then
    local physical_cores_per_pair=$((8 / replicas_per_gpu))
    local first_core=$((gpu * 8 + replica * physical_cores_per_pair))
    local last_core=$((first_core + physical_cores_per_pair - 1))
    echo "${first_core}-${last_core},$((first_core + 64))-$((last_core + 64))"
    return
  fi
  local first_core=$((worker * cores_per_pair))
  local last_core=$((first_core + cores_per_pair - 1))
  ((last_core < cpu_count)) || last_core=$((cpu_count - 1))
  echo "${first_core}-${last_core}"
}

export PYTHONPATH="${PROJECT_ROOT}/src:${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
for worker in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((worker % NUM_GPUS))
  replica=$((worker / NUM_GPUS))
  port=$((PORT_BASE + worker))
  cpu_set="$(cpu_set_for_worker "${worker}")"
  cache_leaf="gpu${gpu}"
  ((replica == 0)) || cache_leaf="gpu${gpu}_replica${replica}"
  mkdir -p "${TORCH_COMPILE_CACHE_ROOT}/${cache_leaf}/inductor" "${TORCH_COMPILE_CACHE_ROOT}/${cache_leaf}/triton"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS="${threads_per_process}" \
    MKL_NUM_THREADS="${threads_per_process}" \
    OPENBLAS_NUM_THREADS="${threads_per_process}" \
    TORCHINDUCTOR_CACHE_DIR="${TORCH_COMPILE_CACHE_ROOT}/${cache_leaf}/inductor" \
    TRITON_CACHE_DIR="${TORCH_COMPILE_CACHE_ROOT}/${cache_leaf}/triton" \
    taskset -c "${cpu_set}" \
    "${PYTHON_BIN}" -m four_suite_experiments.serve \
      --variant "${VARIANT}" \
      --checkpoint "${CHECKPOINT}" \
      --artifact-dir "${ARTIFACT_DIR}" \
      --base-weight-path "${BASE_WEIGHT_PATH}" \
      --libero-assets-dir "${LIBERO_ASSETS_DIR}" \
      --port "${port}" \
      --num-steps 10 \
      >"${RUN_ROOT}/logs/server_worker${worker}_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
done

ready=0
for _ in $(seq 1 300); do
  ready=0
  for worker in $(seq 0 $((NUM_SHARDS - 1))); do
    if curl --silent --fail --max-time 1 "http://127.0.0.1:$((PORT_BASE + worker))/healthz" >/dev/null; then
      ready=$((ready + 1))
    fi
  done
  [[ "${ready}" -eq "${NUM_SHARDS}" ]] && break
  for pid in "${server_pids[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "A policy server exited before readiness; inspect ${RUN_ROOT}/logs/server_worker*.log" >&2
      exit 1
    fi
  done
  sleep 1
done
if [[ "${ready}" -ne "${NUM_SHARDS}" ]]; then
  echo "Only ${ready}/${NUM_SHARDS} policy servers became ready within 300 seconds" >&2
  exit 1
fi

for suite in "${suites[@]}"; do
  suite_root="${RUN_ROOT}/${suite}"
  if [[ -f "${suite_root}/.complete" ]]; then
    echo "Already complete; skipping ${CHECKPOINT_STEP}/${suite}"
    continue
  fi
  mkdir -p "${suite_root}/results" "${suite_root}/videos" "${suite_root}/logs"
  client_pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=$((shard % NUM_GPUS))
    cpu_set="$(cpu_set_for_worker "${shard}")"
    video_arg="--args.no-save-video"
    [[ "${SAVE_VIDEO}" == 1 ]] && video_arg="--args.save-video"
    env \
      PYTHONPATH="${LIBERO_SOURCE_ROOT}:${REPO_ROOT}/packages/openpi-client/src" \
      LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      MUJOCO_GL=egl \
      PYOPENGL_PLATFORM=egl \
      EGL_DEVICE_ID="${gpu}" \
      MUJOCO_EGL_DEVICE_ID="${gpu}" \
      OMP_NUM_THREADS="${threads_per_process}" \
      MKL_NUM_THREADS="${threads_per_process}" \
      OPENBLAS_NUM_THREADS="${threads_per_process}" \
      taskset -c "${cpu_set}" \
      "${PYTHON_BIN}" "${REPO_ROOT}/examples/libero/main.py" \
        --args.host 127.0.0.1 \
        --args.port "$((PORT_BASE + shard))" \
        --args.task-suite-name "${suite}" \
        --args.task-ids "${task_ids[@]}" \
        --args.num-shards "${NUM_SHARDS}" \
        --args.shard-index "${shard}" \
        --args.seed 7 \
        --args.output-jsonl "${suite_root}/results/shard_${shard}.jsonl" \
        --args.video-out-path "${suite_root}/videos" \
        "${video_arg}" \
        >"${suite_root}/logs/shard_${shard}.log" 2>&1 &
    client_pids+=("$!")
  done

  status=0
  for pid in "${client_pids[@]}"; do
    wait "${pid}" || status=1
  done
  client_pids=()
  if [[ "${status}" -ne 0 ]]; then
    echo "At least one ${suite} shard failed; durable partial results can resume with RESUME=1" >&2
    exit "${status}"
  fi
  "${PYTHON_BIN}" -m four_suite_experiments.summarize_eval suite \
    --suite "${suite}" \
    --output "${suite_root}/summary.json" \
    "${suite_root}"/results/shard_*.jsonl \
    >"${suite_root}/summary.txt"
  touch "${suite_root}/.complete"
  echo "Completed ${CHECKPOINT_STEP}/${suite}"
done

summary_inputs=()
for suite in "${suites[@]}"; do
  summary_inputs+=("${RUN_ROOT}/${suite}/summary.json")
done
"${PYTHON_BIN}" -m four_suite_experiments.summarize_eval checkpoint \
  --checkpoint-step "${CHECKPOINT_STEP}" \
  --output "${RUN_ROOT}/summary.json" \
  "${summary_inputs[@]}" \
  >"${RUN_ROOT}/summary.txt"
touch "${RUN_ROOT}/.complete"
echo "Four-suite evaluation complete: ${RUN_ROOT}/summary.json"
