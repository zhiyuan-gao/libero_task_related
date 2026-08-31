#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${project_root}/../.." && pwd)"
openpi_root="${OPENPI_ROOT:-${repo_root}}"
python_bin="${FOUR_SUITE_PYTHON:-${openpi_root}/.venv/bin/python}"
runner="${project_root}/jobs/run_8gpu.sh"
work_dir="${FOUR_SUITE_WORK_DIR:-${project_root}/runtime}"
checkpoint_base="${FOUR_SUITE_CHECKPOINT_BASE_DIR:-${work_dir}/checkpoints}"
log_dir="${FOUR_SUITE_CONTINUATION_LOG_DIR:-${work_dir}/formal_logs/dual_continuation}"

main_parent="${FOUR_SUITE_MAIN_PARENT:-${checkpoint_base}/pi05_libero40_trqc/libero40_trqc_seed42/30000}"
old115_parent="${FOUR_SUITE_OLD115_PARENT:-${checkpoint_base}/pi05_libero40_trqc_supplemental_finetune/libero40_trqc_supplemental115_from30000_seed42/3000}"
libero_assets="${FOUR_SUITE_LIBERO_ASSETS:-${project_root}/external_assets/models/pi05_libero_pytorch/assets}"
completed_root="${FOUR_SUITE_COMPLETED_ROOT:-${work_dir}/prepared/official_completion1932}"
old115_root="${FOUR_SUITE_OLD115_ROOT:-${work_dir}/prepared/supplemental115}"
revision="a4336d589d589045d1c56423ffdf3b88a0e19b1f"

completed_lerobot="${completed_root}/lerobot/${revision}"
completed_artifacts="${completed_root}/artifacts/task_relevant"
old115_lerobot="${old115_root}/lerobot/${revision}"
old115_artifacts="${old115_root}/artifacts/task_relevant"

full_exp="libero40_trqc_official_completion1932_from_main30000_seed42"
old115_exp="libero40_trqc_supplemental115_from30000_seed42"
a_config="pi05_libero40_trqc_official_completion_finetune"
b_config="pi05_libero40_trqc_supplemental_finetune"
full_dir="${checkpoint_base}/${a_config}/${full_exp}"
old115_dir="${checkpoint_base}/${b_config}/${old115_exp}"

mode="${1:-plan}"
if [[ "${mode}" != "plan" && "${mode}" != "smoke" && "${mode}" != "formal" ]]; then
  echo "Usage: $0 {plan|smoke|formal}" >&2
  exit 2
fi

print_plan() {
  printf '1\t1932_from_main\tpopulation=1932\tparent=%s\tcontinuous_updates=6000\tfinal=6000\toutput=%s\n' "${main_parent}" "${full_dir}"
  printf '2\told115_exact_continue\tpopulation=1808\tresume=%s\tadditional_updates=3000\tfinal=6000\toutput=%s\n' "${old115_parent}" "${old115_dir}"
}

if [[ "${mode}" == "plan" ]]; then
  print_plan
  exit 0
fi

if [[ ! -x "${python_bin}" || ! -x "${runner}" ]]; then
  echo "Training environment or runner is unavailable" >&2
  exit 2
fi

required_paths=(
  "${main_parent}/model.safetensors"
  "${old115_parent}/model.safetensors"
  "${old115_parent}/optimizer.pt"
  "${old115_parent}/training_state.pt"
  "${old115_parent}/metadata.pt"
  "${completed_lerobot}/meta/info.json"
  "${completed_artifacts}/validation.json"
  "${old115_lerobot}/meta/info.json"
  "${old115_artifacts}/validation.json"
  "${libero_assets}/physical-intelligence/libero/norm_stats.json"
)
for path in "${required_paths[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required input is missing: ${path}" >&2
    exit 4
  fi
done

mkdir -p "${log_dir}"
exec 9>"${log_dir}/dual_continuation.lock"
if ! flock -n 9; then
  echo "Another dual-continuation controller is already running" >&2
  exit 5
fi

check_gpu_idle() {
  local active
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "${active}" ]]; then
    echo "GPU compute processes are still active; refusing to overlap: ${active}" >&2
    return 1
  fi
}

check_free_space() {
  local available_kib
  available_kib="$(df -Pk "${checkpoint_base}" | awk 'NR==2 {print $4}')"
  if (( available_kib < 100 * 1024 * 1024 )); then
    echo "Less than 100 GiB is free before the next stage; stopping safely" >&2
    return 1
  fi
}

has_resumable_checkpoint() {
  local exp_dir="$1"
  local checkpoint
  shopt -s nullglob
  for checkpoint in "${exp_dir}"/[0-9]*; do
    if [[ -f "${checkpoint}/model.safetensors" && -f "${checkpoint}/optimizer.pt" && -f "${checkpoint}/training_state.pt" ]]; then
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

run_formal_stage() {
  local label="$1"
  local population="$2"
  local lerobot_root="$3"
  local artifact_dir="$4"
  local parent="$5"
  local exp_name="$6"
  local exp_dir="$7"
  local final_step="$8"
  local schedule_steps="$9"
  local log_path="${log_dir}/${label}.log"
  local population_flag=()
  local resume_flag=()

  if [[ "${population}" == "1932" ]]; then
    population_flag=(--official-completion)
  fi
  if [[ -f "${exp_dir}/${final_step}/model.safetensors" ]]; then
    printf 'SKIP completed stage %s at %s\n' "${label}" "${exp_dir}/${final_step}"
    return 0
  fi
  if [[ -d "${exp_dir}" ]]; then
    if has_resumable_checkpoint "${exp_dir}"; then
      resume_flag=(--resume)
    else
      echo "Existing stage directory has no resumable checkpoint; refusing overwrite: ${exp_dir}" >&2
      return 6
    fi
  fi

  check_gpu_idle
  check_free_space
  if [[ ! -f "${parent}/model.safetensors" ]]; then
    echo "Stage parent is missing after the previous stage: ${parent}" >&2
    return 7
  fi
  printf 'START %s population=%s parent=%s final=%s schedule_steps=%s\n' \
    "${label}" "${population}" "${parent}" "${final_step}" "${schedule_steps}"
  "${runner}" supplemental-finetune \
    "${population_flag[@]}" \
    --lerobot-root "${lerobot_root}" \
    --artifact-dir "${artifact_dir}" \
    --parent-checkpoint "${parent}" \
    --libero-assets-dir "${libero_assets}" \
    --exp-name "${exp_name}" \
    --num-workers 8 \
    --num-updates "${schedule_steps}" \
    --target-step "${final_step}" \
    --decay-steps "${schedule_steps}" \
    --disable-wandb \
    "${resume_flag[@]}" >"${log_path}" 2>&1
  if [[ ! -f "${exp_dir}/${final_step}/model.safetensors" ]]; then
    echo "Stage returned success without its final checkpoint: ${exp_dir}/${final_step}" >&2
    return 8
  fi
  printf 'PASS %s final=%s\n' "${label}" "${exp_dir}/${final_step}"
}

run_smoke_stage() {
  local label="$1"
  local population="$2"
  local lerobot_root="$3"
  local artifact_dir="$4"
  local parent="$5"
  local smoke_base="$6"
  local population_flag=()
  if [[ "${population}" == "1932" ]]; then
    population_flag=(--official-completion)
  fi
  FOUR_SUITE_CHECKPOINT_BASE_DIR="${smoke_base}" "${runner}" supplemental-finetune \
    "${population_flag[@]}" \
    --lerobot-root "${lerobot_root}" \
    --artifact-dir "${artifact_dir}" \
    --parent-checkpoint "${parent}" \
    --libero-assets-dir "${libero_assets}" \
    --exp-name "smoke_${label}" \
    --num-workers 8 \
    --disable-wandb \
    --smoke >"${log_dir}/smoke_${label}.log" 2>&1
  printf 'PASS smoke %s\n' "${label}"
}

export OPENPI_ROOT="${openpi_root}"
export FOUR_SUITE_PYTHON="${python_bin}"
export FOUR_SUITE_TORCHRUN="${FOUR_SUITE_TORCHRUN:-${openpi_root}/.venv/bin/torchrun}"
export FOUR_SUITE_SUPPLEMENTAL_FINETUNE_APPROVED=YES

if [[ "${mode}" == "smoke" ]]; then
  check_gpu_idle
  smoke_root="${FOUR_SUITE_SMOKE_CHECKPOINT_DIR:-${work_dir}/smoke_checkpoints}"
  mkdir -p "${smoke_root}"
  smoke_base="$(mktemp -d "${smoke_root}/dual_continuation.XXXXXX")"
  cleanup_smoke() {
    if [[ -d "${smoke_base}" ]]; then
      find "${smoke_base}" -xdev -depth -delete
    fi
  }
  trap cleanup_smoke EXIT
  run_smoke_stage "1932_continuous" "1932" "${completed_lerobot}" "${completed_artifacts}" "${main_parent}" "${smoke_base}"
  old115_state_before="$(stat -c '%s:%Y' "${old115_parent}/model.safetensors" "${old115_parent}/optimizer.pt" "${old115_parent}/training_state.pt")"
  FOUR_SUITE_CHECKPOINT_BASE_DIR="${checkpoint_base}" "${runner}" supplemental-finetune \
    --lerobot-root "${old115_lerobot}" \
    --artifact-dir "${old115_artifacts}" \
    --parent-checkpoint "${main_parent}" \
    --libero-assets-dir "${libero_assets}" \
    --exp-name "${old115_exp}" \
    --num-workers 8 \
    --num-updates 3000 \
    --target-step 3002 \
    --decay-steps 3000 \
    --disable-wandb \
    --resume \
    --smoke >"${log_dir}/smoke_old115_exact_resume.log" 2>&1
  old115_state_after="$(stat -c '%s:%Y' "${old115_parent}/model.safetensors" "${old115_parent}/optimizer.pt" "${old115_parent}/training_state.pt")"
  if [[ "${old115_state_before}" != "${old115_state_after}" ]]; then
    echo "Exact-resume smoke unexpectedly modified the step-3000 parent" >&2
    exit 9
  fi
  printf 'PASS smoke old115_exact_resume\n'
  exit 0
fi

if [[ "${LIBERO_DUAL_CONTINUATION_APPROVED:-}" != "YES" ]]; then
  echo "Formal dual continuation is gated; set LIBERO_DUAL_CONTINUATION_APPROVED=YES only after user confirmation" >&2
  exit 3
fi

run_formal_stage "01_1932_from_main" "1932" "${completed_lerobot}" "${completed_artifacts}" "${main_parent}" "${full_exp}" "${full_dir}" 6000 6000
run_formal_stage "02_old115_exact_continue" "1808" "${old115_lerobot}" "${old115_artifacts}" "${main_parent}" "${old115_exp}" "${old115_dir}" 6000 3000
printf 'ALL TRAINING STAGES COMPLETED; closed-loop evaluation was not started.\n'
