#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${LIBERO_DUAL_CONTINUATION_APPROVED:-}" != "YES" ]]; then
  echo "Background formal launch is gated; set LIBERO_DUAL_CONTINUATION_APPROVED=YES after final confirmation" >&2
  exit 3
fi

jobs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${jobs_dir}/.." && pwd)"
work_dir="${FOUR_SUITE_WORK_DIR:-${project_root}/runtime}"
log_dir="${FOUR_SUITE_CONTINUATION_LOG_DIR:-${work_dir}/formal_logs/dual_continuation}"
export FOUR_SUITE_WORK_DIR="${work_dir}"
export FOUR_SUITE_CHECKPOINT_BASE_DIR="${FOUR_SUITE_CHECKPOINT_BASE_DIR:-${work_dir}/checkpoints}"
export FOUR_SUITE_CONTINUATION_LOG_DIR="${log_dir}"
mkdir -p "${log_dir}"
controller_log="${log_dir}/controller_$(date -u +%Y%m%d_%H%M%SUTC).log"
session_name="${FOUR_SUITE_CONTINUATION_TMUX_SESSION:-libero_dual_continuation}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for a disconnect-safe background launch" >&2
  exit 4
fi
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 5
fi

command=(
  env
  LIBERO_DUAL_CONTINUATION_APPROVED=YES
  "FOUR_SUITE_WORK_DIR=${FOUR_SUITE_WORK_DIR}"
  "FOUR_SUITE_CHECKPOINT_BASE_DIR=${FOUR_SUITE_CHECKPOINT_BASE_DIR}"
  "FOUR_SUITE_CONTINUATION_LOG_DIR=${FOUR_SUITE_CONTINUATION_LOG_DIR}"
  "${jobs_dir}/run_dual_continuation_8gpu.sh"
  formal
)
printf -v command_quoted '%q ' "${command[@]}"
printf -v log_quoted '%q' "${controller_log}"
tmux new-session -d -s "${session_name}" "exec ${command_quoted}>${log_quoted} 2>&1"
pid="$(tmux list-panes -t "${session_name}" -F '#{pane_pid}')"
printf 'Started dual-continuation tmux=%s controller_pid=%s log=%s\n' \
  "${session_name}" "${pid}" "${controller_log}"
