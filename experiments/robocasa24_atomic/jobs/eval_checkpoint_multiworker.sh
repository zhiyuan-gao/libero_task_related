#!/usr/bin/env bash
set -euo pipefail

finetune_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repository_root=$(cd "$finetune_root/../.." && pwd)
openpi_root=${OPENPI_ROOT:-$repository_root}
checkpoint=${CHECKPOINT:?Set CHECKPOINT to a RoboCasa raw checkpoint directory}
run_root=${RUN_ROOT:?Set RUN_ROOT to a new evaluation output directory}
server_python=${SERVER_PYTHON:-$openpi_root/.venv/bin/python}
runtime_root=${ROBOCASA_EVAL_RUNTIME_ROOT:-$finetune_root/.runtime/eval}
client_python=${CLIENT_PYTHON:-$runtime_root/.venv/bin/python}
robocasa_root=${ROBOCASA_ROOT:-$runtime_root/robocasa}
robosuite_root=${ROBOSUITE_ROOT:-$runtime_root/robosuite}
port_base=${PORT_BASE:-8600}
num_gpus=${NUM_GPUS:-8}
num_workers=${NUM_WORKERS:-8}
shard_mode=${SHARD_MODE:-task}
resume=${RESUME:-0}
save_video=${SAVE_VIDEO:-0}
dry_run=${DRY_RUN:-0}
eval_seed=${EVAL_SEED:-7}
trials_per_task=${TRIALS_PER_TASK:-50}
execution_horizon=${EXECUTION_HORIZON:-25}
max_episode_steps=${MAX_EPISODE_STEPS:-1000}
protocol_mode=${EVAL_PROTOCOL_MODE:-formal}
tasks_csv=${TASKS_CSV:-PnPCounterToCab,PnPCabToCounter,PnPCounterToSink,PnPSinkToCounter,PnPCounterToMicrowave,PnPMicrowaveToCounter,PnPCounterToStove,PnPStoveToCounter,OpenSingleDoor,CloseSingleDoor,OpenDoubleDoor,CloseDoubleDoor,OpenDrawer,CloseDrawer,TurnOnSinkFaucet,TurnOffSinkFaucet,TurnSinkSpout,TurnOnStove,TurnOffStove,CoffeeSetupMug,CoffeeServeMug,CoffeePressButton,TurnOnMicrowave,TurnOffMicrowave}

expected_robocasa_commit=756598a5be52e052339bb2d957426e39015c2afb
expected_robosuite_commit=cb173eb465089b1b4d7038dc8e913f18817f2b0f

for integer in "$port_base" "$num_gpus" "$num_workers" "$eval_seed" "$trials_per_task" "$execution_horizon" "$max_episode_steps"; do
  [[ "$integer" =~ ^[0-9]+$ ]] || { echo "Expected a non-negative integer, found: $integer" >&2; exit 1; }
done
(( num_gpus >= 1 && num_gpus <= 8 )) || { echo "NUM_GPUS must be in [1,8]" >&2; exit 1; }
(( num_workers >= 1 && num_workers <= 24 )) || { echo "NUM_WORKERS must be in [1,24]" >&2; exit 1; }
(( execution_horizon >= 1 && execution_horizon <= 50 )) || {
  echo "EXECUTION_HORIZON must be in [1,50]" >&2
  exit 1
}
[[ "$shard_mode" == task || "$shard_mode" == episode ]] || {
  echo "SHARD_MODE must be task or episode" >&2
  exit 1
}
(( port_base >= 1024 && port_base + num_gpus - 1 <= 65535 )) || { echo "Invalid PORT_BASE" >&2; exit 1; }
[[ -x "$server_python" ]] || { echo "Policy-server Python is missing: $server_python" >&2; exit 1; }
[[ -x "$client_python" || "$dry_run" == 1 ]] || { echo "Simulator Python is missing: $client_python" >&2; exit 1; }
[[ -s "$checkpoint/model.safetensors" ]] || { echo "Checkpoint model is missing: $checkpoint/model.safetensors" >&2; exit 1; }
[[ -s "$checkpoint/metadata.pt" ]] || { echo "Checkpoint metadata is missing: $checkpoint/metadata.pt" >&2; exit 1; }
[[ -s "$checkpoint/assets/robocasa24_atomic_hdf5_base50/norm_stats.json" ]] || {
  echo "Checkpoint RoboCasa normalization stats are missing" >&2
  exit 1
}
[[ -d "$robocasa_root/.git" ]] || { echo "Pinned RoboCasa checkout is missing: $robocasa_root" >&2; exit 1; }
[[ -d "$robosuite_root/.git" ]] || { echo "Pinned RoboSuite checkout is missing: $robosuite_root" >&2; exit 1; }
[[ "$(git -C "$robocasa_root" rev-parse HEAD)" == "$expected_robocasa_commit" ]] || {
  echo "RoboCasa checkout is not at the frozen v0.2 commit" >&2
  exit 1
}
[[ "$(git -C "$robosuite_root" rev-parse HEAD)" == "$expected_robosuite_commit" ]] || {
  echo "RoboSuite checkout is not at the frozen dependency commit" >&2
  exit 1
}

IFS=',' read -r -a tasks <<<"$tasks_csv"
(( ${#tasks[@]} >= 1 )) || { echo "TASKS_CSV is empty" >&2; exit 1; }
if [[ "$shard_mode" == episode && "$num_workers" -lt "${#tasks[@]}" ]]; then
  echo "SHARD_MODE=episode requires at least one worker per task" >&2
  exit 1
fi
case "$protocol_mode" in
  formal) formal_arg=--formal ;;
  subset_diagnostic) formal_arg=--no-formal ;;
  *) echo "EVAL_PROTOCOL_MODE must be formal or subset_diagnostic" >&2; exit 1 ;;
esac

if [[ "$dry_run" == 1 ]]; then
  echo "DRY RUN OK"
  echo "checkpoint=$checkpoint"
  echo "tasks=${tasks[*]}"
  echo "rollouts=$((${#tasks[@]} * trials_per_task))"
  echo "policy_servers=$num_gpus simulator_workers=$num_workers"
  echo "shard_mode=$shard_mode"
  echo "action_chunk=50 execute=$execution_horizon flow_steps=10"
  echo "seed=$eval_seed max_episode_steps=$max_episode_steps protocol=$protocol_mode"
  exit 0
fi

if [[ "$resume" != 1 && -e "$run_root" ]]; then
  echo "RUN_ROOT already exists; choose a new directory or set RESUME=1: $run_root" >&2
  exit 1
fi
mkdir -p "$run_root/results" "$run_root/logs" "$run_root/videos"

manifest=$(printf '%s\n' \
  "schema=robocasa24.atomic24.multiworker_eval.v1" \
  "checkpoint=$(realpath "$checkpoint")" \
  "tasks=$tasks_csv" \
  "trials_per_task=$trials_per_task" \
  "seed=$eval_seed" \
  "predicted_action_horizon=50" \
  "execution_horizon=$execution_horizon" \
  "flow_steps=10" \
  "max_episode_steps=$max_episode_steps" \
  "resize=224" \
  "camera_row_transform=vertical_flip_axis_height" \
  "object_instance_split=B" \
  "camera_randomization=false" \
  "layout_style_pairs=(1,1),(2,2),(4,4),(6,9),(7,10)" \
  "num_policy_servers=$num_gpus" \
  "num_simulator_workers=$num_workers" \
  "shard_mode=$shard_mode" \
  "protocol_mode=$protocol_mode" \
  "robocasa_commit=$expected_robocasa_commit" \
  "robosuite_commit=$expected_robosuite_commit")
if [[ -f "$run_root/manifest.txt" && "$(<"$run_root/manifest.txt")" != "$manifest" ]]; then
  echo "Existing evaluation manifest does not match this run" >&2
  exit 1
fi
printf '%s\n' "$manifest" >"$run_root/manifest.txt"

server_pids=()
worker_pids=()
cleanup() {
  local pid
  for pid in "${worker_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  wait "${server_pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cpu_count=$(nproc)
cores_per_worker=$((cpu_count / num_workers))
(( cores_per_worker >= 1 )) || cores_per_worker=1
threads_per_process=$((cores_per_worker / 2))
(( threads_per_process >= 1 )) || threads_per_process=1
cpu_set_for_worker() {
  local worker=$1
  local first=$((worker * cores_per_worker))
  local last=$((first + cores_per_worker - 1))
  (( last < cpu_count )) || last=$((cpu_count - 1))
  echo "$first-$last"
}

server_path_args=()
[[ -z "${DATA_ROOT:-}" ]] || server_path_args+=(--data-root "$DATA_ROOT")
[[ -z "${MANIFEST_ROOT:-}" ]] || server_path_args+=(--manifest-root "$MANIFEST_ROOT")
[[ -z "${POLICY_ASSETS_ROOT:-}" ]] || server_path_args+=(--policy-assets-root "$POLICY_ASSETS_ROOT")
[[ -z "${ARTIFACT_DIR:-}" ]] || server_path_args+=(--artifact-dir "$ARTIFACT_DIR")
[[ -z "${BASE_WEIGHT_DIR:-}" ]] || server_path_args+=(--base-weight-dir "$BASE_WEIGHT_DIR")

for gpu in $(seq 0 $((num_gpus - 1))); do
  port=$((port_base + gpu))
  cpu_set=$(cpu_set_for_worker "$((gpu % num_workers))")
  env \
    PYTHONPATH="$finetune_root/src:$openpi_root/src" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$threads_per_process" \
    MKL_NUM_THREADS="$threads_per_process" \
    OPENBLAS_NUM_THREADS="$threads_per_process" \
    TOKENIZERS_PARALLELISM=false \
    taskset -c "$cpu_set" \
    "$server_python" -m robocasa24_finetune.eval_server \
      --checkpoint "$checkpoint" \
      --port "$port" \
      "${server_path_args[@]}" \
      >"$run_root/logs/server_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
done

ready=0
for _ in $(seq 1 600); do
  ready=0
  for gpu in $(seq 0 $((num_gpus - 1))); do
    if curl --silent --fail --max-time 1 "http://127.0.0.1:$((port_base + gpu))/healthz" >/dev/null; then
      ready=$((ready + 1))
    fi
  done
  [[ "$ready" -eq "$num_gpus" ]] && break
  for pid in "${server_pids[@]}"; do
    kill -0 "$pid" 2>/dev/null || {
      echo "A policy server exited before readiness; inspect $run_root/logs/server_gpu*.log" >&2
      exit 1
    }
  done
  sleep 1
done
[[ "$ready" -eq "$num_gpus" ]] || { echo "Only $ready/$num_gpus policy servers became ready" >&2; exit 1; }

video_arg=--no-save-video
[[ "$save_video" == 1 ]] && video_arg=--save-video
for worker in $(seq 0 $((num_workers - 1))); do
  gpu=$((worker % num_gpus))
  cpu_set=$(cpu_set_for_worker "$worker")
  env \
    PYTHONPATH="$finetune_root/src:$openpi_root/packages/openpi-client/src:$robocasa_root:$robosuite_root" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    EGL_DEVICE_ID="$gpu" \
    MUJOCO_EGL_DEVICE_ID="$gpu" \
    OMP_NUM_THREADS="$threads_per_process" \
    MKL_NUM_THREADS="$threads_per_process" \
    OPENBLAS_NUM_THREADS="$threads_per_process" \
    taskset -c "$cpu_set" \
    "$client_python" -m robocasa24_finetune.eval_worker \
      --host 127.0.0.1 \
      --port "$((port_base + gpu))" \
      --tasks "${tasks[@]}" \
      --num-workers "$num_workers" \
      --worker-index "$worker" \
      --shard-mode "$shard_mode" \
      --trials-per-task "$trials_per_task" \
      --execution-horizon "$execution_horizon" \
      --resize-size 224 \
      --max-episode-steps "$max_episode_steps" \
      --seed "$eval_seed" \
      --output-jsonl "$run_root/results/worker_${worker}.jsonl" \
      --video-root "$run_root/videos" \
      "$video_arg" \
      "$formal_arg" \
      >"$run_root/logs/worker_${worker}.log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one simulator worker failed; completed JSONL records are preserved for RESUME=1" >&2
  exit "$status"
fi

summary_formal_arg=--formal
[[ "$protocol_mode" == subset_diagnostic ]] && summary_formal_arg=--no-formal
env PYTHONPATH="$finetune_root/src" \
  "$client_python" -m robocasa24_finetune.summarize_eval \
  "$run_root"/results/worker_*.jsonl \
  --output "$run_root/summary.json" \
  --tasks "${tasks[@]}" \
  --trials-per-task "$trials_per_task" \
  --execution-horizon "$execution_horizon" \
  "$summary_formal_arg" \
  >"$run_root/summary.txt"
touch "$run_root/.complete"
echo "Evaluation complete: $run_root/summary.json"
