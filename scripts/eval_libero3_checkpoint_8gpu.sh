#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=${CHECKPOINT:?Set CHECKPOINT to a raw serving checkpoint directory}
policy_config=${POLICY_CONFIG:?Set POLICY_CONFIG to the matching training config}
run_root=${RUN_ROOT:?Set RUN_ROOT to the checkpoint evaluation output directory}
server_python=${SERVER_PYTHON:-$repo_root/.venv/bin/python}
client_python=${CLIENT_PYTHON:-$repo_root/.venv/bin/python}
port_base=${PORT_BASE:-8200}
resume=${RESUME:-0}
save_video=${SAVE_VIDEO:-1}
dry_run=${DRY_RUN:-0}
num_shards=8
torch_compile_cache_root=${TORCH_COMPILE_CACHE_ROOT:-$run_root/torch_compile_cache}

if [[ ! -x "$server_python" || ! -x "$client_python" ]]; then
  echo "Python environment is missing: server=$server_python client=$client_python" >&2
  exit 1
fi
if [[ ! -s "$checkpoint/model.safetensors" ]]; then
  echo "Raw serving checkpoint is missing model.safetensors: $checkpoint" >&2
  exit 1
fi
if [[ ! -s "$checkpoint/assets/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "Checkpoint is missing LIBERO norm stats: $checkpoint" >&2
  exit 1
fi
if [[ ! "$policy_config" =~ ^(pi05_libero3_semantic_geometry(_motion)?_aux|pi05_libero3_p3_binary_ground_aux)$ ]]; then
  echo "Unexpected LIBERO-3 policy config: $policy_config" >&2
  exit 1
fi
if [[ "$resume" != 1 && -e "$run_root" ]]; then
  echo "RUN_ROOT already exists; use a new path or set RESUME=1: $run_root" >&2
  exit 1
fi
if [[ "$port_base" -lt 1024 || $((port_base + 7)) -gt 65535 ]]; then
  echo "Invalid PORT_BASE: $port_base" >&2
  exit 1
fi

libero_root="$repo_root/third_party/libero/libero/libero"
for required in assets bddl_files init_files; do
  if [[ ! -d "$libero_root/$required" ]]; then
    echo "LIBERO asset directory is missing: $libero_root/$required" >&2
    exit 1
  fi
done

if [[ "$dry_run" == 1 ]]; then
  echo "DRY RUN OK: config=$policy_config checkpoint=$checkpoint output=$run_root"
  echo "Would launch 8 policy servers on GPUs 0-7 and ports $port_base-$((port_base + 7))."
  echo "Would evaluate 150 deterministic rollouts in 8 shards (task IDs 4,2,3; 50 each)."
  exit 0
fi

mkdir -p \
  "$run_root/results" \
  "$run_root/videos" \
  "$run_root/logs" \
  "$run_root/libero_config" \
  "$torch_compile_cache_root"
printf '%s\n' \
  "benchmark_root: $libero_root" \
  "bddl_files: $libero_root/bddl_files" \
  "init_states: $libero_root/init_files" \
  "datasets: $repo_root/third_party/libero/libero/datasets" \
  "assets: $libero_root/assets" \
  >"$run_root/libero_config/config.yaml"

manifest=$(printf '%s\n' \
  "checkpoint=$checkpoint" \
  "policy_config=$policy_config" \
  "task_ids=4,2,3" \
  "trials_per_task=50" \
  "seed=7" \
  "num_shards=8" \
  "resize=224" \
  "replan_steps=5" \
  "policy_flow_steps=10")
if [[ -f "$run_root/manifest.txt" && "$(<"$run_root/manifest.txt")" != "$manifest" ]]; then
  echo "Existing run manifest does not match this evaluation: $run_root/manifest.txt" >&2
  exit 1
fi
printf '%s\n' "$manifest" >"$run_root/manifest.txt"

server_pids=()
client_pids=()
cleanup() {
  local pid
  for pid in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  wait "${server_pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cpu_count=$(nproc)
cores_per_pair=$((cpu_count / num_shards))
if (( cores_per_pair < 1 )); then
  cores_per_pair=1
fi
threads_per_process=$((cores_per_pair / 2))
if (( threads_per_process < 1 )); then
  threads_per_process=1
fi

cpu_set_for_gpu() {
  local gpu=$1
  if (( cpu_count == 128 )); then
    # Four GPUs are local to each NUMA node. Give every GPU eight physical
    # cores plus their SMT siblings without crossing sockets.
    case "$gpu" in
      0) echo "0-7,64-71" ;;
      1) echo "8-15,72-79" ;;
      2) echo "16-23,80-87" ;;
      3) echo "24-31,88-95" ;;
      4) echo "32-39,96-103" ;;
      5) echo "40-47,104-111" ;;
      6) echo "48-55,112-119" ;;
      7) echo "56-63,120-127" ;;
    esac
    return
  fi
  local first_core=$((gpu * cores_per_pair))
  local last_core=$((first_core + cores_per_pair - 1))
  if (( last_core >= cpu_count )); then
    last_core=$((cpu_count - 1))
  fi
  echo "$first_core-$last_core"
}

cd "$repo_root"
for gpu in $(seq 0 7); do
  port=$((port_base + gpu))
  cpu_set=$(cpu_set_for_gpu "$gpu")
  mkdir -p "$torch_compile_cache_root/gpu${gpu}/inductor" "$torch_compile_cache_root/gpu${gpu}/triton"
  env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$threads_per_process" MKL_NUM_THREADS="$threads_per_process" \
    OPENBLAS_NUM_THREADS="$threads_per_process" \
    TORCHINDUCTOR_CACHE_DIR="$torch_compile_cache_root/gpu${gpu}/inductor" \
    TRITON_CACHE_DIR="$torch_compile_cache_root/gpu${gpu}/triton" \
    taskset -c "$cpu_set" \
    "$server_python" scripts/serve_policy.py \
      --env LIBERO \
      --port "$port" \
      --num-steps 10 \
      policy:checkpoint \
      --policy.config "$policy_config" \
      --policy.dir "$checkpoint" \
      >"$run_root/logs/server_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
done

ready=0
for _ in $(seq 1 300); do
  ready=0
  for gpu in $(seq 0 7); do
    if curl --silent --fail --max-time 1 "http://127.0.0.1:$((port_base + gpu))/healthz" >/dev/null; then
      ready=$((ready + 1))
    fi
  done
  [[ "$ready" -eq 8 ]] && break
  for pid in "${server_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A policy server exited before readiness; inspect $run_root/logs/server_gpu*.log" >&2
      exit 1
    fi
  done
  sleep 1
done
if [[ "$ready" -ne 8 ]]; then
  echo "Only $ready/8 policy servers became ready within 300 seconds" >&2
  exit 1
fi

video_arg=--save-video
if [[ "$save_video" != 1 ]]; then
  video_arg=--no-save-video
fi
for shard in $(seq 0 7); do
  cpu_set=$(cpu_set_for_gpu "$shard")
  env \
    PYTHONPATH="$repo_root/third_party/libero:$repo_root/packages/openpi-client/src" \
    LIBERO_CONFIG_PATH="$run_root/libero_config" \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    EGL_DEVICE_ID="$shard" \
    MUJOCO_EGL_DEVICE_ID="$shard" \
    OMP_NUM_THREADS="$threads_per_process" \
    MKL_NUM_THREADS="$threads_per_process" \
    OPENBLAS_NUM_THREADS="$threads_per_process" \
    taskset -c "$cpu_set" \
    "$client_python" examples/libero/main.py \
      --args.host 127.0.0.1 \
      --args.port "$((port_base + shard))" \
      --args.num-shards "$num_shards" \
      --args.shard-index "$shard" \
      --args.output-jsonl "$run_root/results/shard_${shard}.jsonl" \
      --args.video-out-path "$run_root/videos" \
      "--args.${video_arg#--}" \
      >"$run_root/logs/shard_${shard}.log" 2>&1 &
  client_pids+=("$!")
done

status=0
for pid in "${client_pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one evaluation shard failed; partial JSONL results are preserved and RESUME=1 can continue them." >&2
  exit "$status"
fi

"$client_python" scripts/summarize_libero3_eval.py \
  "$run_root"/results/shard_*.jsonl \
  --output "$run_root/summary.json" \
  >"$run_root/summary.txt"
touch "$run_root/.complete"
echo "Evaluation complete: $run_root/summary.json"
