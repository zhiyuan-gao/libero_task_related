#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
libero_plus_root=${LIBERO_PLUS_ROOT:-/workspace/vla/third_party/LIBERO-plus}
libero_config_path=${LIBERO_CONFIG_PATH:-/workspace/vla/eval/libero_plus/config}
checkpoint=${CHECKPOINT:?Set CHECKPOINT to the P1 raw serving checkpoint directory}
run_root=${RUN_ROOT:?Set RUN_ROOT to a new output directory}
log_root=${LOG_ROOT:-$run_root/logs}
server_python=${SERVER_PYTHON:-$repo_root/.venv/bin/python}
client_python=${CLIENT_PYTHON:-$repo_root/examples/libero/.venv/bin/python}
port_base=${PORT_BASE:-8000}
num_shards=8

if [[ -e "$run_root" ]]; then
  echo "RUN_ROOT already exists: $run_root" >&2
  exit 1
fi
if [[ ! -f "$checkpoint/model.safetensors" ]]; then
  echo "Raw serving checkpoint is missing model.safetensors: $checkpoint" >&2
  exit 1
fi
if [[ ! -d "$libero_plus_root/libero/libero/assets" ]]; then
  echo "LIBERO-Plus assets are missing beneath: $libero_plus_root" >&2
  exit 1
fi
if [[ ! -f "$libero_config_path/config.yaml" ]]; then
  echo "LIBERO-Plus config is missing: $libero_config_path/config.yaml" >&2
  exit 1
fi

mkdir -p "$run_root/results" "$run_root/videos" "$log_root"
server_pids=()
client_pids=()

cleanup() {
  for pid in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

cd "$repo_root"
for gpu in $(seq 0 7); do
  port=$((port_base + gpu))
  env CUDA_VISIBLE_DEVICES="$gpu" \
    "$server_python" scripts/serve_policy.py \
      --env LIBERO \
      --port "$port" \
      policy:checkpoint \
      --policy.config pi05_libero3_p1_aux \
      --policy.dir "$checkpoint" \
      >"$log_root/server_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
done

for _ in $(seq 1 240); do
  ready=0
  for gpu in $(seq 0 7); do
    if rg -q "server listening on 0.0.0.0:$((port_base + gpu))" "$log_root/server_gpu${gpu}.log" 2>/dev/null; then
      ready=$((ready + 1))
    fi
  done
  [[ "$ready" -eq 8 ]] && break
  sleep 1
done
if [[ "${ready:-0}" -ne 8 ]]; then
  echo "Only ${ready:-0}/8 policy servers became ready" >&2
  exit 1
fi

for shard in $(seq 0 7); do
  env \
    PYTHONPATH="$libero_plus_root:$libero_plus_root/.deps" \
    LIBERO_CONFIG_PATH="$libero_config_path" \
    MUJOCO_GL=egl \
    EGL_DEVICE_ID="$shard" \
    MUJOCO_EGL_DEVICE_ID="$shard" \
    "$client_python" examples/libero_plus/main.py \
      --host 127.0.0.1 \
      --port "$((port_base + shard))" \
      --num-shards "$num_shards" \
      --shard-index "$shard" \
      --output-jsonl "$run_root/results/shard_${shard}.jsonl" \
      --video-out-path "$run_root/videos" \
      >"$log_root/shard_${shard}.log" 2>&1 &
  client_pids+=("$!")
done

status=0
for pid in "${client_pids[@]}"; do
  wait "$pid" || status=1
done

jq -s '
  {
    total: length,
    successes: (map(select(.success)) | length),
    success_rate: ((map(select(.success)) | length) / length),
    errors: (map(select(.error != null)) | length),
    by_task: (group_by(.base_task) | map({key: .[0].base_task, value: {
      total: length,
      successes: (map(select(.success)) | length),
      success_rate: ((map(select(.success)) | length) / length)
    }}) | from_entries),
    by_category: (group_by(.category) | map({key: .[0].category, value: {
      total: length,
      successes: (map(select(.success)) | length),
      success_rate: ((map(select(.success)) | length) / length)
    }}) | from_entries)
  }
' "$run_root"/results/*.jsonl >"$run_root/summary.json"
jq -sc '.[]' "$run_root"/results/*.jsonl >"$run_root/results.jsonl"

if [[ "$(jq -r '.total' "$run_root/summary.json")" != 872 ]]; then
  echo "Formal evaluation incomplete: expected 872 records" >&2
  status=1
fi
exit "$status"
