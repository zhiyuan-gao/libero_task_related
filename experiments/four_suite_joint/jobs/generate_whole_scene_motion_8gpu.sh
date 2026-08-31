#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || ( $1 != gate && $1 != smoke && $1 != run ) ]]; then
  echo "usage: $0 <gate|smoke|run>" >&2
  exit 2
fi

phase=$1
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
p3_project=${P3_PROJECT:-/workspace/vla/p3}
p3_vla_root=${P3_VLA_ROOT:-${p3_project}/workspace}
teacher_python=${MOTION_TEACHER_PYTHON:-${p3_vla_root}/_motion_bundle_meta/uv_python/python_install/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}
teacher_site=${MOTION_TEACHER_SITE:-${p3_vla_root}/envs/track4world_teacher/lib/python3.11/site-packages}
manifest=${FOUR_SUITE_MOTION_MANIFEST:-${p3_project}/runtime_metadata/four_suite_policy_motion_manifest.parquet}
task_related_index=${TASK_RELATED_MOTION_INDEX:-/workspace/vla/four_suite_joint_experiments/artifacts/task_relevant/motion_index.parquet}
output_root=${WHOLE_SCENE_MOTION_CACHE_ROOT:-${p3_vla_root}/data/libero_four_suite_annotation/policy_aux_v1/motion_whole_scene_four_suite_v1}
smoke_root=${WHOLE_SCENE_MOTION_SMOKE_ROOT:-${p3_project}/reports/whole_scene_motion/smoke_cache_v1}
track_repo=${TRACK4WORLD_REPO:-${p3_vla_root}/third_party/Track4World}
utils_repo=${UTILS3D_REPO:-${p3_vla_root}/third_party/utils3d}
track_checkpoint=${TRACK4WORLD_CHECKPOINT:-${p3_vla_root}/models/Track4World/track4world_da3.pth}
da3_snapshot=${DA3_SNAPSHOT:-${p3_vla_root}/cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7}
teacher_reference=${MOTION_TEACHER_REFERENCE:-${p3_project}/runtime_metadata/teacher_preprocessing_v1/motion_pilot}
run_id=${WHOLE_SCENE_MOTION_RUN_ID:-whole_scene_motion_$(date +%Y%m%d_%H%M%S)}
log_root=${WHOLE_SCENE_MOTION_LOG_ROOT:-${p3_project}/logs/${run_id}}

readonly worker_count=8
readonly expected_count=256401
readonly selection_column=motion_valid

required_paths=(
  "$teacher_python"
  "$manifest"
  "$task_related_index"
  "$track_repo"
  "$utils_repo"
  "$track_checkpoint"
  "$da3_snapshot"
  "$teacher_reference/hook1/introspection.json"
  "$teacher_reference/smoke16/smoke_report.json"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e $required_path ]]; then
    echo "Missing Whole-scene Motion prerequisite: $required_path" >&2
    exit 1
  fi
done

export PYTHONPATH="$project_root/src:$teacher_site:$track_repo:$p3_vla_root/third_party:$p3_vla_root/third_party/robocasa-video-progress/src:$p3_vla_root/third_party/robocasa-video-progress/scripts"
export HF_HOME="$p3_vla_root/cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

"$teacher_python" - "$manifest" "$task_related_index" "$expected_count" <<'PY'
import sys
from collections import Counter

import pyarrow.parquet as pq

manifest_path, task_index_path, expected_text = sys.argv[1:]
expected = int(expected_text)
manifest = pq.read_table(manifest_path, columns=["sample_id", "suite", "motion_valid"])
valid = manifest.filter(manifest["motion_valid"])
task_index = pq.read_table(task_index_path, columns=["sample_id"])
valid_ids = valid["sample_id"].to_pylist()
task_ids = task_index["sample_id"].to_pylist()
expected_by_suite = {
    "libero_10": 97_547,
    "libero_goal": 47_760,
    "libero_object": 62_444,
    "libero_spatial": 48_650,
}
actual_by_suite = Counter(valid["suite"].to_pylist())
if len(valid_ids) != expected or len(set(valid_ids)) != expected:
    raise SystemExit("Whole-scene Motion manifest population gate failed")
if dict(actual_by_suite) != expected_by_suite:
    raise SystemExit(f"Whole-scene suite population mismatch: {dict(actual_by_suite)}")
if len(task_ids) != expected or set(task_ids) != set(valid_ids):
    raise SystemExit("Task-related and Whole-scene Motion populations are not exactly paired")
print(f"WHOLE_SCENE_MOTION_GATE=PASS selected={expected} suites={dict(sorted(actual_by_suite.items()))}")
PY

if [[ $phase == gate ]]; then
  exit 0
fi

if [[ $phase == smoke ]]; then
  smoke_gpu=${WHOLE_SCENE_MOTION_SMOKE_GPU:-1}
  used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$smoke_gpu" | tr -d ' ')
  if [[ $used_mib -gt 1024 ]]; then
    echo "Smoke GPU $smoke_gpu is not idle (${used_mib} MiB used); refusing to disturb another job" >&2
    exit 1
  fi
  mkdir -p "$smoke_root" "$log_root"
  CUDA_VISIBLE_DEVICES="$smoke_gpu" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    "$teacher_python" -m four_suite_experiments.whole_scene_motion \
      --mode worker \
      --manifest "$manifest" \
      --source-manifest "$manifest" \
      --selection-column "$selection_column" \
      --track4world-repo "$track_repo" \
      --utils3d-repo "$utils_repo" \
      --checkpoint "$track_checkpoint" \
      --da3-snapshot "$da3_snapshot" \
      --teacher-reference-root "$teacher_reference" \
      --output-root "$smoke_root" \
      --worker-index 0 \
      --num-workers 1 \
      --device cuda:0 \
      --shard-size 8 \
      --diagnostic-samples-per-suite 2 \
      --diagnostic-compare-task-related \
      --resume \
      >"$log_root/smoke_worker.log" 2>&1
  "$teacher_python" -m four_suite_experiments.whole_scene_motion \
    --mode finalize \
    --manifest "$manifest" \
    --selection-column "$selection_column" \
    --task-related-index "$task_related_index" \
    --output-root "$smoke_root" \
    --num-workers 1 \
    --final-shard-size 8 \
    --diagnostic-samples-per-suite 2 \
    >"$log_root/smoke_finalize.log" 2>&1
  "$teacher_python" - "$smoke_root/cache_validation.json" <<'PY'
import json
import sys

validation = json.load(open(sys.argv[1]))
assert validation["status"] == "PASS"
assert validation["target_scope"] == "whole_scene"
assert validation["selected_samples"] == 8
assert validation["selected_by_suite"] == {
    "libero_10": 2,
    "libero_goal": 2,
    "libero_object": 2,
    "libero_spatial": 2,
}
assert validation["shape"] == [8, 256]
assert validation["dtype"] == "float32"
assert validation["all_finite"] and validation["no_missing_targets"]
match = validation["same_forward_task_related_cache_check"]
assert match["exact_equal_to_existing_cache"] and match["max_abs"] == 0.0
assert validation["whole_scene_vs_task_related"]["different_rows"] > 0
print(json.dumps({
    "status": "PASS",
    "samples": validation["selected_samples"],
    "selected_by_suite": validation["selected_by_suite"],
    "same_forward_task_cache": match,
    "whole_scene_vs_task_related": validation["whole_scene_vs_task_related"],
    "peak_vram_bytes": validation["runtime"]["sum_peak_vram_bytes"],
    "clips_per_second": validation["runtime"]["overall_clips_per_second"],
}, indent=2, sort_keys=True))
PY
  echo "WHOLE_SCENE_MOTION_SMOKE=PASS output=$smoke_root logs=$log_root"
  exit 0
fi

if [[ ${WHOLE_SCENE_MOTION_CACHE_APPROVED:-NO} != YES ]]; then
  echo "Full cache is gated. Set WHOLE_SCENE_MOTION_CACHE_APPROVED=YES only after explicit approval." >&2
  exit 2
fi

visible_gpus=$($teacher_python -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $worker_count ]]; then
  echo "Expected exactly 8 visible CUDA devices, found $visible_gpus" >&2
  exit 1
fi
if [[ $(nproc) -lt 128 ]]; then
  echo "Expected at least 128 logical CPUs for eight workers" >&2
  exit 1
fi

gpu_idle_gate() {
  "$teacher_python" - <<'PY'
import subprocess
import sys

text = subprocess.check_output([
    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
], text=True)
used = [int(line.strip()) for line in text.splitlines() if line.strip()]
if len(used) != 8 or any(value > 1024 for value in used):
    print(f"WHOLE_SCENE_MOTION_GPU_IDLE_GATE=BUSY memory_mib={used}", flush=True)
    sys.exit(1)
print(f"WHOLE_SCENE_MOTION_GPU_IDLE_GATE=PASS memory_mib={used}")
PY
}

if [[ ${WHOLE_SCENE_MOTION_WAIT_FOR_IDLE:-NO} == YES ]]; then
  idle_poll_seconds=${WHOLE_SCENE_MOTION_IDLE_POLL_SECONDS:-1800}
  if [[ ! $idle_poll_seconds =~ ^[1-9][0-9]*$ ]]; then
    echo "WHOLE_SCENE_MOTION_IDLE_POLL_SECONDS must be a positive integer" >&2
    exit 2
  fi
  until gpu_idle_gate; do
    echo "WHOLE_SCENE_MOTION_GPU_IDLE_RETRY=$(date --iso-8601=seconds) wait_seconds=$idle_poll_seconds"
    sleep "$idle_poll_seconds"
  done
else
  gpu_idle_gate
fi

cache_ready=$(
  "$teacher_python" - "$output_root" "$expected_count" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    validation = json.loads((root / "cache_validation.json").read_text())
    index = root / "index.parquet"
    ready = (
        validation.get("status") == "PASS"
        and validation.get("target_scope") == "whole_scene"
        and validation.get("selected_samples") == expected
        and validation.get("shape") == [expected, 256]
        and validation.get("all_finite") is True
        and validation.get("no_missing_targets") is True
        and validation.get("task_related_population", {}).get("population_exact") is True
        and index.is_file()
        and hashlib.sha256(index.read_bytes()).hexdigest() == validation.get("index_sha256")
    )
except (FileNotFoundError, json.JSONDecodeError):
    ready = False
print("YES" if ready else "NO")
PY
)
if [[ $cache_ready == YES ]]; then
  echo "WHOLE_SCENE_MOTION_CACHE_ALREADY_COMPLETE=SKIP root=$output_root"
  exit 0
fi

mkdir -p "$output_root" "$log_root"
cpu_sets=(
  0-7,64-71
  8-15,72-79
  16-23,80-87
  24-31,88-95
  32-39,96-103
  40-47,104-111
  48-55,112-119
  56-63,120-127
)

echo "WHOLE_SCENE_MOTION_START=$(date --iso-8601=seconds) selected=$expected_count logs=$log_root"
pids=()
for worker_index in 0 1 2 3 4 5 6 7; do
  taskset -c "${cpu_sets[$worker_index]}" env \
    CUDA_VISIBLE_DEVICES="$worker_index" \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
    "$teacher_python" -m four_suite_experiments.whole_scene_motion \
      --mode worker \
      --manifest "$manifest" \
      --source-manifest "$manifest" \
      --selection-column "$selection_column" \
      --track4world-repo "$track_repo" \
      --utils3d-repo "$utils_repo" \
      --checkpoint "$track_checkpoint" \
      --da3-snapshot "$da3_snapshot" \
      --teacher-reference-root "$teacher_reference" \
      --output-root "$output_root" \
      --worker-index "$worker_index" \
      --num-workers "$worker_count" \
      --device cuda:0 \
      --shard-size 50 \
      --resume \
      >"$log_root/worker_${worker_index}.log" 2>&1 &
  pids+=("$!")
done

worker_failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || worker_failed=1
done
if [[ $worker_failed -ne 0 ]]; then
  echo "At least one Whole-scene Motion worker failed; completed shards remain resumable: $log_root" >&2
  exit 1
fi

"$teacher_python" -m four_suite_experiments.whole_scene_motion \
  --mode finalize \
  --manifest "$manifest" \
  --selection-column "$selection_column" \
  --task-related-index "$task_related_index" \
  --output-root "$output_root" \
  --num-workers "$worker_count" \
  --final-shard-size 1000 \
  >"$log_root/finalize.log" 2>&1

"$teacher_python" - "$output_root/cache_validation.json" "$expected_count" <<'PY'
import json
import sys

validation = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
assert validation["status"] == "PASS"
assert validation["target_scope"] == "whole_scene"
assert validation["selected_samples"] == expected
assert validation["shape"] == [expected, 256]
assert validation["dtype"] == "float32"
assert validation["all_finite"] and validation["no_missing_targets"]
assert validation["sample_ids_unique"]
assert validation["task_related_population"]["population_exact"]
assert validation["whole_scene_vs_task_related"]["different_rows"] > 0
print(f"WHOLE_SCENE_MOTION_FINAL_GATE=PASS samples={expected}")
PY
echo "WHOLE_SCENE_MOTION_FINISH=$(date --iso-8601=seconds) output=$output_root logs=$log_root"
