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
python=${GEOMETRY_PYTHON:-/workspace/vla/libero_task_related/.venv/bin/python}
manifest=${FOUR_SUITE_GEOMETRY_MANIFEST:-${p3_project}/runtime_metadata/four_suite_policy_geometry_manifest.parquet}
task_related_index=${TASK_RELEVANT_GEOMETRY_INDEX:-/workspace/vla/four_suite_joint_experiments/artifacts/task_relevant/geometry_index.parquet}
output_root=${WHOLE_SCENE_GEOMETRY_CACHE_ROOT:-${p3_vla_root}/data/libero_four_suite_annotation/policy_aux_v1/geometry_whole_scene_four_suite_v1}
smoke_root=${WHOLE_SCENE_GEOMETRY_SMOKE_ROOT:-${p3_project}/reports/whole_scene_geometry/smoke_cache_v1}
vggt_repo=${VGGT_REPO:-${p3_vla_root}/third_party/vggt}
checkpoint=${VGGT_CHECKPOINT:-${p3_vla_root}/models/VGGT-1B/model.pt}
teacher_reference=${VGGT_GEOMETRY_REFERENCE_ROOT:-${p3_project}/runtime_metadata/vggt_geometry_reference_v1}
geometry_helper_root=${VGGT_GEOMETRY_HELPER_ROOT:-${p3_vla_root}/third_party/robocasa-video-progress/scripts}
run_id=${WHOLE_SCENE_GEOMETRY_RUN_ID:-whole_scene_geometry_$(date +%Y%m%d_%H%M%S)}
log_root=${WHOLE_SCENE_GEOMETRY_LOG_ROOT:-${p3_project}/logs/${run_id}}

readonly worker_count=8
readonly expected_count=273377
readonly selection_column=geometry_valid

required_paths=(
  "$python"
  "$manifest"
  "$task_related_index"
  "$vggt_repo/.git"
  "$checkpoint"
  "$geometry_helper_root/run_vggt_geometry_hook1.py"
  "$teacher_reference/hook1/introspection.json"
  "$teacher_reference/smoke16/smoke_report.json"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e $required_path ]]; then
    echo "Missing Whole-scene Geometry prerequisite: $required_path" >&2
    exit 1
  fi
done

export PYTHONPATH="$project_root/src:$geometry_helper_root:$vggt_repo"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

"$python" - "$manifest" "$task_related_index" "$expected_count" <<'PY'
import sys
from collections import Counter

import pyarrow.parquet as pq

manifest_path, task_index_path, expected_text = sys.argv[1:]
expected = int(expected_text)
manifest = pq.read_table(
    manifest_path,
    columns=["sample_id", "suite", "geometry_valid"],
)
valid = manifest.filter(manifest["geometry_valid"])
task = pq.read_table(task_index_path, columns=["sample_id", "geometry_valid"])
task_valid = task.filter(task["geometry_valid"])
valid_ids = valid["sample_id"].to_pylist()
task_valid_ids = task_valid["sample_id"].to_pylist()
expected_by_suite = {
    "libero_10": 101_381,
    "libero_goal": 52_042,
    "libero_object": 66_984,
    "libero_spatial": 52_970,
}
actual_by_suite = Counter(valid["suite"].to_pylist())
if manifest.num_rows != 273_465 or task.num_rows != 273_465:
    raise SystemExit("Whole-scene Geometry policy population gate failed")
if len(valid_ids) != expected or len(set(valid_ids)) != expected:
    raise SystemExit("Whole-scene Geometry valid population gate failed")
if dict(actual_by_suite) != expected_by_suite:
    raise SystemExit(f"Whole-scene Geometry suite population mismatch: {dict(actual_by_suite)}")
if len(task_valid_ids) != expected or set(task_valid_ids) != set(valid_ids):
    raise SystemExit("Task-related and Whole-scene Geometry populations are not exactly paired")
print(
    "WHOLE_SCENE_GEOMETRY_GATE=PASS "
    f"selected={expected} suites={dict(sorted(actual_by_suite.items()))}"
)
PY

if [[ $phase == gate ]]; then
  exit 0
fi

if [[ $phase == smoke ]]; then
  smoke_gpu=${WHOLE_SCENE_GEOMETRY_SMOKE_GPU:-0}
  used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$smoke_gpu" | tr -d ' ')
  if [[ $used_mib -gt 1024 ]]; then
    echo "Smoke GPU $smoke_gpu is not idle (${used_mib} MiB used); refusing to disturb another job" >&2
    exit 1
  fi
  mkdir -p "$smoke_root" "$log_root"
  CUDA_VISIBLE_DEVICES="$smoke_gpu" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    "$python" -m four_suite_experiments.whole_scene_geometry \
      --mode worker \
      --manifest "$manifest" \
      --selection-column "$selection_column" \
      --vggt-repo "$vggt_repo" \
      --checkpoint "$checkpoint" \
      --teacher-reference-root "$teacher_reference" \
      --output-root "$smoke_root" \
      --worker-index 0 \
      --num-workers 1 \
      --device cuda:0 \
      --batch-size 8 \
      --loader-workers 8 \
      --shard-size 8 \
      --diagnostic-samples-per-suite 2 \
      --diagnostic-compare-task-related \
      --resume \
      >"$log_root/smoke_worker.log" 2>&1
  "$python" -m four_suite_experiments.whole_scene_geometry \
    --mode finalize \
    --manifest "$manifest" \
    --selection-column "$selection_column" \
    --task-related-index "$task_related_index" \
    --output-root "$smoke_root" \
    --num-workers 1 \
    --diagnostic-samples-per-suite 2 \
    >"$log_root/smoke_finalize.log" 2>&1
  "$python" - "$smoke_root/cache_validation.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    validation = json.load(stream)
assert validation["status"] == "PASS"
assert validation["target_scope"] == "whole_scene"
assert validation["policy_samples"] == 8
assert validation["geometry_valid_samples"] == 8
assert validation["geometry_invalid_samples"] == 0
assert validation["selected_by_suite"] == {
    "libero_10": 2,
    "libero_goal": 2,
    "libero_object": 2,
    "libero_spatial": 2,
}
assert validation["shape"] == [8, 2048]
assert validation["dtype"] == "float32"
assert validation["all_finite"] and validation["no_missing_targets"]
match = validation["same_forward_task_related_cache_check"]
assert match["within_atol_1e-5"] and match["max_abs"] <= 1e-5
assert validation["whole_scene_vs_task_related"]["different_rows"] > 0
print(json.dumps({
    "status": "PASS",
    "samples": validation["geometry_valid_samples"],
    "selected_by_suite": validation["selected_by_suite"],
    "same_forward_task_cache": match,
    "whole_scene_vs_task_related": validation["whole_scene_vs_task_related"],
}, indent=2, sort_keys=True))
PY
  echo "WHOLE_SCENE_GEOMETRY_SMOKE=PASS output=$smoke_root logs=$log_root"
  exit 0
fi

if [[ ${WHOLE_SCENE_GEOMETRY_CACHE_APPROVED:-NO} != YES ]]; then
  echo "Full cache is gated. Set WHOLE_SCENE_GEOMETRY_CACHE_APPROVED=YES only after explicit approval." >&2
  exit 2
fi

visible_gpus=$($python -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $worker_count ]]; then
  echo "Expected exactly 8 visible CUDA devices, found $visible_gpus" >&2
  exit 1
fi
if [[ $(nproc) -lt 128 ]]; then
  echo "Expected at least 128 logical CPUs for eight workers" >&2
  exit 1
fi

"$python" - <<'PY'
import subprocess
import sys

text = subprocess.check_output([
    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
], text=True)
used = [int(line.strip()) for line in text.splitlines() if line.strip()]
if len(used) != 8 or any(value > 1024 for value in used):
    print(f"WHOLE_SCENE_GEOMETRY_GPU_IDLE_GATE=BUSY memory_mib={used}", flush=True)
    sys.exit(1)
print(f"WHOLE_SCENE_GEOMETRY_GPU_IDLE_GATE=PASS memory_mib={used}")
PY

cache_ready=$(
  "$python" - "$output_root" "$expected_count" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    with (root / "cache_validation.json").open(encoding="utf-8") as stream:
        validation = json.load(stream)
    index = root / "target_index.parquet"
    ready = (
        validation.get("status") == "PASS"
        and validation.get("target_scope") == "whole_scene"
        and validation.get("policy_samples") == 273_465
        and validation.get("geometry_valid_samples") == expected
        and validation.get("shape") == [expected, 2048]
        and validation.get("all_finite") is True
        and validation.get("no_missing_targets") is True
        and validation.get("task_related_population_exact") is True
        and index.is_file()
        and hashlib.sha256(index.read_bytes()).hexdigest()
        == validation.get("target_index_sha256")
    )
except (FileNotFoundError, json.JSONDecodeError):
    ready = False
print("YES" if ready else "NO")
PY
)
if [[ $cache_ready == YES ]]; then
  echo "WHOLE_SCENE_GEOMETRY_CACHE_ALREADY_COMPLETE=SKIP root=$output_root"
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

echo "WHOLE_SCENE_GEOMETRY_START=$(date --iso-8601=seconds) selected=$expected_count logs=$log_root"
pids=()
for worker_index in 0 1 2 3 4 5 6 7; do
  taskset -c "${cpu_sets[$worker_index]}" env \
    CUDA_VISIBLE_DEVICES="$worker_index" \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
    "$python" -m four_suite_experiments.whole_scene_geometry \
      --mode worker \
      --manifest "$manifest" \
      --selection-column "$selection_column" \
      --vggt-repo "$vggt_repo" \
      --checkpoint "$checkpoint" \
      --teacher-reference-root "$teacher_reference" \
      --output-root "$output_root" \
      --worker-index "$worker_index" \
      --num-workers "$worker_count" \
      --device cuda:0 \
      --batch-size 32 \
      --loader-workers 8 \
      --shard-size 500 \
      --resume \
      >"$log_root/worker_${worker_index}.log" 2>&1 &
  pids+=("$!")
done

worker_failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || worker_failed=1
done
if [[ $worker_failed -ne 0 ]]; then
  echo "At least one Whole-scene Geometry worker failed; completed shards remain resumable: $log_root" >&2
  exit 1
fi

"$python" -m four_suite_experiments.whole_scene_geometry \
  --mode finalize \
  --manifest "$manifest" \
  --selection-column "$selection_column" \
  --task-related-index "$task_related_index" \
  --output-root "$output_root" \
  --num-workers "$worker_count" \
  >"$log_root/finalize.log" 2>&1

"$python" - "$output_root/cache_validation.json" "$expected_count" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    validation = json.load(stream)
expected = int(sys.argv[2])
assert validation["status"] == "PASS"
assert validation["target_scope"] == "whole_scene"
assert validation["policy_samples"] == 273_465
assert validation["geometry_valid_samples"] == expected
assert validation["shape"] == [expected, 2048]
assert validation["dtype"] == "float32"
assert validation["all_finite"] and validation["no_missing_targets"]
assert validation["sample_ids_unique"]
assert validation["task_related_population_exact"]
assert validation["whole_scene_vs_task_related"] is None
print(f"WHOLE_SCENE_GEOMETRY_FINAL_GATE=PASS samples={expected}")
PY
echo "WHOLE_SCENE_GEOMETRY_FINISH=$(date --iso-8601=seconds) output=$output_root logs=$log_root"
