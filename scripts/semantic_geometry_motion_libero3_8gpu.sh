#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != preflight && $1 != full ]]; then
  echo "usage: $0 <preflight|full>" >&2
  exit 2
fi

phase=$1
openpi_root=${OPENPI_ROOT:-/workspace/vla/libero_task_related}
openpi_python=${OPENPI_PYTHON:-${openpi_root}/.venv/bin/python}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/p3/checkpoints/semantic_geometry_motion}
num_workers=${NUM_WORKERS:-8}

readonly config_name=pi05_libero3_semantic_geometry_motion_aux
readonly nproc_per_node=8
readonly global_batch=256
readonly accumulation_steps=1
readonly total_updates=3209
readonly lambda_geo=0.15
readonly lambda_sem=0.01
readonly lambda_motion=0.10

export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=1
export OPENPI_LOG_MEMORY_STATS=0
export TOKENIZERS_PARALLELISM=false

if [[ ! $num_workers =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be a non-negative integer, found '$num_workers'" >&2
  exit 2
fi
visible_gpus=$("$openpi_python" -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $nproc_per_node ]]; then
  echo "Expected exactly 8 visible CUDA devices, found $visible_gpus" >&2
  exit 2
fi

"$openpi_python" - <<'PY'
import hashlib
from pathlib import Path

from openpi.training import config
from safetensors import safe_open

cfg = config.get_config("pi05_libero3_semantic_geometry_motion_aux")
aux = cfg.policy_aux
assert cfg.ema_decay is None
assert aux.mode == "semantic_geometry_motion"
assert aux.num_ground_queries == 0 and aux.num_motion_queries == 8
assert aux.lambda_geo == 0.15 and aux.lambda_sem == 0.01 and aux.lambda_motion == 0.10
assert aux.lambda_ground is None and aux.lerobot_task_indices == (0, 3, 8)
assert cfg.batch_size == 256 and cfg.gradient_accumulation_steps == 1
assert cfg.num_train_steps == 3209 and cfg.lr_schedule.warmup_steps == 1069
checkpoint = Path(cfg.pytorch_weight_path) / "model.safetensors"
assert checkpoint.stat().st_size == 14_467_165_872
with safe_open(checkpoint, framework="pt", device="cpu") as handle:
    names = list(handle.keys())
    assert len(names) == 812
    assert all(handle.get_slice(name).get_dtype() == "F32" for name in names)
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == "4f5facee8d0897bcc95900929e2cdb978a4cebd63651d80155444d4c086c23ee"
print("SEMANTIC_GEOMETRY_MOTION_CONFIG_GATE=PASS")
print("STRICT_FP32_BASE_GATE=PASS")
PY

common_args=(
  "$config_name"
  --batch-size "$global_batch"
  --gradient-accumulation-steps "$accumulation_steps"
  --num-workers "$num_workers"
  --checkpoint-base-dir "$checkpoint_base_dir"
  --keep-period 1000
  --policy-aux.loss-coefficients-approved
  --policy-aux.lambda-geo "$lambda_geo"
  --policy-aux.lambda-sem "$lambda_sem"
  --policy-aux.lambda-motion "$lambda_motion"
  --no-wandb-enabled
)

run_train() {
  "$openpi_python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="$nproc_per_node" \
    "$openpi_root/scripts/train_pytorch.py" "${common_args[@]}" "$@"
}

case "$phase" in
  preflight)
    exp_name=${EXP_NAME:-sgm_libero3_8gpu_preflight}
    run_train --exp-name "$exp_name" --num-train-steps 25 --save-interval 25 --log-interval 1 --overwrite
    run_train --exp-name "$exp_name" --num-train-steps 26 --save-interval 1000 \
      --no-save-final-checkpoint --log-interval 1 --resume
    ;;
  full)
    if [[ ${FULL_TRAINING_APPROVED:-NO} != YES ]]; then
      echo "Set FULL_TRAINING_APPROVED=YES only after explicit approval and a passing 25+1 preflight" >&2
      exit 2
    fi
    if [[ -z ${EXP_NAME:-} ]]; then
      echo "EXP_NAME is required for a full run" >&2
      exit 2
    fi
    resume_args=(--overwrite)
    if [[ ${RESUME:-NO} == YES ]]; then
      resume_args=(--resume)
    fi
    run_train --exp-name "$EXP_NAME" --num-train-steps "$total_updates" \
      --save-interval 1000 --log-interval 1 "${resume_args[@]}"
    ;;
esac
