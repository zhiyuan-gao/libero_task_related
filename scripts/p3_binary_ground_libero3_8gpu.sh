#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != gate && $1 != full ) ]]; then
  echo "usage: $0 <gate|full>" >&2
  exit 2
fi

phase=$1
openpi_root=${OPENPI_ROOT:-/workspace/vla/libero_task_related}
openpi_python=${OPENPI_PYTHON:-${openpi_root}/.venv/bin/python}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-/workspace/vla/p3/checkpoints/p3_binary_ground}
num_workers=${NUM_WORKERS:-8}

readonly config_name=pi05_libero3_p3_binary_ground_aux
readonly nproc_per_node=8
readonly global_batch=256
readonly accumulation_steps=1
readonly total_updates=3209
readonly lambda_geo=0.15
readonly lambda_sem=0.01
readonly lambda_motion=0.05
readonly lambda_ground=0.05
readonly ground_positive_weight=5.9399674343

export OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=1
export OPENPI_LOG_MEMORY_STATS=0
export TOKENIZERS_PARALLELISM=false

if [[ ! $num_workers =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be a non-negative integer, found '$num_workers'" >&2
  exit 2
fi
visible_gpus=$($openpi_python -c 'import torch; print(torch.cuda.device_count())')
if [[ $visible_gpus -ne $nproc_per_node ]]; then
  echo "Expected exactly 8 visible CUDA devices, found $visible_gpus" >&2
  exit 2
fi

$openpi_python - <<'PY'
import hashlib
import json
from pathlib import Path

from openpi.training import config
from openpi.training.policy_aux_dataset import LIBERO3_BINARY_GROUND_NEGATIVE_PATCHES
from openpi.training.policy_aux_dataset import LIBERO3_BINARY_GROUND_POSITIVE_PATCHES
from openpi.training.policy_aux_dataset import LIBERO3_BINARY_GROUND_POSITIVE_WEIGHT
from openpi.training.policy_aux_dataset import MotionPolicyTargetIndex
from safetensors import safe_open

cfg = config.get_config("pi05_libero3_p3_binary_ground_aux")
aux = cfg.policy_aux
assert cfg.ema_decay is None
assert aux.mode == "semantic_geometry_motion_binary_ground"
assert aux.num_ground_queries == 8 and aux.num_motion_queries == 8
assert aux.lambda_geo == 0.15 and aux.lambda_sem == 0.01
assert aux.lambda_motion == 0.05 and aux.lambda_ground == 0.05
assert aux.ground_objective == "binary_fixed_balanced_bce"
assert LIBERO3_BINARY_GROUND_POSITIVE_PATCHES == 2_094_230
assert LIBERO3_BINARY_GROUND_NEGATIVE_PATCHES == 12_439_658
assert LIBERO3_BINARY_GROUND_POSITIVE_WEIGHT == 5.9399674343
assert aux.ground_positive_weight == LIBERO3_BINARY_GROUND_POSITIVE_WEIGHT
assert aux.lerobot_task_indices == (0, 3, 8)
assert cfg.batch_size == 256 and cfg.gradient_accumulation_steps == 1
assert cfg.num_train_steps == 3209 and cfg.lr_schedule.warmup_steps == 1069
assert cfg.pytorch_training_precision == "bfloat16"
assert cfg.optimizer.clip_gradient_norm == 1.0
assert cfg.save_interval == 500 and cfg.keep_period is None
assert cfg.checkpoint_keep_steps == (500, 1000, 2000, 3209)

motion_index = Path(aux.motion_target_index_path).resolve(strict=True)
motion_root = motion_index.parent
validation = json.loads((motion_root / "cache_validation.json").read_text())
assert validation["selected_samples"] == 28_110
assert validation["shape"] == [28_110, 256]
assert validation["dtype"] == "float32"
assert validation["all_finite"] and validation["no_missing_targets"] and validation["sample_ids_unique"]
MotionPolicyTargetIndex(motion_index, aux.motion_normalization_path)

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

print("P3_ARCHITECTURE_AND_LOSS_GATE=PASS")
print("P3_DATASET_GLOBAL_WEIGHT_GATE=PASS")
print("P3_MOTION_CACHE_GATE=PASS")
print("P3_STRICT_FP32_BASE_GATE=PASS")
print("P3_CHECKPOINT_MILESTONE_GATE=PASS")
PY

if [[ $phase == gate ]]; then
  exit 0
fi

if [[ ${FULL_TRAINING_APPROVED:-NO} != YES ]]; then
  echo "Set FULL_TRAINING_APPROVED=YES only after explicit user approval" >&2
  exit 2
fi
if [[ -z ${EXP_NAME:-} ]]; then
  echo "EXP_NAME is required for a full run" >&2
  exit 2
fi

common_args=(
  "$config_name"
  --exp-name "$EXP_NAME"
  --batch-size "$global_batch"
  --gradient-accumulation-steps "$accumulation_steps"
  --num-workers "$num_workers"
  --checkpoint-base-dir "$checkpoint_base_dir"
  --num-train-steps "$total_updates"
  --save-interval 500
  --log-interval 1
  --policy-aux.loss-coefficients-approved
  --policy-aux.lambda-geo "$lambda_geo"
  --policy-aux.lambda-sem "$lambda_sem"
  --policy-aux.lambda-motion "$lambda_motion"
  --policy-aux.lambda-ground "$lambda_ground"
  --policy-aux.ground-objective binary_fixed_balanced_bce
  --policy-aux.ground-positive-weight "$ground_positive_weight"
  --no-wandb-enabled
)

resume_args=(--overwrite)
if [[ ${RESUME:-NO} == YES ]]; then
  resume_args=(--resume)
fi

$openpi_python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="$nproc_per_node" \
  "$openpi_root/scripts/train_pytorch.py" "${common_args[@]}" "${resume_args[@]}"
