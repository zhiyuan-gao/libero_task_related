# LIBERO-10 P1/P2: 8-GPU migration and training

This repository contains the code for the frozen LIBERO-10 P1/P2 protocol. The
training data, auxiliary targets, base weights, and checkpoints are intentionally
not stored in Git.

## Frozen 8-GPU recipe

| Setting | Value |
| --- | ---: |
| GPUs / DDP processes | 8 |
| Local micro-batch per GPU | 32 |
| Global micro-batch | 256 |
| Gradient accumulation | 1 |
| Effective global batch | 256 |
| Optimizer updates | 11,132 |
| Warmup updates | 3,710 |
| EMA decay | 0.999 |
| Precision | bfloat16 |

P1 uses `L_action + 0.15 L_geo`. P2 uses
`L_action + 0.15 L_geo + 0.50 L_ground + 0.01 L_sem`. P2 production training
uses the validated joint-masked semantic implementation; the two-pass path is
retained only as a correctness reference.

## 1. Recreate the fixed filesystem layout

The checked-in configs use `/workspace/vla` paths. On the source machine, copy
the following four directories to the same paths on the target machine:

```text
/workspace/vla/data/libero_four_suite_annotation/policy_aux_v1
/workspace/vla/cache/huggingface/hub/datasets--physical-intelligence--libero
/workspace/vla/models/openpi/pi05_base_pytorch
/workspace/vla/models/openpi/pi05_libero_pytorch/assets
```

The Hugging Face dataset directory must be copied as a whole because its
snapshot uses links into the adjacent `blobs` directory. From the target
machine, commands of this form are safe to resume after interruption:

```bash
mkdir -p /workspace/vla/data/libero_four_suite_annotation
mkdir -p /workspace/vla/cache/huggingface/hub
mkdir -p /workspace/vla/models/openpi/pi05_libero_pytorch

rsync -aH --partial --info=progress2 \
  <SOURCE_HOST>:/workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/ \
  /workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/

rsync -aH --partial --info=progress2 \
  <SOURCE_HOST>:/workspace/vla/cache/huggingface/hub/datasets--physical-intelligence--libero/ \
  /workspace/vla/cache/huggingface/hub/datasets--physical-intelligence--libero/

rsync -aH --partial --info=progress2 \
  <SOURCE_HOST>:/workspace/vla/models/openpi/pi05_base_pytorch/ \
  /workspace/vla/models/openpi/pi05_base_pytorch/

rsync -aH --partial --info=progress2 \
  <SOURCE_HOST>:/workspace/vla/models/openpi/pi05_libero_pytorch/assets/ \
  /workspace/vla/models/openpi/pi05_libero_pytorch/assets/
```

The source payload is currently about 20.4 GB. Checkpoints are separate and are
not required for a new run. Do not resume a 4-GPU checkpoint as an 8-GPU run:
exact continuation records rank-local state and requires the original world
size.

Verify the base weights:

```bash
sha256sum /workspace/vla/models/openpi/pi05_base_pytorch/model.safetensors
```

Expected SHA-256:

```text
6dbc20690a4c391f3a2ae811aa216797a705d82656c54fe0ed4f041a032522c7
```

## 2. Clone and build the environment

```bash
mkdir -p /workspace/vla/third_party
cd /workspace/vla/third_party
git clone --recurse-submodules \
  --branch agent/p1-p2-8gpu-migration \
  https://github.com/zhiyuan-gao/libero_task_related.git openpi
cd /workspace/vla/third_party/openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

The repository pins Python and package revisions through `.python-version` and
`uv.lock`. Confirm that all eight GPUs are visible:

```bash
nvidia-smi -L
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

The second command must report `8` CUDA devices.

## 3. Run the mandatory 25+1 preflight

Run P1 and P2 separately. Each command trains 25 optimizer updates, saves a
raw/EMA checkpoint, restores the complete optimizer/RNG/data state, and then
runs one more update.

```bash
cd /workspace/vla/third_party/openpi

EXP_NAME=p1_8gpu_preflight_$(date -u +%Y%m%d) \
  scripts/preflight_p1_libero10_8gpu.sh

EXP_NAME=p2_8gpu_preflight_$(date -u +%Y%m%d) \
  scripts/preflight_p2_libero10_8gpu.sh
```

Before formal training, confirm all ranks remain alive, losses and gradient
norms are finite, step 25 is saved, resume reaches step 26, and per-GPU peak
allocated memory stays below the physical limit. Use the steady updates, not
the first initialization update, to estimate runtime on the new machine.

## 4. Start formal training

Both runs require all eight GPUs, so run them sequentially unless a second
8-GPU node is available.

```bash
cd /workspace/vla/third_party/openpi

FULL_TRAINING_APPROVED=YES \
EXP_NAME=p1_libero10_8gpu_v1 \
  scripts/launch_p1_libero10_8gpu.sh

FULL_TRAINING_APPROVED=YES \
EXP_NAME=p2_libero10_joint_masked_8gpu_v1 \
  scripts/launch_p2_libero10_8gpu.sh
```

For exact continuation of an interrupted 8-GPU formal run, use the identical
command and experiment name with `RESUME=YES`. Never add `--overwrite` to a
resume:

```bash
FULL_TRAINING_APPROVED=YES \
RESUME=YES \
EXP_NAME=p2_libero10_joint_masked_8gpu_v1 \
  scripts/launch_p2_libero10_8gpu.sh
```

By default, checkpoints are written beneath
`/workspace/vla/checkpoints/openpi_policy_aux`. Set `CHECKPOINT_BASE_DIR` before
launch if the target machine uses a different high-capacity disk. Keep at least
100 GiB free per active run, including room for atomic checkpoint writes.

## 5. Runtime expectation

The latest validated 4xA100 steady measurements were approximately 13.6
s/update for P1 and 14.4 s/update for joint-masked P2. Ideal 8-GPU scaling would
put each 11,132-update run near 21--23 hours of compute, but this is not an
8-GPU measurement. Re-estimate from the target machine's 25-step preflight and
add checkpoint/startup margin before scheduling the formal runs.
