# From-zero Slurm HPC setup

This runbook assumes a single Slurm node with eight 80 GB GPUs. It prepares the
LIBERO-40 TRQC release but does not authorize or launch formal training.
Replace every `REPLACE_*` value with the site's real account, partition, login,
and persistent scratch paths.

## 1. Site prerequisites

Ask the HPC documentation or administrator for the correct GPU partition,
account, maximum wall time, memory flag, and GPU resource spelling. The supplied
template uses `--gpus-per-node=8`; sites that expose GPUs through GRES may require
`--gres=gpu:a100:8` at submission instead.

Required on the login/compute environment:

- Linux x86_64 and an NVIDIA driver compatible with the CUDA 12.6 PyTorch wheel
- one Slurm node exposing exactly eight GPUs
- outbound access to GitHub, PyPI, and Hugging Face from either the login node or
  a data-transfer node
- `git`, `curl`, `rsync`, and `sha256sum`
- at least 220 GB free persistent high-throughput storage; 250 GB is safer

Do not install a private CUDA toolkit unless the site requires it. The frozen
Python environment installs PyTorch 2.7.1 and its CUDA 12.6 userspace runtime;
the site still supplies the NVIDIA driver.

Before setup, inspect the site:

```bash
sinfo
scontrol show partition REPLACE_GPU_PARTITION
nvidia-smi  # run this inside a small GPU allocation, not necessarily on login
```

## 2. Clone the exact release and install Python

Use persistent scratch, not node-local temporary storage:

```bash
mkdir -p /scratch/REPLACE_USER/libero40
cd /scratch/REPLACE_USER/libero40
git clone \
  --branch libero40-task-relevant-query-conditioning-release \
  --recurse-submodules \
  https://github.com/zhiyuan-gao/libero_task_related.git
cd libero_task_related
git rev-parse HEAD
```

Install `uv` without root privileges, then create the exact Python 3.11
environment and apply the required Transformers patch:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
cd /scratch/REPLACE_USER/libero40/libero_task_related/experiments/four_suite_joint
./jobs/setup_python_env.sh
./jobs/run_8gpu.sh test
```

The expected release commit when this runbook was written is recorded by Git;
do not silently switch branches or update dependencies. `uv sync --frozen`
uses the committed `uv.lock`.

## 3. Download the official LeRobot snapshot on the HPC

The complete public snapshot is 34,938,730,091 bytes (about 32.54 GiB). It has
1,693 episode parquet files, 273,465 frames, and 40 tasks. Download it into the
Hugging Face cache so that the snapshot directory retains the required revision
name:

```bash
cd /scratch/REPLACE_USER/libero40/libero_task_related/experiments/four_suite_joint
./jobs/run_8gpu.sh download-data \
  --cache-dir /scratch/REPLACE_USER/libero40/hf/hub
```

The downloader pins both the repository and revision and finishes by checking
all 1,693 parquet files. The resulting path must be:

```text
/scratch/REPLACE_USER/libero40/hf/hub/
  datasets--physical-intelligence--libero/snapshots/
  a4336d589d589045d1c56423ffdf3b88a0e19b1f
```

If compute nodes have no internet, run this step on a login/data-transfer node
against the same shared scratch. Do not download into node-local `$TMPDIR`.

## 4. Copy the frozen local assets

Do not regenerate these teachers on the HPC. Copy exactly the following from
the current 8-A100 machine while preserving each directory name:

| Component | Local source | Size |
|---|---|---:|
| Geometry LIBERO-10 | `/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/geometry_libero10` | 1,604,248,829 B |
| Geometry other suites | `/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/geometry_libero_goal_object_spatial_v1` | 2,723,365,481 B |
| Motion LIBERO-10 | `/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/motion_libero10_full_v1` | 93,165,574 B |
| Motion other suites | `/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/motion_libero_goal_object_spatial_v1` | 312,061,465 B |
| Semantic/identity manifest | `/workspace/vla/p3/runtime_metadata/four_suite_policy_geometry_manifest.parquet` | 6,306,839 B |
| strict FP32 base | `/workspace/vla/models/openpi/pi05_base_pytorch_fp32` | 14,467,166,020 B |
| LIBERO normalization assets | `/workspace/vla/models/openpi/pi05_libero_pytorch/assets` | 1,914 B |

The transfer total is 19,206,316,122 bytes (about 17.89 GiB). From the current
machine, set a real SSH destination and run:

```bash
export HPC_LOGIN=REPLACE_USER@REPLACE_HPC_LOGIN
export HPC_ASSET_ROOT=/scratch/REPLACE_USER/libero40/assets

ssh "${HPC_LOGIN}" "mkdir -p \
  '${HPC_ASSET_ROOT}/annotation/policy_aux_v1' \
  '${HPC_ASSET_ROOT}/runtime_metadata' \
  '${HPC_ASSET_ROOT}/models'"

rsync -a --partial --info=progress2 \
  /workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/geometry_libero10/ \
  "${HPC_LOGIN}:${HPC_ASSET_ROOT}/annotation/policy_aux_v1/geometry_libero10/"
rsync -a --partial --info=progress2 \
  /workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/geometry_libero_goal_object_spatial_v1/ \
  "${HPC_LOGIN}:${HPC_ASSET_ROOT}/annotation/policy_aux_v1/geometry_libero_goal_object_spatial_v1/"
rsync -a --partial --info=progress2 \
  /workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/motion_libero10_full_v1/ \
  "${HPC_LOGIN}:${HPC_ASSET_ROOT}/annotation/policy_aux_v1/motion_libero10_full_v1/"
rsync -a --partial --info=progress2 \
  /workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/motion_libero_goal_object_spatial_v1/ \
  "${HPC_LOGIN}:${HPC_ASSET_ROOT}/annotation/policy_aux_v1/motion_libero_goal_object_spatial_v1/"
rsync -a --partial --info=progress2 \
  /workspace/vla/p3/runtime_metadata/four_suite_policy_geometry_manifest.parquet \
  "${HPC_LOGIN}:${HPC_ASSET_ROOT}/runtime_metadata/"
rsync -a --partial --info=progress2 \
  /workspace/vla/models/openpi/pi05_base_pytorch_fp32/ \
  "${HPC_LOGIN}:${HPC_ASSET_ROOT}/models/pi05_base_pytorch_fp32/"
rsync -a --partial --info=progress2 \
  /workspace/vla/models/openpi/pi05_libero_pytorch/assets/ \
  "${HPC_LOGIN}:${HPC_ASSET_ROOT}/models/pi05_libero_pytorch/assets/"
```

After the first transfer, rerun the same commands with `--checksum --dry-run
--itemize-changes` added. No listed changes means local and remote file contents
match. The later `prepare` and `preflight` steps also validate target counts,
shapes, finite normalization values, identities, and artifact hashes.

Do **not** copy the old `four_suite_joint_experiments/artifacts` directory: its
combined indices contain old absolute paths. The release rebuilds these small
portable metadata files on the HPC. Do not copy prior reports, evaluations,
checkpoints, Hugging Face partial caches, or diagnostic outputs.

## 5. Configure paths and rebuild portable metadata

On the HPC:

```bash
cd /scratch/REPLACE_USER/libero40/libero_task_related/experiments/four_suite_joint
cp hpc.env.example hpc.env
# Edit every REPLACE_* path in hpc.env, then:
source hpc.env
./jobs/run_8gpu.sh prepare --target-scope task_relevant
./jobs/run_8gpu.sh preflight --variant trqc
./jobs/run_8gpu.sh preflight --variant no_query_access
```

Both task-relevant preflights must report `status: PASS`, `training_data_ready: true`, and
`optimizer_steps_executed: 0`. The `prepare` command rebases stale absolute
Motion shard paths onto the transferred HPC cache and refuses unresolved paths.

After both Whole-scene caches are transferred, prepare and validate their
independent bundle:

```bash
./jobs/run_8gpu.sh prepare --target-scope whole_scene
./jobs/run_8gpu.sh preflight --variant whole_scene
```

The two eventual cache roots to transfer are:

```text
/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/
  geometry_whole_scene_four_suite_v1/
/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/
  motion_whole_scene_four_suite_v1/
```

Copy each directory under the HPC
`assets/annotation/policy_aux_v1/` directory with `rsync -a --partial`, then
repeat with `--checksum --dry-run --itemize-changes`. Do not transfer either
cache until its `cache_validation.json` reports `status: PASS`. The TRQC and No
Query Access runs do not require these Whole-scene caches.

## 6. Slurm submission

The repository supplies `jobs/slurm_8gpu.sbatch`. It deliberately omits
site-specific account, partition, memory, and wall-time directives. Supply those
through `sbatch`; command-line options override template directives.

First verify an allocated node and the release tests:

```bash
cd /scratch/REPLACE_USER/libero40/libero_task_related/experiments/four_suite_joint
sbatch \
  --account=REPLACE_ACCOUNT \
  --partition=REPLACE_GPU_PARTITION \
  --time=00:30:00 \
  --mem=REPLACE_MEMORY \
  --export=ALL,FOUR_SUITE_HPC_ENV_FILE="$(pwd)/hpc.env" \
  jobs/slurm_8gpu.sbatch test
```

If the site uses GRES, add its required `--gres=gpu:a100:8` form. The job fails
before training unless exactly eight GPUs are visible. Inspect `slurm-libero40-<job>.out`
for Python test and GPU inventory output.

Formal training remains gated. Only after the real-batch and short optimizer
preflights pass, storage and wall time are confirmed, and the researcher gives
explicit approval should a reviewed command include:

```text
--export=ALL,FOUR_SUITE_HPC_ENV_FILE=/absolute/path/hpc.env,FOUR_SUITE_FULL_TRAINING_APPROVED=YES
```

Use `--disable-wandb` when the compute node has no W&B credentials/network. If a
Slurm time limit stops a run after a checkpoint, resubmit the identical command
with `--resume`; do not use `--overwrite`.

## 7. Storage budget

Approximate persistent allocation:

| Item | Space |
|---|---:|
| Python environment and package cache | 10-15 GB |
| official LeRobot snapshot | 32.54 GiB |
| transferred frozen assets | 17.89 GiB |
| three resumable checkpoints for one run | about 60 GB |
| three resumable checkpoints for all three variants | about 180 GB |
| temporary checkpoint write and safety margin | at least 20-40 GB |

A complete three-variant campaign should reserve at least 300 GB; 320 GB is the
recommended minimum if checkpoints from all runs coexist. A full checkpoint includes roughly 7.5 GB model weights
and 14 GB optimizer state, so budgeting only for `model.safetensors` is unsafe.
