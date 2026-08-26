# LIBERO-40 Task-Relevant Query Conditioning

This release branch is the self-contained code and runbook for the three
frozen LIBERO-40 experiments below. It intentionally excludes historical
reports, evaluation outputs, checkpoints, generated caches, and unrelated
OpenPI examples. Upstream and Gemma license files remain in the repository.

## Experiments

All variants use the official 40-task LeRobot training set: 1,693 episodes and
273,465 frames across `libero_10`, `libero_goal`, `libero_object`, and
`libero_spatial`.

| CLI variant | Semantic target | Geometry target | Motion target | Action reads Geometry/Motion queries |
| --- | --- | --- | --- | --- |
| `trqc` | task-relevant | task-relevant | task-relevant | yes |
| `whole_scene` | task-relevant | whole-scene | whole-scene | yes |
| `no_query_access` | task-relevant | task-relevant | task-relevant | no |

The frozen auxiliary objective is

```text
L = L_action + 0.01 L_semantic + 0.05 L_geometry + 0.05 L_motion
```

Every variant has eight Geometry queries and eight Motion queries. Grounding is
disabled. `no_query_access` changes only the Action Expert attention access;
its targets and losses are identical to `trqc`.

The candidate formal recipe is 30,000 optimizer updates, 10,000 warmup
updates, global batch 256, seed 42, BF16, AdamW, gradient clipping at 1.0, and
no EMA. Training saves every 1,000 updates through step 20,000, then every 500
updates through step 30,000. A rolling cap retains 30 evaluation models, so the
final set is `11k..20k` at 1k intervals plus `20.5k..30k` at 500-step intervals.
Only the newest two checkpoints retain AdamW, RNG, sampler, and DataLoader state
for exact continuation; older retained checkpoints are evaluation-only. Formal
training is gated and is never started by installation,
download, preparation, testing, or preflight commands.

## Required HPC environment

- Linux x86_64 with Git and `curl`
- Slurm allocation for one node with exactly 8 visible NVIDIA GPUs
  (the validated target is 8 x A100 80 GB)
- 64 CPU cores per training job and persistent scratch storage
- An NVIDIA driver compatible with the CUDA runtime resolved by the lockfile
- Access to the private Hugging Face dataset
  `Zhiyuan17/libero40-trqc-assets`

The repository lockfile installs Python 3.11, PyTorch 2.7.1, and Transformers
4.53.2. Do not create a different environment by hand.

## 1. Clone and install

```bash
export LIBERO40_ROOT=/scratch/$USER/libero40
mkdir -p "$LIBERO40_ROOT"
cd "$LIBERO40_ROOT"

git clone --recurse-submodules \
  --branch libero40-task-relevant-query-conditioning-release \
  https://github.com/zhiyuan-gao/libero_task_related.git
cd libero_task_related

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
./experiments/four_suite_joint/jobs/setup_python_env.sh

FOUR_SUITE_PYTHON="$PWD/.venv/bin/python" \
  ./experiments/four_suite_joint/jobs/run_8gpu.sh test
```

The setup script initializes submodules, performs `uv sync --frozen`, installs
the repository's Transformers replacement, and verifies the pinned versions.

## 2. Download the frozen inputs

There are two independent downloads.

### Public LeRobot data

The downloader pins `physical-intelligence/libero` to revision
`a4336d589d589045d1c56423ffdf3b88a0e19b1f` and rejects an incomplete or
different 1,693-episode snapshot.

```bash
cd "$LIBERO40_ROOT/libero_task_related/experiments/four_suite_joint"
./jobs/run_8gpu.sh download-data --cache-dir "$LIBERO40_ROOT/hf/hub"
```

Record the absolute `downloaded_snapshot` path printed by this command; it is
used as `FOUR_SUITE_LEROBOT_ROOT` below.

### Private research assets

Login is interactive and can use the browser verification flow. Download the
immutable tag, not the moving repository head.

```bash
cd "$LIBERO40_ROOT/libero_task_related"
uvx --from 'huggingface_hub>=1.0' hf auth login
uvx --from 'huggingface_hub>=1.0' hf download \
  Zhiyuan17/libero40-trqc-assets \
  --repo-type dataset \
  --revision task-and-whole-scene-v1 \
  --local-dir "$LIBERO40_ROOT/assets"

cd "$LIBERO40_ROOT/assets"
sha256sum -c SHA256SUMS
```

`task-and-whole-scene-v1` resolves to asset commit
`cb1c086c7928556af7c2d08ee99f226102c67692`. It contains:

- the strict FP32-converted pi0.5 PyTorch base;
- official LIBERO policy normalization statistics;
- the frozen sample-identity and Semantic manifest;
- task-relevant Geometry targets for 273,377 valid frames;
- task-relevant Motion targets for 256,401 valid frames;
- Whole-scene Geometry targets for the same 273,377 valid frames;
- Whole-scene Motion targets for the same 256,401 valid frames.

The 19.45 GB asset release does not duplicate the public LeRobot data and does
not contain checkpoints, reports, worker logs, smoke outputs, or temporary
generation shards. The earlier `task-relevant-v1` tag remains frozen for exact
reproduction of the two task-relevant variants alone.

## 3. Configure local paths

```bash
cd "$LIBERO40_ROOT/libero_task_related/experiments/four_suite_joint"
cp hpc.env.example hpc.env
```

Replace every `REPLACE_USER` value in `hpc.env`. Also set
`FOUR_SUITE_LEROBOT_ROOT` to the exact snapshot path printed by the public-data
download. The important path mapping is:

```text
FOUR_SUITE_ANNOTATION_ROOT=$LIBERO40_ROOT/assets/annotation
FOUR_SUITE_JOINT_MANIFEST=$LIBERO40_ROOT/assets/runtime_metadata/four_suite_policy_geometry_manifest.parquet
FOUR_SUITE_BASE_WEIGHTS=$LIBERO40_ROOT/assets/models/pi05_base_pytorch_fp32
FOUR_SUITE_LIBERO_ASSETS=$LIBERO40_ROOT/assets/models/pi05_libero_pytorch/assets
FOUR_SUITE_WHOLE_SCENE_GEOMETRY_ROOT=$LIBERO40_ROOT/assets/annotation/policy_aux_v1/geometry_whole_scene_four_suite_v1
FOUR_SUITE_WHOLE_SCENE_MOTION_ROOT=$LIBERO40_ROOT/assets/annotation/policy_aux_v1/motion_whole_scene_four_suite_v1
```

Keep `hpc.env` untracked. All preparation and training jobs must receive its
absolute path through `FOUR_SUITE_HPC_ENV_FILE`.

## 4. Prepare and preflight

Preparation rebuilds small portable indices on the HPC filesystem and rewrites
legacy cache paths. Do not copy a prepared artifact directory from another
machine.

```bash
cd "$LIBERO40_ROOT/libero_task_related/experiments/four_suite_joint"
source hpc.env

./jobs/run_8gpu.sh prepare --target-scope task_relevant
./jobs/run_8gpu.sh preflight \
  --variant trqc --num-train-steps 30000 --warmup-steps 10000
./jobs/run_8gpu.sh preflight \
  --variant no_query_access --num-train-steps 30000 --warmup-steps 10000
./jobs/run_8gpu.sh prepare --target-scope whole_scene
./jobs/run_8gpu.sh preflight \
  --variant whole_scene --num-train-steps 30000 --warmup-steps 10000
```

All three preflights are read-only and report `optimizer_steps_executed: 0`. Then
verify the actual Slurm node, environment, and eight-GPU visibility:

```bash
sbatch --account=ACCOUNT --partition=PARTITION --time=00:30:00 --mem=64G \
  --export=ALL,FOUR_SUITE_HPC_ENV_FILE="$PWD/hpc.env" \
  jobs/slurm_8gpu.sbatch test
```

Use site-specific account, partition, wall-time, and memory values. The Slurm
launcher itself requests one node, eight GPUs, and 64 CPU cores.

## 5. Run the frozen experiments

Only set the approval variable for a reviewed formal optimizer job. The main
method is launched as follows:

```bash
sbatch --account=ACCOUNT --partition=PARTITION --time=TIME --mem=MEMORY \
  --export=ALL,FOUR_SUITE_HPC_ENV_FILE="$PWD/hpc.env",FOUR_SUITE_FULL_TRAINING_APPROVED=YES \
  jobs/slurm_8gpu.sbatch train \
    --variant trqc \
    --exp-name libero40_trqc_seed42 \
    --num-train-steps 30000 \
    --warmup-steps 10000 \
    --seed 42 \
    --batch-size 256 \
    --disable-wandb
```

Run the two ablations by changing only these arguments:

```text
--variant no_query_access --exp-name libero40_no_query_access_seed42
--variant whole_scene --exp-name libero40_whole_scene_seed42
```

Do not add `--overwrite` to an existing experiment. Use `--resume` only when
resuming the same configuration and checkpoint directory.

The Whole-scene source loader requires the explicit Whole-scene cache roots
and target-scope provenance. It never falls back to task-relevant targets.

The only authoritative instructions for this release branch are this README
and the checked-in `experiments/four_suite_joint/hpc.env.example` template.
