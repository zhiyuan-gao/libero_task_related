# RoboCasa Atomic-24 pi0.5 fine-tuning

This is an isolated experiment package inside the shared OpenPI repository. It
imports the reviewed OpenPI/PyTorch model core without changing the LIBERO
experiment package or `robocasa24_cache_tools`.

## Frozen experiment contract

| Field | Value |
|---|---|
| Population | 24 non-navigation Atomic tasks × exactly 50 `base50` demos |
| Policy views | left agent, wrist, right agent |
| State/action | raw 16D state and 12D action; normalized, then padded to 32D |
| Action target | predict 50 steps; evaluation executes 25 steps, then replans |
| Sampling | task probability `N_task^0.4`; uniform episode; uniform timestep |
| Normalization | raw-frame z-score per task, then equal-weight task moment merge |
| Initialization | strictly FP32-converted `pi05_base` PyTorch safetensors |
| Training | 30,000 optimizer updates, global batch 128, BF16, AdamW |
| LR | peak `5e-5`, 10,000-update warmup, cosine to `5e-6` |
| Parallelism | eight-GPU DDP (`fsdp_devices=1`) |
| EMA | disabled |
| Checkpoints | save every 1,000; safely retain the latest 4 full resume states by default |

Matched variants:

- `baseline`: Action only;
- `task_relevant`: Action + task-relevant Geometry (`0.05`) + Semantic
  (`0.01`) + Motion (`0.05`);
- `whole_scene`: Action + whole-scene Geometry (`0.05`) + the identical
  Semantic target (`0.01`) + whole-scene Motion (`0.05`).

Whole-scene preparation enforces the same Geometry/Motion valid rows as the
task-relevant source manifest. It cannot silently gain extra supervision.

## Environment

From the repository root, install this experiment into the locked OpenPI
environment and export portable paths:

```bash
uv sync --frozen
uv pip install --python .venv/bin/python --no-deps -e experiments/robocasa24_atomic

export OPENPI_ROOT="$PWD"
export ROBOCASA24_ROOT="$OPENPI_ROOT/experiments/robocasa24_atomic"
export PYTHONPATH="$ROBOCASA24_ROOT/src:$OPENPI_ROOT/src"
export ROBOCASA24_PYTHON="$OPENPI_ROOT/.venv/bin/python"
cd "$ROBOCASA24_ROOT"
```

The environment must contain `h5py`, `numpy`, `pandas`, `pyarrow`, `einops`,
`safetensors`, `torch`, and the OpenPI checkout. For development only:

```bash
$ROBOCASA24_PYTHON -m pytest tests ablations/tests
```

All optimizer entrypoints automatically use the validated LIBERO performance
profile: the default CUDA allocator, disabled training-loop memory-stat
synchronization, and disabled tokenizer thread fan-out. These settings only
affect runtime allocation/logging behavior; model, losses, batches, precision,
optimizer, and sampling remain unchanged.

## Required data and exact sources

Use immutable revisions rather than moving repository heads.

| Required input | Source | Pinned revision | Required subset |
|---|---|---|---|
| Atomic-24 policy data | [`Zhiyuan17/robocasa24-atomic-success100-256`](https://huggingface.co/datasets/Zhiyuan17/robocasa24-atomic-success100-256) | `7236e704a04ebe477cc06d0a06ad540cd968fa5d` | `data/base50/**` |
| Source manifests + task-relevant Semantic/Geometry/Motion | [`Zhiyuan17/robocasa24-cache-batch1-base50`](https://huggingface.co/datasets/Zhiyuan17/robocasa24-cache-batch1-base50) | `d1028edd9094ec7f61e42d40babf74d971113948` | complete repository |
| Strict FP32 pi0.5 base | private dataset `Zhiyuan17/libero40-trqc-assets` | `84fd8b5849a976b08b36dc328141de88f483193a` | `models/pi05_base_pytorch_fp32/**` |
| Whole-scene Geometry/Motion | not yet published on HF | n/a | validated Batch-1 Whole-scene cache |

The HDF5 subset contains exactly 24 task files, 1,200 episodes, and 332,859
frames (about 42.9 GiB). The task-relevant cache is about 2.61 GiB. The FP32
base is about 13.5 GiB. Do not use the `success100` population, a different
Human-50 selection, or a BF16-converted base.

Create a machine-local asset layout and download only the required files:

```bash
export ROBOCASA24_ASSET_ROOT=/large_disk/robocasa24_assets
export RAW_DATA_ROOT="$ROBOCASA24_ASSET_ROOT/raw"
export TASK_CACHE_ROOT="$ROBOCASA24_ASSET_ROOT/cache/task_relevant"
export WHOLE_CACHE_ROOT="$ROBOCASA24_ASSET_ROOT/cache/whole_scene"
export MODEL_BUNDLE_ROOT="$ROBOCASA24_ASSET_ROOT/model_bundle"
export POLICY_ASSETS_ROOT="$ROBOCASA24_ASSET_ROOT/policy_assets"
export PREPARED_ROOT="$ROBOCASA24_ASSET_ROOT/prepared"
export CHECKPOINT_ROOT="$ROBOCASA24_ASSET_ROOT/checkpoints"

mkdir -p "$RAW_DATA_ROOT" "$TASK_CACHE_ROOT" "$WHOLE_CACHE_ROOT" \
  "$MODEL_BUNDLE_ROOT" "$POLICY_ASSETS_ROOT" "$PREPARED_ROOT" \
  "$CHECKPOINT_ROOT"

uvx --from 'huggingface_hub>=1.0' hf download \
  Zhiyuan17/robocasa24-atomic-success100-256 \
  --repo-type dataset \
  --revision 7236e704a04ebe477cc06d0a06ad540cd968fa5d \
  --include 'data/base50/**' \
  --local-dir "$RAW_DATA_ROOT"

uvx --from 'huggingface_hub>=1.0' hf download \
  Zhiyuan17/robocasa24-cache-batch1-base50 \
  --repo-type dataset \
  --revision d1028edd9094ec7f61e42d40babf74d971113948 \
  --local-dir "$TASK_CACHE_ROOT"

uvx --from 'huggingface_hub>=1.0' hf auth login
uvx --from 'huggingface_hub>=1.0' hf download \
  Zhiyuan17/libero40-trqc-assets \
  --repo-type dataset \
  --revision 84fd8b5849a976b08b36dc328141de88f483193a \
  --include 'models/pi05_base_pytorch_fp32/**' \
  --local-dir "$MODEL_BUNDLE_ROOT"
```

Then define the paths consumed by every training and validation command:

```bash
export DATA_ROOT="$RAW_DATA_ROOT/data/base50"
export MANIFEST_ROOT="$TASK_CACHE_ROOT"
export SEMANTIC_ROOT="$TASK_CACHE_ROOT"
export GEOMETRY_ROOT="$TASK_CACHE_ROOT"
export MOTION_ROOT="$TASK_CACHE_ROOT"
export PI05_FP32_BASE="$MODEL_BUNDLE_ROOT/models/pi05_base_pytorch_fp32"
```

The Whole-scene cache is the only large research input not currently available
from HF. Until a pinned repository/revision is published, place its 24 task
directories under `$WHOLE_CACHE_ROOT`. It must contain validated
`geometry_whole_scene/final` and `motion_whole_scene/final` outputs aligned to
the same valid rows as the task-relevant cache. Generate it using
[`robocasa-atomic24-cache-tools`](https://github.com/zhiyuan-gao/robocasa-atomic24-cache-tools)
and its `WHOLE_SCENE_CACHE.md`; do not synthesize or substitute targets on the
training machine.

The following are deliberately not downloaded for training once the two cache
roots exist: raw segmentation masks, teacher model checkpoints, VGGT,
Track4World, worker shards, cache-generation logs, and `additional50` data.

## One-time CPU preparation

`DATA_ROOT` may be either the dataset root containing
`data/base50/<task>/*.hdf5` or the `base50` directory itself. Cache roots may be
different directories; stale absolute paths inside cache indices are safely
relocated under the supplied root.

Compute the exact policy normalization statistics. This follows the official
RoboCasa OpenPI convention: calculate raw-frame state/action moments inside
each task and merge tasks with equal weight. Action chunks still repeat the
last full 12D action at an episode tail during training, but padded chunk
tokens are deliberately not counted again in normalization:

```bash
$ROBOCASA24_PYTHON -m robocasa24_finetune.compute_norm_stats \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --output-root "$POLICY_ASSETS_ROOT"
```

Consolidate the task-relevant target cache into read-only memory maps:

```bash
$ROBOCASA24_PYTHON -m robocasa24_finetune.prepare_artifacts \
  --scope task_relevant \
  --manifest-root "$MANIFEST_ROOT" \
  --semantic-root "$SEMANTIC_ROOT" \
  --geometry-root "$GEOMETRY_ROOT" \
  --motion-root "$MOTION_ROOT" \
  --output-dir "$PREPARED_ROOT/task_relevant"
```

Prepare the Whole-scene control while reusing the identical source manifests
and Semantic labels:

```bash
$ROBOCASA24_PYTHON -m robocasa24_finetune.prepare_artifacts \
  --scope whole_scene \
  --manifest-root "$MANIFEST_ROOT" \
  --semantic-root "$SEMANTIC_ROOT" \
  --geometry-root "$WHOLE_CACHE_ROOT" \
  --motion-root "$WHOLE_CACHE_ROOT" \
  --output-dir "$PREPARED_ROOT/whole_scene"
```

Source caches are never rewritten and an existing output directory is never
overwritten. The generated prepared directories are portable and together use
about 6.2 GB. They are intentionally not stored in Git or HF because they can
be reconstructed deterministically from the smaller source caches.

Run the read-only population, HDF5, norm-stat, and target-alignment audit:

```bash
$ROBOCASA24_PYTHON -m robocasa24_finetune.validate \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant"
```

Repeat the validator with `--artifact-dir "$PREPARED_ROOT/whole_scene"` before
the Whole-scene smoke. The expected policy population is 24 tasks, 1,200
episodes, and 332,859 frames.

## Dry run, smoke, and formal training

Configuration construction is CPU-only:

```bash
jobs/run_8gpu.sh dry-run \
  --variant task_relevant --exp-name rc24_main \
  --data-root "$DATA_ROOT" --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant" \
  --base-weight-dir "$PI05_FP32_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_ROOT"
```

The real optimizer smoke and formal run are separately gated. Neither can be
started accidentally:

```bash
ROBOCASA24_SMOKE_APPROVED=YES jobs/run_8gpu.sh smoke --updates 2 ...
ROBOCASA24_FULL_TRAINING_APPROVED=YES jobs/run_8gpu.sh train ...
```

Do not set either approval variable until all eight GPUs are free and the
read-only validator has returned `PASS`.

Each full checkpoint includes both model and AdamW state (about 21 GB with the
current model), so the default rolling retention is deliberate. Pass
`--max-checkpoints-to-keep N` only after checking disk space and deciding the
evaluation grid.

## Multi-worker closed-loop evaluation

The frozen Atomic-24 protocol predicts 50 actions, executes the first 25, and
then replans. It evaluates 50 seeded episodes for every task with the three
training views. Evaluation uses object split B, fixed cameras, and the five
reference layout/style pairs `(1,1), (2,2), (4,4), (6,9), (7,10)`.
Native RoboSuite camera rows receive the same vertical-axis flip recorded in
the training HDF5 provenance; no horizontal flip or 180-degree rotation is used.

Policy inference and simulation use separate environments. The default launcher
starts eight policy servers (one checkpoint copy per A100) and eight RoboCasa
simulator workers. Every worker owns complete tasks rather than arbitrary
episodes, preserving each task's seeded reset stream. `NUM_WORKERS` is
configurable up to 24. For reduced task populations, `SHARD_MODE=episode`
divides the workers into near-equal groups per task and shards that task's
episodes across the group. Every shard still performs all resets in canonical
order, so parallelism does not change the seeded episode population.

Prepare the pinned simulator once. Asset download is about 5 GB:

```bash
DOWNLOAD_ASSETS=1 jobs/setup_eval_runtime.sh
```

Run a non-mutating launcher preflight:

```bash
CHECKPOINT=/absolute/checkpoint/30000 \
RUN_ROOT=/absolute/eval/output \
DRY_RUN=1 \
jobs/eval_checkpoint_multiworker.sh
```

The real command is identical without `DRY_RUN=1`. Formal evaluation always
means 24 × 50 = 1,200 rollouts. Each completed rollout is durably appended to a
worker JSONL file. If interrupted, rerun the exact command with `RESUME=1`; the
launcher checks the immutable manifest and fills only missing rollouts. The final
`summary.json` contains overall, category, and every-task success rates. Videos
are disabled by default to avoid I/O overhead; set `SAVE_VIDEO=1` only for a
separate qualitative run.

The predicted chunk stays frozen at 50. Reduced-task diagnostics may override
how many actions are executed before replanning with `EXECUTION_HORIZON=N`.
The launcher records this value in the immutable manifest and passes it to both
workers and the result validator. Full Atomic-24 formal mode still requires the
frozen execution horizon of 25.
