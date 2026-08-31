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

Use the same command with `--scope whole_scene` and the Whole-scene Geometry /
Motion roots for the ablation. Source caches are never rewritten and an
existing output directory is never overwritten.

Run the read-only population, HDF5, norm-stat, and target-alignment audit:

```bash
$ROBOCASA24_PYTHON -m robocasa24_finetune.validate \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant"
```

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
