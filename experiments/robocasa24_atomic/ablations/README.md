# RoboCasa Atomic-24 ablation code

This directory documents the isolated implementation under
`src/robocasa24_finetune/ablations/`.  It does not modify the reviewed main
training entrypoint or the existing task-relevant / Whole-scene cache
generators.

## Matched variants

| CLI name | Auxiliary losses | Action Expert reads | Target scope |
|---|---|---|---|
| `geometry_only` | Geometry `0.05` | `Zg` | task-relevant |
| `semantic_geometry` | Semantic `0.01` + Geometry `0.05` | `Zg` | task-relevant |
| `semantic_motion` | Semantic `0.01` + Motion `0.05` | `Zm` | task-relevant |
| `full` | Semantic `0.01` + Geometry `0.05` + Motion `0.05` | `Zg, Zm` | task-relevant (main-result reference only) |
| `supervision_only` | same as Full | native image/language context only | task-relevant |
| `whole_scene` | same as Full | `Zg, Zm` | Whole-scene Geometry/Motion |

Semantic teacher tokens remain hidden from Action in every Semantic variant.
`semantic_motion` removes Geometry queries and the Geometry head before loading
the official base.  It does not emulate the ablation by setting a zero loss.
`supervision_only` retains both query branches and their losses, but removes
Action-to-query attention edges in both optimizer forwards and checkpoint
inference.  Its Action RoPE positions use the native context length.

Everything not listed in the table comes from the same reviewed main config:
Atomic-24 Human-50 population, three cameras, task-alpha sampling, z-score
normalization, FP32-converted pi0.5 initialization, 50-step action prediction,
global batch 128, 30k optimizer updates, 10k warmup, BF16 AdamW, no EMA, and
8-GPU DDP.

The formal ablation queue contains only:

```text
geometry_only
semantic_geometry
semantic_motion
supervision_only
whole_scene
```

`full` remains available for topology compatibility and smoke checks, but its
formal result is taken from the main experiment and it must not be trained a
second time as an ablation.

## Artifact selection

Use `prepared/task_relevant` for `geometry_only`, `semantic_geometry`,
`semantic_motion`, `supervision_only`, and the `full` reference/smoke config.
Use `prepared/whole_scene` only for `whole_scene`. Whole-scene Semantic reuses
the same Semantic labels; only Geometry and Motion teacher pooling changes.
Preparation must already have enforced exactly matched valid rows.

This matched design intentionally keeps Semantic in `whole_scene`: the formal
comparison with Full must change only the Geometry/Motion target scope. A lone
Whole-scene-without-Semantic run would change two factors at once and is not a
valid isolated scope ablation. Testing a Semantic-by-scope interaction would
instead require a paired task-relevant-without-Semantic and
Whole-scene-without-Semantic experiment; neither is in the current formal run
queue.

## Written entrypoints

Configuration-only construction remains optimizer-free:

```bash
jobs/ablations/run_8gpu.sh dry-run \
  --ablation semantic_motion \
  --exp-name rc24_ablation_semantic_motion \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant" \
  --base-weight-dir "$PI05_FP32_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_ROOT"
```

The dedicated optimizer smoke and formal launcher are separately gated:

```bash
ROBOCASA24_ABLATION_SMOKE_APPROVED=YES \
  jobs/ablations/run_8gpu.sh smoke --updates 2 ...

ROBOCASA24_ABLATION_TRAINING_APPROVED=YES \
  jobs/ablations/run_8gpu.sh train ...
```

An ablation checkpoint must be served through the matching topology loader.
The single-server entry is useful for debugging:

```bash
jobs/ablations/serve_checkpoint.sh \
  --checkpoint "$CHECKPOINT" --port 8600
```

Formal evaluation reuses the existing simulator workers and summarizer while
routing only policy-server construction to the ablation loader:

```bash
CHECKPOINT="$CHECKPOINT" RUN_ROOT="$RUN_ROOT" \
NUM_GPUS=8 NUM_WORKERS=24 SHARD_MODE=episode \
  jobs/ablations/eval_checkpoint_multiworker.sh
```

This wrapper does not copy or alter the common evaluation protocol.  The
checkpoint metadata selects the exact ablation topology.

## Current verification status

Validated on 2026-08-31:

- all common and ablation CPU tests passed (`20 passed`);
- all six configurations passed construction/dry-run checks with the frozen
  30k recipe;
- `geometry_only`, `semantic_geometry`, `semantic_motion`, `full`, and
  `supervision_only` each completed one real 8-GPU optimizer update on the
  formal Batch-1 task-relevant population with finite losses and gradients;
- the official FP32 base load reported exactly the auxiliary query/head keys
  expected by each topology;
- prepared-artifact scope is now a hard configuration gate: task-relevant and
  Whole-scene artifacts cannot be interchanged silently;
- `whole_scene` completed a one-update *plumbing-only* smoke using a temporary
  artifact explicitly marked as a non-scientific proxy. The proxy was deleted
  after the run. A real Whole-scene optimizer smoke still requires the full
  Batch-1 Whole-scene cache and `prepared/whole_scene`.

No formal training checkpoint or closed-loop evaluation was produced. Smoke
checkpoint directories and generated test artifacts were removed after the
checks.

Five-update timing smokes used the formal eight-GPU global batch of 128 and
logged every update. The first optimizer update includes DDP/static-graph
initialization and is not a steady-state measurement. Mean time over updates
2--5 was:

| Ablation | First update | Mean updates 2--5 |
|---|---:|---:|
| `geometry_only` | 16.8 s | 2.71 s/update |
| `semantic_geometry` | 18.7 s | 2.87 s/update |
| `semantic_motion` | 18.4 s | 2.87 s/update |
| `full` | 17.8 s | 2.91 s/update |
| `supervision_only` | 16.9 s | 2.91 s/update |
| `whole_scene` diagnostic proxy | 18.0 s | 2.91 s/update |

All 30 timing updates completed with finite losses and gradients and without
OOM or data-loader stalls. `whole_scene` remains a plumbing-only timing result
until the real Batch-1 Whole-scene prepared artifact exists.
