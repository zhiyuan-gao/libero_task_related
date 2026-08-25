# LIBERO-40 Task-Relevant Query Conditioning release

This directory contains only the code and protocol for joint training on the
official 40-task LIBERO population. It does not contain earlier experiment
reports, checkpoints, evaluation results, diagnostic scripts, or generated
teacher targets.

## Method

The main method is Task-Relevant Query Conditioning (TRQC):

```text
Action + 0.01 Semantic + 0.05 Geometry + 0.05 Motion
```

- Geometry and Motion each use eight independent learned queries.
- The Action Expert reads both query groups in the `trqc` and `whole_scene`
  variants.
- `no_query_access` keeps the same queries, heads, losses, parameters, data,
  and gradients, but blocks only Action-to-Geometry/Motion attention.
- Ground queries and Ground loss are absent. This is not P3.
- Training initializes from the strict FP32-converted pi0.5 base, runs in BF16,
  and does not use EMA.

The frozen experiment family has three variants:

- `trqc`: task-relevant Semantic, Geometry, and Motion; Action reads the
  Geometry/Motion queries.
- `whole_scene`: identical Semantic and query access, but both Geometry and
  Motion use independently normalized Whole-scene targets.
- `no_query_access`: identical task-relevant targets and losses to `trqc`, but
  Action-to-Geometry/Motion attention is blocked.

The matched Whole-scene teacher recipes are documented in
[`WHOLE_SCENE_MOTION.md`](WHOLE_SCENE_MOTION.md) and
[`WHOLE_SCENE_GEOMETRY.md`](WHOLE_SCENE_GEOMETRY.md).

## Frozen population

- Hugging Face dataset: `physical-intelligence/libero`
- Revision: `a4336d589d589045d1c56423ffdf3b88a0e19b1f`
- 1,693 episodes, 273,465 frames, 40 tasks
- LIBERO-10: 379 episodes / 101,469 frames
- LIBERO-Goal: 428 episodes / 52,042 frames
- LIBERO-Object: 454 episodes / 66,984 frames
- LIBERO-Spatial: 432 episodes / 52,970 frames

The LeRobot parquet files may be downloaded directly on the HPC system. They do
not require teacher preprocessing. Geometry, Motion, Semantic metadata, the
FP32-converted base, and LIBERO normalization assets are generated or frozen
assets that must be transferred separately with checksums.

## Required external assets

Set paths explicitly on the training machine:

```bash
export FOUR_SUITE_LEROBOT_ROOT=/path/to/hf_snapshot/a4336d589d589045d1c56423ffdf3b88a0e19b1f
export FOUR_SUITE_ANNOTATION_ROOT=/path/to/libero_four_suite_annotation
export FOUR_SUITE_JOINT_MANIFEST=/path/to/four_suite_policy_geometry_manifest.parquet
export FOUR_SUITE_BASE_WEIGHTS=/path/to/pi05_base_pytorch_fp32
export FOUR_SUITE_LIBERO_ASSETS=/path/to/pi05_libero_pytorch/assets
export FOUR_SUITE_CHECKPOINT_BASE_DIR=/path/to/checkpoints/libero40_joint
```

Expected auxiliary populations:

- Geometry: 273,377 valid targets; 88 frames are masked.
- Motion: 256,401 valid targets; missing future-horizon frames are masked.
- Semantic: a non-empty target for all 273,465 frames.

## Environment and tests

For a machine that has never run this repository, follow the complete
[`HPC_SETUP_SLURM.md`](HPC_SETUP_SLURM.md) runbook. It covers the frozen Python
environment, exact dataset download, local-to-HPC asset transfer, integrity
checks, path rebasing, storage, and Slurm submission.

From the repository root, create the normal OpenPI environment. The launcher
defaults to that repository and its `.venv`; both may be overridden:

```bash
cd experiments/four_suite_joint
export FOUR_SUITE_PYTHON=/path/to/python
export FOUR_SUITE_TORCHRUN=/path/to/torchrun
./jobs/run_8gpu.sh test
```

Tests do not initialize a model or perform optimizer updates.

The complete LeRobot snapshot can be downloaded and validated directly on the
HPC with:

```bash
./jobs/run_8gpu.sh download-data --cache-dir /persistent/path/to/hf/hub
```

## Prepare portable metadata

Generated indices contain absolute target-store paths. Run `prepare` once per
target scope on the HPC filesystem after transferring that auxiliary bundle:

```bash
./jobs/run_8gpu.sh prepare --target-scope task_relevant
./jobs/run_8gpu.sh preflight --variant trqc
./jobs/run_8gpu.sh preflight --variant no_query_access
```

`prepare` performs CPU-only metadata joins and pooled train normalization. It
does not run VGGT, Track4World, or any other teacher. Static preflight validates
artifacts and configuration but performs zero optimizer updates. A real
forward/backward and short multi-GPU preflight is still required before formal
training.

Only after both Whole-scene caches have been generated or transferred:

```bash
./jobs/run_8gpu.sh prepare --target-scope whole_scene
./jobs/run_8gpu.sh preflight --variant whole_scene
```

There is no fallback between target scopes: missing Whole-scene assets stop the
command instead of silently using task-relevant targets.

## Formal training gate

Formal training requires explicit step and warmup budgets plus an approval
environment variable. The matched-epoch candidate is 30,000 updates with
10,000 warmup updates, but it must be approved before launch.

```bash
export FOUR_SUITE_FULL_TRAINING_APPROVED=YES
./jobs/run_8gpu.sh train \
  --variant trqc \
  --exp-name libero40_trqc_seed42 \
  --num-train-steps 30000 \
  --warmup-steps 10000
```

Run `whole_scene` and `no_query_access` with the identical seed, update count,
warmup, batch, optimizer, and checkpoint policy. `whole_scene` intentionally
uses its own Geometry/Motion artifact bundle; the other two variants share the
task-relevant bundle. The current release retains the latest three full resume
checkpoints at a 1,000-update cadence; revise this only after storage and
checkpoint-selection policy are frozen.

## Deliberately out of scope

- P1, P2, P3, Binary Ground, and Geometry-conditioned Motion experiments
- Prior training/evaluation reports and numerical results
- Diagnostic and tiny-overfit scripts
- Generated caches, checkpoints, logs, and videos
