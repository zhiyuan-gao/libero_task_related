# LIBERO-40 dual continuation runbook

This runbook covers two sequential, approval-gated continuations.  It does not
launch closed-loop evaluation.

## Frozen sequence

1. `1932_from_main`: main TRQC step 30,000 -> 1,932 episodes for one continuous 6,000-update run.
2. `old115_exact_continue`: exact-resume the existing 115-addition step 3,000 on the same old 1,808-episode population and continue to step 6,000.

Every stage uses 8 GPUs, global batch 256, BF16, seed 42, no EMA, AdamW,
gradient clipping at 1.0, and the frozen Semantic/Geometry/Motion objectives.
The 1,932 run creates AdamW once and uses one 6,000-update schedule (200-update
warmup, peak LR `1e-5`, cosine decay to `1e-6`).  The old-115 continuation
restores its existing model, AdamW, RNG, and DataLoader state exactly.  Its
original 3,000-update LR schedule has already reached `1e-6`, so the additional
updates remain at that terminal LR rather than restarting warmup.

Checkpoints are saved every 500 updates.  All model weights remain evaluable;
only the newest two checkpoints in each experiment keep optimizer and exact
DataLoader/RNG state.  During old-115 continuation, step 3,000 remains an
evaluation model after its resume payload is naturally superseded.

## Safety behavior

- Formal execution requires `LIBERO_DUAL_CONTINUATION_APPROVED=YES`.
- A file lock prevents duplicate controllers.
- Every stage checks that all GPUs are idle and at least 100 GiB is free.
- A failed stage stops the chain; the next stage is never started.
- A partial stage resumes only from its own exact checkpoint.
- An existing non-resumable partial directory is never overwritten automatically.
- A completed stage is skipped only when its final model exists.
- Formal logs are separate for both stages.

The public adopted step-3,000 checkpoint is intentionally evaluation-only.  An
exact second-stage resume additionally requires that checkpoint's original
`optimizer.pt`, `training_state.pt`, and `metadata.pt`; the controller checks
all three before touching a GPU.  If those private resume payloads are not
available, reproduce the adopted result as a fresh 3,000-update warm start from
the public main step 30,000 model instead of claiming an exact step-3,000 to
step-6,000 continuation.

## Portable path configuration

The controller has no machine-specific absolute paths.  By default it uses
`experiments/four_suite_joint/runtime`; every input may instead be supplied by
environment variable:

```text
FOUR_SUITE_CHECKPOINT_BASE_DIR  parent and output checkpoint tree
FOUR_SUITE_MAIN_PARENT          complete main step-30,000 checkpoint
FOUR_SUITE_OLD115_PARENT        complete old-115 step-3,000 resume checkpoint
FOUR_SUITE_COMPLETED_ROOT       prepared 1,932-population root
FOUR_SUITE_OLD115_ROOT          assembled exact 1,808-population root
FOUR_SUITE_LIBERO_ASSETS        official LIBERO normalization asset directory
FOUR_SUITE_CONTINUATION_LOG_DIR controller and per-stage logs
```

`FOUR_SUITE_OLD115_ROOT` must be the output of the checked-in
`assemble-supplemental115` command.  Its two relevant subdirectories are
resolved automatically.

## Commands

Inspect the resolved plan without training:

```bash
experiments/four_suite_joint/jobs/run_dual_continuation_8gpu.sh plan
```

Run a fresh 1,932 2-update smoke followed by a no-checkpoint exact-resume smoke
from the old-115 step 3,000:

```bash
experiments/four_suite_joint/jobs/run_dual_continuation_8gpu.sh smoke
```

Only after explicit researcher confirmation, launch the formal chain in a
detached background session:

```bash
LIBERO_DUAL_CONTINUATION_APPROVED=YES \
  experiments/four_suite_joint/jobs/launch_dual_continuation_background.sh
```
