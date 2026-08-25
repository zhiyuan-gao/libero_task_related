# Whole-scene Motion ablation

This is the Motion half of the matched Whole-scene Geometry + Motion ablation
for TRQC. Semantic supervision remains task-relevant.

## Frozen comparison

Both targets use exactly the same:

- 256,401 `motion_valid` samples from the 1,693-episode LIBERO population;
- agent-camera-only 11-frame clip `t:t+10`;
- Track4World checkpoint, model input preprocessing, and forward pass;
- final aggregator level (`level=3`) and source time index (`time=0`);
- 256-dimensional FP32 target.

The only change is pooling over the 45 x 45 source token grid:

```text
Task-related: sum(mask_coverage_i * token_i) / sum(mask_coverage_i)
Whole-scene:   mean(token_i), i=1..2025
```

The task mask is still decoded during generation solely to preserve and audit
the matched valid-sample set. It never weights the Whole-scene output. No wrist
Motion target is introduced.

Whole-scene outputs are written to an independent cache root and receive their
own train-only mean/std in `target_statistics_train.json`. The existing
task-related cache is read-only.

## Safety-gated commands

From `experiments/four_suite_joint`:

```bash
./jobs/generate_whole_scene_motion_8gpu.sh gate
WHOLE_SCENE_MOTION_SMOKE_GPU=1 \
  ./jobs/generate_whole_scene_motion_8gpu.sh smoke
```

The smoke uses two deterministic valid samples per suite (eight total) on one
GPU. For each sample, it computes both pooling variants from the same captured
teacher features and requires the recomputed task-related vector to be exactly
equal to the existing cache vector.

The full run is deliberately blocked unless explicitly approved, and also
refuses to start unless all eight GPUs have at most 1 GiB allocated:

```bash
export WHOLE_SCENE_MOTION_CACHE_APPROVED=YES
./jobs/generate_whole_scene_motion_8gpu.sh run
```

For an unattended launch that must wait for another workload, enable the
explicit idle retry. The default retry interval below is 30 minutes:

```bash
export WHOLE_SCENE_MOTION_CACHE_APPROVED=YES
export WHOLE_SCENE_MOTION_WAIT_FOR_IDLE=YES
export WHOLE_SCENE_MOTION_IDLE_POLL_SECONDS=1800
./jobs/generate_whole_scene_motion_8gpu.sh run
```

The full job is eight-way resumable. It finalizes only after all workers pass,
verifies exact sample-ID equality against the task-related cache, writes the
portable index/shards, and computes train-only normalization.

Default full output:

```text
/workspace/vla/p3/workspace/data/libero_four_suite_annotation/policy_aux_v1/
  motion_whole_scene_four_suite_v1/
```

No formal training is started by this launcher.
