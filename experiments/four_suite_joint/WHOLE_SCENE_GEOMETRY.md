# Matched Whole-scene Geometry target

This cache supplies the Geometry half of the `whole_scene` ablation. The
ablation keeps task-relevant Semantic supervision and the same query topology,
losses, coefficients, training population, and Action access as TRQC. Only the
spatial scope of the Geometry and Motion teacher targets changes.

## Frozen matched-control definition

For Geometry, the implementation reuses the exact frozen `facebook/VGGT-1B`
checkpoint, repository revision, RGB preprocessing, final aggregator layer,
agent/wrist inputs, and view-validity semantics used by the task-relevant
cache. A frame remains Geometry-valid only when the existing task annotation
marks at least one view valid, so the whole-scene and task-relevant variants
have the same 273,377 supervised frames and the same 88 masked frames.

For each matched-valid view, let

```text
H[v, i] in R^2048, i = 1..1369
```

be the 37 x 37 final-layer spatial patch tokens from the same VGGT forward.
The Whole-scene view target is

```text
G_view[v] = mean_i H[v, i].
```

Valid agent/wrist view targets are then averaged with equal view weight. The
task-relevant target instead uses task-mask coverage to weight those same
patches. Therefore the only intended teacher change is patch pooling.

The finalized cache contains:

- `targets_valid_fp32.npy`, shape `[273377, 2048]`;
- `target_index.parquet`, covering all 273,465 policy frames and explicitly
  masking the 88 invalid frames;
- `normalization/train_mean_std.json`, computed only from the whole-scene
  training targets;
- `target_statistics_train.json` and `cache_validation.json`.

## Gates and eventual generation

The code is ready, but this document does not authorize cache generation.
Run the CPU-only population/provenance gate at any time:

```bash
cd experiments/four_suite_joint
./jobs/generate_whole_scene_geometry_8gpu.sh gate
```

When one GPU is idle, the eight-sample smoke verifies that a same-forward
reconstruction of the task-relevant target matches the existing cache (absolute
tolerance `1e-5`) and that Whole-scene pooling actually changes the target:

```bash
WHOLE_SCENE_GEOMETRY_SMOKE_GPU=0 \
  ./jobs/generate_whole_scene_geometry_8gpu.sh smoke
```

The resumable full run is deliberately approval-gated and refuses to start
unless all eight GPUs use at most 1 GiB each:

```bash
WHOLE_SCENE_GEOMETRY_CACHE_APPROVED=YES \
  ./jobs/generate_whole_scene_geometry_8gpu.sh run
```

Eight workers are assigned disjoint samples by global-index modulus. Each
worker writes atomic shards, validates a shard before resuming it, and leaves
completed shards intact after a failure. The finalizer runs only after all
workers succeed and refuses population, shape, identity, dtype, finite-value,
or frozen-teacher provenance drift.

The `whole_scene` training variant remains unavailable until both this cache
and the separately documented Whole-scene Motion cache have passed their final
gates and `prepare --target-scope whole_scene` has built the portable metadata
bundle.
