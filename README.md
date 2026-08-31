# RoboCasa Atomic-24 SGeM-VLA release

This branch packages the RoboCasa Atomic-24 main experiment and its matched
ablations with the exact shared OpenPI/PyTorch model core used by the LIBERO
experiments. Benchmark-specific code is isolated under
`experiments/robocasa24_atomic`; generated data, caches, checkpoints, logs, and
evaluation outputs are not stored in Git.

The LIBERO-40 release remains available on the separate
`libero40-task-relevant-query-conditioning-release` branch. RoboCasa cache
generation remains in the independent
[`robocasa-atomic24-cache-tools`](https://github.com/zhiyuan-gao/robocasa-atomic24-cache-tools)
repository.

## Included experiments

- Main SGeM-VLA: task-relevant Semantic, Geometry, and Motion supervision;
  Action reads the Geometry and Motion query tokens.
- Geometry only.
- Semantic + Geometry.
- Semantic + Motion.
- Supervision-only: identical auxiliary targets and losses, but Action cannot
  read the auxiliary query tokens.
- Whole-scene control: Semantic is unchanged; only Geometry and Motion use
  Whole-scene targets.

The `full` ablation topology remains available for smoke/compatibility checks,
but formal ablation training rejects it because the main-experiment result must
be reused instead of trained twice.

## Clone and install

```bash
git clone --recurse-submodules \
  --branch robocasa24-atomic-release \
  https://github.com/zhiyuan-gao/libero_task_related.git
cd libero_task_related

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv pip install --python .venv/bin/python --no-deps -e experiments/robocasa24_atomic

.venv/bin/python -m pytest \
  experiments/robocasa24_atomic/tests \
  experiments/robocasa24_atomic/ablations/tests
```

The release uses Python 3.11, the repository lockfile, the local Transformers
replacement, and the same FP32-converted pi0.5 base contract as LIBERO. Do not
substitute a BF16-converted base checkpoint.

## Inputs kept outside Git

The training run requires:

1. RoboCasa Atomic-24 `base50` HDF5 policy data: 24 tasks and exactly 50
   successful demonstrations per task;
2. the immutable source manifests plus Semantic/Geometry/Motion cache produced
   by the cache-tools repository;
3. the strict FP32-converted pi0.5 PyTorch base;
4. the generated per-task policy normalization statistics; and
5. a writable checkpoint directory.

Task-relevant and Whole-scene source caches are converted locally into portable
read-only memmaps by the preparation command. These generated `prepared/`
directories are ignored by Git.

See
[`experiments/robocasa24_atomic/README.md`](experiments/robocasa24_atomic/README.md)
for the frozen recipe, preparation, smoke, formal training, and multi-worker
closed-loop evaluation commands. See
[`experiments/robocasa24_atomic/ablations/README.md`](experiments/robocasa24_atomic/ablations/README.md)
for the five-run formal ablation queue.
