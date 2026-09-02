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

## Data availability

The exact 24-task `base50` HDF5 population is public on Hugging Face:

- [`Zhiyuan17/robocasa24-atomic-success100-256`](https://huggingface.co/datasets/Zhiyuan17/robocasa24-atomic-success100-256)
- pinned revision: `7236e704a04ebe477cc06d0a06ad540cd968fa5d`
- training needs only `data/base50/**` (24 HDF5 files, 1,200 episodes,
  332,859 frames, about 42.9 GiB).

The source manifests and complete task-relevant Semantic/Geometry/Motion cache
are also public:

- [`Zhiyuan17/robocasa24-cache-batch1-base50`](https://huggingface.co/datasets/Zhiyuan17/robocasa24-cache-batch1-base50)
- pinned revision: `d1028edd9094ec7f61e42d40babf74d971113948`
- about 2.61 GiB.

The strict FP32-converted pi0.5 PyTorch base is already uploaded under
`models/pi05_base_pytorch_fp32/` in the public dataset
`Zhiyuan17/libero40-trqc-assets`, pinned at
`84fd8b5849a976b08b36dc328141de88f483193a`. No Hugging Face authentication is
required; do not download its unrelated LIBERO assets.

The RoboCasa Whole-scene Geometry/Motion cache is **not currently published on
Hugging Face**. Generate it with the cache-tools repository or transfer the
validated cache before running the Whole-scene ablation. It reuses the public
task-relevant Semantic cache and source manifests.

Prepared memmaps and the exact equal-task-weight policy normalization are not
uploaded. Both are deterministic CPU products generated on the training
machine from the pinned inputs above. The detailed README provides selective
download and preparation commands. No teacher checkpoints, segmentation masks,
VGGT, or Track4World environment are needed once both source caches exist.

See
[`experiments/robocasa24_atomic/README.md`](experiments/robocasa24_atomic/README.md)
for the frozen recipe, preparation, smoke, formal training, and multi-worker
closed-loop evaluation commands. See
[`experiments/robocasa24_atomic/ablations/README.md`](experiments/robocasa24_atomic/ablations/README.md)
for the five-run formal ablation queue.
