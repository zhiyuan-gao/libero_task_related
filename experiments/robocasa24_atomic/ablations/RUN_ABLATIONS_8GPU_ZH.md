# 在一台新的 8-GPU 机器上运行 RoboCasa Atomic-24 消融实验

本文给出从空机器到正式训练、断点续训和评估前检查的完整命令。所有命令都从
`robocasa24-atomic-release` 分支运行，禁止使用 `main`，也不要从旧工作区复制临时代码。
本文只改变消融拓扑和既定辅助监督组合，不改变冻结的模型主体、loss 系数、数据定义或
30k 训练 recipe。`full` 是主实验结果的引用，**不能作为消融重复训练**。

## 1. 运行范围和前置条件

| 消融 | prepared 目录 | 当前能否运行 |
|---|---|---|
| `geometry_only` | `prepared/task_relevant` | 可以 |
| `semantic_geometry` | `prepared/task_relevant` | 可以 |
| `semantic_motion` | `prepared/task_relevant` | 可以 |
| `supervision_only` | `prepared/task_relevant` | 可以 |
| `whole_scene` | `prepared/whole_scene` | 只有真实 Whole-scene cache 验证通过后才可以 |

不要训练 `full`。不要用 task-relevant cache 冒充 Whole-scene cache，也不要用 plumbing
smoke 的临时 proxy 跑正式实验。

每个正式实验使用全部 8 张 GPU，建议逐个运行。每个最终实验保留 6 个完整 checkpoint，
按每个约 21 GB 估算约占 126 GB；5 个消融约占 630 GB。加上原始数据、base、prepared、
日志和评估输出，建议训练盘至少预留 1 TB 可用空间。

## 2. 检查新机器并确定绝对路径

```bash
hostname
nvidia-smi -L
nvidia-smi
lscpu
free -h
df -hT
nvcc --version || true
command -v sinfo >/dev/null && sinfo || echo "未检测到 Slurm"

# 开始 smoke 或训练前应无其他 GPU 进程。
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

确认系统能看到计划用于训练的 8 张 CUDA GPU 即可。A100 80GB 是已经验证过的参考配置，
不是硬性型号要求；H100、H800 等显存足够并支持 BF16 的卡也可以使用，最终以 8-GPU
optimizer smoke 是否正常为准。不同型号的 update 时间和显存占用会有所不同。

下文采用以下绝对路径；若训练盘不在 `/workspace`，只修改 `WORK_ROOT`：

```bash
export WORK_ROOT=/workspace/vla
export REPO_ROOT="$WORK_ROOT/repos/libero_task_related"
export ASSET_ROOT="$WORK_ROOT/assets/robocasa24_atomic"
export LOG_ROOT="$WORK_ROOT/logs/robocasa24_atomic/ablations"
mkdir -p "$WORK_ROOT/repos" "$ASSET_ROOT" "$LOG_ROOT"
```

## 3. 获取经过审核的代码

`804663528a10cf0d473cdaf3a1de1dda764148c6` 是 Atomic-24 release 的固定基线。消融
checkpoint 新保留策略是在该基线上新增的代码，因此新机器应 checkout
`robocasa24-atomic-release` 上包含本手册的最新审核 commit，而不是退回旧基线。

```bash
cd "$WORK_ROOT/repos"
git clone --recurse-submodules \
  --branch robocasa24-atomic-release \
  https://github.com/zhiyuan-gao/libero_task_related.git
cd "$REPO_ROOT"
git submodule update --init --recursive

git merge-base --is-ancestor \
  804663528a10cf0d473cdaf3a1de1dda764148c6 HEAD

test -f experiments/robocasa24_atomic/ablations/RUN_ABLATIONS_8GPU_ZH.md
grep -q 'ABLATION_SAVE_INTERVAL = 1_000' \
  experiments/robocasa24_atomic/src/robocasa24_finetune/ablations/configs.py
grep -q 'ABLATION_KEEP_PERIOD = 5_000' \
  experiments/robocasa24_atomic/src/robocasa24_finetune/ablations/configs.py

git rev-parse HEAD | tee "$LOG_ROOT/code_commit.txt"
git status --short
```

`git status --short` 应为空。全部消融使用同一个已记录 commit，不要在队列中途更新代码。

## 4. 按 lockfile 安装环境

```bash
cd "$REPO_ROOT"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv pip install --python .venv/bin/python --no-deps \
  -e experiments/robocasa24_atomic

export OPENPI_ROOT="$REPO_ROOT"
export ROBOCASA24_ROOT="$REPO_ROOT/experiments/robocasa24_atomic"
export PYTHONPATH="$ROBOCASA24_ROOT/src:$OPENPI_ROOT/src"
export ROBOCASA24_PYTHON="$OPENPI_ROOT/.venv/bin/python"
```

不要自行升级或替换 PyTorch、Transformers、CUDA Python 包或其他 lockfile 依赖。

## 5. 下载固定数据、task-relevant cache 和 FP32 base

```bash
export RAW_DATA_ROOT="$ASSET_ROOT/raw"
export BATCH1_CACHE_ROOT="$ASSET_ROOT/cache/batch1_base50"
export TASK_CACHE_ROOT="$BATCH1_CACHE_ROOT"
export MODEL_BUNDLE_ROOT="$ASSET_ROOT/model_bundle"
export POLICY_ASSETS_ROOT="$ASSET_ROOT/policy_assets"
export PREPARED_ROOT="$ASSET_ROOT/prepared"
export CHECKPOINT_ROOT="$ASSET_ROOT/checkpoints"

mkdir -p "$RAW_DATA_ROOT" "$BATCH1_CACHE_ROOT" \
  "$MODEL_BUNDLE_ROOT" "$POLICY_ASSETS_ROOT" "$PREPARED_ROOT" \
  "$CHECKPOINT_ROOT"

cd "$REPO_ROOT"
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

uvx --from 'huggingface_hub>=1.0' hf download \
  Zhiyuan17/libero40-trqc-assets \
  --repo-type dataset \
  --revision 84fd8b5849a976b08b36dc328141de88f483193a \
  --include 'models/pi05_base_pytorch_fp32/**' \
  --local-dir "$MODEL_BUNDLE_ROOT"
```

以上三个 Hugging Face dataset 当前均为 public，不需要登录。base 只能使用
`models/pi05_base_pytorch_fp32/`，禁止替换成 BF16 base 或其他 checkpoint。

```bash
export DATA_ROOT="$RAW_DATA_ROOT/data/base50"
export MANIFEST_ROOT="$TASK_CACHE_ROOT"
export SEMANTIC_ROOT="$TASK_CACHE_ROOT"
export GEOMETRY_ROOT="$TASK_CACHE_ROOT"
export MOTION_ROOT="$TASK_CACHE_ROOT"
export PI05_FP32_BASE="$MODEL_BUNDLE_ROOT/models/pi05_base_pytorch_fp32"

find "$DATA_ROOT" -type f -name '*.hdf5' | wc -l
du -sh "$RAW_DATA_ROOT" "$TASK_CACHE_ROOT" "$PI05_FP32_BASE"
test -s "$PI05_FP32_BASE/model.safetensors"
```

HDF5 文件应为 24 个。完整 population 必须是 24 tasks、1,200 episodes、332,859 frames，
最终以 validator 结果为准。

## 6. 生成 normalization 和 task-relevant prepared memmap

```bash
cd "$ROBOCASA24_ROOT"

$ROBOCASA24_PYTHON -m robocasa24_finetune.compute_norm_stats \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --output-root "$POLICY_ASSETS_ROOT"

$ROBOCASA24_PYTHON -m robocasa24_finetune.prepare_artifacts \
  --scope task_relevant \
  --manifest-root "$MANIFEST_ROOT" \
  --semantic-root "$SEMANTIC_ROOT" \
  --geometry-root "$GEOMETRY_ROOT" \
  --motion-root "$MOTION_ROOT" \
  --output-dir "$PREPARED_ROOT/task_relevant"
```

生成器不会覆盖已有目录。若重跑，不要直接删除旧目录；先检查 provenance，并将旧目录改名
留档，再重新生成。

## 7. Validator、CPU tests 和消融 dry-run

```bash
cd "$ROBOCASA24_ROOT"
$ROBOCASA24_PYTHON -m robocasa24_finetune.validate \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant" \
  | tee "$LOG_ROOT/task_relevant_validator.json"

cd "$REPO_ROOT"
$ROBOCASA24_PYTHON -m pytest \
  experiments/robocasa24_atomic/tests \
  experiments/robocasa24_atomic/ablations/tests \
  | tee "$LOG_ROOT/cpu_tests.log"
```

validator 必须输出 `status: PASS`、24 tasks、1,200 episodes、332,859 frames。随后构造
4 个 task-relevant 消融的冻结配置：

```bash
cd "$ROBOCASA24_ROOT"
for ABLATION in \
  geometry_only semantic_geometry semantic_motion supervision_only
do
  jobs/ablations/run_8gpu.sh dry-run \
    --ablation "$ABLATION" \
    --exp-name "rc24_ablation_${ABLATION}" \
    --data-root "$DATA_ROOT" \
    --manifest-root "$MANIFEST_ROOT" \
    --policy-assets-root "$POLICY_ASSETS_ROOT" \
    --artifact-dir "$PREPARED_ROOT/task_relevant" \
    --base-weight-dir "$PI05_FP32_BASE" \
    --checkpoint-base-dir "$CHECKPOINT_ROOT" \
    | tee "$LOG_ROOT/${ABLATION}_dry_run.json"
done
```

每份 dry-run 都必须显示：30,000 updates、10,000 warmup、global batch 128、EMA disabled、
`checkpoint_save_interval=1000`、`checkpoint_keep_period=5000` 和普通 checkpoint 保留 1 个。

## 8. 2-update 真实 8-GPU smoke

只在 8 张 GPU 都空闲、validator/tests/dry-run 全部通过后执行。以下以
`semantic_motion` 为例；正式跑某个拓扑前，应先对同一拓扑做 smoke：

```bash
cd "$ROBOCASA24_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROBOCASA24_ABLATION_SMOKE_APPROVED=YES \
jobs/ablations/run_8gpu.sh smoke \
  --updates 2 \
  --ablation semantic_motion \
  --exp-name rc24_ablation_semantic_motion_smoke2 \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant" \
  --base-weight-dir "$PI05_FP32_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_ROOT" \
  --num-workers 2 \
  2>&1 | tee "$LOG_ROOT/semantic_motion_smoke2.log"
```

检查 8 个 rank 都正常、loss 和梯度有限、无 OOM、无 DataLoader stall，并记录 update 时间及
各卡显存。smoke 不写正式 checkpoint，也不能代替正式 Whole-scene cache 的真实 smoke。

## 9. 在 tmux 中后台启动一个正式消融

非 Slurm 机器推荐使用 `tmux`。它在关闭 VS Code、Codex 或 SSH 断开后仍会继续运行。一台
8 卡机器不要同时跑两个正式消融。

```bash
tmux new -s rc24_ablation_semantic_motion
```

进入 tmux 后，重新执行第 2、4、5 节的 `export`，然后运行：

```bash
set -o pipefail
cd "$ROBOCASA24_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROBOCASA24_ABLATION_TRAINING_APPROVED=YES \
jobs/ablations/run_8gpu.sh train \
  --ablation semantic_motion \
  --exp-name rc24_ablation_semantic_motion \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant" \
  --base-weight-dir "$PI05_FP32_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_ROOT" \
  --num-workers 2 \
  2>&1 | tee "$LOG_ROOT/semantic_motion_train.log"
```

按 `Ctrl-b`，再按 `d` detach。检查命令：

```bash
tmux ls
tmux attach -t rc24_ablation_semantic_motion
tail -n 100 "$LOG_ROOT/semantic_motion_train.log"
nvidia-smi
```

其余三个 task-relevant 消融只替换消融名、exp-name、session 名和日志名，artifact 仍是
`$PREPARED_ROOT/task_relevant`。一个实验完成并检查结果后，再启动下一个。

如果机器使用 Slurm，应使用 `sbatch` 提交 1 node、8 GPU 的作业，让 Slurm 管理断线后的
生命周期；不要依赖登录节点上的 tmux。partition、account、CPU 和内存参数按集群实际配置
填写，作业内执行的训练命令与上面相同。

## 10. checkpoint 规则和断点续训

正式消融固定每 1,000 updates 保存一次；5,000 的倍数永久保留，其他普通 checkpoint 只
保留最新一个。所有保留下来的 checkpoint 都包含完整 optimizer、RNG 和 DataLoader resume
state。

```text
# 训练到 28k 时
5k, 10k, 15k, 20k, 25k  # 周期性保护
28k                       # 最新普通 checkpoint
```

30k 正常结束后保留 `5k, 10k, 15k, 20k, 25k, 30k`。不要通过 CLI 改变该规则。

若训练被中断，使用完全相同的 `exp-name` 和路径重新运行，并额外加 `--resume`：

```bash
ROBOCASA24_ABLATION_TRAINING_APPROVED=YES \
jobs/ablations/run_8gpu.sh train \
  --resume \
  --ablation semantic_motion \
  --exp-name rc24_ablation_semantic_motion \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/task_relevant" \
  --base-weight-dir "$PI05_FP32_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_ROOT" \
  --num-workers 2
```

正式训练不要使用 `--overwrite`。resume 前先确认 log 中的最后 update 与 checkpoint 一致。

## 11. Whole-scene 消融的额外门槛

Whole-scene Geometry/Motion cache 不能用 task-relevant cache 替代。只有满足以下条件后才能
运行 `whole_scene`：

1. 24 tasks × 50 demos 的完整 Geometry cache 完成；
2. 完整 Motion cache 完成；
3. valid rows 与 task-relevant source manifest 严格一致；
4. validator 返回 PASS；
5. 已生成 `$PREPARED_ROOT/whole_scene`；
6. 使用真实 cache 完成 8-GPU optimizer smoke。

这里生成的是 **Batch 1 / base50 的 Whole-scene cache**：24 tasks × 每任务 50 demos，
只重新生成 Whole-scene Geometry 和 Motion；Semantic 继续复用同一 Batch 1 task-relevant
Semantic cache。不要使用 `pipeline.batch2.example.json`，也不要混入 `additional50`。

Whole-scene cache 由独立的 cache-tools 仓库生成：

```bash
export CACHE_TOOL_ROOT="$WORK_ROOT/repos/robocasa-atomic24-cache-tools"

git clone \
  https://github.com/zhiyuan-gao/robocasa-atomic24-cache-tools.git \
  "$CACHE_TOOL_ROOT"
git -C "$CACHE_TOOL_ROOT" checkout \
  3b73f8b2977bcc2c17107afef25ada8aab13de27
test "$(git -C "$CACHE_TOOL_ROOT" rev-parse HEAD)" = \
  3b73f8b2977bcc2c17107afef25ada8aab13de27
```

仓库入口是
[robocasa-atomic24-cache-tools](https://github.com/zhiyuan-gao/robocasa-atomic24-cache-tools)。
环境、teacher、数据和完整 pipeline 的安装方法见固定 commit 的
[README.md](https://github.com/zhiyuan-gao/robocasa-atomic24-cache-tools/blob/3b73f8b2977bcc2c17107afef25ada8aab13de27/README.md)；
Whole-scene 定义、单组件入口、完整 launcher、断点续跑和 matched-identity validation 见
[WHOLE_SCENE_CACHE.md](https://github.com/zhiyuan-gao/robocasa-atomic24-cache-tools/blob/3b73f8b2977bcc2c17107afef25ada8aab13de27/WHOLE_SCENE_CACHE.md)。

使用 cache-tools 的 Batch 1 模板：

```bash
cp "$CACHE_TOOL_ROOT/configs/pipeline.batch1.example.json" \
  "$CACHE_TOOL_ROOT/configs/pipeline.batch1.local.json"
cp "$CACHE_TOOL_ROOT/configs/source_schema.example.json" \
  "$CACHE_TOOL_ROOT/configs/source_schema.local.json"

export PIPELINE_CONFIG="$CACHE_TOOL_ROOT/configs/pipeline.batch1.local.json"
export SOURCE_SCHEMA="$CACHE_TOOL_ROOT/configs/source_schema.local.json"
```

将 `pipeline.batch1.local.json` 中的 `/REPLACE/...` 全部改成新机器的真实绝对路径，并确认：

```text
dataset_revision = 7236e704a04ebe477cc06d0a06ad540cd968fa5d
data_root = $RAW_DATA_ROOT
cache_root = $BATCH1_CACHE_ROOT
source_roles = ["base50"]
expected_episodes_per_source = 50
expected_total_episodes_per_task = 50
```

上面的 `$RAW_DATA_ROOT` 和 `$BATCH1_CACHE_ROOT` 是路径说明；JSON 不会展开 shell 变量，文件
中应填写 `realpath` 得到的实际绝对路径。

`cache_root` 必须指向第 5 节下载的完整 Batch 1 task-relevant cache 根目录。这样每个 task
已有的 `source/`、`semantic/`、`geometry/` 和 `motion/` 可作为 identity reference，launcher
会在同一 task 目录中新建且只写入：

```text
<BATCH1_CACHE_ROOT>/<TASK>/geometry_whole_scene/
<BATCH1_CACHE_ROOT>/<TASK>/motion_whole_scene/
<BATCH1_CACHE_ROOT>/<TASK>/whole_scene_cache_validation.json
```

cache 生成还需要 annotations、place masks、VGGT、Track4World 和各自环境；按 cache-tools
`README.md` 的 Batch 1 下载与安装章节准备。它们只用于生成 cache，不进入 policy 训练。

配好 `CONTROL_PYTHON`、`GEOMETRY_PYTHON` 和 `MOTION_PYTHON` 后，典型的 8 卡 launcher 为：

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
MOTION_WORKERS_PER_GPU=2 \
DRY_RUN=1 \
bash "$CACHE_TOOL_ROOT/scripts/run_whole_scene_cache_pipeline.sh"
```

80GB 卡通常可从每卡 2 个 Motion worker 开始；显存较小的卡可设为 1。先运行
`DRY_RUN=1`，确认打印的是 24 个 Batch 1 task 且路径正确，再移除该变量开始生成。

拿到经过验证的 cache 后，生成训练 prepared：

```bash
$ROBOCASA24_PYTHON -m robocasa24_finetune.prepare_artifacts \
  --scope whole_scene \
  --manifest-root "$MANIFEST_ROOT" \
  --semantic-root "$SEMANTIC_ROOT" \
  --geometry-root "$BATCH1_CACHE_ROOT" \
  --motion-root "$BATCH1_CACHE_ROOT" \
  --output-dir "$PREPARED_ROOT/whole_scene"

$ROBOCASA24_PYTHON -m robocasa24_finetune.validate \
  --data-root "$DATA_ROOT" \
  --manifest-root "$MANIFEST_ROOT" \
  --policy-assets-root "$POLICY_ASSETS_ROOT" \
  --artifact-dir "$PREPARED_ROOT/whole_scene"
```

`whole_scene` 的 dry-run、smoke 和训练命令与前文相同，但使用
`--ablation whole_scene` 和 `--artifact-dir "$PREPARED_ROOT/whole_scene"`。

## 12. 训练完成后的评估

训练期间不要自动启动 closed-loop evaluation。正式评估前先确认 checkpoint、执行 chunk 和
worker 数。消融必须使用专用 wrapper，以便按 metadata 恢复对应拓扑：

```bash
cd "$ROBOCASA24_ROOT"
DOWNLOAD_ASSETS=1 jobs/setup_eval_runtime.sh

export CHECKPOINT="$CHECKPOINT_ROOT/pi05_robocasa24_ablation_semantic_motion/rc24_ablation_semantic_motion/30000"
export RUN_ROOT="$ASSET_ROOT/eval/semantic_motion_30000_execute25"

NUM_GPUS=8 NUM_WORKERS=48 SHARD_MODE=episode EXECUTION_HORIZON=25 DRY_RUN=1 \
  jobs/ablations/eval_checkpoint_multiworker.sh
```

确认 dry-run 后，去掉 `DRY_RUN=1` 才会开始 24 tasks × 50 episodes 的真实评估。中断后使用
相同的 `RUN_ROOT` 并加 `RESUME=1`。评估固定预测 50 actions、执行 25 actions 后 replan。

## 13. 每次正式启动前的最终检查表

- 代码在 `robocasa24-atomic-release`，commit 已记录，worktree 干净；
- 8 张 GPU 空闲，磁盘空间足够；
- 使用固定 revision 的 base50、task-relevant cache 和严格 FP32 base；
- validator `PASS`，CPU tests 和目标消融 dry-run 通过；
- 同一拓扑的 2-update 真实 smoke 通过；
- `full` 未加入训练队列；
- `whole_scene` 没有使用 proxy 或 task-relevant cache；
- checkpoint 根目录、exp-name、日志路径和 resume/overwrite 选择已复核；
- 已明确确认可以启动该次正式训练。
