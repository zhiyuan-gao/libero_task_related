# LIBERO-10 P1/P2 最终 8×A100 Handoff

> **性能配置更新：** base conversion 精度和 trainer 启动环境以
> `PI05_FP32_CONVERSION_BF16_8GPU_SPEED_HANDOFF_2026-08-19.md` 为准；本文的数据、loss、
> update 数和 checkpoint 规则保持有效。

更新时间：2026-08-19 UTC

目标机器：单机 8×A100-80GB

仓库：<https://github.com/zhiyuan-gao/libero_task_related>

状态：代码和 4 卡验证已完成；8 卡 preflight 与正式训练尚未执行

## 0. 本文件的权威性

这是停用当前机器前的最终 handoff。若旧聊天、旧 README、旧 transfer
manifest 或以下文件与本文件冲突，以本文件和 GitHub `main` 为准：

```text
LIBERO10_P1_P2_MASTER_8GPU_CODEX_HANDOFF_2026-08-18.md
LIBERO10_P1_P2_TRAINING_RECIPE_SPEED_SUMMARY_2026-08-19.md
P1_P2_TRANSFER_README.md
```

必须忽略的过期状态包括：

```text
从 c73e503 直接训练
正式训练 30,000 updates
EMA 未实现或禁用
P2 正式路径使用两次独立 PaliGemma forward
8 卡每卡 batch 1、gradient accumulation 32
```

正确状态是：

```text
validated implementation commit: 7d53f6f0462d4b1c276d58838addd34827496c51
implementation merge commit:      631370cf9ac936bb82b941da0815993368dd484f
8-GPU local micro-batch:           32
8-GPU gradient accumulation:       1
effective/global batch:            256
formal optimizer updates:          11,132
warmup updates:                    3,710
EMA decay:                         0.999
P2 semantic implementation:        joint-masked, one PaliGemma forward
```

本 handoff 文档提交会是 `631370c` 的后代。克隆后应验证：

```bash
git merge-base --is-ancestor \
  631370cf9ac936bb82b941da0815993368dd484f HEAD
```

退出码必须为 0。

## 1. 当前完成状态

已完成：

- P1/P2 辅助训练实现和 serving 支持。
- 4 卡与 8 卡独立启动脚本，公共 recipe 被统一冻结。
- 官方每卡 micro-batch 32 和 effective/global batch 256 对齐。
- LIBERO-10 数据量缩放后的 11,132 updates / 3,710 warmup。
- 官方 EMA=0.999；checkpoint 同时保存 raw train weights 和 EMA serving weights。
- optimizer、EMA、RNG、data-loader position 和 rank-local state 的精确恢复。
- P2 joint-masked semantic 优化及两遍 reference 实现。
- 4×A100 上的 P1/P2 preflight、P2 语义消融、联合语义 A/B/C/D/E 验证。
- 31 个提交前相关单测、ruff、shell 语法和 4/8 卡静态 recipe gate。

尚未完成：

- 在目标 8×A100 机器验证 transfer payload。
- 在目标机器重建 Python 环境。
- 8 卡 P1 25+1 preflight。
- 8 卡 P2 25+1 preflight。
- 根据 8 卡实测速度重新估算总训练时间。
- 正式 11,132-update P1/P2 训练。
- 正式 checkpoint 的 LIBERO closed-loop evaluation。

在 8 卡 preflight 通过并人工检查前，不要启动正式训练。

## 2. GitHub 和 Git 状态

仓库是独立 public repository，不是 GitHub fork network 中的 fork：

```text
https://github.com/zhiyuan-gao/libero_task_related
```

关键提交：

```text
15a9616  OpenPI upstream base
5abae9f  initial frozen P1/P2 auxiliary-query implementation
c73e503  auxiliary serving and exact-resume baseline
7d53f6f  validated 4/8-GPU recipe, EMA/resume, joint P2, tests
631370c  merge 7d53f6f into main
```

已合并 PR：

```text
https://github.com/zhiyuan-gao/libero_task_related/pull/1
```

功能分支 `agent/p1-p2-8gpu-migration` 暂时保留用于溯源。新机器应直接克隆
默认分支 `main`，不要恢复旧 payload 中的 `c73e503` Git bundle。

## 3. 科学问题与数据边界

研究目标：在不增加额外机器人轨迹的情况下，使用 task-relevant
offline/privileged supervision 改善 VLA action policy。

正式 policy population 只包含 LIBERO-10：

| 项目 | 冻结值 |
| --- | ---: |
| Hugging Face repo | `physical-intelligence/libero` |
| Revision | `a4336d589d589045d1c56423ffdf3b88a0e19b1f` |
| Episodes | `0..378`，共 379 |
| Frames / samples | 101,469 |
| Effective epochs | 约 28.09 |

禁止把正式 policy population 扩展到 LIBERO 四个 suite、OXE、DROID、Bridge
或其他机器人轨迹。Geometry/Ground/Semantic 教师目标已经离线物化；训练机器不需要
VGGT、Track4World 或其他 teacher 模型。

## 4. 冻结训练 recipe

| 设置 | P1 | P2 |
| --- | ---: | ---: |
| 初始化 | 官方 `pi05_base_pytorch` | 官方 `pi05_base_pytorch` |
| Optimizer updates | 11,132 | 11,132 |
| Warmup updates | 3,710 | 3,710 |
| Local micro-batch/GPU，8 卡 | 32 | 32 |
| Global micro-batch，8 卡 | 256 | 256 |
| Gradient accumulation，8 卡 | 1 | 1 |
| Effective/global batch | 256 | 256 |
| Precision | bfloat16 | bfloat16 |
| EMA decay | 0.999 | 0.999 |
| Peak/end LR | `5e-5` / `5e-5` | `5e-5` / `5e-5` |
| AdamW betas | 0.9 / 0.95 | 0.9 / 0.95 |
| Weight decay | `1e-10` | `1e-10` |
| Gradient clipping | global norm 1.0 | global norm 1.0 |
| Seed | 42 | 42 |
| Checkpoint interval | 1,000 updates | 1,000 updates |

4 卡 profile 被保留用于复现：每卡 32、global micro-batch 128、accumulation 2、
effective batch 256。不要用 4 卡 profile 启动目标 8 卡实验。

Loss 定义被冻结：

```text
P1: L = L_action + 0.15 L_geo

P2: L = L_action
      + 0.15 L_geo
      + 0.50 L_ground
      + 0.01 L_sem
```

不做 loss coefficient sweep、GradNorm、动态调权、额外 query warmup 或 P1→P2
checkpoint warm-start。P1 和 P2 都从同一官方 base checkpoint 独立初始化。

## 5. P1/P2 架构冻结点

共同 π0.5 设置：

```text
pi05=true
action_horizon=10
discrete_state_input=false
PaliGemma=Gemma 2B
action expert=Gemma 300M
internal action_dim=32
LIBERO output action shape=10x7
```

P1 layout：

```text
Context | Geometry x8 | Action
```

P2 joint training layout：

```text
PaliGemma: Context | Geometry x8 | Ground x8 | Semantic teacher tokens
Expert:    Action
```

P2 semantic 是原生自回归 PaliGemma LM objective，不是 Semantic Query。joint mask
保证 Action/Geometry/Ground 都不能读取 Semantic GT；Semantic teacher tokens 只在训练路径
构造。正式默认是 `semantic_impl="joint_masked"`；
`semantic_impl="two_pass_reference"` 只用于数值 reference，不能用于正式训练。

## 6. 必须从旧电脑迁移的非 Git 资产

GitHub 不包含训练数据、模型权重、辅助标注、日志或 checkpoint。

旧电脑上原始 transfer payload 应为：

```text
/workspace/vla/p1_p2_policy_transfer_v1
约 20.18 GiB
```

重要：停用前这台 4 卡机器上的同名目录是空目录（0 bytes，mtime
2026-08-18 22:36 UTC）。不要从本机复制这个空目录；必须使用更早那台电脑上保存的
完整约 20.18 GiB payload。

在 8 卡目标机器上执行：

```bash
mkdir -p /workspace/vla

rsync -aH --partial --info=progress2 \
  <OLD_SOURCE_HOST>:/workspace/vla/p1_p2_policy_transfer_v1/ \
  /workspace/vla/
```

必须保留源目录末尾 `/`。不要添加 `--delete`，不要添加 `-L`；Hugging Face cache
含有必须保留的相对 symlink。

若 payload 已整体上传到目标机器的
`/workspace/vla/p1_p2_policy_transfer_v1/`，展开方式为：

```bash
rsync -aH --info=progress2 \
  /workspace/vla/p1_p2_policy_transfer_v1/ \
  /workspace/vla/
```

展开后必须出现：

```text
/workspace/vla/data/libero_four_suite_annotation/policy_aux_v1
/workspace/vla/cache/huggingface/hub/datasets--physical-intelligence--libero
/workspace/vla/models/openpi/pi05_base_pytorch
/workspace/vla/models/openpi/pi05_libero_pytorch/assets
```

参考体积：

```text
policy_aux_v1:                         about 1.6 GB
Hugging Face LIBERO dataset cache:     about 12 GB
pi05_base_pytorch:                     about 6.8 GB
pi05_libero_pytorch/assets:            small normalization assets
```

不要传输或恢复这台 4 卡机器产生的 checkpoint 来做 8 卡 resume。精确恢复记录了
rank-local state 和 world size；4 卡 checkpoint 不能作为 8 卡精确续训点。8 卡正式实验
应从 `pi05_base_pytorch` 新开。

## 7. 在目标机器验证资产

先克隆最新代码，再运行 manifest verifier：

```bash
cd /workspace/vla/third_party/openpi

python3 scripts/verify_policy_aux_transfer_manifest.py \
  /workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/handoff/transfer_manifest.json \
  --workspace-root /workspace/vla
```

旧 payload 的预期结果：

```text
PASS: verified 22 required items
```

然后运行：

```bash
sha256sum -c \
  /workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/handoff/handoff_core_sha256.txt

sha256sum /workspace/vla/models/openpi/pi05_base_pytorch/model.safetensors
```

所有 core 项必须为 `OK`。base model 的 SHA-256 必须为：

```text
6dbc20690a4c391f3a2ae811aa216797a705d82656c54fe0ed4f041a032522c7
```

旧 transfer manifest 和 Git bundle 中的 code provenance 停留在 `c73e503`；这不影响
它验证 data/model/cache。代码 provenance 必须单独以 GitHub `main` 和
`631370c` ancestor check 为准。

## 8. 克隆代码与重建环境

```bash
mkdir -p /workspace/vla/third_party
cd /workspace/vla/third_party

git clone --recurse-submodules \
  https://github.com/zhiyuan-gao/libero_task_related.git openpi

cd /workspace/vla/third_party/openpi

git merge-base --is-ancestor \
  631370cf9ac936bb82b941da0815993368dd484f HEAD

git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

仓库固定 Python/依赖版本于 `.python-version` 和 `uv.lock`。transformers replacement
是当前 PyTorch π0.5 路径的必要步骤，不能省略。

确认硬件和软件：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
.venv/bin/python -c 'import transformers; print(transformers.__version__)'
df -h /workspace
```

CUDA device count 必须为 8，transformers 必须为 4.53.2。建议为 P1/P2 正式
checkpoint 合计预留至少 200 GiB；每个 active run 还需要约 27 GiB 原子写入余量。

正式训练默认启用 Weights & Biases。开始前在目标机器完成 `wandb login`，或明确记录
经批准的离线 logging 方案；logging 模式不能改变科学 recipe。

## 9. 提交前验证证据

合并 `7d53f6f` 前执行并通过：

```text
31 targeted pytest tests: PASS
ruff on changed Python files: PASS
shell syntax for 4/8-GPU launchers: PASS
4/8-GPU frozen launch-template gate: PASS
```

P2 joint-masked A/B/C/D/E 结果：

```text
fixed-batch forward parity: PASS
adversarial Semantic no-leakage: PASS (Action/Geo/Ground exact zero change)
representative gradient parity: PASS
4xA100 15-update stability: PASS
raw/EMA exact resume: PASS
```

4 卡 joint-masked P2 稳态为约 14.40 s/update、57.28 GB allocated/GPU；旧两遍 P2
为 25.18 s/update、80.99 GB allocated/GPU。joint-masked 实现约 1.75× 更快，且没有
发现 scientific-semantic 变化。

## 10. 目标 8 卡机器的 mandatory preflight

不要直接启动 11,132 updates。先依次运行 P1、P2；两者都使用全部 8 张 GPU，不能在
同一节点并行。

建议在 `tmux` 或其他持久终端中运行，并把完整 stdout/stderr 保存到持久磁盘。

```bash
cd /workspace/vla/third_party/openpi
mkdir -p /workspace/vla/logs/8gpu_preflight
set -o pipefail

EXP_NAME=p1_8gpu_preflight_$(date -u +%Y%m%d) \
  scripts/preflight_p1_libero10_8gpu.sh 2>&1 | \
  tee /workspace/vla/logs/8gpu_preflight/p1_8gpu_preflight.log

EXP_NAME=p2_8gpu_preflight_$(date -u +%Y%m%d) \
  scripts/preflight_p2_libero10_8gpu.sh 2>&1 | \
  tee /workspace/vla/logs/8gpu_preflight/p2_8gpu_preflight.log
```

每个脚本会：

1. 从官方 `pi05_base_pytorch` 新开实验；
2. 训练 25 optimizer updates；
3. 保存 raw/EMA/optimizer/rank-local continuation state；
4. 从 step 25 精确恢复；
5. 再完成第 26 个 update。

验收条件：

- 八个 rank 全程存活，没有 NCCL/DDP error。
- local/per-GPU batch=32、accumulation=1、effective batch=256。
- Action/Geo/Ground/Semantic（按实验适用）和 total loss 全部 finite。
- gradient norm 非零且 finite，无 NaN/Inf。
- step 25 checkpoint 成功写完。
- resume 明确加载 raw weights、EMA、optimizer、RNG/data state。
- EMA update count 从 25 连续进入 26。
- resume 完成 step 26。
- 峰值 allocated memory 不超过每张卡物理显存。
- 用初始化 update 之后的稳态 update 重新计算 seconds/update。

完成两个 preflight 后先停止，把日志、稳态速度、峰值显存和 checkpoint metadata 交给
人工/Codex 检查。preflight 脚本不会自动进入正式训练。

## 11. 正式训练命令

只有 8 卡 P1/P2 preflight 均通过并获得人工批准后，才设置
`FULL_TRAINING_APPROVED=YES`。

P1：

```bash
cd /workspace/vla/third_party/openpi

FULL_TRAINING_APPROVED=YES \
EXP_NAME=p1_libero10_8gpu_v1 \
  scripts/launch_p1_libero10_8gpu.sh
```

P2：

```bash
cd /workspace/vla/third_party/openpi

FULL_TRAINING_APPROVED=YES \
EXP_NAME=p2_libero10_joint_masked_8gpu_v1 \
  scripts/launch_p2_libero10_8gpu.sh
```

默认 checkpoint 位置：

```text
/workspace/vla/checkpoints/openpi_policy_aux/pi05_libero_p1_aux/<EXP_NAME>/
/workspace/vla/checkpoints/openpi_policy_aux/pi05_libero_p2_aux/<EXP_NAME>/
```

若 checkpoint 使用另一块持久磁盘，只允许在首次启动前设置
`CHECKPOINT_BASE_DIR=<persistent-path>`。resume 时必须使用完全相同的路径、实验名、
world size 和 trajectory-affecting config。

## 12. 中断后的精确恢复

只能恢复同一个 8 卡实验：

```bash
cd /workspace/vla/third_party/openpi

FULL_TRAINING_APPROVED=YES \
RESUME=YES \
EXP_NAME=p2_libero10_joint_masked_8gpu_v1 \
  scripts/launch_p2_libero10_8gpu.sh
```

P1 使用对应 P1 脚本和相同 P1 experiment name。不要在 resume 时使用 `--overwrite`，
不要改变 batch、accumulation、GPU 数、seed、optimizer、EMA、数据范围、warmup 或 loss
系数。trainer 会拒绝不满足 exact-continuation 的 checkpoint。

EMA checkpoint schema：

```text
train_model.safetensors   raw optimizer-updated model，用于精确续训
model.safetensors         EMA model，用于 serving/evaluation
optimizer.pt              optimizer state
metadata.pt               step/config/EMA metadata
training_state_rank*.pt   rank-local RNG/data continuation state
```

## 13. 耗时预期

已验证的 4×A100 稳态：

| 实验 | 4 卡稳态 |
| --- | ---: |
| P1 | 约 13.61 s/update |
| P2 joint-masked | 约 14.40 s/update |

理想 8 卡缩放对应每个 11,132-update run 约 21--23 小时纯计算。该数字不是 8 卡实测；
应以目标机器 preflight 的稳态速度为准，并加入 checkpoint、启动、filesystem 和调度余量。
当前合理规划是每个 run 约一天，P1/P2 串行约 44--50 小时，之后再根据实测修正。

## 14. 关键代码入口

```text
scripts/policy_aux_gpu_common.sh
scripts/policy_aux_8gpu_common.sh
scripts/preflight_p1_libero10_8gpu.sh
scripts/preflight_p2_libero10_8gpu.sh
scripts/launch_p1_libero10_8gpu.sh
scripts/launch_p2_libero10_8gpu.sh
scripts/train_pytorch.py
scripts/serve_policy.py
scripts/validate_policy_aux_launch_templates.py
scripts/validate_joint_p2_semantic.py
src/openpi/training/config.py
src/openpi/training/policy_aux_dataset.py
src/openpi/training/pytorch_ema.py
src/openpi/models_pytorch/pi05_aux_queries.py
docs/libero10_p1_p2_8gpu_migration.md
```

## 15. 新机器交给 Codex 的首条消息

可直接粘贴：

```text
我正在单机 8×A100-80GB 上继续 LIBERO-10 P1/P2 训练。

权威 handoff：
/workspace/vla/third_party/openpi/docs/LIBERO10_P1_P2_FINAL_8GPU_HANDOFF_2026-08-19.md

仓库：
https://github.com/zhiyuan-gao/libero_task_related

请先只读检查 handoff、Git HEAD、8 张 GPU、数据/模型 manifest、base model SHA256、
Python/torch/transformers 环境和磁盘空间。确认实现 merge commit 631370c 是 HEAD 的祖先。
然后运行 8 卡 P1 25+1 preflight 和 P2 25+1 preflight，记录每个 loss、gradient norm、
稳态 seconds/update、峰值 allocated/reserved memory、step-25 checkpoint 和 step-26 exact
resume 结果。preflight 后停止并向我汇报，不要自动启动 11,132-update 正式训练。
```

## 16. 最终禁止事项

- 不从旧 `c73e503` bundle 启动训练。
- 不把正式训练改回 30,000 updates。
- 不把 LIBERO-10 扩成四个 suite。
- 不改变每卡 micro-batch 32 或 effective batch 256。
- 不关闭 EMA，不用 EMA weights 做 optimizer resume。
- 不把 P2 正式 semantic path 改回 two-pass reference。
- 不恢复 4 卡 checkpoint 为 8 卡 run。
- 不覆盖 P1 checkpoint 作为 P2 初始化，反之亦然。
- 不在 preflight 失败或尚未人工审核时设置 `FULL_TRAINING_APPROVED=YES`。
- 不把数据、权重、日志或 checkpoint 提交进 Git。

到这里为止，当前机器可安全退出项目；后续唯一需要的持久来源是 GitHub `main`、旧电脑
保存的完整约 20.18 GiB transfer payload，以及目标 8 卡机器新产生的日志/checkpoint。
