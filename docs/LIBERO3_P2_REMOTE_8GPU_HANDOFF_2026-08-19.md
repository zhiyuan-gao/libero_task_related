# LIBERO 三任务 P2：另一台 8×A100 独立训练 Handoff

更新时间：2026-08-19（CEST）

目标机器：单机 8×NVIDIA A100-80GB

仓库：<https://github.com/zhiyuan-gao/libero_task_related>

## 0. 任务边界和权威性

本文件只服务于以下任务：在另一台单机 8×A100-80GB 机器上，独立训练已经批准的
LIBERO 三任务 **P2**。不在该机器训练 P1，不训练 action-only baseline。

对于这个 P2-only 三任务 run，本文件覆盖旧 handoff 中关于“LIBERO-10 全十任务、
11,132 updates、P1/P2 串行”的内容。底层 P1/P2 架构、资产 provenance 和 exact-resume
规则仍继承：

```text
docs/LIBERO10_P1_P2_FINAL_8GPU_HANDOFF_2026-08-19.md
```

代码必须来自 GitHub `main`。三任务实现的 merge anchor 是：

```text
97424d88d2fe162293af4f7813c8b35136f407f3
```

本文档合并后 GitHub `main` 的 HEAD 会更新，因此新机器不要要求 HEAD 恰好等于上述
SHA；应要求它是 HEAD 的祖先：

```bash
git merge-base --is-ancestor \
  97424d88d2fe162293af4f7813c8b35136f407f3 HEAD
```

退出码必须为 0。原始 P1/P2 实现 merge commit 也必须仍是祖先：

```bash
git merge-base --is-ancestor \
  631370cf9ac936bb82b941da0815993368dd484f HEAD
```

不要从 transfer payload 中的旧 `c73e503` Git bundle 恢复代码。

## 1. 冻结的科学协议

三个任务在官方 LeRobot LIBERO 数据中的 task indices、episodes 和 frames 为：

| Task index | Task | Episodes | Frames |
| ---: | --- | ---: | ---: |
| 0 | put the white mug on the left plate and put the yellow and white mug on the right plate | 38 | 9,807 |
| 3 | turn on the stove and put the moka pot on it | 41 | 10,866 |
| 8 | put the black bowl in the bottom drawer of the cabinet and close it | 35 | 8,577 |
| 合计 |  | 114 | 29,250 |

三任务是在训练前冻结的，不允许改成其他三个任务。训练数据仍来自官方 LeRobot revision：

```text
repo:     physical-intelligence/libero
revision: a4336d589d589045d1c56423ffdf3b88a0e19b1f
```

P2 必须直接从官方 `pi05_base_pytorch` 独立初始化；不需要、也禁止使用 P1 checkpoint
warm-start。

冻结 recipe：

| 设置 | 值 |
| --- | ---: |
| Config | `pi05_libero3_p2_aux` |
| Optimizer updates | 3,209 |
| Warmup updates | 1,069 |
| GPU 数 | 8 |
| Local/per-GPU micro-batch | 32 |
| Global micro-batch | 256 |
| Gradient accumulation | 1 |
| Effective/global batch | 256 |
| Precision | bfloat16 |
| EMA decay | 0.999 |
| Seed | 42 |
| Peak/end LR | `5e-5` / `5e-5` |
| Gradient clipping | global norm 1.0 |
| Checkpoint interval | 1,000 updates，加 final step 3,209 |
| Gradient checkpointing | 开启 |
| W&B | 当前 launcher 明确关闭 |

Loss：

```text
L = L_action
    + 0.15 L_geo
    + 0.50 L_ground
    + 0.01 L_sem
```

P2 正式路径必须是 `joint_masked`：

```text
PaliGemma: Context | Geometry x8 | Ground x8 | Semantic teacher tokens
Expert:    Action
```

`two_pass_reference` 只用于数值 reference。不要添加 Semantic Query，不要改 loss 系数，
不要改成 30k 或 11,132 updates，不要扩展到四个 suite。

正式训练循环不运行 closed-loop eval。训练完成后的 LIBERO evaluation 是独立阶段。

## 2. 需要上传到新机器的非 Git 资产

可直接复用之前旧机器保存的完整 payload：

```text
/workspace/vla/p1_p2_policy_transfer_v1
约 21 GiB
```

该 payload 已在 2026-08-19 重新验证：

```text
transfer manifest:  PASS，22/22 required items
handoff core SHA:   全部 OK
base model SHA-256: 6dbc20690a4c391f3a2ae811aa216797a705d82656c54fe0ed4f041a032522c7
```

这次三任务改动只修改 Git 代码，没有修改、覆盖或重新生成 payload 内的数据、标注、
模型和 Hugging Face cache。

上传时必须保留 symlink/hardlink：

```bash
rsync -aH --partial --info=progress2 \
  /workspace/vla/p1_p2_policy_transfer_v1/ \
  <NEW_HOST>:/workspace/vla/p1_p2_policy_transfer_v1/
```

禁止使用 `rsync --delete`，禁止使用 `rsync -L`。

若新机器上的 payload 仍嵌套在 `/workspace/vla/p1_p2_policy_transfer_v1/`，展开：

```bash
mkdir -p /workspace/vla
rsync -aH --info=progress2 \
  /workspace/vla/p1_p2_policy_transfer_v1/ \
  /workspace/vla/
```

展开后必须存在：

```text
/workspace/vla/data/libero_four_suite_annotation/policy_aux_v1
/workspace/vla/cache/huggingface/hub/datasets--physical-intelligence--libero
/workspace/vla/models/openpi/pi05_base_pytorch
/workspace/vla/models/openpi/pi05_libero_pytorch/assets
```

不要上传 P1 checkpoint；P2 不依赖它。不要上传当前机器的 P2 preflight checkpoint
作为新机器正式 run 的初始化。

## 3. 克隆 GitHub `main`

如果仓库不存在：

```bash
mkdir -p /workspace/vla/third_party
cd /workspace/vla/third_party
git clone --recurse-submodules \
  https://github.com/zhiyuan-gao/libero_task_related.git openpi
```

如果已经存在，只允许在确认没有本地用户改动后更新 `main`：

```bash
cd /workspace/vla/third_party/openpi
git status --short
git switch main
git pull --ff-only origin main
git submodule update --init --recursive
```

然后确认：

```bash
cd /workspace/vla/third_party/openpi
git status --short
git rev-parse HEAD
git merge-base --is-ancestor \
  97424d88d2fe162293af4f7813c8b35136f407f3 HEAD
git merge-base --is-ancestor \
  631370cf9ac936bb82b941da0815993368dd484f HEAD
```

`git status --short` 应为空，两个 ancestor check 的退出码都必须是 0。

## 4. 验证资产

```bash
cd /workspace/vla/third_party/openpi

python3 scripts/verify_policy_aux_transfer_manifest.py \
  /workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/handoff/transfer_manifest.json \
  --workspace-root /workspace/vla

sha256sum -c \
  /workspace/vla/data/libero_four_suite_annotation/policy_aux_v1/handoff/handoff_core_sha256.txt

sha256sum \
  /workspace/vla/models/openpi/pi05_base_pytorch/model.safetensors
```

预期：

```text
PASS: verified 22 required items
所有 core 项：OK
base SHA-256：6dbc20690a4c391f3a2ae811aa216797a705d82656c54fe0ed4f041a032522c7
```

旧 manifest 会验证 payload 内的旧 Git bundle；它只作为历史文件存在，不能用于代码恢复。

## 5. 重建环境

路径必须保持 `/workspace/vla`，因为当前冻结 config 使用这些显式资产路径。

```bash
cd /workspace/vla/third_party/openpi
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

当前已验证软件组合：

```text
Python       3.11.16
torch        2.7.1+cu126
torch CUDA   12.6
transformers 4.53.2
```

系统 driver 不要求与来源机器逐字相同，但必须支持 CUDA 12.6。来源机器的参考 driver
为 `595.71.05`。

确认硬件、版本、磁盘和 GPU 空闲：

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

.venv/bin/python - <<'PY'
import sys
import torch
import transformers

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_devices", torch.cuda.device_count())
print("transformers", transformers.__version__)
PY

df -h /workspace
```

必须正好可见 8 张 A100-80GB，且没有其他训练进程占用 GPU。

一个完整 checkpoint 约 27 GB；预计保留 step 1000、2000、3000、3209。资产展开后建议
`/workspace` 至少还有 150 GiB 可用空间，避免 checkpoint 原子写入时空间不足。

## 6. 不占 GPU 的静态检查和定向单测

即使机器当前空闲，也显式隐藏 CUDA/JAX GPU，避免测试预分配显存：

```bash
cd /workspace/vla/third_party/openpi

uvx --from pre-commit==4.6.2 pre-commit run --all-files

CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/pytest -q \
  src/openpi/training/policy_aux_dataset_test.py \
  src/openpi/training/data_loader_test.py::test_indexed_subset_dataset_preserves_base_indices \
  src/openpi/training/data_loader_test.py::test_torch_data_loader \
  src/openpi/training/data_loader_test.py::test_torch_data_loader_infinite \
  src/openpi/training/data_loader_test.py::test_distributed_sampler_epoch_advances_when_infinite_loader_restarts \
  src/openpi/training/data_loader_test.py::test_torch_data_loader_parallel

bash -n \
  scripts/launch_p2_libero3_8gpu.sh \
  scripts/preflight_p2_libero3_8gpu.sh \
  scripts/policy_aux_libero3_8gpu_common.sh
```

预期定向单测为 `11 passed`。不要在正式训练运行期间取消 CUDA 隐藏来运行测试；曾验证
某些 JAX tests 会在每张 GPU 预分配大量显存并与训练冲突。

## 7. 新机器必须运行 P2 25+1 preflight

已有机器的 P1/P2 preflight 已通过，但新机器仍需验证自己的 driver、NCCL、filesystem
和 checkpoint/resume。该机器只承担 P2，因此只需重新运行 P2 preflight。

使用 detached `tmux` 和直接文件重定向，避免依赖 Codex、VS Code 或 SSH：

```bash
cd /workspace/vla/third_party/openpi
mkdir -p /workspace/vla/logs/8gpu_preflight

PREFLIGHT_EXP=p2_libero3_8gpu_preflight_$(date -u +%Y%m%dT%H%M%SZ)
PREFLIGHT_LOG=/workspace/vla/logs/8gpu_preflight/${PREFLIGHT_EXP}.log

tmux new-session -d -s libero3_p2_preflight \
  "bash -lc 'cd /workspace/vla/third_party/openpi && \
  EXP_NAME=${PREFLIGHT_EXP} scripts/preflight_p2_libero3_8gpu.sh \
  >>${PREFLIGHT_LOG} 2>&1'"
```

监控：

```bash
tmux has-session -t libero3_p2_preflight && echo RUNNING || echo EXITED
tail -n 100 "$PREFLIGHT_LOG"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

脚本会连续完成：

1. 从官方 base 初始化；
2. 训练 25 optimizer updates；
3. 保存 step 25 checkpoint；
4. 从 step 25 精确恢复；
5. 完成 step 26，EMA update 从 25 连续到 26。

step 25 checkpoint 目录：

```text
/workspace/vla/checkpoints/openpi_policy_aux/pi05_libero3_p2_aux/<PREFLIGHT_EXP>/25/
```

必须至少含五个非空文件：

```text
model.safetensors
train_model.safetensors
optimizer.pt
metadata.pt
training_state.pt
```

日志必须确认：

- 8 个 rank 全程存活，world size=8；
- local batch=32、global/effective batch=256、accumulation=1；
- `loss_action`、`loss_geometry`、`loss_ground`、`loss_semantic` 和 `loss_total` 全部 finite；
- `grad_norm` 非零且 finite；
- 没有 NaN、Inf、OOM、NCCL/DDP error 或 rank crash；
- `Saved checkpoint at step 25`；
- 8 个 rank 都显示 `Successfully loaded all checkpoint components from step 25`；
- `Restored EMA parameters at update 25`；
- 恢复后的日志显示 `ema_updates=26` 并完成 step 26。

可以用以下命令提取关键证据：

```bash
tr '\r' '\n' < "$PREFLIGHT_LOG" | rg \
  'Training config|loss_action|loss_geometry|loss_ground|loss_semantic|loss_total|grad_norm|Saved checkpoint at step 25|Successfully loaded all checkpoint components from step 25|Restored EMA parameters at update 25|ema_updates=26|OutOfMemory|NCCL|Traceback'
```

不要在 preflight 失败时启动正式训练。

## 8. detached 正式 P2 启动

只有资产、环境、静态检查和新机器 P2 25+1 preflight 全部通过后才执行。

正式 run 必须使用新实验名；不得复用 preflight 名称，不得设置 `RESUME=YES`：

```bash
cd /workspace/vla/third_party/openpi
mkdir -p /workspace/vla/logs/8gpu_formal

EXP_NAME=p2_libero3_8gpu_formal_$(date -u +%Y%m%dT%H%M%SZ)
FORMAL_LOG=/workspace/vla/logs/8gpu_formal/${EXP_NAME}.log
EXP_RECORD=/workspace/vla/logs/8gpu_formal/p2_current_exp_name.txt

printf '%s\n' "$EXP_NAME" > "$EXP_RECORD"

test ! -e \
  "/workspace/vla/checkpoints/openpi_policy_aux/pi05_libero3_p2_aux/${EXP_NAME}"

tmux new-session -d -s libero3_p2_formal \
  "bash -lc 'cd /workspace/vla/third_party/openpi && \
  exec env FULL_TRAINING_APPROVED=YES EXP_NAME=${EXP_NAME} \
  scripts/launch_p2_libero3_8gpu.sh >>${FORMAL_LOG} 2>&1'"
```

该 tmux server 与 shell/torchrun 会独立于 Codex、VS Code 和 SSH。关闭这些客户端后训练
继续。机器重启、容器停止或 tmux server 被杀仍会中断训练。

启动后必须核对：

```bash
tmux has-session -t libero3_p2_formal && echo RUNNING || echo EXITED
pgrep -af 'torch.distributed.run|train_pytorch.py.*pi05_libero3_p2_aux'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
tail -n 100 "$FORMAL_LOG"
```

日志开头必须明确显示：

```text
world_size=8
micro_batch_per_gpu=32
effective_global_batch=256
num_train_steps=3209
gradient_checkpointing=True
warmup=1069
EMA decay=0.999
bfloat16
```

正式 launcher 已传入 `--no-wandb-enabled`，不需要 `wandb login`。

## 9. 进度、性能和异常监控

进度：

```bash
EXP_NAME=$(< /workspace/vla/logs/8gpu_formal/p2_current_exp_name.txt)
FORMAL_LOG=/workspace/vla/logs/8gpu_formal/${EXP_NAME}.log

tr '\r' '\n' < "$FORMAL_LOG" | rg 'step=|Auxiliary metrics|Training:' | tail -n 30
```

错误扫描：

```bash
rg -n \
  'Traceback|CUDA out of memory|OutOfMemory|NCCL.*(WARN|ERROR)|ChildFailed|ProcessExited|NaN|Killed|SIG[A-Z]+' \
  "$FORMAL_LOG"
```

每个 logging interval 应报告 action/geometry/ground/semantic/total loss、gradient norm、
EMA count、seconds/update、samples/s 和显存；进度条会逐 update 前进。所有
loss/gradient 必须 finite。

来源 8 卡机器的 P2 25-step preflight 在初始化后约 11 s/update，但该数字不是新机器保证。
用新机器初始化后的稳态中位数估时：

```text
预计纯计算小时 = 3,209 × 稳态 seconds/update ÷ 3,600
```

若稳态为 11--14 s/update，纯计算约 9.8--12.5 小时，另加启动和 checkpoint I/O。

## 10. checkpoint 和完成验收

正式 checkpoint 根目录：

```text
/workspace/vla/checkpoints/openpi_policy_aux/pi05_libero3_p2_aux/<EXP_NAME>/
```

预期 step 目录：

```text
1000/
2000/
3000/
3209/
```

每个完整 checkpoint 应至少有：

```text
model.safetensors         EMA serving/evaluation weights
train_model.safetensors   raw train weights，用于精确续训
optimizer.pt              optimizer state
metadata.pt               step/config/EMA metadata
training_state.pt         RNG/data-loader/rank continuation state
```

检查 final：

```bash
EXP_NAME=$(< /workspace/vla/logs/8gpu_formal/p2_current_exp_name.txt)
FINAL=/workspace/vla/checkpoints/openpi_policy_aux/pi05_libero3_p2_aux/${EXP_NAME}/3209

find "$FINAL" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
test -s "$FINAL/model.safetensors"
test -s "$FINAL/train_model.safetensors"
test -s "$FINAL/optimizer.pt"
test -s "$FINAL/metadata.pt"
test -s "$FINAL/training_state.pt"
```

用于 evaluation/serving 的是 `model.safetensors`（EMA）；不是 raw train weights。

## 11. 中断后的 exact resume

先确认原 torchrun 已完全退出，8 张 GPU 没有残留训练进程。只恢复同一个 8 卡 P2
experiment，不改变任何 recipe 或路径：

```bash
cd /workspace/vla/third_party/openpi

EXP_NAME=$(< /workspace/vla/logs/8gpu_formal/p2_current_exp_name.txt)
RESUME_LOG=/workspace/vla/logs/8gpu_formal/${EXP_NAME}.resume.log

tmux new-session -d -s libero3_p2_formal \
  "bash -lc 'cd /workspace/vla/third_party/openpi && \
  exec env FULL_TRAINING_APPROVED=YES RESUME=YES EXP_NAME=${EXP_NAME} \
  scripts/launch_p2_libero3_8gpu.sh >>${RESUME_LOG} 2>&1'"
```

trainer 会选择最新的**完整** checkpoint。上一个 checkpoint 后尚未保存的 updates 需要
重算。resume 日志必须确认恢复 raw train weights、optimizer、EMA、RNG、data-loader
position 和 rank-local state。

禁止事项：

- 不在 resume 时使用新 `EXP_NAME`；
- 不把 `RESUME=YES` 用于第一次正式启动；
- 不添加 `--overwrite`；
- 不改变 GPU 数、batch、accumulation、seed、warmup、EMA 或 loss 系数；
- 不用 `model.safetensors` 代替 raw train weights 做 optimizer resume；
- 不从 P1 checkpoint 恢复。

## 12. 新机器最终需要汇报

在启动正式 P2 前汇报：

1. Git HEAD，以及 `97424d88...`、`631370cf...` 两个 ancestor checks；
2. 8 张 GPU 的型号、显存和 driver；
3. manifest、core checksums 和 base model SHA；
4. Python、torch、torch CUDA、transformers 版本；
5. `/workspace` 可用空间和是否有其他 GPU 进程；
6. P2 25+1 preflight 的五个 loss、grad norm、显存、step 25 checkpoint 和 step 26 resume；
7. 新机器稳态 seconds/update、samples/s 和 3,209-update 预计耗时；
8. preflight 日志路径。

正式 run 启动后汇报：

1. `EXP_NAME`、tmux session 和日志路径；
2. 8 个 rank/8 张 GPU 是否都在；
3. 实际起始 config；
4. 首个 optimizer update 的 finite loss/grad/EMA；
5. 每次 checkpoint 写入和异常扫描结果；
6. final step 3,209 checkpoint 的文件、大小和 metadata。

## 13. 可粘贴给另一台机器 Codex 的首条消息

```text
我正在另一台单机 8×A100-80GB 机器上独立运行 LIBERO 三任务 P2。

权威仓库：
https://github.com/zhiyuan-gao/libero_task_related

权威 handoff：
docs/LIBERO3_P2_REMOTE_8GPU_HANDOFF_2026-08-19.md

请完整阅读 handoff，然后直接检查机器状态并执行资产展开/校验、环境安装、CPU 隔离静态
检查和 P2 25+1 preflight。不要使用 payload 内旧 Git bundle，不要删除或重新生成资产，
不要使用 rsync --delete 或 rsync -L。确认 GitHub main 中
97424d88d2fe162293af4f7813c8b35136f407f3 是 HEAD 的祖先。

preflight 通过后先向我汇报所有 loss、grad norm、显存、速度、step 25 checkpoint 和
step 26 exact resume。获得批准后，用 detached tmux 和直接文件重定向启动 P2 3,209-update
正式训练。只跑 P2；不跑 P1 或 action-only。P2 必须从官方 pi05_base_pytorch 独立初始化，
使用 joint-masked semantic implementation。训练进程不能依赖 Codex、VS Code 或 SSH。
```

## 14. 最终禁止事项

- 不使用 payload 的 `c73e503` bundle 作为代码。
- 不删除、覆盖或重新生成已有数据/模型。
- 不使用 `rsync --delete` 或 `rsync -L`。
- 不运行 P1 或 action-only。
- 不从 P1 checkpoint warm-start P2。
- 不改变三个 task indices `(0, 3, 8)`。
- 不改变 3,209 updates、1,069 warmup、batch 256、EMA 0.999 或 loss 系数。
- 不关闭 gradient checkpointing。
- 不把 P2 semantic path 改成 `two_pass_reference`。
- 不添加 Semantic Query。
- 不把 policy population 扩展到其他 LIBERO tasks 或四个 suite。
- 不在 preflight 失败时启动正式训练。
- 不让正式 torchrun 依赖 Codex、VS Code、SSH 或前台 pipe/`tee`。
- 不把数据、模型、日志或 checkpoint 提交到 Git。
