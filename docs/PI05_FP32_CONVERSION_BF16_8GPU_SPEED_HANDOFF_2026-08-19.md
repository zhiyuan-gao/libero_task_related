# π₀.₅ FP32 Conversion + BF16 Finetuning：8×A100 性能 Handoff

更新时间：2026-08-19 UTC

适用机器：单机 8×NVIDIA A100-80GB

仓库：<https://github.com/zhiyuan-gao/libero_task_related>

## 0. 结论与覆盖范围

后续 P1/P2 训练统一采用：

```text
official π₀.₅ JAX base
  -> 一次性 float32 PyTorch conversion
  -> bfloat16 PyTorch finetuning
```

不要使用 full-float32 finetuning。相同的 P2、每卡 batch 32、global batch 256、EMA 0.999
在 8×A100-80GB 上会在第一个 AdamW step 达到约 79.23 GiB/GPU 后 OOM。

本文件只覆盖以下新设置：

- FP32-converted base checkpoint；
- PyTorch finetuning 仍为 BF16；
- 默认 CUDA allocator；
- 关闭逐 update 的 CUDA memory-stats 查询；
- foreach 批量 EMA；
- 每 update 只调用一次全模型 `clip_grad_norm_`。

这些设置不改变 action/P1/P2 loss、loss 系数、global batch、learning-rate schedule、EMA
公式、梯度裁剪阈值、update 数或 checkpoint 语义。三任务 P2 的数据和正式 update 数继续遵循
`LIBERO3_P2_REMOTE_8GPU_HANDOFF_2026-08-19.md`；LIBERO-10 P1/P2 继续遵循
`LIBERO10_P1_P2_FINAL_8GPU_HANDOFF_2026-08-19.md`。

## 1. 已验证性能

本机实测配置：8×A100-80GB、P2、每卡 micro-batch 32、global/effective batch 256、
accumulation 1、EMA 0.999、gradient checkpointing enabled。

| 配置 | 结果 | 稳态 seconds/update |
|---|---|---:|
| FP32 conversion + BF16 finetuning | 25 updates 完成 | mean 5.3125，median 5.2，range 5.2–6.0 |
| FP32 conversion + full FP32 finetuning | 第一个 AdamW step OOM | 不可用 |

首个 BF16 update 的 64.7 秒包含 `torch.compile(max-autotune)`，不计入稳态统计。此前旧
allocator/逐步显存查询组合会出现约 6–16 秒周期抖动；关闭 EMA 也不能消除该抖动。因此远端
机器仍为 10 秒以上时，先检查本文件第 5 节中的启动环境，而不是删除 EMA 或辅助 loss。

## 2. 更新代码

PR 合并后使用 `main`：

```bash
cd /workspace/vla/third_party/openpi
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
```

如果 PR 尚未合并，可直接检出发布分支：

```bash
git fetch origin agent/optimize-pi05-8gpu
git switch --detach origin/agent/optimize-pi05-8gpu
```

不要覆盖或重新生成 `/workspace/vla/data` 下已经传输完成的 P1/P2 资产。

## 3. 准备 FP32-converted base

训练配置要求以下固定路径：

```text
/workspace/vla/models/openpi/pi05_base_pytorch_fp32
```

推荐从已验证机器传输该目录：

```bash
mkdir -p /workspace/vla/models/openpi/pi05_base_pytorch_fp32
rsync -ah --info=progress2 \
  <SOURCE_HOST>:/workspace/vla/models/openpi/pi05_base_pytorch_fp32/ \
  /workspace/vla/models/openpi/pi05_base_pytorch_fp32/
```

也可以在目标机器从官方 JAX base 重新转换一次：

```bash
cd /workspace/vla/third_party/openpi

OPENPI_DATA_HOME=/workspace/vla/models/openpi_jax_cache \
  .venv/bin/python -c \
  'from openpi.shared.download import maybe_download; print(maybe_download("gs://openpi-assets/checkpoints/pi05_base"))'

.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /workspace/vla/models/openpi_jax_cache/openpi-assets/checkpoints/pi05_base \
  --config-name pi05_libero \
  --output-path /workspace/vla/models/openpi/pi05_base_pytorch_fp32 \
  --precision float32
```

转换只需执行一次。不要把已有 BF16 safetensors 用 `.float()` 上采样；那不能恢复转换时丢失
的精度。

源机器上已验证、直接传输时应精确匹配的 checkpoint：

```text
model.safetensors bytes: 14,467,165,872
model.safetensors SHA-256: c7b765acc64b419f81b751d40d9771e4afc5cfa525229c9ca900a68baf247f77
tensor count: 812
parameter count: 3,616,757,520
all tensors: torch.float32
```

### 3.1 本地重转换 checksum 与正式运行 provenance

另一台 8×A100 机器按照上面的同一条 JAX→PyTorch 命令重新转换后，得到：

```text
model.safetensors bytes: 14,467,165,872
model.safetensors SHA-256: 16d790217b2846abe13d1a2372790c5cd7b2a21a96f94d50ca22190823d2c4ab
tensor count: 812
parameter count: 3,616,757,520
all tensors: torch.float32
strict auxiliary-base validation: PASS
```

该 `16d790...` artifact 已完成三任务 P2 的 25+1 preflight，并用于正式实验
`p2_libero3_fp32base_bf16_full_20260819T165517Z`。正式实验仍为 BF16 finetuning，并非
full-FP32 training。

两个整文件 SHA 不同的原因已经定位到
`paligemma_with_expert.gemma_expert.lm_head.weight`。当前转换器从 JAX checkpoint 加载参数时
不会填充这个 action-expert LM head，因此它保留 PyTorch 模型构造时的随机初始化值。当前 action
expert 调用内部 `.model.forward`，不调用这个 LM head；它不参与 action forward、loss 或梯度。
其余 811 个 tensor 在转换为 BF16 后与原已验证官方 BF16 base 逐元素完全一致。

因此 checksum 的使用规则是：

- 从另一台机器传输 artifact 时，必须与源文件记录的 SHA 完全一致；
- 在目标机器从同一官方 JAX base 本地重转换时，必须记录实际 SHA，并通过下面的严格结构、dtype、
  tensor 数和参数数验证；不要仅因 unused LM head 造成的整文件 SHA 不同而拒绝 artifact；
- 每个正式 run 必须在启动清单中记录实际使用的 base SHA，运行中不得替换 base；
- 除上述已定位的 unused LM head 外，任何参与 forward 的 tensor 差异都不在本说明的允许范围内。

校验：

```bash
sha256sum /workspace/vla/models/openpi/pi05_base_pytorch_fp32/model.safetensors

cd /workspace/vla/third_party/openpi
.venv/bin/python scripts/validate_pi05_aux_base_checkpoint.py \
  --checkpoint /workspace/vla/models/openpi/pi05_base_pytorch_fp32/model.safetensors \
  --output /tmp/pi05_fp32_base_validation.json
```

## 4. 确认训练精度

FP32 只用于训练前的 conversion checkpoint。正式 trainer 必须显示：

```text
Training precision: bfloat16
```

配置常量已经指向 `pi05_base_pytorch_fp32`，而 `pytorch_training_precision` 保持
`bfloat16`。P1 和 P2 必须使用同一个 base 和同一种 finetuning precision。

## 5. 确认性能环境

正式 launcher 默认导出：

```bash
OPENPI_USE_DEFAULT_CUDA_ALLOCATOR=1
OPENPI_LOG_MEMORY_STATS=0
```

第一项会在 CUDA 初始化前移除继承的 `PYTORCH_CUDA_ALLOC_CONF`。第二项只关闭训练循环中的
逐 update 显存诊断，不影响 forward、backward、梯度或 optimizer。检查当前 shell 没有用外部
脚本覆盖它们：

```bash
env | rg 'OPENPI_USE_DEFAULT_CUDA_ALLOCATOR|OPENPI_LOG_MEMORY_STATS|PYTORCH_CUDA_ALLOC_CONF' || true
```

不要重新加入以下旧设置：

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
逐 update torch.cuda.memory_stats()
逐辅助模块遍历并同步 gradient norm
逐参数 EMA kernel
```

需要诊断显存时，可以为一次短跑显式设置 `OPENPI_LOG_MEMORY_STATS=1`，但不要用该结果作为正式
速度基线。

## 6. 静态检查

```bash
cd /workspace/vla/third_party/openpi

.venv/bin/python scripts/validate_policy_aux_launch_templates.py \
  --output /tmp/policy_aux_launch_templates.json

.venv/bin/pytest -q \
  src/openpi/training/pytorch_ema_test.py \
  src/openpi/models_pytorch/pi05_aux_queries_test.py

bash -n \
  scripts/policy_aux_gpu_common.sh \
  scripts/policy_aux_libero3_8gpu_common.sh
```

全部命令必须成功后才能启动 preflight。

## 7. 目标机器 25+1 preflight

三任务 P2：

```bash
cd /workspace/vla/third_party/openpi
mkdir -p /workspace/vla/logs/8gpu_preflight

EXP_NAME=p2_libero3_fp32base_bf16_8gpu_preflight_$(date -u +%Y%m%dT%H%M%SZ)
LOG=/workspace/vla/logs/8gpu_preflight/${EXP_NAME}.log

EXP_NAME="$EXP_NAME" scripts/preflight_p2_libero3_8gpu.sh >"$LOG" 2>&1
```

LIBERO-10 P1/P2 使用对应的 `preflight_p1_libero10_8gpu.sh` 和
`preflight_p2_libero10_8gpu.sh`。preflight 会训练 25 updates、保存 step 25、恢复并完成
step 26。

检查：

```bash
tr '\r' '\n' < "$LOG" | rg \
  'Training precision|Training config|step=[0-9]+ loss=|ema_updates=|Saved checkpoint|Successfully loaded|OutOfMemory|Traceback'
```

验收条件：

- `Training precision: bfloat16`；
- local batch 32、effective global batch 256；
- action/P1/P2 loss 和 global grad norm 均 finite；
- EMA 从 update 1 连续到 26；
- step 25 保存和 step 26 exact resume 成功；
- 排除首个 compile update 后，P2 应大致在 5–6 秒/update。不同 driver、CPU 和存储可能带来
  小幅差异；若稳定超过 8 秒，停止正式训练并重新检查第 5 节。

## 8. 正式启动

只有目标机器 preflight 通过后再启动。三任务 P2：

```bash
cd /workspace/vla/third_party/openpi
FULL_TRAINING_APPROVED=YES \
EXP_NAME=p2_libero3_fp32base_bf16_full_$(date -u +%Y%m%dT%H%M%SZ) \
scripts/launch_p2_libero3_8gpu.sh
```

不要把失败的 full-FP32 run 当作可恢复 checkpoint，也不要从 P1 checkpoint 初始化 P2。

## 9. 变更的训练影响

- FP32 conversion 会改变 base 权重的保存精度，因此属于明确记录的初始化 recipe 更新。
- BF16 finetuning、loss、梯度和 optimizer 逻辑保持不变。
- foreach EMA 与原公式逐元素等价，仍在每个 optimizer update 后更新一次。
- 单次 `clip_grad_norm_` 仍以全模型 global norm 1.0 裁剪；删除的只是额外诊断遍历。
- allocator 与显存统计调整只影响内存管理、同步和日志，不改变参数更新公式。
