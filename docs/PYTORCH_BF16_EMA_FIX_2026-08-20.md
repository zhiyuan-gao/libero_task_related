# PyTorch BF16 EMA precision fix（2026-08-20）

## 问题

旧 `openpi.pytorch_ema.v1` 在初始化 shadow 时直接 clone 模型参数，因此 BF16
finetuning 会得到 BF16 EMA shadow。对于 `decay=0.999`，单步 EMA 增量经常小于
BF16 在当前权重尺度上的一个 ULP；每一步就地舍入可能使 shadow 长时间不动，而不是
只在最终保存时产生一次 BF16 量化误差。

本机 P1/P2 旧 checkpoint 中的 `model.safetensors` 因此不能用于 serving/evaluation。
RAW `train_model.safetensors` 不受 EMA shadow 累积精度影响。

## 当前默认与验证状态

自 2026-08-20 起，本项目自研 PyTorch P1/P2 配置默认 `ema_decay=None`，训练、恢复、
评估和部署均使用 optimizer 更新后的 RAW 权重。官方 `pi05_libero` 配置保持原样，
没有被改写。

下面的 FP32 EMA v2 是**实验实现**。目前只通过 CPU 单元测试（更新公式、dtype、保存/
恢复与旧 schema 拒绝）；尚未通过 8×A100 的 25+1 checkpoint/resume preflight、长程
训练或闭环成功率评估。因此当前正式 P1/P2 训练不得启用它，也不得将其描述为已验证的
训练 recipe。若未来要启用，必须先完成上述三类验证并单独记录结果。

## 实验性修复

`openpi.pytorch_ema.v2` 采用常见的 mixed-precision EMA 数值模式：

- BF16/FP16 模型参数对应的 shadow 初始化为 FP32；
- shadow 仍在每个 optimizer update 后更新一次，decay 和更新公式不变；
- 继续使用 `torch._foreach_*` 批量 kernel；FP32 destination 与 BF16/FP16 source
  做 mixed-dtype update；
- FP32/FP64、complex 和非浮点参数保持原 dtype；
- 保存的 EMA `model.safetensors` 为 FP32，加载进 BF16 serving model 时才做一次
  最终 cast；同一个 FP32 文件也用于 exact resume，避免丢失 EMA 累积状态；
- metadata schema 升级为 `openpi.pytorch_ema.v2` 并记录所有 shadow dtype，旧 v1
  checkpoint 会被明确拒绝，不能被静默上采样后继续训练。

这不改变 action/P1/P2 loss、optimizer、梯度、学习率、EMA decay 或更新频率。代价是
每张 DDP GPU 上的 EMA shadow 从约 7.5 GB 增加到约 15 GB，EMA checkpoint 文件也会
从约 7.5 GB 增加到约 15 GB。若重新启用，必须先做 25+1 preflight，确认显存、保存和
step 26 exact resume；该项目前尚未执行。

## 旧 checkpoint 迁移规则

旧 v1 BF16 EMA 没有可靠方式恢复已经在 3,209 次 update 中丢失的低位增量，不能通过
`.float()` 修复。继续训练时应：

1. 加载 `train_model.safetensors` RAW 权重；
2. 按当前默认保持 no-EMA，并直接保存 RAW `model.safetensors`；
3. 若未来经验证后选择启用 v2，则从 RAW 权重重新建 FP32 shadow，并把 EMA update count
   作为新阶段重新开始，不能声称与旧 EMA exact-continuation；
4. 无论是否启用 EMA，都保留 optimizer 和 rank-local RNG/data-loader state；启用时再
   额外保留 RAW train weights 与 FP32 EMA weights。

若要求严格延续原优化轨迹，应将旧 RAW 权重视为新实验阶段的 initialization，并在
provenance 中记录边界，而不是绕过 v2 schema 检查。

## 参考实现

- PyTorch `AveragedModel` 官方支持使用 `get_ema_multi_avg_fn` 做批量 EMA；
- Hugging Face Diffusers 的 mixed-precision 训练文档明确把 EMA 作为模型参数的额外
  full-precision copy，并提供 foreach EMA；
- `fadel/pytorch_ema` 提供将内部 EMA state 显式转换到指定 floating dtype 的 `.to()`。

本实验实现保留项目现有的按参数名 strict topology、原子 checkpoint 和 exact-resume
语义，只采用上述实现共有的“全精度 shadow + batched update”数值策略。这里的参考
只说明数值策略有先例，不构成对本项目实现正确性或模型表现的验证。
