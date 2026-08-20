# P1 三任务训练机器最终交接（2026-08-20）

本文档冻结本机完成的 P1 三任务训练、标准 LIBERO-10 闭环评估和
LIBERO-Plus 零样本评估。后续工作不应依赖本机仍然在线。

## 1. 代码身份

- GitHub：`https://github.com/zhiyuan-gao/libero_task_related`
- 交接分支：`p1-training-this-machine-8xa100-20260819`
- 本分支包含此前的 8-GPU π0.5 工程优化、P1/P2 实现、标准三任务并行评估和
  LIBERO-Plus 三任务评估入口。
- 本机只推送该分支，不自动合并到 `main`；应在另一台机器验证后再决定是否合并。

离线 Git bundle 也保存在迁移目录的 `code/` 下。优先从 GitHub 分支恢复；GitHub
不可用时再使用 bundle。

## 2. 冻结的 P1 训练集合与 recipe

训练集合来自 `physical-intelligence/libero` revision
`a4336d589d589045d1c56423ffdf3b88a0e19b1f`：

| LeRobot task index | Task | Episodes | Policy frames |
| ---: | --- | ---: | ---: |
| 0 | put the white mug on the left plate and put the yellow and white mug on the right plate | 38 | 9,807 |
| 3 | turn on the stove and put the moka pot on it | 41 | 10,866 |
| 8 | put the black bowl in the bottom drawer of the cabinet and close it | 35 | 8,577 |
| 合计 |  | 114 | 29,250 |

正式训练设置：

| 设置 | 值 |
| --- | --- |
| Config | `pi05_libero3_p1_aux` |
| Initialization | FP32-converted official π0.5 base |
| Finetuning compute dtype | BF16 |
| Optimizer updates | 3,209 |
| Warmup updates | 1,069 |
| GPUs | 8 × A100 80 GB |
| Global batch | 256 |
| Effective epochs | about 28.09 |
| Typical steady-state speed | about 4.8–5.5 s/update, about 47 samples/s |

训练从 2026-08-19 16:34:18 UTC 到 21:25:15 UTC，约 4 小时 50 分钟。
日志中的 loss 从 step 0 的 `1.4806` 降到 step 3200 的 `0.0197`；中间值包括
step 1000 `0.0601`、step 2000 `0.0313`、step 3000 `0.0205`。

## 3. Checkpoint：必须使用 RAW 权重

迁移目录中有两个视图：

- `checkpoints/p1_libero3_3209/raw_serving_checkpoint/`：评估和部署使用。
- `checkpoints/p1_libero3_3209/full_training_checkpoint/`：保留 optimizer、随机状态、
  data-loader metadata 等训练恢复材料。

RAW 权重为 `train_model.safetensors`，SHA-256：

```text
293d26e0e50499731ed38de7baea7b133d2f1c09e7203eb8386cff1b0f89bbed
```

`full_training_checkpoint/model.safetensors` 是受 BF16 EMA 累积精度问题影响的
shadow，SHA-256：

```text
0e3d2d6947ed4e463c79955449d0179ea23d2925cd1e564ba5fd04ec49d0b0ef
```

它不能用于评估或部署。若继续训练，应显式载入 RAW `train_model.safetensors`
作为 trainable weights。当前自研 P1/P2 默认关闭 EMA；不要继续载入旧的 BF16 EMA
shadow。仓库中的 FP32 EMA v2 仅通过单元测试，尚未通过 8-GPU preflight、长程训练
和闭环评估，不属于当前正式 recipe。原因、实验实现和旧 checkpoint 迁移规则见
`docs/PYTORCH_BF16_EMA_FIX_2026-08-20.md`；迁移目录中的 `CHECKPOINT_README.md`
也重复记录了这一要求。

## 4. 标准 LIBERO-10 三任务闭环结果

P1 RAW checkpoint 在相同 seed、初始状态、horizon、`replan=5`、每任务 50 次协议下：

| Task | P1 RAW | Official `pi05_libero` comparison |
| --- | ---: | ---: |
| moka pot | 50/50 (100%) | 48/50 (96%) |
| black bowl | 49/50 (98%) | 49/50 (98%) |
| two mugs | 44/50 (88%) | 47/50 (94%) |
| 合计 | 143/150 (95.33%) | 144/150 (96.00%) |

完整视频、逐 rollout 结果和失败分析都在迁移目录的 `artifacts/eval/` 与
`artifacts/logs/eval/` 中。

## 5. LIBERO-Plus 零样本评估

协议冻结为三项训练任务派生出的全部 872 个 Plus variants，每个 variant 使用第一个
官方初始状态各跑一次：8 路 GPU/仿真并行、seed 7、输入 resize 224、`replan=5`、
最大 520 policy steps。评估按用户决定提前停止；以下是部分结果，不是完整 benchmark
结果：

<!-- LIBERO_PLUS_RESULT_START -->
| Task | Completed | Successes | Partial rate |
| --- | ---: | ---: | ---: |
| bowl | 184 | 44 | 23.91% |
| moka | 203 | 43 | 21.18% |
| mugs | 171 | 33 | 19.30% |
| 合计 | 558/872 | 120 | 21.51% |
<!-- LIBERO_PLUS_RESULT_END -->

variant 按类别排列，因此 aggregate 不是完整 872 variants 的无偏估计；Light Conditions
和 Objects Layout 未开始，Sensor Noise 仅完成 38/144。详细类别统计和解释见
`docs/P1_LIBERO_PLUS_PARTIAL_EVAL_2026-08-20.md`，机器可读汇总见
`docs/results/p1_libero_plus_partial_summary_2026-08-20.json`。GitHub 不上传 rollout 视频。

本机原始目录含逐 variant JSONL、summary、日志和视频；Google Drive 精简迁移包只保留
结构化结果与日志，不含 LIBERO-Plus 视频：

```text
artifacts/eval/libero_plus/p1_raw_formal_872_20260819T234400Z/
artifacts/logs/eval/p1_libero_plus_formal_872_20260819T234400Z/
```

## 6. 迁移目录

本机集中目录：

```text
/workspace/vla/p1_training_this_machine_transfer_20260819
```

其中包括：

- `checkpoints/`：完整训练恢复 checkpoint 和 RAW serving checkpoint；
- `data/`：P1/P2 辅助监督、calibration、manifests 和 provenance；
- `artifacts/eval/`：标准评估、官方同协议对照和失败分析；Google Drive 精简包另含
  LIBERO-Plus 的结构化结果，但不含其视频；
- `artifacts/logs/`：训练、preflight、速度诊断和评估日志；
- `code/`：此分支的离线 Git bundle；
- `SHA256SUMS`：迁移文件的完整校验表；
- `PUBLIC_ASSETS_NOT_INCLUDED.md`：未重复打包的公开资产及精确 revision/hash。

目录使用 hardlink 避免 RAW 权重与完整 checkpoint 重复占空间。迁移时保留 hardlink：

```bash
rsync -aH --info=progress2 \
  /workspace/vla/p1_training_this_machine_transfer_20260819/ \
  USER@DEST_HOST:/DEST/PATH/p1_training_this_machine_transfer_20260819/
```

传输后校验：

```bash
cd /DEST/PATH/p1_training_this_machine_transfer_20260819
sha256sum -c SHA256SUMS
git bundle verify code/*.bundle
```

## 7. 未打包的可重下载资产

为了避免无意义地增加迁移体积，没有重复打包 LeRobot/Hugging Face cache、官方
π0.5 base、LIBERO-Plus 的约 9.5 GB 展开资产、Python virtualenv 或系统依赖。
精确 source revision、资产 SHA-256 和大小见
`PUBLIC_ASSETS_NOT_INCLUDED.md`。RAW serving checkpoint 自带所需的 LIBERO norm stats。
