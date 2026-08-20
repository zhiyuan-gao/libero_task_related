# P1 RAW LIBERO-Plus 部分评估（2026-08-20）

本次评估按用户决定提前停止，不是完整的 872-variant 正式结果。

- checkpoint：P1 RAW step 3209
- 开始：2026-08-19 23:44:26 UTC
- 停止：2026-08-20 00:37:45 UTC
- 协议：8 路 GPU/仿真并行，seed 7，resize 224，`replan=5`，最多 520 policy steps
- 完成：558/872（64.0%）
- 成功：120/558（21.51%）
- runtime errors：0

## 按基础任务

| Task | Completed | Successes | Partial success rate |
| --- | ---: | ---: | ---: |
| bowl | 184 | 44 | 23.91% |
| moka | 203 | 43 | 21.18% |
| mugs | 171 | 33 | 19.30% |

## 按类别

| Category | Completed | Successes | Partial success rate |
| --- | ---: | ---: | ---: |
| Background Textures | 122 | 53 | 43.44% |
| Camera Viewpoints | 142 | 4 | 2.82% |
| Language Instructions | 130 | 50 | 38.46% |
| Robot Initial States | 126 | 13 | 10.32% |
| Sensor Noise | 38 | 0 | 0.00% |

variant 顺序按类别分组，因此 120/558 不能作为完整 benchmark 成功率的无偏估计。
Light Conditions 和 Objects Layout 尚未开始，Sensor Noise 只完成 38/144；上表只描述
实际完成的 variants。评估未出现 runtime error，停止原因是当前部分结果明显低于预期，
由用户决定不再消耗机器时间。

仓库只保留本报告和结构化汇总，不提交 558 个 rollout 视频。迁移包中另保留逐 variant
JSONL、8 个 shard JSONL、manifest 和运行日志，便于复核。
