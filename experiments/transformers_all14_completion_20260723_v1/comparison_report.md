# Ascend-Maze 与 Ray baseline：14 Workflow 纯 Transformers 对比

日期：2026-07-23

## 结论

- 14 个迁移 Workflow 在 Ascend-Maze 与 Ray baseline 上均成功完成，合计
  `28/28` 次成功。
- 两条路径共执行 `124` 次模型调用，输入/输出 token 数逐项一致；去除最终业务
  输出中的 `start_time`、`end_time` 后，`14/14` 个 Workflow 的退出输出完全一致。
- 统一使用 Transformers manual-greedy、`max_tokens=4096`、temperature `0`；
  Ray baseline 的每个 Ray Task 均为 `max_calls=1`。
- 14 项单样本的简单平均中，Ascend-Maze E2E 为 `152.637 s`，Ray 为
  `159.421 s`，Ascend-Maze 少 `6.784 s`。文本和视觉分组中，Ascend-Maze
  分别少 `6.592 s` 和 `7.487 s`。
- 这些数字证明当前系统路径已经可以执行相同 Workflow、模型和输入，并得到相同
  业务输出。它们不是吞吐 benchmark，也不能替代数据集 accuracy/reward evaluator。

## 实验口径

- 文本模型：`Qwen3-4B`，BF16，`max_model_len=10240`。
- 视觉模型：`Qwen2.5-VL-3B-Instruct`，BF16，
  `max_model_len=12288`。
- 推理方式：两边均为 Transformers manual-greedy。
- 生成参数：`max_tokens=4096`、temperature `0`。
- Ray 约束：普通 Ray Task，`max_calls=1`。
- 每个 Workflow 使用 `sample_offset=0` 的固定单样本。
- E2E 使用 `latency_metrics.client_e2e_ms`，覆盖提交准备、提交、调度、Task 执行
  和最终结果返回，不包含事后的详细证据采集与 `destroy_run`。
- 模型时间使用 `model_request_ms`；非模型时间为
  `client_e2e_minus_model_ms`。
- Transformers 模型冷加载发生在模型 Task 的第一次 `chat()` 内，因此计入 E2E
  和模型时间。同一 Task 内的后续顺序 `chat()` 复用该 Task 的模型 session；Task
  结束后显式释放模型。

本表复用了 10 个只有单次 chat/Task 的既有文本结果。它们不受“同一 Task 内复用
模型 session”修复影响。`openagi.document_qa` 是既有文本结果中唯一包含同一 Task
多次 chat 的 Workflow，已在修复后重新运行。`gaia.vision` 使用纯 Transformers
三轮验证的 R1；两个 OpenAGI 复杂视觉 Workflow 使用本目录下的新结果。

## 14 Workflow 结果

单位均为秒。`M-R` 小于零表示 Ascend-Maze 更快。`calls (cold/reused)` 表示模型
调用总数，以及其中 Task 首次冷加载和 Task 内复用次数。输出 SHA 是规范化退出
业务对象 SHA-256 的前 12 位。

| Workflow | 类型 | Maze E2E | Ray E2E | M-R | Maze 模型 | Ray 模型 | Maze 非模型 | Ray 非模型 | calls (cold/reused) | 输出 SHA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gaia.file` | text | 78.279 | 83.920 | -5.641 | 74.125 | 74.718 | 4.154 | 9.202 | 3 (3/0) | `fe03dc8ec3c3` |
| `gaia.reason` | text | 182.341 | 187.686 | -5.345 | 178.197 | 179.514 | 4.144 | 8.172 | 3 (3/0) | `26b2895cac41` |
| `gaia.speech` | text | 500.430 | 497.147 | +3.283 | 493.663 | 487.218 | 6.767 | 9.929 | 4 (4/0) | `78d1bce20f9f` |
| `gaia.vision` | vision | 89.185 | 89.267 | -0.082 | 85.526 | 83.457 | 3.659 | 5.810 | 1 (1/0) | `1377a17de849` |
| `openagi.document_qa` | text | 111.309 | 123.318 | -12.009 | 98.442 | 97.942 | 12.867 | 25.376 | 21 (4/17) | `e6496aac42fa` |
| `openagi.image_captioning_complex` | vision | 309.552 | 327.293 | -17.741 | 295.960 | 302.136 | 13.592 | 25.157 | 60 (6/54) | `5c2de903215c` |
| `openagi.multimodal_vqa_complex` | vision | 453.678 | 458.315 | -4.637 | 443.123 | 438.778 | 10.555 | 19.537 | 20 (4/16) | `cb5d2f20ba92` |
| `openagi.text_processing_multilingual` | text | 174.466 | 183.102 | -8.636 | 161.925 | 158.017 | 12.541 | 25.085 | 4 (4/0) | `95e40c876307` |
| `tbench.airline_book` | text | 55.416 | 65.260 | -9.844 | 48.701 | 48.946 | 6.715 | 16.314 | 2 (2/0) | `adc86ec88414` |
| `tbench.airline_cancel` | text | 57.324 | 66.255 | -8.931 | 48.671 | 50.655 | 8.653 | 15.600 | 2 (2/0) | `ad3ad855b9d9` |
| `tbench.retail_cancel` | text | 21.265 | 26.322 | -5.057 | 16.499 | 17.041 | 4.766 | 9.281 | 1 (1/0) | `944c929950bc` |
| `tbench.retail_cancel_modify` | text | 39.210 | 47.638 | -8.428 | 33.467 | 34.486 | 5.743 | 13.152 | 1 (1/0) | `632f19b5d10d` |
| `tbench.retail_modify` | text | 36.260 | 43.016 | -6.756 | 30.167 | 30.232 | 6.093 | 12.784 | 1 (1/0) | `6be5a823b6ac` |
| `tbench.retail_return` | text | 28.203 | 33.351 | -5.148 | 21.391 | 20.885 | 6.812 | 12.466 | 1 (1/0) | `c2a4a53a9937` |

## 分组汇总

分组结果是 Workflow 单样本的简单平均。不同 Workflow 的 Task 数、图片数、prompt
和输出长度差异很大，因此不能把该平均值解释成加权总分或负载吞吐结论。

| 分组 | 数量 | Maze E2E | Ray E2E | M-R | Maze 模型 | Ray 模型 | Maze 非模型 | Ray 非模型 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GAIA | 4 | 212.559 | 214.505 | -1.946 | 207.878 | 206.227 | 4.681 | 8.278 |
| OpenAGI | 4 | 262.251 | 273.007 | -10.756 | 249.862 | 249.218 | 12.389 | 23.789 |
| tau-bench | 6 | 39.613 | 46.974 | -7.361 | 33.149 | 33.707 | 6.464 | 13.266 |
| 文本 | 11 | 116.773 | 123.365 | -6.592 | 109.568 | 109.059 | 7.205 | 14.306 |
| 视觉 | 3 | 284.138 | 291.625 | -7.487 | 274.870 | 274.790 | 9.269 | 16.835 |
| 全部 | 14 | 152.637 | 159.421 | -6.784 | 144.990 | 144.573 | 7.647 | 14.848 |

模型时间的全部 14 项平均仅相差 `0.417 s`，明显小于 Workflow 间和单轮模型生成
波动；这符合两条路径调用同一 Transformers 实现的预期。当前 E2E 差异主要来自
非模型路径，其中 Ascend-Maze 平均少 `7.201 s`。但本次各 Workflow 只有固定单样本，
尚不足以形成稳定的性能优劣结论。

## Task 内模型复用

三个 Workflow 会在同一个模型 Task 内顺序调用多次 `chat()`。两条路径都遵循
“每个模型 Task 冷加载一次，Task 内复用，Task 结束清理”的相同生命周期。

| Workflow | 模型 Task | 调用 | 冷加载 | Task 内复用 | 输入 token | 输出 token |
|---|---:|---:|---:|---:|---:|---:|
| `openagi.document_qa` | 4 | 21 | 4 | 17 | 17,652 | 463 |
| `openagi.image_captioning_complex` | 6 | 60 | 6 | 54 | 31,798 | 2,413 |
| `openagi.multimodal_vqa_complex` | 4 | 20 | 4 | 16 | 9,182 | 4,700 |

全部 14 项合计为 `124` 次调用、`37` 次 Task 冷加载和 `87` 次 Task 内复用；两边
数字完全一致。Ray 的 `max_calls=1` 仍然成立：它阻止 Ray Worker 跨 Task 复用，
不阻止一个用户 Task 在自身生命周期内顺序调用多次 `chat()`。

## 输出一致性

输出比较以 Ascend-Maze 记录的 exit Task 集合为准，在 Ray 的同名 Task 结果中取出
相同对象，只移除业务结构里的 `start_time` 和 `end_time`，然后按排序键编码 JSON。
结果为 `14/14` 字节一致。没有忽略模型文本、业务状态、特征字段或 Task 结果。

四个含 `final_answer` 的本次重点结果，其最终文本 SHA-256 为：

| Workflow | final_answer SHA-256 |
|---|---|
| `gaia.vision` | `d320979f5fadc0e760d0b3c54a10dca2cc62f5129b57c40f0100c2da994a8636` |
| `openagi.document_qa` | `885b15ac13b4be89443121becd44fc90cd6488843bba621743ebfac35f3a2b99` |
| `openagi.image_captioning_complex` | `2b9bfd6ff2505c8995fe0b39e316954ad038ffd4a61ad46bd1530662ce48179d` |
| `openagi.multimodal_vqa_complex` | `3151498795daf8c53f9855d4b518cba9e7fcbe49cdb20beb293c81627e972a3a` |

输出一致只证明两条执行路径没有改变本次模型和 Workflow 的结果。它不证明结果满足
数据集 expected answer。GAIA、OpenAGI 和 tau-bench 的正式 accuracy/reward 仍需
各自 evaluator 单独验证。

## 数据与资源回收

逐项审计 14 对记录得到：

- Ascend-Maze `14/14` 的 `RunDataIndex` 终态均为 `active_count=0`、
  `adopted_count=0`。
- Ascend-Maze `14/14` 的 DataStore `stage_count == tombstone_count`，且 destroy
  tombstone 均为 `destroy_succeeded=true`。
- 两边 `28/28` 的 summary 均无 cleanup error、无 residual vLLM process。
  本报告使用 Transformers，字段名沿用通用 runner schema。
- 最终 `npu-smi info` 显示 8 张 NPU 均无运行进程，HBM 为
  `3206-3210 MiB` 的空闲基线。
- 最终进程扫描没有 raylet、GCS、Qwen 模型 Worker、Ascend-Maze Controller 或
  benchmark runner 残留。

## 证据范围

本报告由三个已完成实验集合组成：

- 10 个单次 chat/Task 文本 Workflow：
  `experiments/{qwen_benchmark_smoke,ray_baseline_smoke}/all14_goal_20260723_v1`
- `gaia.vision` 三轮纯 Transformers 验证，本表采用 R1：
  `experiments/vision_transformers_gaia_3x_20260723_v1`
- `openagi.document_qa` 和两个复杂视觉 Workflow：
  `experiments/transformers_all14_completion_20260723_v1/{maze,ray}`

每个目录都保留 `plan.json`、样本 record、summary 和失败/清理字段。当前 Git HEAD 为
`0d68788e0ba28f09102c617d26fa5b72a8288d54`，实验运行于未提交工作区；当前
`src/workflows/tools/tests/tests_ray_host` 源码树摘要为
`fb97635b206b30440a9a0b6beb8a427b37a6a33b8188042c96db18064025b370`。
因此原始 plan/record/summary 是运行事实的权威证据，不能用 Git HEAD 单独重建本次
全部结果。

## 下一阶段

纯 Transformers 的固定单样本系统路径已经闭合。下一阶段应停止继续针对单个
Workflow 做烟雾式修补，转入正式实验设计：

1. 冻结并提交当前实现、配置和本报告，建立可复现实验基线。
2. 分别为文本与视觉运行 batch-like 和 arrival-like 多样本负载，报告吞吐、排队、
   P50/P95/P99、失败率和 NPU 利用率。
3. 对性能实验至少增加重复轮次和换卡/交叉顺序，避免把生成波动解释成系统收益。
4. 独立接入 GAIA、OpenAGI、tau-bench evaluator，报告 accuracy/reward；不要把
   `RunState.SUCCEEDED` 当作模型正确率。
