# Ascend-Maze 与 Ray baseline：14 Workflow 对照报告

日期：2026-07-23

## 结论

- Ascend-Maze 与 Ray baseline 的 14 个迁移 Workflow 均执行成功，正式矩阵为
  `28/28` 成功；3 个代表 Workflow 的两轮复测为 `12/12` 成功。
- 去掉时间戳、trace 和运行期 ID 后，两条路径的最终业务输出 `14/14` 一致。
  这证明当前比较执行的是同一批 Workflow 和同一组模型结果。
- tau-bench 六个样本中 Ascend-Maze 均更快，平均 client E2E 比 Ray 少
  `7.361 s`，主要来自较低的 Worker 启动/调度等待。
- 文本 GAIA/OpenAGI 多数也由 Ascend-Maze 更快，但 `gaia.speech` 是例外。
- 三个视觉 Workflow 中，Ascend-Maze 的非模型时间比 Ray 高约 `43.3-47.9 s`，
  是当前最明确的剩余性能问题。它把 14 项简单平均结果拉成 Ascend-Maze
  比 Ray 慢 `6.062 s`；这个平均值不能代表统一负载下的吞吐结论。
- 系统执行成功不等于模型答案正确。GAIA 的 4 个单样本均未得到可确认的正确
  最终答案；OpenAGI 和 tau-bench 本次没有运行官方 evaluator，不能报告任务
  accuracy/reward。
- 测试结束后，8 张 NPU 均无运行进程；Ascend-Maze 的 Controller、RunDataIndex、
  DataHandle 和模型服务清理审计全部通过。

## 实验口径

两条路径使用相同的 Workflow、样本、模型、推理参数和 Worker 约束：

- 文本模型：`Qwen3-4B`，Transformers manual greedy，BF16；
- 文本参数：`max_tokens=4096`、`max_model_len=10240`、temperature `0`；
- 视觉模型：`Qwen2.5-VL-3B-Instruct`，vLLM-Ascend，BF16；
- 视觉参数：`max_tokens=4096`、`max_model_len=12288`、
  `max_num_batched_tokens=4096`、`max_num_seqs=1`；
- Ray baseline：普通 Ray Task/Actor，`max_calls=1`；
- tau-bench smoke overrides 关闭；
- 每个 Workflow 取一个固定样本，`sample_offset=0`；
- 正式矩阵并行使用全部 8 张卡：Ascend-Maze 使用物理卡 `0/2/4/6`，Ray baseline
  使用 `1/3/5/7`；代表复测第二轮交换两条路径的卡号；
- 正式 E2E 使用 `latency_metrics.client_e2e_ms`，从请求提交准备开始到结果返回；
- 模型时间使用 `model_request_ms`，非模型时间使用
  `client_e2e_minus_model_ms`。

视觉 vLLM 服务在样本计时开始前启动，两条路径口径一致，因此视觉 E2E 不包含
模型服务冷启动。当前记录保留了服务 PID、启动参数和停止结果，但没有独立的服务
启动耗时字段；不能从本报告推断视觉冷启动性能。

环境指纹：
`0dd3eecf97253ba211b4a6d34013d2cd4b0ebf20354e7fd89728caa64d2c7cb9`

源码基线：

- Git HEAD：`0d68788e0ba28f09102c617d26fa5b72a8288d54`
- 被跟踪文件的工作区 diff SHA-256：
  `64121f5982cf5ed7d37c9a6d959a56f4d65344ca7badd6000a58669dd650dd37`
- 实验运行于未提交工作区；原始 plan/record/summary 是本次结果的权威证据。

## 14 Workflow 正式矩阵

单位均为秒。`A-R` 小于零表示 Ascend-Maze 更快。

| Workflow | 类型 | A E2E | Ray E2E | A-R | A 模型 | Ray 模型 | A 非模型 | Ray 非模型 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gaia.file` | text | 78.279 | 83.920 | -5.641 | 74.125 | 74.718 | 4.154 | 9.202 |
| `gaia.reason` | text | 182.341 | 187.686 | -5.345 | 178.197 | 179.514 | 4.144 | 8.172 |
| `gaia.speech` | text | 500.430 | 497.147 | +3.283 | 493.663 | 487.218 | 6.767 | 9.929 |
| `gaia.vision` | vision | 266.873 | 216.250 | +50.623 | 213.098 | 210.341 | 53.775 | 5.909 |
| `openagi.document_qa` | text | 161.698 | 181.616 | -19.918 | 150.492 | 156.040 | 11.206 | 25.576 |
| `openagi.image_captioning_complex` | vision | 203.302 | 151.320 | +51.982 | 137.007 | 128.291 | 66.295 | 23.029 |
| `openagi.multimodal_vqa_complex` | vision | 319.322 | 256.638 | +62.684 | 257.282 | 237.984 | 62.040 | 18.654 |
| `openagi.text_processing_multilingual` | text | 174.466 | 183.102 | -8.636 | 161.925 | 158.017 | 12.541 | 25.085 |
| `tbench.airline_book` | text | 55.416 | 65.260 | -9.844 | 48.701 | 48.946 | 6.715 | 16.314 |
| `tbench.airline_cancel` | text | 57.324 | 66.255 | -8.931 | 48.671 | 50.655 | 8.653 | 15.600 |
| `tbench.retail_cancel` | text | 21.265 | 26.322 | -5.057 | 16.499 | 17.041 | 4.766 | 9.281 |
| `tbench.retail_cancel_modify` | text | 39.210 | 47.638 | -8.428 | 33.467 | 34.486 | 5.743 | 13.152 |
| `tbench.retail_modify` | text | 36.260 | 43.016 | -6.756 | 30.167 | 30.232 | 6.093 | 12.784 |
| `tbench.retail_return` | text | 28.203 | 33.351 | -5.148 | 21.391 | 20.885 | 6.812 | 12.466 |

分组简单平均如下。各 Workflow 的任务数和输出 token 数差异很大，因此这里只用于
定位路径差异，不能当作加权 benchmark 总分。

| 分组 | A E2E | Ray E2E | A-R | A 模型 | Ray 模型 | A 非模型 | Ray 非模型 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GAIA | 256.981 | 246.251 | +10.730 | 239.771 | 237.948 | 17.210 | 8.303 |
| OpenAGI | 214.697 | 193.169 | +21.528 | 176.677 | 170.083 | 38.020 | 23.086 |
| tau-bench | 39.613 | 46.974 | -7.361 | 33.149 | 33.708 | 6.464 | 13.266 |
| 全部 14 项 | 151.742 | 145.680 | +6.062 | 133.192 | 131.026 | 18.550 | 14.654 |

## 代表 Workflow 两轮复测

GAIA、OpenAGI 和 tau-bench 各选一个文本 Workflow。第二轮交换 Ascend-Maze
与 Ray 使用的物理 NPU，以降低单卡差异影响。`均值 +/- SD` 使用两轮样本标准差；
两轮只能展示波动，不能形成高置信度统计结论。

| Workflow | 指标 | A R1 | A R2 | A 均值 +/- SD | Ray R1 | Ray R2 | Ray 均值 +/- SD | 均值 A-R |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gaia.reason` | E2E | 135.366 | 139.978 | 137.672 +/- 3.261 | 140.220 | 133.282 | 136.751 +/- 4.906 | +0.921 |
| `gaia.reason` | 模型 | 130.889 | 132.513 | 131.701 +/- 1.148 | 132.114 | 125.120 | 128.617 +/- 4.946 | +3.084 |
| `gaia.reason` | 非模型 | 4.477 | 7.465 | 5.971 +/- 2.113 | 8.106 | 8.162 | 8.134 +/- 0.040 | -2.163 |
| `openagi.text_processing_multilingual` | E2E | 163.901 | 163.794 | 163.848 +/- 0.076 | 182.633 | 182.041 | 182.337 +/- 0.419 | -18.490 |
| `openagi.text_processing_multilingual` | 模型 | 151.137 | 151.207 | 151.172 +/- 0.049 | 157.719 | 157.886 | 157.803 +/- 0.118 | -6.631 |
| `openagi.text_processing_multilingual` | 非模型 | 12.764 | 12.587 | 12.676 +/- 0.125 | 24.914 | 24.155 | 24.535 +/- 0.537 | -11.859 |
| `tbench.airline_book` | E2E | 54.347 | 54.351 | 54.349 +/- 0.003 | 64.925 | 62.707 | 63.816 +/- 1.568 | -9.467 |
| `tbench.airline_book` | 模型 | 47.752 | 47.242 | 47.497 +/- 0.361 | 48.771 | 46.532 | 47.652 +/- 1.583 | -0.155 |
| `tbench.airline_book` | 非模型 | 6.595 | 7.109 | 6.852 +/- 0.363 | 16.154 | 16.175 | 16.165 +/- 0.015 | -9.313 |

`gaia.reason` 第一轮 Ascend-Maze 更快，交换卡后的第二轮 Ray 更快；两轮均值只差
`0.921 s`，小于模型时间的轮间波动。OpenAGI 与 tau-bench 两轮方向一致，且差距
主要来自非模型路径。

## Task 级拆分

下表是两轮均值，单位为秒。

| 指标 | GAIA A | GAIA Ray | OpenAGI A | OpenAGI Ray | tau A | tau Ray |
|---|---:|---:|---:|---:|---:|---:|
| Worker startup | 3.100 | 6.927 | 8.328 | 20.777 | 5.164 | 12.547 |
| Input fetch/binding | 1.067 | 0.000 | 2.452 | 0.000 | 1.900 | 0.000 |
| Callable | 131.743 | 128.634 | 151.223 | 157.820 | 48.083 | 48.218 |
| Output put | 0.420 | 0.012 | 0.392 | 0.040 | 0.265 | 0.155 |
| Dispatch prepare | 0.029 | 0.107 | 0.050 | 0.101 | 0.031 | 0.232 |
| Dispatch wait | 3.129 | 6.927 | 8.378 | 20.777 | 5.195 | 12.547 |
| Model load | 39.116 | 39.193 | 52.758 | 52.447 | 25.231 | 25.563 |
| Generation | 90.823 | 87.860 | 96.301 | 102.910 | 21.076 | 21.061 |
| Model cleanup | 1.535 | 1.330 | 1.829 | 2.138 | 1.035 | 0.876 |

这些行不能相加：`dispatch_wait` 包含 `worker_startup`，`callable` 包含 chat/model
request。Ray 在进入 Task 函数前物化 ObjectRef 参数，因此其 `input_fetch=0` 是
计时边界差异，相关工作包含在 Ray 的 startup/dispatch 路径中。

代表复测显示，Ascend-Maze 的 `output_put` 已降到 `0.265-0.420 s`，不是当前文本
路径的主要瓶颈。OpenAGI 和 tau-bench 的优势主要来自 Worker startup；GAIA 的
模型生成波动足以覆盖不到一秒的 E2E 均值差。

## DataStore 审计

优化后的普通 submission input 使用 staged handle identity，不再对用户大对象做
隐式内容 canonicalization。两轮代表复测均记录：

| Workflow | 每轮 stage | 每轮 tombstone | submission canonicalize | submission Ray put 均值 | runtime output Ray put 均值 | 全部 Ray put 均值 |
|---|---:|---:|---:|---:|---:|---:|
| `gaia.reason` | 37 | 37 | 0.000 s | 0.015 s | 0.129 s | 0.180 s |
| `openagi.text_processing_multilingual` | 90 | 90 | 0.000 s | 0.014 s | 0.184 s | 0.232 s |
| `tbench.airline_book` | 55 | 55 | 0.000 s | 0.047 s | 0.134 s | 0.201 s |

每轮结束时 `active_count=0`、`adopted_count=0`，stage 与 tombstone 数量相等。
因此此前约 `3.5 s` 的 submission canonicalization 已被消除，且没有通过服务端
缓存用户内容来换取结果。

## 正确性

正确性分为两个层次：

1. 系统执行正确性：两条路径均为 `14/14` 成功，最终业务输出 `14/14` 一致。
2. 模型/任务正确性：需要数据集 evaluator 或明确 expected answer，不能由
   `RunState.SUCCEEDED` 推断。

GAIA 单样本观察如下：

| Workflow | expected | 两条路径最终结果 | 判断 |
|---|---|---|---|
| `gaia.file` | `Time-Parking 2: Parallel Universe` | `cannot determine` | 不正确；本次 file smoke 明确只传文件摘要，不是 accuracy 测试 |
| `gaia.reason` | `3` | `1` | 不正确 |
| `gaia.speech` | 指定 5 种配料 | 生成了不同配料列表 | 不正确 |
| `gaia.vision` | `17.056` | 输出达到 4096 token，未形成可解析 final answer | 不正确/不可解析 |

OpenAGI 和 tau-bench 的本次 sample record 没有可直接使用的 expected answer，且没有
运行官方 evaluator。tau-bench 的业务动作成功完成只能说明执行路径成立，不能替代
官方 reward。后续 accuracy/reward 实验应独立于本次系统路径与性能报告。

## Worker 与资源回收

- Ray baseline 的 14 条正式记录都声明 `worker_max_calls=1`。
- Ray baseline 共执行 96 个 DAG Task；每个 Workflow 内所有 Task 的 Worker PID
  均不同，验证没有在普通 Ray Worker 进程间复用 Task。
- 文本模型记录包含每次请求的 model load、generation、cleanup 和 Worker PID。
- 三个视觉 Workflow 每条路径各启动一个独立 vLLM 服务，均记录启动 PID；Ray
  服务停止 return code 为 0，Ascend-Maze 由 Controller shutdown 回收。
- Ascend-Maze：`14/14` Controller shutdown `cleanup_confirmed=true`；`14/14`
  destroy tombstone `destroy_succeeded=true`；`14/14` DataStore 终态 active/adopted
  计数为零；无 residual vLLM。
- Ray baseline：无 cleanup error、无 residual vLLM。
- 全部正式矩阵与代表复测结束后，`npu-smi info` 显示 8 张 NPU 均无运行进程，
  HBM 回到约 `3205-3210 MiB` 的空闲基线。

## 实现回归与真实硬件验收

正式矩阵完成后，又对当前实现进行了独立回归和真实硬件验收：

- CPU/Fake 单元测试：`504 passed`；
- Ray Host 集成测试：`62 passed`；
- 真实 Ascend 测试：`19 passed`，包括进程绑定、OOM/故障、HBM 回落以及
  vLLM-Ascend 的 chat、drain、restart、crash 和启动失败回收；
- mypy：152 个源码文件无问题；
- 本次清理修复涉及文件的 ruff 检查通过；全目录 ruff 仍报告一个本次未修改的
  `workflows/_common.py:225` 旧未使用变量；
- 测试结束后再次检查，Ray 未初始化，无 raylet、GCS、vLLM、Controller 或测试服务
  残留；8 张 NPU 均无进程，HBM 为 `3205-3211 MiB`。

真实服务测试还验证并修复了一个故障路径竞态：NodeAgent 已先释放同一个完整
`PortLease` 时，Controller 侧重复 release 现在按幂等成功处理，但未知或身份不一致
的 Lease 仍被拒绝；若并发清理已经把 ModelInstance 推进到 `STOPPED`，迟到异常不再
尝试非法的 `STOPPED -> FAILED` 转换。这个修复只改变资源清理路径，没有修改
Workflow、`@task` 定义、模型参数或正式实验计时结果。

## 已修复问题与排除项

正式成功结果之前保留了以下失败证据：

- `.pre_multicard_fix_failure`：物理 NPU ID 与进程内逻辑设备映射错误；
- `.pre_timeout_failure`：视觉长请求 timeout；
- `.pre_kv_cache_failure`：manual greedy 未正确复用 KV cache；
- `.pre_owned_cleanup_failure`：并行 runner 把其他 runner 的进程误判为残留；
- 代表复测 `run2.pre_acl_env_failure`：重启命令覆盖 CANN `PYTHONPATH`，预检时
  无法 import `acl`。该次没有启动 Ray、没有占用 NPU，已从正式统计排除。

对应修复包括：物理卡通过 `ASCEND_RT_VISIBLE_DEVICES` 映射后在子进程内使用
`npu:0`；manual greedy 首步使用完整 prompt、后续只输入新 token 并复用
`past_key_values`；残留扫描只检查当前 runner 拥有的进程组；PortLease release
采用 generation 精确的幂等语义，并允许已经完成的 `STOPPED` 终态赢得并发清理。

## 剩余工作

当前 14 个 Workflow 的单样本系统路径已经闭合，但还不能称为完整性能或准确率
benchmark。下一阶段应按以下顺序推进：

1. 给视觉路径增加 service-ready、submission、dispatch、首请求和清理阶段的独立
   计时，定位 Ascend-Maze 多出的 `43.3-47.9 s` 非模型时间。
2. 修复视觉计时问题后，对文本和视觉分别运行 batch-like 与 arrival-like 多样本
   实验，报告吞吐、排队、P50/P95/P99 和资源利用率。
3. 接入 GAIA、OpenAGI、tau-bench 官方 evaluator，单独报告 accuracy/reward；不要
   把系统成功率当作模型正确率。
4. 增加更多重复次数，并串行/交叉换卡复测，以分离 NPU 和主机并发波动。

## 原始证据

- Ascend-Maze 正式矩阵：
  `experiments/qwen_benchmark_smoke/all14_goal_20260723_v1`
- Ray baseline 正式矩阵：
  `experiments/ray_baseline_smoke/all14_goal_20260723_v1`
- Ascend-Maze 代表复测：
  `experiments/qwen_benchmark_smoke/representative_2x_goal_20260723_v1`
- Ray baseline 代表复测：
  `experiments/ray_baseline_smoke/representative_2x_goal_20260723_v1`
