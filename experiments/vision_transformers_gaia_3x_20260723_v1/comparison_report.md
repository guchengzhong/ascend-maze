# GAIA 视觉 Workflow：Ascend-Maze 与 Ray baseline 对比

日期：2026-07-23

## 结论

- `gaia.vision` 在 Ascend-Maze 和 Ray baseline 上各运行 3 次，`6/6` 成功。
- Ascend-Maze 平均 client E2E 为 `89.530 +/- 0.586 s`，Ray baseline 为
  `91.006 +/- 1.534 s`；本次样本中 Ascend-Maze 平均快 `1.475 s`（`1.62%`）。
- Ascend-Maze 的模型请求平均慢 `0.776 s`，但非模型路径平均快 `2.252 s`，最终
  E2E 略快。模型生成本身存在约一秒级波动，因此不能把 `1.475 s` 解释成稳定的
  吞吐优势。
- 两条路径每轮均使用 `max_tokens=4096`、temperature `0`，并自然生成 `885`
  个输出 token 后遇到 EOS。`4096` 是最大输出上限，不要求模型强制生成 4096 token。
- 6 次最终输出的 SHA-256 完全一致：
  `d320979f5fadc0e760d0b3c54a10dca2cc62f5129b57c40f0100c2da994a8636`。

## 实验口径

- Workflow：`gaia.vision`
- 样本：`gaia.vision.df6561b2-7ee5-4540-baab-5095f742716a`
- 模型：`Qwen2.5-VL-3B-Instruct`
- 推理：Transformers manual greedy、BF16、单张 Ascend 910B3 NPU
- 参数：`max_tokens=4096`、`max_model_len=12288`、temperature `0`
- 输入：`434` token 和 1 张相同图片
- 输出：每轮 `885` token
- Ray baseline：普通 Ray Task，`max_calls=1`
- 运行顺序：Ray R1、Ascend-Maze R1、Ray R2、Ascend-Maze R2、Ray R3、
  Ascend-Maze R3，均使用物理 NPU 0
- E2E：从请求开始提交/派发到退出 Task 结果返回，不含 Workflow 构建、详细证据采集
  和 `destroy_run`

表中的 `+/-` 是 3 次运行的样本标准差；`Maze - Ray` 小于零表示 Ascend-Maze
更快。本报告只比较性能，不评价模型答案准确率。

## 逐轮结果

单位均为秒。

| 轮次 | Maze E2E | Ray E2E | Maze - Ray | Maze 模型 | Ray 模型 | Maze 非模型 | Ray 非模型 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 89.185 | 89.267 | -0.082 | 85.526 | 83.457 | 3.659 | 5.810 |
| R2 | 90.207 | 91.581 | -1.374 | 86.653 | 85.733 | 3.554 | 5.848 |
| R3 | 89.199 | 92.169 | -2.970 | 85.648 | 86.308 | 3.551 | 5.861 |

## 汇总结果

| 指标 | Ascend-Maze | Ray baseline | Maze - Ray | 相对 Ray |
|---|---:|---:|---:|---:|
| Client E2E | 89.530 +/- 0.586 s | 91.006 +/- 1.534 s | -1.475 s | -1.62% |
| Model request | 85.942 +/- 0.618 s | 85.166 +/- 1.508 s | +0.776 s | +0.91% |
| Non-model | 3.588 +/- 0.062 s | 5.840 +/- 0.027 s | -2.252 s | -38.56% |

E2E 差异不是来自精度或输出长度：两边的输入 token、输出 token 和最终文本完全
相同。逐轮 `Maze - Ray` 从 `-0.082 s` 到 `-2.970 s`，主要原因是模型生成时间会
波动，而两条路径是顺序运行，不是同一时刻在两张卡上运行。因此，本次最可靠的
判断是两者处于同一性能量级，未观察到此前视觉 vLLM 路径约 50 秒的 Maze 劣势。

## 模型分项

| 指标 | Ascend-Maze | Ray baseline | Maze - Ray |
|---|---:|---:|---:|
| Model load | 1.833 +/- 0.117 s | 2.125 +/- 0.032 s | -0.292 s |
| Processor load | 1.317 +/- 0.058 s | 1.358 +/- 0.025 s | -0.041 s |
| Multimodal preprocess | 0.058 +/- 0.002 s | 0.055 +/- 0.003 s | +0.003 s |
| Generation | 71.291 +/- 0.623 s | 70.236 +/- 1.388 s | +1.054 s |
| Cleanup | 0.411 +/- 0.016 s | 0.426 +/- 0.002 s | -0.015 s |
| Adapter total | 85.523 +/- 0.623 s | 84.728 +/- 1.509 s | +0.795 s |

模型分项不能简单相加为 Adapter total；模型 forward、decode 和框架内部工作还包含
在总时间中。两条路径调用同一个 Transformers 实现，主要差异是轮间 generation
波动，而不是模型加载路径发生了结构性变化。

## Task 与数据路径

| 指标 | Ascend-Maze | Ray baseline | Maze - Ray |
|---|---:|---:|---:|
| Worker startup | 2.023 +/- 0.017 s | 5.020 +/- 0.021 s | -2.996 s |
| Input fetch/binding | 0.434 +/- 0.004 s | 0.000 +/- 0.000 s | +0.434 s |
| Output put | 0.089 +/- 0.002 s | 0.008 +/- 0.001 s | +0.081 s |
| Callable minus chat | 0.043 +/- 0.001 s | 0.037 +/- 0.001 s | +0.007 s |
| Task wrapper overhead | 0.003 +/- 0.001 s | 0.772 +/- 0.005 s | -0.770 s |

这些行不能相加。`dispatch_wait` 包含 Worker startup；Ray 在进入用户 callable 前已
物化 ObjectRef，所以 Ray 的 `input_fetch=0` 是计时边界，不代表没有输入传输。
Ray 的 output put 是 driver 观察到的结果序列化/传输上界估计，Ascend-Maze 的
output put 是 Worker 内 `RayDataStore.put_staged`，两者也不是完全相同的边界。

本次 Ascend-Maze 的 submission input canonicalization 为 `0 ms`。三轮全部
RayDataStore put 平均约 `60 ms`，其中 submission input put 约 `12 ms`、runtime
output put 约 `42 ms`，数据存储已不是该视觉样本的主要瓶颈。非模型优势主要来自
较低的 Worker 启动/派发开销。

## 完整性与回收

- 6 份记录均为 `status=succeeded`。
- 每份记录均明确保存 `max_tokens=4096`、temperature `0`、输入 `434` token、
  输出 `885` token。
- Ascend-Maze 每轮 Controller shutdown 均为 `cleanup_confirmed=true`。
- 6 轮均无 cleanup error、无 residual vLLM 进程。
- 每轮结束时 NPU 设备进程列表为空；最终检查无 Raylet、GCS 或模型 Worker 残留。

原始证据位于本目录的 `maze/run1..3` 和 `ray/run1..3`，每轮均包含 `plan.json`、
`vision_records.jsonl`、`vision_summary.json` 和 `summary.json`。
