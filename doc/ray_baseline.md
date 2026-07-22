# Ray baseline 使用与实验配置说明

本文档用于固定 Ascend-Maze 当前 Ray baseline 的使用口径。它只说明原生 Ray
task/actor 对照路径，不代表 Ascend-Maze Controller、Scheduler、Placement 或
ModelRouter 的性能结论。

## 1. 当前已经验证过什么

截至 2026-07-22，已有两类验证：

1. Ray correctness smoke 已覆盖已迁移的 gaia、openagi、tbench workflows。

   证据目录：

   ```text
   experiments/ray_baseline_smoke/all_workflows_1sample_20260722_1/
   ```

   该次结果为：

   ```text
   result=succeeded
   sample_count=14
   succeeded=14
   failed=0
   discovery_failure_count=0
   text=11/11
   vision=3/3
   ```

   覆盖的 workflow 为：

   | Dataset | Workflow | Family |
   |---|---|---|
   | gaia | file | text |
   | gaia | reason | text |
   | gaia | speech | text |
   | gaia | vision | vision |
   | openagi | document_qa | text |
   | openagi | image_captioning_complex | vision |
   | openagi | multimodal_vqa_complex | vision |
   | openagi | text_processing_multilingual | text |
   | tbench | airline_book | text |
   | tbench | airline_cancel | text |
   | tbench | retail_cancel | text |
   | tbench | retail_cancel_modify | text |
   | tbench | retail_modify | text |
   | tbench | retail_return | text |

   这里的含义是：已迁移 workflow 在 Ray task/actor 路径下，每个 workflow 至少
   1 个样本可以端到端跑通。它不代表全量样本已覆盖，也不代表正式性能结论。
   vision workflow 已切到 `true_multimodal` 请求口径。当前
   Ascend/CANN/torch_npu 环境中的 Qwen2.5-VL 视觉 encoder
   `aclnnUniqueConsecutive` AICPU 失败已通过 repo runtime workaround
   绕开；小规模视觉 Ray smoke 已在
   `experiments/ray_baseline_smoke/vision_smoke_repo_runtime_patch_20260722_1/`
   验证 3/3 成功。

2. Ray performance runner 已验证 batch 和 arrival 两类 Maze 风格负载入口。

   已跑通的最小性能通路为 Qwen3-4B + `tbench.retail_cancel`：

   | 模式 | 证据目录 | 结果 |
   |---|---|---|
   | batch `batch_size=4` | `experiments/ray_baseline_performance/batch4_retail_cancel_ray_20260722_1/` | 4/4 成功 |
   | poisson arrival `arrival_ratio=0.5` | `experiments/ray_baseline_performance/arrival_ratio_0_5_retail_cancel_ray_20260722_1/` | 2/2 成功 |

   `arrival_ratio=0.5` 使用 `avg_workflow_time_seconds=45`，因此：

   ```text
   effective_arrival_rate = 0.5 / 45 = 0.011111... req/s
   ```

## 2. 两个 Ray baseline 脚本的边界

### `tools/ray_baseline_smoke.py`

用于 correctness smoke。它回答的问题是：

```text
同一批已迁移 workflow 是否能通过原生 Ray task/actor + vLLM-Ascend 服务跑通？
```

它不会使用 Ascend-Maze 的：

- Controller；
- Scheduler；
- Placement；
- RuntimeClient；
- C11 ModelRouter。

### `tools/ray_baseline_performance.py`

用于性能实验入口。它复用 smoke 的采样和执行路径，但额外输出：

- `performance_plan.json`；
- `summary.json`；
- `{family}_performance_summary.json`；
- `{family}_performance_records.jsonl`；
- `{family}_performance_failures.jsonl`；
- vLLM service log。

关键指标包括：

- measurement planned/launched/completed；
- success rate；
- workflow latency；
- task latency；
- chat latency；
- input/output/total tokens；
- output tokens per wall second；
- output tokens per chat second；
- failure reasons；
- cleanup errors；
- residual vLLM processes。

## 3. batch 与 arrival 参数语义

### Batch 模式

Batch 模式使用：

```text
--arrival-mode batch
--batch-size N
```

语义是一次 measurement batch 包含 N 个请求，计划发射时间均为
`planned_launch_offset_ms=0`。`--measurement-iterations` 不再表达 batch 大小。

推荐让 Ray workflow 并发和 vLLM 最大序列数与 batch size 对齐，至少在小规模通路
验证时这样更直观：

```text
--concurrency N
--model-actor-concurrency N
--max-num-seqs N
```

### Arrival 模式

Arrival 模式支持：

```text
--arrival-mode paced
--arrival-mode poisson
--arrival-ratio R
--avg-workflow-time-seconds T
```

实际到达率为：

```text
effective_arrival_rate = R / T
```

例如 Maze 旧实验中常用 `T=45s`，则：

| arrival_ratio | arrival_rate |
|---:|---:|
| 0.50 | 0.011111 req/s |
| 0.75 | 0.016667 req/s |
| 1.00 | 0.022222 req/s |
| 1.25 | 0.027778 req/s |

`poisson` 使用指数间隔，并通过 `--seed` 固定计划；`paced` 使用固定间隔。runner 会把每个
measurement item 的 `planned_launch_offset_ms` 写入 `performance_plan.json`。

如果设置了 `--measurement-window-seconds`，arrival measurement 会保持到窗口结束后再汇总，
避免吞吐指标因为最后一个请求提前结束而缩短统计窗口。

`--target-qps` 仍保留为向后兼容的直接到达率参数，但正式 Maze 风格实验优先使用
`--arrival-ratio`。

## 4. 推荐命令

下面命令默认在仓库根目录执行：

```bash
cd /home/user2/workplace/Ascend-Maze
export PYTHONPATH="$PWD/src:$PWD:$PWD/tools:${PYTHONPATH:-}"
PY=/home/user2/workplace/miniconda3/envs/ascend-maze/bin/python
```

### 4.1 correctness smoke 计划检查

只生成计划，不启动 Ray/vLLM：

```bash
$PY tools/ray_baseline_smoke.py \
  --plan-only \
  --samples-per-workflow 1 \
  --output-dir experiments/ray_baseline_smoke/plan_all_<tag>
```

### 4.2 全 workflow correctness smoke

这一步已经跑过，不需要每次重复跑。只有在 workflow 代码、数据加载、模型路径或 Ray
执行逻辑发生实质修改后才重跑：

```bash
$PY tools/ray_baseline_smoke.py \
  --family text \
  --family vision \
  --samples-per-workflow 1 \
  --max-num-seqs 1 \
  --output-dir experiments/ray_baseline_smoke/all_workflows_1sample_<tag>
```

### 4.3 最小 batch performance 通路

已跑通命令：

```bash
$PY tools/ray_baseline_performance.py \
  --family text \
  --dataset tbench \
  --workflow retail_cancel \
  --samples-per-workflow 1 \
  --arrival-mode batch \
  --batch-size 4 \
  --warmup-iterations 1 \
  --concurrency 4 \
  --model-actor-concurrency 4 \
  --max-num-seqs 4 \
  --output-dir experiments/ray_baseline_performance/batch4_retail_cancel_ray_<tag>
```

### 4.4 最小 arrival performance 通路

已跑通短窗口命令：

```bash
$PY tools/ray_baseline_performance.py \
  --family text \
  --dataset tbench \
  --workflow retail_cancel \
  --samples-per-workflow 1 \
  --arrival-mode poisson \
  --arrival-ratio 0.5 \
  --avg-workflow-time-seconds 45 \
  --measurement-window-seconds 90 \
  --seed 1 \
  --warmup-iterations 1 \
  --concurrency 4 \
  --model-actor-concurrency 4 \
  --max-num-seqs 4 \
  --output-dir experiments/ray_baseline_performance/arrival_ratio_0_5_retail_cancel_ray_<tag>
```

## 5. 正式 baseline 推荐矩阵

推荐矩阵文件：

```text
experiments/ray_baseline_performance/ray_baseline_matrix.recommended.json
```

### 5.1 Workload 分层

正式 baseline 建议分三层，而不是现在立即重复全量运行：

| 层级 | 用途 | Workload |
|---|---|---|
| admission | 改完脚本后的快速通路验证 | `tbench.retail_cancel` |
| text_primary | Qwen3-4B 正式文本 Ray baseline | gaia text、openagi text、tbench text，共 11 个 workflow |
| vision_separate | 视觉路径单独验证 | gaia/openagi vision，共 3 个 workflow |

其中 `vision_separate` 不和 Qwen3-4B 文本矩阵混跑。视觉建议使用
Qwen2.5-VL-3B-Instruct 或后续确认的 Ascend VL 模型，并单独记录 vision 口径。

### 5.2 Batch sizes

正式 batch 建议：

```text
batch_size = 1, 2, 4, 8, 16
```

说明：

- `1` 是串行参照；
- `2/4` 是当前硬件和 Qwen3-4B 下的低风险并发区；
- `8/16` 用于观察排队和吞吐平台期，不要求 vLLM 同时执行同等数量请求；
- 若 `max-num-seqs` 小于 batch size，结果语义是“一次性提交 N 个请求，由 Ray/vLLM 排队执行”。

### 5.3 Arrival ratios

正式 arrival 建议沿用 Maze 风格负载点：

```text
arrival_ratio = 0.50, 0.75, 1.00, 1.25, 0.50
avg_workflow_time_seconds = 45
measurement_window_seconds = 600
arrival_mode = poisson
```

最后再次运行 `0.50` 是为了观察升载后回到低负载时是否有残留、退化或长尾污染。

调试阶段可以只跑：

```text
arrival_ratio = 0.50
measurement_window_seconds = 90 或 180
```

## 6. 当前不建议做的事

现在不建议立刻重复跑全部 workflow 的 batch/arrival 矩阵。原因是：

- correctness smoke 已证明 14 个已迁移 workflow 的基本通路可运行；
- performance runner 的 batch/arrival 入口已通过 `tbench.retail_cancel` 验证；
- 正式性能矩阵会花费较长 NPU 时间，并且应在模型、窗口、样本数、seed、并发上限和
  输出目录命名冻结后再跑；
- 视觉 baseline 需要和文本 baseline 分开设计，否则 Qwen3-4B 文本结果与 VL 模型结果
  会混在一个口径里。

因此当前更合理的状态是：代码和入口已准备好，先保存本文档和推荐矩阵；等要产出正式
Ray baseline 时，再按矩阵执行。

## 7. 下一阶段小规模性能实验计划

下一阶段先跑小规模 Ray performance admission，不直接进入完整矩阵。

### 7.1 Text admission

目标是继续使用已经跑通过的轻量文本 workload，确认脚本改动后 batch/arrival
入口仍稳定：

```bash
PY=/home/user2/workplace/miniconda3/envs/ascend-maze/bin/python

PYTHONPATH=$PWD/src:$PWD:$PWD/tools:${PYTHONPATH:-} \
$PY tools/ray_baseline_performance.py \
  --family text \
  --dataset tbench \
  --workflow retail_cancel \
  --samples-per-workflow 1 \
  --arrival-mode batch \
  --batch-size 2 \
  --warmup-iterations 1 \
  --concurrency 2 \
  --model-actor-concurrency 2 \
  --max-num-seqs 2 \
  --output-dir experiments/ray_baseline_performance/text_retail_cancel_batch2_<tag>
```

然后再跑一个短窗口 arrival：

```bash
PYTHONPATH=$PWD/src:$PWD:$PWD/tools:${PYTHONPATH:-} \
$PY tools/ray_baseline_performance.py \
  --family text \
  --dataset tbench \
  --workflow retail_cancel \
  --samples-per-workflow 1 \
  --arrival-mode poisson \
  --arrival-ratio 0.5 \
  --avg-workflow-time-seconds 45 \
  --measurement-window-seconds 90 \
  --warmup-iterations 1 \
  --concurrency 2 \
  --model-actor-concurrency 2 \
  --max-num-seqs 2 \
  --output-dir experiments/ray_baseline_performance/text_retail_cancel_arrival_0_5_<tag>
```

### 7.2 Vision admission

目标是只验证 `true_multimodal` Ray performance 入口，不追求性能结论：

```bash
PYTHONPATH=$PWD/src:$PWD:$PWD/tools:${PYTHONPATH:-} \
$PY tools/ray_baseline_performance.py \
  --family vision \
  --samples-per-workflow 1 \
  --arrival-mode batch \
  --batch-size 1 \
  --warmup-iterations 0 \
  --concurrency 1 \
  --model-actor-concurrency 1 \
  --max-num-seqs 1 \
  --vision-max-num-batched-tokens 4096 \
  --request-timeout-ms 180000 \
  --output-dir experiments/ray_baseline_performance/vision_true_multimodal_batch1_<tag>
```

该 vision admission 必须记录：

- `generation_config=vllm`；
- `qwen2_5_vl_cpu_unique_consecutive_workaround=true`；
- `vision_mode=true_multimodal`；
- 3 个 vision workflow 的 sample records；
- vLLM service log；
- `cleanup_errors=[]`；
- `residual_vllm_processes=[]`；
- 跑后 `npu-smi info` 无残留用户进程。

只有 admission 都通过后，再冻结正式矩阵的样本数、窗口、seed、batch size 和
arrival ratio。
