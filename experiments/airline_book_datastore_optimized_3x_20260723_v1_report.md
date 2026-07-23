# RayDataStore optimization report

Date: 2026-07-23

## Scope

The before and after comparisons use the same:

- workflow: `tbench.airline_book`
- sample: `tbench.airline_book.9a9e376c-0089-4a3e-8480-f05df35ae465`
- model: Qwen3-4B
- inference: Transformers manual greedy, BF16
- `max_tokens=4096`, `max_model_len=10240`, temperature `0`
- Ray Task `max_calls=1`
- tau-bench smoke overrides disabled
- three successful runs for Ascend-Maze and three for the Ray baseline

The environment fingerprint was
`0dd3eecf97253ba211b4a6d34013d2cd4b0ebf20354e7fd89728caa64d2c7cb9`.
All runs completed successfully and left no NPU process behind. Each Ray
baseline run used seven distinct Worker PIDs for its seven Tasks.

## Changes

Submission inputs now produce canonical bytes once. The digest and byte size in
the resulting DataHandle are reused by the local prepared-submission signature
and by `SubmissionContract` input identity, which in turn determines the
submission payload hash.

Runtime outputs no longer calculate a stable content digest. Their identity is
the existing `owner_generation + staged_handle_id`, and the owner-backed Ray
ObjectRef remains the transport path. Code packages are recorded as a separate
source and also skip data content canonicalization because their integrity is
already covered by the CodePackage contract.

The long-lived DataStore owner aggregates these metrics across the Controller
and all one-shot Workers:

- `canonicalize_ms`
- `ray_put_ms`
- `owner_stage_ms`
- `canonicalize_count`
- `value_size_bytes`
- source-specific counters and timings for `submission_input`, `code_package`,
  and `runtime_output`

`owner_stage_ms` is actor-side Handle registration service time. It does not
claim to include the complete actor RPC round trip.

## Before and after

Values are mean +/- sample standard deviation over three runs.

| Metric | Before | After | After - before |
|---|---:|---:|---:|
| E2E | 72.185 +/- 0.602 s | 58.866 +/- 0.994 s | -13.319 s |
| Non-model | 23.692 +/- 0.417 s | 9.877 +/- 0.471 s | -13.815 s |
| Model request | 48.493 +/- 0.503 s | 48.989 +/- 0.645 s | +0.496 s |
| Model load | 24.949 +/- 0.169 s | 24.999 +/- 0.123 s | +0.051 s |
| Generation | 22.487 +/- 0.317 s | 22.895 +/- 0.455 s | +0.407 s |
| Prepare submission | 10.593 +/- 0.049 s | 3.611 +/- 0.026 s | -6.982 s |
| Input staging | 7.052 +/- 0.040 s | 3.609 +/- 0.026 s | -3.443 s |
| Worker startup | 4.992 +/- 0.030 s | 4.930 +/- 0.012 s | -0.063 s |
| Input fetch/binding | 1.763 +/- 0.025 s | 1.764 +/- 0.035 s | +0.001 s |
| Output put | 7.003 +/- 0.338 s | 0.219 +/- 0.003 s | -6.784 s |
| `task0_init` output put | 6.835 +/- 0.339 s | 0.057 +/- 0.001 s | -6.778 s |

The runtime-output optimization independently removed 6.784 s from aggregate
output storage. The submission-input optimization independently removed 6.982 s
from submission preparation. The 0.496 s increase in model request time offsets
part of those gains in E2E; with three runs, the data does not demonstrate a
material model-path regression.

## After versus current Ray baseline

| Metric | Ascend-Maze after | Ray baseline after | Ascend - Ray |
|---|---:|---:|---:|
| E2E | 58.866 +/- 0.994 s | 63.959 +/- 0.446 s | -5.093 s |
| Non-model | 9.877 +/- 0.471 s | 15.460 +/- 0.121 s | -5.583 s |
| Model request | 48.989 +/- 0.645 s | 48.499 +/- 0.392 s | +0.490 s |
| Model load | 24.999 +/- 0.123 s | 24.986 +/- 0.129 s | +0.013 s |
| Generation | 22.895 +/- 0.455 s | 22.482 +/- 0.376 s | +0.413 s |
| Worker startup | 4.930 +/- 0.012 s | 12.001 +/- 0.130 s | -7.071 s |
| Input fetch/binding | 1.764 +/- 0.035 s | 0.000 s | +1.764 s |
| Output put | 0.219 +/- 0.003 s | 0.142 +/- 0.005 s | +0.077 s |

Ray materializes ObjectRef arguments before entering its Worker function, so
its zero `input_fetch` is a measurement-scope result; that work is included in
Ray `worker_startup`. Rows such as `dispatch_wait` and `worker_startup` overlap
and must not be added together.

## DataStore evidence

Each optimized Ascend-Maze run recorded:

| Source | Put count | Canonicalize count |
|---|---:|---:|
| Submission input | 5 | 5 |
| Code package | 7 | 0 |
| Runtime output | 43 | 0 |

Across the three runs, DataStore means were approximately:

| Internal component | Mean |
|---|---:|
| Submission canonicalization | 3.557 s |
| All `ray.put` calls | 0.160 s |
| All owner-stage service | 0.001 s |

The canonical input size recorded per run was 5,935,837 bytes. The remaining
3.557 s is the one content identity required for raw submission data, not a
duplicate DataStore traversal.

## New bottleneck

RayDataStore output storage is no longer the bottleneck. The largest remaining
Ascend-Maze non-model components are:

- aggregate Worker startup: 4.930 s;
- required submission input canonicalization: 3.557 s;
- Worker-side input binding and code loading: 1.764 s;
- output storage: 0.219 s.

The next investigation should split Worker input binding into CodePackage load,
callable reconstruction, and argument binding. SharedFileRef or a stable dataset
handle can later avoid canonicalizing explicitly shared dataset contents, but
that is a separate input representation optimization and must be applied fairly
to both comparison paths.

## Evidence paths

Before Ascend-Maze:
`qwen_benchmark_smoke/airline_book_max_calls_1_3x_20260723_v2`

Before Ray baseline:
`ray_baseline_smoke/airline_book_max_calls_1_3x_20260723_v2`

After Ascend-Maze:
`qwen_benchmark_smoke/airline_book_datastore_optimized_3x_20260723_v1`

After Ray baseline:
`ray_baseline_smoke/airline_book_datastore_optimized_3x_20260723_v1`
