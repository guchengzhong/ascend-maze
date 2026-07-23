# airline_book one-shot Worker comparison

Date: 2026-07-23

## Scope

The comparison uses the same settings on both execution paths:

- workflow: `tbench.airline_book`
- sample: `tbench.airline_book.9a9e376c-0089-4a3e-8480-f05df35ae465`
- model: `/home/user2/workplace/model_weight/model_from_hf/Qwen3-4B`
- inference: Transformers manual greedy, BF16
- `max_tokens=4096`, `max_model_len=10240`, temperature `0`
- Ray Task `max_calls=1`
- no tau-bench smoke overrides
- three successful runs per execution path

Ascend-Maze records:
`qwen_benchmark_smoke/airline_book_max_calls_1_3x_20260723_v2`

Ray baseline records:
`ray_baseline_smoke/airline_book_max_calls_1_3x_20260723_v2`

Every Ray run executed seven Tasks in seven distinct Worker PIDs. Therefore the
baseline did not reuse a Ray Worker process between Tasks.

## Results

Values are mean +/- sample standard deviation across three runs.

| Metric | Ascend-Maze | Ray baseline (`max_calls=1`) |
|---|---:|---:|
| E2E | 72.185 +/- 0.602 s | 63.731 +/- 0.394 s |
| Model request | 48.493 +/- 0.503 s | 48.361 +/- 0.434 s |
| Model load (2 calls) | 24.949 +/- 0.169 s | 25.115 +/- 0.453 s |
| Generation (2 calls) | 22.487 +/- 0.317 s | 22.171 +/- 0.244 s |
| Worker startup | 4.992 +/- 0.030 s | 11.912 +/- 0.092 s |
| Dispatch prepare | 0.023 +/- 0.001 s | 0.202 +/- 0.002 s |
| Dispatch wait | 5.016 +/- 0.030 s | 11.912 +/- 0.092 s |
| Input fetch/binding | 1.763 +/- 0.025 s | 0.000 +/- 0.000 s |
| Callable | 49.064 +/- 0.501 s | 48.930 +/- 0.440 s |
| Callable minus chat | 0.571 +/- 0.004 s | 0.569 +/- 0.012 s |
| Output put | 7.003 +/- 0.338 s | 0.145 +/- 0.006 s |
| Non-model | 23.692 +/- 0.417 s | 15.370 +/- 0.086 s |

The rows are not additive. `dispatch_wait` includes `worker_startup`, and
`callable` includes the chat/model request. Ray materializes top-level ObjectRefs
before the Worker function starts, so its `input_fetch` is zero by definition;
that cost is included in `worker_startup`. Ray `output_put` is a driver-observed
result serialization and transfer upper-bound estimate, while Ascend-Maze
measures `RayDataStore.put_staged` in the Worker.

## Findings

The E2E difference is 8.454 s in favor of the Ray baseline. Model request time
differs by only 0.131 s, while the non-model difference is 8.322 s. The measured
gap is therefore a control/data-path gap, not a model loading or generation gap.

Ascend-Maze submission preparation averages 10.593 s:

| Submission component | Mean |
|---|---:|
| Input signature | 3.539 s |
| Input staging | 7.052 s |
| Controller submit round trip | 0.171 s |
| Final result fetch | 0.014 s |

Ascend-Maze output storage averages 7.003 s. Of that, `task0_init` alone
accounts for 6.835 s; the other six Tasks together account for about 0.168 s.
`task0_init` returns the airline backend data loaded from approximately 5 MB of
JSON. The current RayDataStore path computes both `canonical_digest(value)` and
`canonical_bytes(value)` before `ray.put`, which traverses/serializes the large
value twice.

## Optimization ceilings

RayDataStore/submission has a zero-cost theoretical ceiling of:

```text
input signature + input staging + output put
= 3.539 + 7.052 + 7.003
= 17.594 s
```

This is not an attainable saving: storage, transfer, ownership metadata, and
some hashing remain necessary, and overlapping scopes may not translate
one-for-one to E2E. The observed 8.454 s E2E gap is the relevant first target.
Worker-side input binding adds another measured 1.763 s, but is kept separate
from the RayDataStore zero-cost ceiling.

Standby already reduces Ascend-Maze Worker startup by 6.920 s compared with
one-shot plain Ray (4.992 s versus 11.912 s). Its absolute remaining zero-cost
ceiling is only 4.992 s and realistic savings are lower. Standby is therefore
not the next bottleneck.

## Next optimization

Optimize RayDataStore and submission first, without changing Workflow or Task
definitions:

1. Produce canonical bytes once and derive the digest and size from those bytes.
2. Do not require a stable content digest for large runtime intermediates unless
   the contract needs one.
3. Add internal identity/content deduplication or handle forwarding for repeated
   large values.
4. Avoid restaging `task0_init`'s unchanged backend data through every data-path
   boundary.
5. Repeat the same six-run comparison before changing Standby behavior.
