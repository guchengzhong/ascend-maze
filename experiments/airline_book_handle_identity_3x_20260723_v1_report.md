# Submission Handle identity optimization report

Date: 2026-07-23

## Scope

The comparison uses the same settings on both execution paths:

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
All six measured runs succeeded and released their NPU processes. One earlier
Ascend run-3 command failed during preflight because its shell command replaced
the CANN `PYTHONPATH` and could not import `acl`. It did not start Ray or touch
the NPU, is preserved as `run3_preflight_failed_acl`, and is excluded from all
statistics.

## Contract change

Ordinary submission inputs no longer derive request identity from their
contents. The identities are now separated by responsibility:

```text
submission_id   client-generated request idempotency key
run_id          Controller-generated execution identity
DataHandle      staged object identity and lifetime
digest          explicit content-addressed inputs only
```

The client prepares ordinary inputs through
`put_staged_for_submission_input()`. Their submission identity is
`owner_generation + staged_handle_id`, so staging uses the Ray ObjectRef path
without recursively freezing and encoding the user value. `SharedFileRef` and
other explicit content-addressed values retain stable digests. Canonicalization
also remains unchanged for deterministic control-plane values such as IR,
configuration, Literals, and workflow fingerprints.

Response-loss behavior remains explicit: the same prepared request reuses its
original handles, while a new client first queries `submission_id` through the
new `GetSubmission` RPC. Re-uploading the same bytes creates different handles
and is conservatively reported as a conflict. The implementation does not add
a server-side cache of user contents and does not change Workflow or `@task`
definitions.

## Optimization result

Values are mean +/- sample standard deviation over three runs. E2E here is the
client request interval recorded as `client_e2e_ms`, consistent with the prior
report.

| Metric | Before | Handle identity | Change |
|---|---:|---:|---:|
| E2E | 58.866 +/- 0.994 s | 55.304 +/- 0.008 s | -3.562 s |
| Non-model | 9.877 +/- 0.471 s | 6.434 +/- 0.156 s | -3.443 s |
| Model request | 48.989 +/- 0.645 s | 48.870 +/- 0.161 s | -0.119 s |
| Model load | 24.999 +/- 0.123 s | 25.068 +/- 0.154 s | +0.069 s |
| Generation | 22.895 +/- 0.455 s | 22.765 +/- 0.090 s | -0.130 s |
| Prepare submission | 3.611 +/- 0.026 s | 0.057 +/- 0.002 s | -3.554 s |
| Input staging | 3.609 +/- 0.026 s | 0.055 +/- 0.002 s | -3.554 s |
| Controller submit round trip | 0.176 +/- 0.002 s | 0.171 +/- 0.003 s | -0.005 s |
| Worker startup | 4.930 +/- 0.012 s | 4.964 +/- 0.075 s | +0.034 s |
| Input fetch/binding | 1.764 +/- 0.035 s | 1.750 +/- 0.028 s | -0.014 s |
| Output put | 0.219 +/- 0.003 s | 0.224 +/- 0.007 s | +0.005 s |

The E2E improvement closely tracks the removed submission canonicalization.
Model, Worker, input-fetch, and output-put time stayed within ordinary run
variation. This isolates the improvement to the intended submission path.

## Current Ascend-Maze versus Ray

| Metric | Ascend-Maze | Ray baseline | Ascend - Ray |
|---|---:|---:|---:|
| E2E | 55.304 +/- 0.008 s | 64.370 +/- 0.829 s | -9.066 s |
| Non-model | 6.434 +/- 0.156 s | 15.407 +/- 0.128 s | -8.973 s |
| Model request | 48.870 +/- 0.161 s | 48.963 +/- 0.704 s | -0.093 s |
| Model load | 25.068 +/- 0.154 s | 25.077 +/- 0.150 s | -0.009 s |
| Generation | 22.765 +/- 0.090 s | 22.846 +/- 0.595 s | -0.081 s |
| Worker startup | 4.964 +/- 0.075 s | 11.961 +/- 0.096 s | -6.997 s |
| Input fetch/binding | 1.750 +/- 0.028 s | 0.000 s | +1.750 s |
| Output put | 0.224 +/- 0.007 s | 0.145 +/- 0.004 s | +0.079 s |
| Dispatch prepare | 0.024 +/- 0.002 s | 0.204 +/- 0.007 s | -0.180 s |

Ray materializes ObjectRef arguments before entering its Worker function, so
its zero `input_fetch` is a measurement-scope result; that work is included in
Ray `worker_startup`. `dispatch_wait` includes `worker_startup`, and callable
time includes chat time, so these rows must not be added together.

Both paths cold-load Qwen3-4B in each of the two model Tasks. Their model request
means differ by only 0.093 s. The current E2E advantage is therefore a
non-model execution-path result, dominated by Ascend-Maze standby Workers versus
seven one-shot plain Ray Workers.

## DataStore evidence

Every optimized Ascend-Maze run recorded:

| Source | Put count | Canonicalize count |
|---|---:|---:|
| Submission input | 5 | 0 |
| Code package | 7 | 0 |
| Runtime output | 43 | 0 |

Across the three runs:

| DataStore component | Mean |
|---|---:|
| Submission input canonicalization | 0.000 s |
| Submission input Ray put | 0.042 s |
| Runtime output Ray put | 0.116 s |
| All Ray puts | 0.171 s |

All 55 staged objects per run were tombstoned by the end of the measured
lifecycle. Values without an explicit content-addressing contract report their
byte size as unknown rather than performing a hidden serialization pass just
for metrics.

## Raw E2E values

| Path | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Ascend-Maze client E2E | 55.300 s | 55.298 s | 55.313 s |
| Ray baseline client E2E | 64.654 s | 63.437 s | 65.020 s |

## Evidence paths

Previous Ascend-Maze:
`qwen_benchmark_smoke/airline_book_datastore_optimized_3x_20260723_v1`

Current Ascend-Maze:
`qwen_benchmark_smoke/airline_book_handle_identity_3x_20260723_v1`

Previous Ray baseline:
`ray_baseline_smoke/airline_book_datastore_optimized_3x_20260723_v1`

Current Ray baseline:
`ray_baseline_smoke/airline_book_handle_identity_3x_20260723_v1`
