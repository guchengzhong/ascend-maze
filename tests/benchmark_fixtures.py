from __future__ import annotations

import hashlib
import json
from pathlib import Path


BUILD_REVISION = "a" * 40
WORKFLOW_FINGERPRINT = "b" * 64
MODEL_DIGEST = "c" * 64
ENVIRONMENT_FINGERPRINT = "d" * 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_performance_config(root: Path, *, profile: str = "performance") -> Path:
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    config = root / "performance.toml"
    config.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'profile = "{profile}"',
                "",
                "[control]",
                f'runtime_directory = "{runtime}"',
                f'cluster_token_file = "{runtime / "cluster.token"}"',
                "",
                "[runtime.ray]",
                'namespace = "benchmark-test"',
                "",
                "[cluster]",
                f'environment_fingerprint = "{ENVIRONMENT_FINGERPRINT}"',
                "",
                "[scheduler]",
                'policy = "hacs_no_tp"',
                'partitioner = "heterogeneous"',
                "",
                "[placement]",
                'anchor_strategy = "static"',
                "task_slots_total = 2",
                "allow_colocation = true",
                "",
                "[worker]",
                "max_tasks_per_worker = 1",
                "standby_min_idle = 2",
                "standby_max_idle = 2",
                "",
                "[recording]",
                'backend = "noop"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def write_experiment_spec(
    root: Path,
    *,
    base_config: Path | None = None,
    reverse_matrix: bool = False,
    study_kind: str = "formal",
    block_count: int = 10,
    workflow_factory: str = "benchmarks.workloads:build",
    workflow_fingerprint: str = WORKFLOW_FINGERPRINT,
    arrival_mode: str = "poisson",
    concurrency: int = 2,
    rate_per_second: float = 2.5,
    warmup_runs: int = 10,
    warmup_duration_ms: int = 0,
    measurement_run_count: int = 0,
    measurement_duration_ms: int = 60_000,
    drain_deadline_ms: int = 30_000,
    trace_offsets_ms: tuple[int, ...] = (0, 0, 25, 50),
) -> Path:
    config = base_config or write_performance_config(root)
    dataset = root / "dataset.json"
    if not dataset.exists():
        dataset.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workflow_fingerprint": workflow_fingerprint,
                    "records": [
                        {"record_id": "record-a", "inputs": {"value": 1}},
                        {"record_id": "record-b", "inputs": {"value": 2}},
                        {"record_id": "record-c", "inputs": {"value": 3}},
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    trace = root / "trace.json"
    if arrival_mode == "trace_replay":
        trace.write_text(
            json.dumps(
                {"schema_version": 1, "offsets_ms": list(trace_offsets_ms)},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    factors = [
        ("ordering", ["scheduler.policy"]),
        ("anchor", ["placement.anchor_strategy"]),
        ("partitioner", ["scheduler.partitioner"]),
        (
            "worker_mode",
            ["worker.standby_min_idle", "worker.standby_max_idle"],
        ),
    ]
    cells: list[tuple[str, list[str], list[tuple[str, object]]]] = [
        ("maze_full", [], []),
        ("fcfs", ["ordering"], [("scheduler.policy", "fcfs")]),
        (
            "no_resource_anchor",
            ["anchor"],
            [("placement.anchor_strategy", "declared_only")],
        ),
        (
            "no_heterogeneous_queue",
            ["partitioner"],
            [("scheduler.partitioner", "unified")],
        ),
        (
            "no_standby",
            ["worker_mode"],
            [("worker.standby_min_idle", 0), ("worker.standby_max_idle", 0)],
        ),
    ]
    if reverse_matrix:
        factors.reverse()
        cells.reverse()
    lines = [
        "schema_version = 1",
        'study_name = "scheduler-ablation"',
        f'study_kind = "{study_kind}"',
        "base_seed = 7842",
        f"block_count = {block_count}",
        "repetition_count = 1",
        f'build_revision = "{BUILD_REVISION}"',
        f'base_config = "{config.name}"',
        f'base_config_sha256 = "{sha256(config)}"',
        "",
        "[workload]",
        'name = "qwen-workflow"',
        f'workflow_factory = "{workflow_factory}"',
        f'workflow_fingerprint = "{workflow_fingerprint}"',
        'model_catalog_revision = "no-model-catalog"',
        f'model_artifact_digest = "{MODEL_DIGEST}"',
        f'required_environment_fingerprint = "{ENVIRONMENT_FINGERPRINT}"',
        "",
        "[[workload.inputs]]",
        'logical_name = "dataset"',
        f'path = "{dataset.name}"',
        f'sha256 = "{sha256(dataset)}"',
        f"size_bytes = {dataset.stat().st_size}",
    ]
    if arrival_mode == "trace_replay":
        lines.extend(
            (
                "",
                "[[workload.inputs]]",
                'logical_name = "trace"',
                f'path = "{trace.name}"',
                f'sha256 = "{sha256(trace)}"',
                f"size_bytes = {trace.stat().st_size}",
            )
        )
    arrival_value = (
        f"concurrency = {concurrency}"
        if arrival_mode == "closed_loop"
        else (
            'trace_input = "trace"'
            if arrival_mode == "trace_replay"
            else f"rate_per_second = {rate_per_second}"
        )
    )
    lines.extend(
        (
            "",
            "[arrival]",
            f'mode = "{arrival_mode}"',
            arrival_value,
            "",
            "[windows]",
            f"warmup_runs = {warmup_runs}",
            f"warmup_duration_ms = {warmup_duration_ms}",
            f"measurement_run_count = {measurement_run_count}",
            f"measurement_duration_ms = {measurement_duration_ms}",
            f"drain_deadline_ms = {drain_deadline_ms}",
            "",
            "[analysis]",
            'metric_set = ["dct_ms", "throughput_success_per_s"]',
            'validity_policy = "c14_v1"',
            'statistics_policy = "c14_v1"',
            'performance_budget_set = "c14_v1"',
            'quantile_method = "hyndman_fan_type_7"',
            "bootstrap_samples = 10000",
            "confidence_level = 0.95",
            "familywise_confidence_level = 0.9875",
            "automatic_outlier_removal = false",
            "",
            "[matrix]",
            'kind = "internal_ablation_v1"',
            'baseline_cell = "maze_full"',
        )
    )
    for name, paths in factors:
        rendered_paths = ", ".join(f'"{path}"' for path in paths)
        lines.extend(
            (
                "",
                "[[matrix.factors]]",
                f'name = "{name}"',
                f"allowed_paths = [{rendered_paths}]",
            )
        )
    for name, cell_factors, overrides in cells:
        rendered_factors = ", ".join(f'"{factor}"' for factor in cell_factors)
        lines.extend(
            (
                "",
                "[[matrix.cells]]",
                f'name = "{name}"',
                f"factors = [{rendered_factors}]",
                "confirmatory = true",
            )
        )
        for path, value in overrides:
            rendered_value = f'"{value}"' if isinstance(value, str) else str(value)
            lines.extend(
                (
                    "",
                    "[[matrix.cells.overrides]]",
                    f'path = "{path}"',
                    f"value = {rendered_value}",
                )
            )
    spec = root / ("experiment-reversed.toml" if reverse_matrix else "experiment.toml")
    spec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return spec
