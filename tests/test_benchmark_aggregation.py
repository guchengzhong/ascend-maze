from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

from ascend_maze.benchmark.aggregation import (
    _comparison_rows,
    _finalize_p99_status,
    aggregate_study,
    load_study,
    rebuild_aggregate_csv,
)
from ascend_maze.benchmark.importer import validate_study
from ascend_maze.benchmark.metrics import RunFact, extract_metric
from ascend_maze.benchmark.metrics import CORRECTNESS_GUARD_METRICS
from ascend_maze.benchmark.reporting import rebuild_report_views, report_study
from ascend_maze.benchmark.persistence import atomic_write_json
from ascend_maze.benchmark.statistics import (
    budget_decision,
    deterministic_bootstrap_interval,
    relative_effect_percent,
    summarize_distribution,
    type7_quantile,
)
from test_benchmark_importer import _event, _fixture


ROOT = Path(__file__).resolve().parents[1]


def _run_fact(
    run_id: str,
    *,
    status: str = "succeeded",
    terminal_at_ms: int = 1_500,
) -> RunFact:
    return RunFact(
        run_id=run_id,
        phase="measurement",
        offered_at_ms=1_000,
        issued_at_ms=1_000,
        admitted_at_ms=1_001,
        terminal_at_ms=terminal_at_ms,
        terminal_status=status,
        scheduled_at_ms=1_000,
        scheduled_offset_ms=0,
        arrival_lateness_ms=0,
    )


def _values(name: str, events: tuple[object, ...], runs: tuple[RunFact, ...]) -> list[float]:
    extraction = extract_metric(
        name,
        events=events,  # type: ignore[arg-type]
        runs=runs,
        measurement_duration_ms=1_000,
        recording_complete=True,
    )
    return [sample.value for sample in extraction.samples]


def test_type7_summary_bootstrap_budget_and_outlier_golden() -> None:
    assert type7_quantile([1.0, 2.0, 3.0, 4.0], 0.25) == pytest.approx(1.75)
    assert type7_quantile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    summary = summarize_distribution([1.0, 2.0, 3.0, 1_000.0])
    assert summary.median == pytest.approx(2.5)
    assert summary.mad == pytest.approx(1.0)
    assert summary.maximum == 1_000.0
    interval = deterministic_bootstrap_interval(
        [1.0, -2.0, 4.0, -1.0],
        seed=123,
        samples=10_000,
        confidence_level=0.95,
        one_sided_upper=True,
    )
    assert interval.upper == pytest.approx(2.5)
    assert relative_effect_percent(100.0, 120.0, higher_is_better=False) == -20.0
    assert budget_decision(point_estimate=4.0, upper_bound=4.9, limit=5.0) == "pass"
    assert budget_decision(point_estimate=4.0, upper_bound=5.1, limit=5.0) == "fail"
    assert budget_decision(point_estimate=5.0, upper_bound=5.0, limit=5.0) == "borderline"


def test_metric_golden_values_and_clock_domains() -> None:
    run_id = "run_metric_golden"
    events = (
        _event(1, "run_submitted", run_id=run_id, monotonic_time_ms=100),
        _event(2, "task_queued", run_id=run_id, task_id="task", monotonic_time_ms=110),
        _event(
            3,
            "task_dispatched",
            run_id=run_id,
            task_id="task",
            attempt=1,
            lease_id="lease",
            payload={"dispatch_id": "dispatch"},
            monotonic_time_ms=160,
        ),
        _event(
            4,
            "scheduling_decision",
            run_id=run_id,
            task_id="task",
            payload={"score_compute_ms": 1.0, "policy_select_ms": 2.0, "placement_ms": 3.0},
        ),
        _event(
            5,
            "worker_acquired",
            run_id=run_id,
            task_id="task",
            attempt=1,
            lease_id="lease",
            payload={"worker_acquire_ms": 7, "cold_start_ms": 11, "source": "cold_start"},
        ),
        _event(
            6,
            "inference_request",
            run_id=run_id,
            task_id="task",
            attempt=1,
            lease_id="lease",
            payload={
                "duration_ms": 100,
                "ttft_ms": 20,
                "output_tokens": 5,
                "engine_queue_depth": 3,
                "prefix_cache_hit": True,
            },
        ),
        _event(
            7,
            "device_resource_sample",
            run_id=run_id,
            payload={"observed_free_hbm_mb": 60_000, "utilization": 75.0, "active_lease_count": 2},
        ),
        _event(
            8,
            "recovery_decision",
            run_id=run_id,
            task_id="task",
            attempt=1,
            lease_id="lease",
            payload={"decision_id": "decision", "eligible_at_ms": 230, "cleanup_duration_ms": 9},
            monotonic_time_ms=200,
        ),
        _event(
            9,
            "recovery_succeeded",
            run_id=run_id,
            task_id="task",
            attempt=2,
            lease_id="lease2",
            payload={"decision_id": "decision"},
            monotonic_time_ms=250,
        ),
        _event(
            10,
            "data_transfer",
            run_id=run_id,
            payload={"data_transfer_ms": 13, "data_transfer_bytes": 4_096},
        ),
        _event(
            11,
            "model_instance_requested",
            run_id=run_id,
            model_instance_id="model-instance",
            payload={"model_id": "model"},
            monotonic_time_ms=120,
        ),
        _event(
            12,
            "model_instance_ready",
            run_id=run_id,
            model_instance_id="model-instance",
            payload={"model_id": "model"},
            monotonic_time_ms=180,
        ),
        _event(
            13,
            "recorder_emit",
            run_id=run_id,
            payload={"emit_duration_ms": 4},
        ),
        _event(14, "run_terminal", run_id=run_id, payload={"status": "succeeded"}, monotonic_time_ms=300),
    )
    runs = (_run_fact(run_id),)
    assert _values("dct_ms", events, runs) == [200.0]
    assert _values("queue_ms", events, runs) == [50.0]
    assert _values("scheduler_total_ms", events, runs) == [6.0]
    assert _values("worker_acquire_ms", events, runs) == [7.0]
    assert _values("worker_cold_start_ms", events, runs) == [11.0]
    assert _values("ttft_ms", events, runs) == [20.0]
    assert _values("tpot_ms", events, runs) == [20.0]
    assert _values("inference_token_throughput_per_s", events, runs) == [50.0]
    assert _values("device_hbm_free_mb", events, runs) == [60_000.0]
    assert _values("device_utilization_pct", events, runs) == [75.0]
    assert _values("active_lease_count", events, runs) == [2.0]
    assert _values("fault_cleanup_ms", events, runs) == [9.0]
    assert _values("fault_backoff_ms", events, runs) == [30.0]
    assert _values("fault_recovery_ms", events, runs) == [50.0]
    assert _values("data_transfer_ms", events, runs) == [13.0]
    assert _values("data_transfer_bytes", events, runs) == [4_096.0]
    assert _values("model_cold_start_ms", events, runs) == [60.0]
    assert _values("recorder_emit_ms", events, runs) == [4.0]

    other_producer_terminal = events[:-1] + (
        replace_event_producer(events[-1], "node-agent"),
    )
    assert _values("dct_ms", other_producer_terminal, runs) == []


def replace_event_producer(event: object, producer_id: str) -> object:
    from dataclasses import replace

    return replace(event, producer_id=producer_id)


def test_throughput_counts_only_measurement_window_terminals() -> None:
    runs = (
        _run_fact("run-a", status="succeeded", terminal_at_ms=1_500),
        _run_fact("run-b", status="failed", terminal_at_ms=1_600),
        _run_fact("run-c", status="succeeded", terminal_at_ms=2_100),
    )
    assert _values("throughput_success_per_s", (), runs) == [1.0]
    assert _values("throughput_terminal_per_s", (), runs) == [2.0]
    assert _values("offered_load_per_s", (), runs) == [3.0]


def test_p99_sample_gate_is_study_and_trial_scoped() -> None:
    rows = [
        {
            "metric_name": "dct_ms",
            "metric_valid": True,
            "sample_count": 100,
            "p99_status": "pending_study_sample_check",
        },
        {
            "metric_name": "dct_ms",
            "metric_valid": True,
            "sample_count": 99,
            "p99_status": "pending_study_sample_check",
        },
    ]
    _finalize_p99_status(rows, 1_000)
    assert {row["p99_status"] for row in rows} == {"insufficient_sample"}
    rows[1]["sample_count"] = 100
    _finalize_p99_status(rows, 999)
    assert {row["p99_status"] for row in rows} == {"insufficient_sample"}
    _finalize_p99_status(rows, 1_000)
    assert {row["p99_status"] for row in rows} == {"sufficient"}


def test_block_pairing_excludes_unpaired_and_reports_negative_effect(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    study = load_study(fixture.study)
    full = next(cell for cell in study.cells if cell.name == "maze_full")
    fcfs = next(cell for cell in study.cells if cell.name == "fcfs")
    rows = [
        _synthetic_trial_row(full.cell_id, "maze_full", 0, 10, 120.0),
        _synthetic_trial_row(fcfs.cell_id, "fcfs", 0, 10, 100.0),
        _synthetic_trial_row(full.cell_id, "maze_full", 1, 20, 50.0),
    ]
    comparisons = _comparison_rows(study, rows)
    comparison = next(
        row
        for row in comparisons
        if row["baseline_cell_id"] == fcfs.cell_id and row["metric_name"] == "dct_ms"
    )
    assert comparison["paired_blocks"] == 1
    assert comparison["relative_effect_pct"] == pytest.approx(-20.0)
    assert comparison["familywise_ci_lower"] == pytest.approx(-20.0)
    assert comparison["decision"] == "fail"


def test_performance_benefit_requires_complete_correctness_guards(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    study = load_study(fixture.study)
    full = next(cell for cell in study.cells if cell.name == "maze_full")
    fcfs = next(cell for cell in study.cells if cell.name == "fcfs")
    rows = [
        _synthetic_trial_row(full.cell_id, "maze_full", 0, 10, 80.0),
        _synthetic_trial_row(fcfs.cell_id, "fcfs", 0, 10, 100.0),
    ]
    comparison = next(
        row
        for row in _comparison_rows(study, rows)
        if row["baseline_cell_id"] == fcfs.cell_id and row["metric_name"] == "dct_ms"
    )
    assert comparison["decision"] == "insufficient_sample"
    assert comparison["guard_decision"] == "insufficient_sample"

    for guard_name in sorted(CORRECTNESS_GUARD_METRICS):
        value = 0.99 if guard_name == "success_rate" else 0.0
        rows.extend(
            (
                _synthetic_trial_row(
                    full.cell_id,
                    "maze_full",
                    0,
                    10,
                    value,
                    metric_name=guard_name,
                    higher_is_better=guard_name == "success_rate",
                ),
                _synthetic_trial_row(
                    fcfs.cell_id,
                    "fcfs",
                    0,
                    10,
                    value,
                    metric_name=guard_name,
                    higher_is_better=guard_name == "success_rate",
                ),
            )
        )
    guarded = next(
        row
        for row in _comparison_rows(study, rows)
        if row["baseline_cell_id"] == fcfs.cell_id and row["metric_name"] == "dct_ms"
    )
    assert guarded["guard_decision"] == "pass"
    assert guarded["decision"] == "pass"


def _synthetic_trial_row(
    cell_id: str,
    cell_name: str,
    block_index: int,
    pairing_seed: int,
    value: float,
    *,
    metric_name: str = "dct_ms",
    higher_is_better: bool = False,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "cell_name": cell_name,
        "metric_name": metric_name,
        "metric_valid": True,
        "primary_value": value,
        "block_index": block_index,
        "repetition_index": 0,
        "pairing_seed": pairing_seed,
        "higher_is_better": higher_is_better,
        "unit": "ratio" if metric_name in CORRECTNESS_GUARD_METRICS else "ms",
    }


def test_aggregate_report_missing_invalid_and_offline_rebuild(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "valid")
    atomic_write_json(
        fixture.trial / "state.json",
        {"prompt": "must-not-enter-report", "fabricated_metric": 0},
    )
    assert validate_study(fixture.study)["study_valid"] is True
    aggregate = aggregate_study(fixture.study)
    assert aggregate["valid_trial_count"] == 1
    report = report_study(fixture.study)
    assert report["invalid_trial_count"] == 0
    machine_report = json.loads(
        (fixture.study / "report" / "report.v1.json").read_text(encoding="utf-8")
    )
    assert "must-not-enter-report" not in json.dumps(machine_report, sort_keys=True)
    assert all(
        item["mad_sensitivity"]["primary_analysis_unchanged"] is True
        for item in machine_report["metrics"]
    )
    trial_rows = pq.read_table(
        fixture.study / "aggregates" / "trial_metrics.parquet"
    ).to_pylist()
    dct = next(row for row in trial_rows if row["metric_name"] == "dct_ms")
    assert dct["primary_value"] == 7.0
    assert dct["p99_status"] == "insufficient_sample"

    report_bytes = (fixture.study / "report" / "report.v1.json").read_bytes()
    (fixture.study / "report" / "report.md").unlink()
    shutil.rmtree(fixture.study / "report" / "plot_data")
    rebuild_report_views(fixture.study)
    assert (fixture.study / "report" / "report.v1.json").read_bytes() == report_bytes
    first_csv = (fixture.study / "aggregates" / "cell_metrics.csv").read_bytes()
    (fixture.study / "aggregates" / "cell_metrics.csv").unlink()
    rebuild_aggregate_csv(fixture.study)
    assert (fixture.study / "aggregates" / "cell_metrics.csv").read_bytes() == first_csv

    invalid = _fixture(
        tmp_path / "invalid",
        flush_changes={"dropped_control_event_count": 1},
    )
    assert validate_study(invalid.study)["study_valid"] is False
    aggregate_study(invalid.study)
    invalid_rows = pq.read_table(
        invalid.study / "aggregates" / "trial_metrics.parquet"
    ).to_pylist()
    assert all(row["metric_valid"] is False for row in invalid_rows)
    assert all(row["primary_value"] is None for row in invalid_rows)
    assert all("dropped_control_events" in row["reason_codes"] for row in invalid_rows)


def test_aggregate_and_report_are_hashseed_deterministic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    validate_study(fixture.study)
    snapshots: list[dict[str, bytes]] = []
    for seed in ("1", "991"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        for command in ("aggregate", "report"):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ascend_maze.benchmark.cli",
                    command,
                    str(fixture.study),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
            )
            assert completed.stderr == b""
            json.loads(completed.stdout)
        paths = [
            fixture.study / "aggregates" / "run_metrics.parquet",
            fixture.study / "aggregates" / "trial_metrics.parquet",
            fixture.study / "aggregates" / "cell_metrics.csv",
            fixture.study / "aggregates" / "comparisons.csv",
            fixture.study / "aggregates" / "validity.csv",
            fixture.study / "report" / "report.v1.json",
            fixture.study / "report" / "report.md",
            fixture.study / "report" / "plot_data" / "metrics.csv",
            fixture.study / "report" / "plot_data" / "comparisons.csv",
            fixture.study / "report" / "plot_data" / "validity.csv",
        ]
        snapshots.append({str(path.relative_to(fixture.study)): path.read_bytes() for path in paths})
    assert snapshots[0] == snapshots[1]
