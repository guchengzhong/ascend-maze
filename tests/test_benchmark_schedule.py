from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ascend_maze.benchmark.clock import VirtualBenchmarkClock
from ascend_maze.benchmark.contracts import (
    ArrivalSpec,
    MeasurementWindows,
    TrialManifest,
)
from ascend_maze.benchmark.loader import load_study_plan
from ascend_maze.benchmark.schedule import materialize_trial_schedule
from ascend_maze.benchmark.schedule_parquet import (
    validate_schedule_parquet,
    write_schedule_parquet,
)
from ascend_maze.benchmark.workload import (
    TraceSchedule,
    load_workload_dataset,
)
from benchmark_fixtures import write_experiment_spec


def _schedule(
    tmp_path: Path,
    arrival: ArrivalSpec,
    windows: MeasurementWindows,
    *,
    trace: TraceSchedule | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = load_study_plan(write_experiment_spec(tmp_path))
    trial = replace(plan.trials[0], pairing_seed=1234)
    spec = replace(plan.spec, arrival=arrival, windows=windows)
    attempt = TrialManifest.planned(trial)
    return materialize_trial_schedule(
        spec,
        trial,
        attempt.trial_attempt_id,
        load_workload_dataset(plan.spec.workload),
        trace=trace,
    )


def test_four_arrival_modes_have_deterministic_golden_offsets(tmp_path: Path) -> None:
    closed = _schedule(
        tmp_path / "closed",
        ArrivalSpec("closed_loop", concurrency=2),
        MeasurementWindows(0, 0, 4, 0, 1_000),
    )
    assert [item.scheduled_offset_ms for item in closed.measurement] == [
        None,
        None,
        None,
        None,
    ]

    fixed = _schedule(
        tmp_path / "fixed",
        ArrivalSpec("fixed_rate", rate_per_second=2),
        MeasurementWindows(0, 0, 0, 1_600, 1_000),
    )
    assert [item.scheduled_offset_ms for item in fixed.measurement] == [
        0,
        500,
        1_000,
        1_500,
    ]

    poisson = _schedule(
        tmp_path / "poisson",
        ArrivalSpec("poisson", rate_per_second=5),
        MeasurementWindows(0, 0, 0, 1_000, 1_000),
    )
    assert [item.scheduled_offset_ms for item in poisson.measurement] == [
        27,
        234,
        452,
        486,
        843,
        946,
    ]

    trace = _schedule(
        tmp_path / "trace",
        ArrivalSpec("trace_replay", trace_input="dataset"),
        MeasurementWindows(0, 0, 0, 101, 1_000),
        trace=TraceSchedule(1, (0, 0, 25, 100), "d" * 64),
    )
    assert [item.scheduled_offset_ms for item in trace.measurement] == [0, 0, 25, 100]


def test_paired_cells_share_offsets_and_input_order(tmp_path: Path) -> None:
    plan = load_study_plan(write_experiment_spec(tmp_path))
    block = [item for item in plan.trials if item.block_index == 0]
    dataset = load_workload_dataset(plan.spec.workload)
    schedules = [
        materialize_trial_schedule(
            plan.spec,
            trial,
            TrialManifest.planned(trial).trial_attempt_id,
            dataset,
        )
        for trial in block
    ]
    expected = [
        (item.scheduled_offset_ms, item.record_id, item.input_digest)
        for item in schedules[0].measurement
    ]
    for schedule in schedules[1:]:
        assert [
            (item.scheduled_offset_ms, item.record_id, item.input_digest)
            for item in schedule.measurement
        ] == expected
        assert [item.submission_id for item in schedule.measurement] != [
            item.submission_id for item in schedules[0].measurement
        ]


def test_schedule_parquet_is_stable_and_resume_validates_digest(tmp_path: Path) -> None:
    schedule = _schedule(
        tmp_path / "source",
        ArrivalSpec("fixed_rate", rate_per_second=4),
        MeasurementWindows(1, 0, 0, 1_000, 1_000),
    )
    path = tmp_path / "arrival_schedule.parquet"
    digest = write_schedule_parquet(path, schedule)
    assert validate_schedule_parquet(path, schedule) == digest
    changed = replace(
        schedule,
        measurement=(replace(schedule.measurement[0], record_id="changed"),),
    )
    with pytest.raises(Exception, match="schedule"):
        validate_schedule_parquet(path, changed)


def test_virtual_clock_waits_are_absolute_not_cumulative() -> None:
    async def scenario() -> None:
        clock = VirtualBenchmarkClock(monotonic_ms=100, wall_ms=1_000)
        assert await clock.wait_until(125) == 125
        assert await clock.wait_until(150) == 150
        assert await clock.wait_until(130) == 150
        assert clock.waited_deadlines == [125, 150, 130]

    import asyncio

    asyncio.run(scenario())


def test_poisson_schedule_bytes_are_hashseed_independent(tmp_path: Path) -> None:
    spec = write_experiment_spec(tmp_path)
    script = "\n".join(
        (
            "from ascend_maze.benchmark.canonical import canonical_json_bytes",
            "from ascend_maze.benchmark.contracts import TrialManifest",
            "from ascend_maze.benchmark.loader import load_study_plan",
            "from ascend_maze.benchmark.schedule import materialize_trial_schedule",
            "from ascend_maze.benchmark.workload import load_workload_dataset",
            f"plan = load_study_plan({str(spec)!r})",
            "trial = plan.trials[0]",
            "schedule = materialize_trial_schedule(",
            "    plan.spec, trial, TrialManifest.planned(trial).trial_attempt_id,",
            "    load_workload_dataset(plan.spec.workload),",
            ")",
            "import sys",
            "sys.stdout.buffer.write(canonical_json_bytes(schedule.canonical_payload()))",
        )
    )
    outputs: list[bytes] = []
    for seed in ("1", "777"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                env=environment,
            ).stdout
        )
    assert outputs[0] == outputs[1]
