from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ascend_maze.benchmark.clock import VirtualBenchmarkClock
from ascend_maze.benchmark.contracts import TrialManifest
from ascend_maze.benchmark.fake_runtime import (
    FakeBenchmarkRuntime,
    FakeBenchmarkRuntimeFactory,
)
from ascend_maze.benchmark.loader import load_study_plan
from ascend_maze.benchmark.orchestrator import (
    TrialOrchestrator,
    TrialPaths,
    resume_study,
    run_study,
)
from ascend_maze.benchmark.persistence import atomic_write_json
from benchmark_fixtures import write_experiment_spec
from benchmark_workload_fixtures import build


def _execution_spec(
    root: Path,
    *,
    arrival_mode: str = "closed_loop",
    warmup_runs: int = 1,
    warmup_duration_ms: int = 0,
    measurement_run_count: int = 4,
    measurement_duration_ms: int = 0,
    rate_per_second: float = 100.0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = build().compile().workflow_fingerprint
    return write_experiment_spec(
        root,
        study_kind="pilot",
        block_count=3,
        workflow_factory="benchmark_workload_fixtures:build",
        workflow_fingerprint=fingerprint,
        arrival_mode=arrival_mode,
        concurrency=2,
        rate_per_second=rate_per_second,
        warmup_runs=warmup_runs,
        warmup_duration_ms=warmup_duration_ms,
        measurement_run_count=measurement_run_count,
        measurement_duration_ms=measurement_duration_ms,
        drain_deadline_ms=1_000,
    )


def test_trial_closed_loop_records_counts_and_excludes_warmup(tmp_path: Path) -> None:
    async def scenario() -> None:
        plan = load_study_plan(_execution_spec(tmp_path / "spec"))
        trial = plan.trials[0]
        cell = next(item for item in plan.cells if item.cell_id == trial.cell_id)
        clock = VirtualBenchmarkClock()
        runtime = FakeBenchmarkRuntime(clock, run_duration_ms=10)
        orchestrator = TrialOrchestrator(
            runtime_factory=FakeBenchmarkRuntimeFactory(runtime),
            clock=clock,
        )
        state = await orchestrator.execute(
            plan=plan,
            cell=cell,
            trial=trial,
            paths=TrialPaths(tmp_path / "trial"),
        )

        assert state.state == "valid"
        assert state.warmup_counters.canonical_payload() == {
            "offered": 1,
            "issued": 1,
            "committed": 1,
            "terminal": 1,
            "succeeded": 1,
            "failed": 0,
            "timed_out": 0,
        }
        assert state.measurement_counters.canonical_payload() == {
            "offered": 4,
            "issued": 4,
            "committed": 4,
            "terminal": 4,
            "succeeded": 4,
            "failed": 0,
            "timed_out": 0,
        }
        assert runtime.max_active_runs == 2
        run_manifest = json.loads((tmp_path / "trial/run_manifest.json").read_text())
        assert run_manifest["warmup_excluded_from_measurement"] is True
        assert len(runtime.flush_calls) == 5
        assert len(runtime.destroy_calls) == 5
        assert len({request for _, request in runtime.flush_calls}) == 5

    asyncio.run(scenario())


def test_open_arrivals_use_absolute_deadlines_without_sleep_drift(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        plan = load_study_plan(
            _execution_spec(
                tmp_path / "spec",
                arrival_mode="fixed_rate",
                warmup_runs=0,
                measurement_run_count=0,
                measurement_duration_ms=40,
                rate_per_second=100,
            )
        )
        trial = plan.trials[0]
        cell = next(item for item in plan.cells if item.cell_id == trial.cell_id)
        clock = VirtualBenchmarkClock(monotonic_ms=100, wall_ms=1_000)
        runtime = FakeBenchmarkRuntime(clock, run_duration_ms=1)
        state = await TrialOrchestrator(
            runtime_factory=FakeBenchmarkRuntimeFactory(runtime),
            clock=clock,
        ).execute(
            plan=plan,
            cell=cell,
            trial=trial,
            paths=TrialPaths(tmp_path / "trial"),
        )
        measurement = [item for item in state.runs if item.phase == "measurement"]
        assert [item.scheduled_at_monotonic_ms for item in measurement] == [
            100,
            110,
            120,
            130,
        ]
        assert [item.issued_at_monotonic_ms for item in measurement] == [
            100,
            110,
            120,
            130,
        ]
        assert [item.arrival_lateness_ms for item in measurement] == [0, 0, 0, 0]

    asyncio.run(scenario())


def test_trace_replay_and_duration_warmup_share_the_open_arrival_path(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        plan = load_study_plan(
            _execution_spec(
                tmp_path / "spec",
                arrival_mode="trace_replay",
                warmup_runs=0,
                warmup_duration_ms=51,
                measurement_run_count=0,
                measurement_duration_ms=51,
            )
        )
        trial = plan.trials[0]
        cell = next(item for item in plan.cells if item.cell_id == trial.cell_id)
        clock = VirtualBenchmarkClock(monotonic_ms=100)
        state = await TrialOrchestrator(
            runtime_factory=FakeBenchmarkRuntimeFactory(
                FakeBenchmarkRuntime(clock, run_duration_ms=1)
            ),
            clock=clock,
        ).execute(
            plan=plan,
            cell=cell,
            trial=trial,
            paths=TrialPaths(tmp_path / "trial"),
        )
        assert state.state == "valid"
        assert state.warmup_counters.offered == 4
        assert state.measurement_counters.offered == 4
        for phase in ("warmup", "measurement"):
            assert [
                item.scheduled_offset_ms for item in state.runs if item.phase == phase
            ] == [0, 0, 25, 50]

    asyncio.run(scenario())


def test_commit_response_loss_replays_same_submission_without_second_run(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        plan = load_study_plan(_execution_spec(tmp_path / "spec", warmup_runs=0))
        trial = plan.trials[0]
        cell = next(item for item in plan.cells if item.cell_id == trial.cell_id)
        clock = VirtualBenchmarkClock()
        runtime = FakeBenchmarkRuntime(
            clock,
            lose_first_commit_response=True,
        )
        factory = FakeBenchmarkRuntimeFactory(runtime)
        orchestrator = TrialOrchestrator(runtime_factory=factory, clock=clock)
        paths = TrialPaths(tmp_path / "trial")
        state = await orchestrator.execute(
            plan=plan, cell=cell, trial=trial, paths=paths
        )
        assert state.state == "valid"
        assert len(runtime.runs_by_id) == 4
        assert len(runtime.submit_calls) == 8
        assert all(
            runtime.submit_calls.count(item.submission_id) == 2 for item in state.runs
        )

        calls = list(runtime.submit_calls)
        resumed = await TrialOrchestrator(runtime_factory=factory, clock=clock).execute(
            plan=plan, cell=cell, trial=trial, paths=paths
        )
        assert resumed == state
        assert runtime.submit_calls == calls

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "interrupted_state",
    ["planned", "preparing", "warming", "measuring", "draining", "flushing"],
)
def test_every_nonterminal_state_resumes_without_duplicate_runs(
    tmp_path: Path, interrupted_state: str
) -> None:
    class InterruptAfterState:
        def __init__(self) -> None:
            self.fired = False

        def __call__(self, stage: str, path: Path) -> None:
            if self.fired or stage != "after_replace" or path.name != "state.json":
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["state"] == interrupted_state:
                self.fired = True
                raise RuntimeError(f"interrupted at {interrupted_state}")

    async def scenario() -> None:
        root = tmp_path / interrupted_state
        plan = load_study_plan(
            _execution_spec(root / "spec", warmup_runs=1, measurement_run_count=3)
        )
        trial = plan.trials[0]
        cell = next(item for item in plan.cells if item.cell_id == trial.cell_id)
        clock = VirtualBenchmarkClock()
        runtime = FakeBenchmarkRuntime(clock)
        factory = FakeBenchmarkRuntimeFactory(runtime)
        paths = TrialPaths(root / "trial")
        failpoint = InterruptAfterState()
        with pytest.raises(RuntimeError, match="interrupted"):
            await TrialOrchestrator(
                runtime_factory=factory,
                clock=clock,
                failpoint=failpoint,
            ).execute(plan=plan, cell=cell, trial=trial, paths=paths)
        assert failpoint.fired

        completed = await TrialOrchestrator(
            runtime_factory=factory,
            clock=clock,
        ).execute(plan=plan, cell=cell, trial=trial, paths=paths)
        assert completed.state == "valid"
        assert len(runtime.runs_by_id) == 4
        assert len({item.submission_id for item in completed.runs}) == 4

    asyncio.run(scenario())


def test_atomic_write_failure_before_replace_preserves_old_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"revision": 1})

    def fail(stage: str, target: Path) -> None:
        del target
        if stage == "after_file_fsync":
            raise RuntimeError("injected atomic write failure")

    with pytest.raises(RuntimeError, match="injected"):
        atomic_write_json(path, {"revision": 2}, failpoint=fail)
    assert json.loads(path.read_text()) == {"revision": 1}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_resource_leak_marks_trial_invalid_and_blocks_next_cell(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec = _execution_spec(
            tmp_path / "spec", warmup_runs=0, measurement_run_count=1
        )
        clock = VirtualBenchmarkClock()
        runtime = FakeBenchmarkRuntime(clock, leak_resources=True)
        result = await run_study(
            spec,
            runtime_factory=FakeBenchmarkRuntimeFactory(runtime),
            output_root=tmp_path / "output",
            clock_factory=lambda: clock,
        )
        assert result.state == "blocked"
        assert result.completed_trials == 1
        assert result.blocked_reason == "resource_recovery_failed"
        with pytest.raises(Exception, match="blocked"):
            await resume_study(
                result.study_directory,
                runtime_factory=FakeBenchmarkRuntimeFactory(runtime),
                clock_factory=lambda: clock,
            )

    asyncio.run(scenario())


def test_study_resume_preserves_remaining_trial_order_and_is_idempotent(
    tmp_path: Path,
) -> None:
    class InterruptFirstMeasurement:
        fired = False

        def __call__(self, stage: str, path: Path) -> None:
            if self.fired or stage != "after_replace" or path.name != "state.json":
                return
            if json.loads(path.read_text(encoding="utf-8"))["state"] == "measuring":
                self.fired = True
                raise RuntimeError("study interrupted")

    async def scenario() -> None:
        spec = _execution_spec(
            tmp_path / "spec", warmup_runs=0, measurement_run_count=1
        )
        plan = load_study_plan(spec)
        clock = VirtualBenchmarkClock()
        runtime = FakeBenchmarkRuntime(clock)
        factory = FakeBenchmarkRuntimeFactory(runtime)
        failpoint = InterruptFirstMeasurement()
        with pytest.raises(RuntimeError, match="study interrupted"):
            await run_study(
                spec,
                runtime_factory=factory,
                output_root=tmp_path / "output",
                clock_factory=lambda: clock,
                failpoint=failpoint,
            )
        study_root = tmp_path / "output" / plan.spec.study_id
        result = await resume_study(
            study_root,
            runtime_factory=factory,
            clock_factory=lambda: clock,
        )
        assert result.state == "completed"
        opened = [item[0] for item in factory.open_calls]
        expected = [
            TrialManifest.planned(trial).trial_attempt_id for trial in plan.trials
        ]
        assert opened == [expected[0], *expected]
        manifest = json.loads((study_root / "study_manifest.json").read_text())
        assert [item["trial_id"] for item in manifest["trials"]] == [
            item.trial_id for item in plan.trials
        ]

        calls = list(factory.open_calls)
        repeated = await resume_study(
            study_root,
            runtime_factory=factory,
            clock_factory=lambda: clock,
        )
        assert repeated == result
        assert factory.open_calls == calls

    asyncio.run(scenario())
