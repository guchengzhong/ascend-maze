from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event

import pyarrow.parquet as pq
import pytest

from ascend_maze import Workflow, task
from ascend_maze.control.client import InMemoryRuntimeClient
from ascend_maze.control.controller import InMemoryController
from ascend_maze.contracts.recording import (
    ExecutionEvent,
    FlushResult,
    ParquetRecorderConfig,
    RunRecordingContext,
)
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.lifecycle import RunStatus
from ascend_maze.placement import NodeCapacity
from ascend_maze.recording import ParquetRecorder


def _context(
    run_id: str = "run_1",
    *,
    expected: tuple[str, ...] = ("controller",),
) -> RunRecordingContext:
    return RunRecordingContext(
        schema_version=1,
        experiment_id=f"experiment_{run_id}",
        run_id=run_id,
        workflow_fingerprint="w" * 64,
        config_fingerprint="c" * 64,
        environment_fingerprint="e" * 64,
        build_revision="stage5d-test",
        started_wall_time_ms=10,
        initial_expected_producer_ids=expected,
    )


def _event(
    sequence: int,
    *,
    run_id: str = "run_1",
    producer_id: str = "controller",
    event_type: str = "task_running",
) -> ExecutionEvent:
    return ExecutionEvent(
        schema_version=1,
        event_id=f"event_{producer_id}_{sequence}",
        experiment_id=f"experiment_{run_id}",
        run_id=run_id,
        task_id="task_1",
        attempt=1,
        lease_id="lease_1",
        route_lease_id=None,
        model_instance_id=None,
        event_type=event_type,
        producer_id=producer_id,
        producer_sequence=sequence,
        node_id=None if producer_id == "controller" else "node_a",
        device_id=None,
        monotonic_time_ms=100 + sequence,
        wall_time_ms=1_000 + sequence,
        duration_ms=None,
        payload={"sequence": sequence, "safe_metadata": "value"},
    )


def _config(root: Path, **overrides: object) -> ParquetRecorderConfig:
    values: dict[str, object] = {
        "root_directory": str(root),
        "control_queue_capacity": 32,
        "telemetry_queue_capacity": 16,
        "batch_size": 2,
        "flush_interval_ms": 5,
        "compression": "zstd",
        "max_page_size": 100,
    }
    values.update(overrides)
    return ParquetRecorderConfig(**values)  # type: ignore[arg-type]


def test_parquet_flush_is_atomic_and_query_is_committed_only(tmp_path: Path) -> None:
    async def scenario() -> None:
        recorder = ParquetRecorder(
            _config(tmp_path),
            cursor_signing_key=b"stage5d-cursor-signing-key",
        )
        recorder.open_run(_context())
        events = tuple(
            _event(
                sequence,
                event_type=(
                    "device_resource_sample" if sequence == 5 else "task_running"
                ),
            )
            for sequence in range(1, 6)
        )
        assert all(recorder.emit(event) for event in events)
        assert recorder.emit(events[0])
        with pytest.raises(RuntimeError, match="before flush"):
            recorder.get_run_events("run_1")

        result = await recorder.flush_run("run_1", 5_000)
        assert result.recording_complete
        assert result.dropped_control_event_count == 0
        assert result.dropped_telemetry_count == 0
        assert len(result.committed_files) >= 3
        assert result == await recorder.flush_run("run_1", 5_000)
        assert all(Path(path).is_file() for path in result.committed_files)
        assert not tuple(tmp_path.rglob("*.tmp"))

        context_file = next(
            Path(path) for path in result.committed_files if ".context." in path
        )
        context_table = pq.read_table(context_file)
        assert context_table.schema.metadata == {
            b"ascend_maze_schema": b"run_recording_context_v1"
        }
        assert context_table.to_pylist()[0]["run_id"] == "run_1"
        event_file = next(
            Path(path) for path in result.committed_files if ".control." in path
        )
        assert pq.read_schema(event_file).metadata == {
            b"ascend_maze_schema": b"execution_event_v1"
        }

        collected: list[ExecutionEvent] = []
        cursor = None
        while True:
            page = recorder.get_run_events("run_1", cursor=cursor, limit=2)
            collected.extend(page.events)
            if page.exhausted:
                assert page.next_cursor is None
                break
            assert page.next_cursor is not None
            assert not page.next_cursor.isdigit()
            cursor = page.next_cursor
        assert {event.event_id for event in collected} == {
            event.event_id for event in events
        }
        assert all(event.payload["safe_metadata"] == "value" for event in collected)

        first = recorder.get_run_events("run_1", limit=1)
        assert first.next_cursor is not None
        payload, signature = first.next_cursor.split(".", 1)
        tampered = f"{payload}.{'A' if signature[0] != 'A' else 'B'}{signature[1:]}"
        with pytest.raises(ContractValidationError, match="signature"):
            recorder.get_run_events("run_1", cursor=tampered, limit=1)

        recorder.open_run(_context("run_2"))
        assert recorder.emit(_event(6, run_id="run_2"))
        await recorder.flush_run("run_2", 5_000)
        with pytest.raises(ContractValidationError, match="another run"):
            recorder.get_run_events(
                "run_2",
                cursor=first.next_cursor,
                limit=1,
            )
        await recorder.close(5_000)

    asyncio.run(scenario())


def test_control_capacity_is_reserved_when_telemetry_queue_is_full(
    tmp_path: Path,
) -> None:
    class BlockingRecorder(ParquetRecorder):
        def __init__(self, config: ParquetRecorderConfig) -> None:
            self.writer_entered = Event()
            self.release_writer = Event()
            super().__init__(config)

        def _write_event_group(self, key, events) -> None:
            self.writer_entered.set()
            assert self.release_writer.wait(5)
            super()._write_event_group(key, events)

    async def scenario() -> None:
        recorder = BlockingRecorder(
            _config(
                tmp_path,
                control_queue_capacity=1,
                telemetry_queue_capacity=1,
                batch_size=1,
            )
        )
        recorder.open_run(_context())
        assert recorder.emit(_event(1))
        assert await asyncio.to_thread(recorder.writer_entered.wait, 5)
        assert recorder.emit(_event(2, event_type="device_resource_sample"))
        assert not recorder.emit(_event(3, event_type="device_resource_sample"))
        assert recorder.emit(_event(4, event_type="task_succeeded"))
        assert not recorder.emit(_event(5, event_type="task_failed"))
        recorder.release_writer.set()

        result = await recorder.flush_run("run_1", 5_000)
        assert not result.recording_complete
        assert result.dropped_telemetry_count == 1
        assert result.dropped_control_event_count == 1
        await recorder.close(5_000)

    asyncio.run(scenario())


def test_writer_failure_is_reported_without_raising_from_emit(tmp_path: Path) -> None:
    async def scenario() -> None:
        recorder = ParquetRecorder(_config(tmp_path))
        recorder.open_run(_context())
        recorder.inject_write_failure()
        assert recorder.emit(_event(1))
        result = await recorder.flush_run("run_1", 5_000)
        assert not result.recording_complete
        assert result.writer_errors
        assert "injected Parquet writer failure" in result.writer_errors[0]
        assert not tuple(tmp_path.rglob("*.tmp"))
        await recorder.close(5_000)

    asyncio.run(scenario())


def test_remote_producer_flush_is_merged_without_directory_scanning(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        recorder = ParquetRecorder(_config(tmp_path))
        recorder.open_run(_context(expected=("controller", "node_agent:a")))
        assert recorder.emit(_event(1))
        remote_file = str(tmp_path / "remote" / "node.control.00000001.parquet")
        recorder.merge_producer_flush(
            "run_1",
            "node_agent:a",
            FlushResult("run_1", (remote_file,), 0, 0, 0, 0, (), True, 3),
        )
        result = await recorder.flush_run("run_1", 5_000)
        assert result.recording_complete
        assert remote_file in result.committed_files
        assert result.missing_producer_count == 0
        await recorder.close(5_000)

        missing = ParquetRecorder(_config(tmp_path / "missing"))
        missing.open_run(_context(expected=("controller", "node_agent:a")))
        assert missing.emit(_event(2))
        incomplete = await missing.flush_run("run_1", 5_000)
        assert not incomplete.recording_complete
        assert incomplete.missing_producer_count == 1
        await missing.close(5_000)

    asyncio.run(scenario())


def test_writer_failure_does_not_change_business_result_and_requires_force_destroy(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        @task
        def echo(value: str):
            return {"result": value}

        recorder = ParquetRecorder(_config(tmp_path))
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="stage5d-test",
            node_capacities=(
                NodeCapacity(
                    node_id="local",
                    boot_id="boot_1",
                    node_ip="127.0.0.1",
                    cpu_total=2,
                    mem_total_mb=1_024,
                    cpu_system_reserved=0,
                    mem_system_reserved_mb=0,
                    io_slots_total=2,
                    observed_free_mem_mb=1_024,
                ),
            ),
            recorder=recorder,
        )
        try:
            await controller.start()
            recorder.inject_write_failure()
            workflow = Workflow("recording-failure-isolation")
            output = workflow.add_task(echo, inputs={"value": "ok"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage5d_writer_failure",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=5)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, output.task_id) == {"result": "ok"}

            with pytest.raises(RuntimeError, match="recording is incomplete"):
                await controller.destroy_run(outcome.run_id)
            destroyed = await controller.destroy_run(outcome.run_id, force=True)
            assert not destroyed.flush_result.recording_complete
            assert destroyed.flush_result.writer_errors
            assert controller.indexes.active_count == 0
        finally:
            await controller.close()

    asyncio.run(scenario())
