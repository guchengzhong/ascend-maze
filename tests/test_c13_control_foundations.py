from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ascend_maze import Workflow
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.control.contracts import RequestJournal
from ascend_maze.control.process_lock import ControllerProcessLock
from ascend_maze.contracts.resources import ReservationVector
from ascend_maze.core.errors import SubmissionConflictError
from ascend_maze.lifecycle import RunStatus, TaskStatus
from ascend_maze.placement import NodeCapacity
from ascend_maze.recording import InMemoryRecorder
from task_fixtures import barrier


def _node() -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_a",
        node_ip="127.0.0.1",
        cpu_total=2,
        mem_total_mb=512,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=1,
        observed_free_mem_mb=512,
    )


def test_controller_uses_configured_recorder_flush_timeout() -> None:
    controller = InMemoryController(
        config_fingerprint="c" * 64,
        environment_fingerprint="e" * 64,
        build_revision="test",
        node_capacities=(_node(),),
        recorder_flush_timeout_ms=60_000,
    )
    assert controller.core.recorder_flush_timeout_ms == 60_000

    with pytest.raises(ValueError, match="recorder_flush_timeout_ms"):
        InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(_node(),),
            recorder_flush_timeout_ms=0,
        )


def test_request_journal_replays_only_the_same_operation_and_payload() -> None:
    journal = RequestJournal(capacity=2)
    result = {"status": "cancelled"}
    assert journal.remember(
        "request_1",
        operation="cancel_run",
        payload_digest="a" * 64,
        result=result,
    ) is result
    assert journal.lookup(
        "request_1", operation="cancel_run", payload_digest="a" * 64
    ) is result
    with pytest.raises(SubmissionConflictError, match="different operation or payload"):
        journal.lookup(
            "request_1", operation="destroy_run", payload_digest="a" * 64
        )

    journal.remember(
        "request_2", operation="cancel_run", payload_digest="b" * 64, result={}
    )
    journal.remember(
        "request_3", operation="cancel_run", payload_digest="c" * 64, result={}
    )
    assert journal.lookup(
        "request_1", operation="cancel_run", payload_digest="a" * 64
    ) is None


def test_process_lock_rejects_duplicate_and_preserves_replacement_inode(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "runtime" / "controller.pid").resolve()
    first = ControllerProcessLock(path, controller_generation="generation_1")
    first.acquire()
    second = ControllerProcessLock(path, controller_generation="generation_2")
    with pytest.raises(RuntimeError, match="another Controller owns"):
        second.acquire()

    replacement = path.with_suffix(".replacement")
    replacement.write_text(
        '{"controller_generation":"generation_new"}\n', encoding="utf-8"
    )
    replacement.replace(path)
    first.close()
    assert path.exists()
    assert "generation_new" in path.read_text(encoding="utf-8")


def test_watch_uses_controller_sequence_and_survives_recorder_failure() -> None:
    async def scenario() -> None:
        recorder = InMemoryRecorder()
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(_node(),),
            recorder=recorder,
            control_event_retention_count=100,
        )
        await controller.start()
        workflow = Workflow("watch-control-sequence")
        workflow.add_task(barrier)
        recorder.inject_emit_failure()
        run_id = await InMemoryRuntimeClient(controller).run(workflow, inputs={})
        terminal = await controller.wait_run(run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED

        first = controller.watch_run(run_id, after_sequence=0, limit=1)
        assert len(first.events) == 1
        assert first.events[0].sequence == first.next_sequence
        second = controller.watch_run(
            run_id,
            after_sequence=first.next_sequence,
            limit=100,
        )
        events = first.events + second.events
        assert [item.sequence for item in events] == sorted(
            item.sequence for item in events
        )
        assert any(item.event_type == "run_terminal" for item in events)
        assert second.run_terminal
        await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_watch_retention_requires_fresh_snapshot() -> None:
    async def scenario() -> None:
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(_node(),),
            control_event_retention_count=1,
        )
        await controller.start()
        workflow = Workflow("watch-retention")
        workflow.add_task(barrier)
        run_id = await InMemoryRuntimeClient(controller).run(workflow, inputs={})
        await controller.wait_run(run_id, timeout_seconds=2)
        page = controller.watch_run(run_id, after_sequence=1)
        assert page.snapshot_required
        assert page.events == ()
        await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_queue_snapshot_is_owned_by_scheduler_and_advances_cluster_version() -> None:
    async def scenario() -> None:
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(_node(),),
        )
        await controller.start()
        standby = controller.placement.reserve_standby(
            worker_id="blocking_worker",
            worker_generation=1,
            profile="cpu",
            node_id="node_a",
            boot_id="boot_a",
            resources=ReservationVector(2, 0, 0, 0, 0),
            now_ms=controller.clock.monotonic_ms(),
            startup_deadline_ms=controller.clock.monotonic_ms() + 5_000,
        )
        assert standby is not None
        placement_version = controller.placement.snapshot().snapshot_version
        workflow = Workflow("queue-snapshot")
        task = workflow.add_task(barrier)
        run_id = await InMemoryRuntimeClient(controller).run(workflow, inputs={})
        for _ in range(500):
            if controller.snapshot(run_id).task(task.task_id).status is TaskStatus.QUEUED:
                break
            await asyncio.sleep(0.002)
        else:
            raise AssertionError("Task did not remain queued behind reserved capacity")

        queue = controller.queue_snapshot()
        assert queue.snapshot_version == controller.cluster_snapshot_version()
        assert queue.snapshot_version > placement_version
        assert len(queue.tasks) == 1
        queued = queue.tasks[0]
        assert queued.run_id == run_id
        assert queued.task_id == task.task_id
        assert queued.status is TaskStatus.QUEUED
        assert queued.partition == "cpu"
        assert queued.queue_generation == 1
        assert queued.pending_reason is not None

        controller.placement.release_lease(
            standby.lease_id,
            now_ms=controller.clock.monotonic_ms(),
            reason="test_unblock",
        )
        assert controller.core.post_resource_changed("test_unblock")
        terminal = await controller.wait_run(run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        await controller.destroy_run(run_id)
        await controller.close()

    asyncio.run(scenario())
