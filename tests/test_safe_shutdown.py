from __future__ import annotations

import asyncio

import pytest

from ascend_maze import Workflow
from ascend_maze.control import (
    ControllerLifecycleState,
    InMemoryController,
    InMemoryRuntimeClient,
    ShutdownMode,
)
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.data import InMemoryDataStore
from ascend_maze.lifecycle import RunStatus, TaskStatus
from ascend_maze.inference import ModelInstanceState
from ascend_maze.placement import NodeCapacity
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime import FakeExecutionPlan, FakeRuntimeBackend
from task_fixtures import finish
from inference_helpers import make_controller as make_inference_controller
from inference_helpers import make_spec


CONFIG_FINGERPRINT = "c" * 64
ENVIRONMENT_FINGERPRINT = "e" * 64


def _node(*, cpu: int = 2) -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_a",
        node_ip="127.0.0.1",
        cpu_total=cpu,
        mem_total_mb=512,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=2,
        observed_free_mem_mb=512,
    )


def _controller(
    *,
    node: NodeCapacity | None = None,
    recorder: InMemoryRecorder | None = None,
    runtime: FakeRuntimeBackend | None = None,
    data_store: InMemoryDataStore | None = None,
) -> InMemoryController:
    return InMemoryController(
        config_fingerprint=CONFIG_FINGERPRINT,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        build_revision="stage7b_test",
        node_capacities=(node or _node(),),
        controller_generation="controller_stage7b",
        data_owner_generation="controller_stage7b",
        recorder=recorder,
        runtime=runtime,
        data_store=data_store,
        shutdown_drain_timeout_ms=500,
    )


async def _wait_task_status(
    controller: InMemoryController,
    run_id: str,
    task_id: str,
    status: TaskStatus,
) -> None:
    for _ in range(1_000):
        if controller.snapshot(run_id).task(task_id).status is status:
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"Task did not reach {status.value}")


def test_graceful_shutdown_drains_running_attempt_and_rejects_new_submissions() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        workflow = Workflow("graceful-shutdown")
        node = workflow.add_task(finish, inputs={"summary": "done"})
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=100),
        )
        client = InMemoryRuntimeClient(controller)
        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="graceful_shutdown",
        )
        assert outcome.run_id is not None
        await _wait_task_status(
            controller,
            outcome.run_id,
            node.task_id,
            TaskStatus.RUNNING,
        )

        shutdown_task = asyncio.create_task(
            controller.shutdown(drain_timeout_ms=1_000)
        )
        for _ in range(1_000):
            if controller.lifecycle_state is ControllerLifecycleState.DRAINING:
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("Controller did not enter draining")

        second = Workflow("rejected-during-drain")
        second.add_task(finish, inputs={"summary": "no"})
        with pytest.raises(StateTransitionError, match="submissions are closed"):
            await client.submit(
                second,
                inputs={},
                submission_id="rejected_during_drain",
            )
        result = await shutdown_task

        assert result.mode is ShutdownMode.GRACEFUL
        assert result.lifecycle_state is ControllerLifecycleState.STOPPED
        assert result.active_run_ids_at_start == (outcome.run_id,)
        assert result.drained_run_ids == (outcome.run_id,)
        assert result.terminated_run_ids == ()
        assert result.cleanup_confirmed
        assert result.recording_complete
        assert result.exit_code == 0
        assert controller.snapshot(outcome.run_id).status is RunStatus.SUCCEEDED
        assert controller.placement.active_lease_count() == 0
        assert controller.deadlines.active_count == 0
        assert result.steps == (
            "draining",
            "new_work_rejected",
            "dispatch_stopped",
            "runs_drained",
            "remaining_runs_cancelled",
            "models_stopped",
            "worker_pool_stopped",
            "leases_confirmed_or_quarantined",
            "recorder_flushed_closed",
            "runtime_generation_stopped",
            "control_transports_stopped",
            "stopped",
        )
        assert await controller.close() is result

    asyncio.run(scenario())


def test_graceful_shutdown_deadline_cancels_queued_run_without_attempt() -> None:
    async def scenario() -> None:
        controller = _controller(node=_node(cpu=0))
        await controller.start()
        workflow = Workflow("shutdown-queued")
        node = workflow.add_task(finish, inputs={"summary": "never"})
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="shutdown_queued",
        )
        assert outcome.run_id is not None
        result = await controller.shutdown(drain_timeout_ms=10)
        task = controller.snapshot(outcome.run_id).task(node.task_id)
        assert controller.snapshot(outcome.run_id).status is RunStatus.CANCELLED
        assert task.status is TaskStatus.CANCELLED
        assert task.attempt_count == 0
        assert result.drained_run_ids == ()
        assert result.terminated_run_ids == (outcome.run_id,)
        assert result.cleanup_confirmed
        assert result.exit_code == 0

    asyncio.run(scenario())


class _UncleanRuntime(FakeRuntimeBackend):
    async def close(self) -> None:
        raise RuntimeError("injected runtime close failure")


def test_force_shutdown_reports_live_dispatch_and_returns_nonzero() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        runtime = _UncleanRuntime(
            data_store=store,
            owner_generation="controller_stage7b",
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        )
        controller = _controller(runtime=runtime, data_store=store)
        await controller.start()
        workflow = Workflow("force-shutdown-incomplete")
        node = workflow.add_task(finish, inputs={"summary": "late"})
        runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=10_000, ignore_cancel=True),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="force_shutdown_incomplete",
        )
        assert outcome.run_id is not None
        await _wait_task_status(
            controller,
            outcome.run_id,
            node.task_id,
            TaskStatus.RUNNING,
        )
        result = await controller.shutdown(force=True)

        assert result.mode is ShutdownMode.FORCE
        assert controller.snapshot(outcome.run_id).status is RunStatus.CANCELLED
        assert result.terminated_run_ids == (outcome.run_id,)
        assert not result.cleanup_confirmed
        assert result.recording_complete
        assert result.exit_code == 1
        assert any(error.startswith("worker_runtime_stop:") for error in result.errors)
        assert any(
            item.kind == "runtime_dispatch"
            and item.state == "cleanup_unconfirmed"
            for item in result.incomplete_resources
        )
        attempt = controller.snapshot(outcome.run_id).task(node.task_id).attempts[0]
        assert controller.placement.lease_snapshot(attempt.lease_id).status.value == (
            "invalidated"
        )
        await FakeRuntimeBackend.close(runtime)

    asyncio.run(scenario())


def test_c8_failure_does_not_change_business_result_and_marks_shutdown_incomplete() -> None:
    async def scenario() -> None:
        recorder = InMemoryRecorder()
        controller = _controller(recorder=recorder)
        await controller.start()
        workflow = Workflow("shutdown-recorder-failure")
        node = workflow.add_task(finish, inputs={"summary": "ok"})
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="shutdown_recorder_failure",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        recorder.inject_emit_failure()
        result = await controller.shutdown()

        assert controller.snapshot(outcome.run_id).status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, node.task_id) == {"result": "ok"}
        assert result.cleanup_confirmed
        assert not result.recording_complete
        assert result.exit_code == 1
        assert result.flush_results[0].writer_errors

    asyncio.run(scenario())


def test_safe_shutdown_stops_idle_model_instance_before_releasing_lease(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", min_replicas=1)
        controller, inference, adapter = make_inference_controller(spec)
        await controller.start()
        for _ in range(1_000):
            instances = inference.model_instances()
            if instances and instances[0].state is ModelInstanceState.READY:
                break
            await asyncio.sleep(0.002)
        else:
            raise AssertionError("minimum model replica did not become ready")
        assert controller.placement.active_lease_count() == 1

        result = await controller.shutdown()

        assert result.cleanup_confirmed
        assert result.exit_code == 0
        assert adapter.stop_count == 1
        assert inference.model_instances()[0].state is ModelInstanceState.STOPPED
        assert controller.placement.active_lease_count() == 0
        assert result.steps.index("models_stopped") < result.steps.index(
            "worker_pool_stopped"
        )

    asyncio.run(scenario())
