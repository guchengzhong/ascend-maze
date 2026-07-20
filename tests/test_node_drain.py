from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ascend_maze import Workflow, task
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.inference import ModelInstanceState
from ascend_maze.lifecycle import RunStatus, TaskStatus
from ascend_maze.placement import NodeCapacity, NodeStatus
from ascend_maze.runtime import FakeExecutionPlan, FakeRuntimeBackend

from inference_helpers import make_controller, make_node, make_spec


@task(resources={"cpu_num": 1, "mem": 32})
def _drain_value(value: str):
    return {"value": value}


def _cpu_node() -> NodeCapacity:
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
        capabilities={"environment_fingerprint": "e" * 64},
    )


def _controller() -> InMemoryController:
    return InMemoryController(
        config_fingerprint="c" * 64,
        environment_fingerprint="e" * 64,
        build_revision="test",
        node_capacities=(_cpu_node(),),
    )


async def _wait_for_bound_task(controller: InMemoryController, run_id: str) -> None:
    for _ in range(200):
        if any(
            item.lease.run_id == run_id and item.status.value == "bound"
            for item in controller.placement.lease_snapshots()
        ):
            return
        await asyncio.sleep(0.005)
    raise AssertionError("Task did not bind before the test deadline")


def test_graceful_node_drain_waits_for_active_task_and_resume_wakes_queue() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        try:
            workflow = Workflow("node-drain-graceful")
            value = workflow.input("value")
            first = workflow.add_task(_drain_value, inputs={"value": value})
            runtime = controller.runtime
            assert isinstance(runtime, FakeRuntimeBackend)
            runtime.set_plan(
                first.task_id,
                1,
                FakeExecutionPlan(execution_delay_ms=150),
            )
            first_run = await InMemoryRuntimeClient(controller).run(
                workflow,
                inputs={"value": "first"},
                submission_id="node_drain_first",
            )
            await _wait_for_bound_task(controller, first_run)

            draining = asyncio.create_task(
                controller.drain_node(
                    "node_a",
                    boot_id="boot_a",
                    timeout_ms=2_000,
                )
            )
            await asyncio.sleep(0.025)
            assert controller._node_snapshot("node_a").status is NodeStatus.DRAINING
            assert not draining.done()

            queued_workflow = Workflow("node-drain-queued")
            queued_value = queued_workflow.input("value")
            queued_workflow.add_task(
                _drain_value, inputs={"value": queued_value}
            )
            queued_run = await InMemoryRuntimeClient(controller).run(
                queued_workflow,
                inputs={"value": "second"},
                submission_id="node_drain_second",
            )
            await asyncio.sleep(0.025)
            assert controller.snapshot(queued_run).task_states[0].status in {
                TaskStatus.QUEUED,
                TaskStatus.READY,
            }
            assert not any(
                item.lease.run_id == queued_run
                and item.status.value in {"reserved", "bound"}
                for item in controller.placement.lease_snapshots()
            )

            assert (
                await controller.wait_run(first_run, timeout_seconds=2)
            ).status is RunStatus.SUCCEEDED
            drained = await draining
            assert drained.status == "drained"
            assert drained.cleanup_confirmed
            assert not drained.incomplete_resources
            assert not controller.snapshot(queued_run).terminal

            resumed = await controller.resume_node("node_a", boot_id="boot_a")
            assert resumed.status == "healthy"
            assert (
                await controller.wait_run(queued_run, timeout_seconds=2)
            ).status is RunStatus.SUCCEEDED
        finally:
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_node_drain_timeout_is_structured_and_retry_is_idempotent() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        try:
            workflow = Workflow("node-drain-timeout")
            value = workflow.input("value")
            node = workflow.add_task(_drain_value, inputs={"value": value})
            runtime = controller.runtime
            assert isinstance(runtime, FakeRuntimeBackend)
            runtime.set_plan(
                node.task_id,
                1,
                FakeExecutionPlan(execution_delay_ms=150),
            )
            run_id = await InMemoryRuntimeClient(controller).run(
                workflow,
                inputs={"value": "payload"},
                submission_id="node_drain_timeout",
            )
            await _wait_for_bound_task(controller, run_id)

            timed_out = await controller.drain_node(
                "node_a", boot_id="boot_a", timeout_ms=1
            )
            assert timed_out.status == "draining"
            assert timed_out.timed_out
            assert timed_out.exit_code == 1
            assert any(
                item.kind == "placement_lease"
                for item in timed_out.incomplete_resources
            )

            await controller.wait_run(run_id, timeout_seconds=2)
            completed = await controller.drain_node(
                "node_a", boot_id="boot_a", timeout_ms=1_000
            )
            replayed = await controller.drain_node(
                "node_a", boot_id="boot_a", timeout_ms=1_000
            )
            assert completed.status == replayed.status == "drained"
            assert completed.cleanup_confirmed and replayed.cleanup_confirmed
        finally:
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_force_node_drain_cancels_affected_run_through_scheduler_cleanup() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        try:
            workflow = Workflow("node-drain-force")
            value = workflow.input("value")
            node = workflow.add_task(_drain_value, inputs={"value": value})
            runtime = controller.runtime
            assert isinstance(runtime, FakeRuntimeBackend)
            runtime.set_plan(
                node.task_id,
                1,
                FakeExecutionPlan(execution_delay_ms=10_000),
            )
            run_id = await InMemoryRuntimeClient(controller).run(
                workflow,
                inputs={"value": "payload"},
                submission_id="node_drain_force",
            )
            await _wait_for_bound_task(controller, run_id)

            result = await controller.drain_node(
                "node_a", boot_id="boot_a", force=True, timeout_ms=2_000
            )
            assert result.status == "drained"
            assert result.cancelled_run_ids == (run_id,)
            assert controller.snapshot(run_id).status is RunStatus.CANCELLED
            assert controller.placement.active_lease_count(run_id) == 0
        finally:
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_node_actions_require_current_boot_and_full_drain() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        try:
            with pytest.raises(ValueError, match="boot_id"):
                await controller.drain_node("node_a")
            with pytest.raises(StateTransitionError, match="boot_id changed"):
                await controller.drain_node("node_a", boot_id="stale_boot")
            with pytest.raises(StateTransitionError, match="fully drained"):
                controller.placement.set_node_status(
                    "node_a", NodeStatus.DRAINING, now_ms=0
                )
                await controller.resume_node("node_a", boot_id="boot_a")
        finally:
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_node_drain_waits_for_model_route_then_stops_instance(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        spec = make_spec(
            tmp_path / "model",
            min_replicas=1,
            scale_cooldown_ms=0,
        )
        controller, inference, _ = make_controller(
            spec,
            nodes=(make_node("node_a"),),
        )
        await controller.start()
        try:
            ready = await inference.wait_ready(
                spec.model_id, timeout_seconds=2
            )
            instance = ready[0]
            route = inference.router.acquire(
                run_id="external_run",
                task_id="external_task",
                attempt=1,
                model_id=spec.model_id,
                session_key_hash=None,
                dispatch_deadline_ms=inference.clock.monotonic_ms() + 5_000,
            ).lease
            assert route is not None
            assert inference.activate_route(route.route_lease_id)

            timed_out = await controller.drain_node(
                "node_a",
                boot_id="boot_node_a",
                timeout_ms=1,
            )
            assert timed_out.timed_out
            current = inference.instances.snapshot(instance.instance_id)
            assert current.state is ModelInstanceState.DRAINING
            assert current.route_occupancy == 1

            assert await inference.release_route(route, reason="test_complete")
            completed = await controller.drain_node(
                "node_a",
                boot_id="boot_node_a",
                timeout_ms=2_000,
            )
            assert completed.status == "drained"
            stopped = inference.instances.snapshot(instance.instance_id)
            assert stopped.state is ModelInstanceState.STOPPED
            assert stopped.placement_lease_id is None
            assert controller.placement.active_lease_count() == 0
            await asyncio.sleep(0.1)
            assert not any(
                item.node_id == "node_a"
                and item.state is ModelInstanceState.READY
                for item in inference.model_instances()
            )

            resumed = await controller.resume_node(
                "node_a", boot_id="boot_node_a"
            )
            assert resumed.status == "healthy"
            ready_after_resume = await inference.wait_ready(
                spec.model_id, timeout_seconds=2
            )
            assert ready_after_resume[0].node_id == "node_a"
            assert ready_after_resume[0].generation > instance.generation
        finally:
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())
