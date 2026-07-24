from __future__ import annotations

import asyncio
from dataclasses import replace

from ascend_maze import Workflow
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.control import InMemoryControllerRecoveryStore
from ascend_maze.core.clock import ManualClock
from ascend_maze.data import InMemoryDataStore
from ascend_maze.lifecycle import AttemptStatus, RunStatus, TaskStatus
from ascend_maze.placement import NodeCapacity
from ascend_maze.runtime import FakeRuntimeBackend
from task_fixtures import barrier, timeout_task


CONFIG_FINGERPRINT = "c" * 64
ENVIRONMENT_FINGERPRINT = "e" * 64
OWNER_GENERATION = "async_dispatch_owner"


def _node(node_id: str) -> NodeCapacity:
    return NodeCapacity(
        node_id=node_id,
        boot_id=f"boot_{node_id}",
        node_ip="127.0.0.1",
        cpu_total=1,
        mem_total_mb=256,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=1,
        observed_free_mem_mb=256,
    )


class _FirstDispatchBlockedRuntime(FakeRuntimeBackend):
    def __init__(self, data_store: InMemoryDataStore) -> None:
        super().__init__(
            data_store=data_store,
            owner_generation=OWNER_GENERATION,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        )
        self.first_dispatch_entered = asyncio.Event()
        self.second_dispatch_entered = asyncio.Event()
        self.release_first_dispatch = asyncio.Event()
        self.dispatch_order: list[str] = []

    async def dispatch(self, request, lease):  # type: ignore[no-untyped-def]
        self.dispatch_order.append(request.task_id)
        if len(self.dispatch_order) == 1:
            self.first_dispatch_entered.set()
            await self.release_first_dispatch.wait()
        else:
            self.second_dispatch_entered.set()
        return await super().dispatch(request, lease)


def _controller(runtime: FakeRuntimeBackend, store: InMemoryDataStore):
    return InMemoryController(
        config_fingerprint=CONFIG_FINGERPRINT,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        build_revision="async_dispatch_test",
        node_capacities=(_node("node_a"), _node("node_b")),
        controller_generation=OWNER_GENERATION,
        data_owner_generation=OWNER_GENERATION,
        data_store=store,
        runtime=runtime,
    )


def test_blocked_cold_dispatch_does_not_block_other_ready_task_or_events() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        runtime = _FirstDispatchBlockedRuntime(store)
        controller = _controller(runtime, store)
        await controller.start()

        workflow = Workflow("async-dispatch-independent-entries")
        first = workflow.add_task(barrier, task_name="first")
        second = workflow.add_task(barrier, task_name="second")
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="async_dispatch_independent_entries",
        )
        assert outcome.run_id is not None

        await asyncio.wait_for(runtime.first_dispatch_entered.wait(), 1)
        await asyncio.wait_for(runtime.second_dispatch_entered.wait(), 1)

        second_task_id = runtime.dispatch_order[1]
        for _ in range(1_000):
            if (
                controller.snapshot(outcome.run_id).task(second_task_id).status
                is TaskStatus.SUCCEEDED
            ):
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("second Task event was blocked by first dispatch")

        first_task_id = runtime.dispatch_order[0]
        assert controller.snapshot(outcome.run_id).task(first_task_id).status.value == (
            "starting"
        )
        runtime.release_first_dispatch.set()
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert {first.task_id, second.task_id} == set(runtime.dispatch_order)
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_cancel_interrupts_pending_dispatch_and_releases_lease() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        runtime = _FirstDispatchBlockedRuntime(store)
        controller = _controller(runtime, store)
        await controller.start()

        workflow = Workflow("cancel-pending-dispatch")
        node = workflow.add_task(barrier)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="cancel_pending_dispatch",
        )
        assert outcome.run_id is not None
        await asyncio.wait_for(runtime.first_dispatch_entered.wait(), 1)

        cancelled = await asyncio.wait_for(
            controller.cancel_run(outcome.run_id),
            timeout=1,
        )
        assert cancelled.status is RunStatus.CANCELLED
        assert cancelled.task(node.task_id).status is TaskStatus.CANCELLED
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        assert controller.core.active_dispatch_ids() == ()
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_dispatch_deadline_interrupts_pending_startup_without_blocking_loop() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        runtime = _FirstDispatchBlockedRuntime(store)
        clock = ManualClock(monotonic_ms=1_000, wall_ms=2_000)
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="async_dispatch_deadline_test",
            node_capacities=(_node("node_a"),),
            controller_generation=OWNER_GENERATION,
            data_owner_generation=OWNER_GENERATION,
            data_store=store,
            runtime=runtime,
            clock=clock,
            dispatch_timeout_ms=10,
        )
        await controller.start()

        workflow = Workflow("deadline-pending-dispatch")
        node = workflow.add_task(timeout_task, inputs={"value": "never-started"})
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="deadline_pending_dispatch",
        )
        assert outcome.run_id is not None
        await runtime.first_dispatch_entered.wait()
        assert controller.snapshot(outcome.run_id).task(node.task_id).status.value == (
            "starting"
        )

        clock.advance(11)
        await asyncio.wait_for(controller.core.wake_deadlines(), timeout=1)
        terminal = controller.snapshot(outcome.run_id)
        assert terminal.status is RunStatus.FAILED
        assert terminal.task(node.task_id).status is TaskStatus.FAILED
        assert terminal.task(node.task_id).attempts[0].error is not None
        assert (
            terminal.task(node.task_id).attempts[0].error.error_code
            == "worker_start_failed"
        )
        assert controller.core.pending_dispatch_count(outcome.run_id) == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_recovery_fences_starting_attempt_from_previous_generation() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        recovery = InMemoryControllerRecoveryStore()
        runtime = _FirstDispatchBlockedRuntime(store)
        first = InMemoryController(
            cluster_id="async_dispatch_recovery",
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="async_dispatch_recovery_test",
            node_capacities=(_node("node_a"),),
            controller_generation="controller_1",
            data_owner_generation=OWNER_GENERATION,
            data_store=store,
            runtime=runtime,
            recovery_store=recovery,
        )
        await first.start()
        workflow = Workflow("recover-pending-dispatch")
        node = workflow.add_task(barrier)
        outcome = await InMemoryRuntimeClient(first).submit(
            workflow,
            inputs={},
            submission_id="recover_pending_dispatch",
        )
        assert outcome.run_id is not None
        await runtime.first_dispatch_entered.wait()
        before = first.snapshot(outcome.run_id).task(node.task_id)
        assert before.status.value == "starting"
        assert before.attempts[0].status.value == "dispatched"
        await first.crash()

        second = InMemoryController(
            cluster_id="async_dispatch_recovery",
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="async_dispatch_recovery_test",
            node_capacities=(_node("node_a"),),
            controller_generation="controller_2",
            data_store=store,
            recovery_store=recovery,
        )
        await second.start()
        recovered = second.snapshot(outcome.run_id)
        assert recovered.status is RunStatus.INTERRUPTED
        assert recovered.task(node.task_id).status is TaskStatus.CANCELLED
        assert recovered.task(node.task_id).attempts[0].status.value == "cancelled"
        assert second.core.pending_dispatch_count() == 0
        assert second.placement.active_lease_count() == 0
        assert second.deadlines.active_count == 0
        await second.destroy_run(outcome.run_id)
        await second.close()

    asyncio.run(scenario())


def test_binding_generation_change_cancels_pending_startup_and_retries() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        runtime = _FirstDispatchBlockedRuntime(store)
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="async_dispatch_binding_test",
            node_capacities=(_node("node_a"),),
            controller_generation=OWNER_GENERATION,
            data_owner_generation=OWNER_GENERATION,
            data_store=store,
            runtime=runtime,
        )
        await controller.start()
        workflow = Workflow("binding-change-pending-dispatch")
        node = workflow.add_task(barrier)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="binding_change_pending_dispatch",
        )
        assert outcome.run_id is not None
        await runtime.first_dispatch_entered.wait()

        controller.placement.register_node(
            replace(_node("node_a"), boot_id="boot_node_a_restarted")
        )
        assert controller.core.post_runtime_binding_invalidated(
            "node_a",
            "boot_node_a",
            reason="test binding generation changed",
        )
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        attempts = terminal.task(node.task_id).attempts
        assert terminal.status is RunStatus.SUCCEEDED
        assert [item.status for item in attempts] == [
            AttemptStatus.FAILED,
            AttemptStatus.SUCCEEDED,
        ]
        assert attempts[0].node_id == attempts[1].node_id == "node_a"
        assert controller.core.pending_dispatch_count() == 0
        assert controller.placement.active_lease_count() == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())
