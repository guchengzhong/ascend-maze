from __future__ import annotations

import asyncio
import os
from pathlib import Path

import ray
import pytest

from ascend_maze import Workflow, task
from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.control.node_rpc import (
    NodeAgent,
    NodeAgentIdentity,
    control_pb2,
    report_worker_event,
)
from ascend_maze.control.ray_controller import RayHostController
from ascend_maze.contracts.resources import ReservationVector
from ascend_maze.contracts.worker import (
    StandbyWorkerState,
    WarmupManifest,
    WorkerPoolConfig,
    WorkerPoolProfileConfig,
    WorkerProfile,
)
from ascend_maze.lifecycle import AttemptStatus, RunStatus, TaskStatus
from ascend_maze.core.errors import RunDataIndexError, StateTransitionError
from ascend_maze.placement import LeaseStatus, NodeCapacity, NodeStatus
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind


CONFIG = "c" * 64
ENVIRONMENT = "e" * 64


def _node() -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_1",
        node_ip="127.0.0.1",
        cpu_total=2,
        mem_total_mb=1_024,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=2,
        observed_free_mem_mb=1_024,
    )


def _controller(
    *,
    ray_namespace: str,
    recovery_path: Path,
    generation: str,
    worker_pool_config: WorkerPoolConfig | None = None,
) -> RayHostController:
    return RayHostController(
        cluster_id="cluster_controller_recovery",
        authorization_token=b"recovery-token",
        ray_namespace=ray_namespace,
        config_fingerprint=CONFIG,
        environment_fingerprint=ENVIRONMENT,
        build_revision="test",
        node_capacities=(_node(),),
        controller_generation=generation,
        recovery_path=recovery_path,
        worker_pool_config=worker_pool_config,
    )


def _pool_config() -> WorkerPoolConfig:
    return WorkerPoolConfig(
        mode="zero_hbm_standby",
        profiles=(
            WorkerPoolProfileConfig(
                profile=WorkerProfile.CPU,
                min_idle=1,
                max_idle=1,
                max_total=1,
                replenish_concurrency=1,
                idle_ttl_ms=60_000,
                acquire_timeout_ms=10_000,
                max_tasks_per_worker=4,
                max_worker_lifetime_ms=120_000,
                max_rss_growth_mb=256,
                standby_resources=ReservationVector(1, 64, 0, 0, 0),
                warmup_manifest=WarmupManifest(("json",)),
            ),
        ),
        reconcile_interval_ms=25,
    )


async def _wait_task_running(
    controller: RayHostController,
    run_id: str,
    task_id: str,
) -> None:
    for _ in range(1_000):
        snapshot = controller.snapshot(run_id)
        if snapshot.task(task_id).status is TaskStatus.RUNNING:
            return
        if snapshot.terminal:
            raise AssertionError(f"Ray recovery Run terminated early: {snapshot}")
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"Ray recovery Task did not enter running: {controller.snapshot(run_id)}"
    )


async def _wait_reconciled(controller: RayHostController) -> None:
    for _ in range(1_000):
        if not controller.recovery_pending_nodes:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"Controller recovery did not reconcile: {controller.recovery_pending_nodes}"
    )


def test_ray_controller_restart_recovers_owner_fences_events_and_avoids_redispatch(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        @task
        def recovery_load(value: str):
            return {"text": value}

        @task
        def recovery_slow(text: str):
            import time

            time.sleep(30)
            return {"result": text}

        recovery_path = tmp_path / "controller-recovery.sqlite3"
        first = _controller(
            ray_namespace=ray_namespace,
            recovery_path=recovery_path,
            generation="controller_1",
        )
        await first.start()
        agent = NodeAgent(
            identity=NodeAgentIdentity(
                cluster_id="cluster_controller_recovery",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            ),
            authorization_token=b"recovery-token",
            heartbeat_interval_ms=25,
        )
        await agent.start(controller_endpoint=first.node_rpc_endpoint)

        workflow = Workflow("ray-controller-recovery")
        value = workflow.input("value")
        loaded = workflow.add_task(recovery_load, inputs={"value": value})
        slow = workflow.add_task(
            recovery_slow,
            inputs={"text": loaded.outputs["text"]},
        )
        outcome = await InMemoryRuntimeClient(first).submit(
            workflow,
            inputs={"value": "payload"},
            submission_id="submission_ray_controller_recovery",
        )
        assert outcome.run_id is not None
        run_id = outcome.run_id
        await _wait_task_running(first, run_id, slow.task_id)
        before = first.snapshot(run_id)
        assert before.task(loaded.task_id).status is TaskStatus.SUCCEEDED
        old_attempt = before.task(slow.task_id).attempts[-1]
        assert old_attempt.status is AttemptStatus.RUNNING
        assert first.result(run_id, loaded.task_id) == {"text": "payload"}
        owner_descriptor = first.ray_data_store.descriptor
        old_runtime_generation = agent.runtime_generation
        assert old_runtime_generation is not None

        await first.crash()
        late_handle = first.ray_data_store.put_staged_for_runtime_node(
            "late",
            owner_descriptor.owner_generation,
            node_id="node_a",
            boot_id="boot_1",
            runtime_generation=old_runtime_generation,
        )
        stale_meta = agent._next_meta()
        stale_message_id = str(stale_meta.message_id)
        stale_message = control_pb2.AgentStreamMessage(
            heartbeat=control_pb2.NodeHeartbeat(meta=stale_meta)
        )

        second = _controller(
            ray_namespace=ray_namespace,
            recovery_path=recovery_path,
            generation="controller_2",
        )
        try:
            await second.start()
            assert second.ray_data_store.descriptor == owner_descriptor
            await agent.reconnect(second.node_rpc_endpoint)
            await agent._queue.put(stale_message)
            stale_worker_event = RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_RESULT,
                dispatch_id=old_attempt.dispatch_id,
                run_id=run_id,
                task_id=slow.task_id,
                attempt=old_attempt.attempt,
                lease_id=old_attempt.lease_id,
                route_lease_id=None,
                occurred_at_ms=old_attempt.dispatched_at_ms,
                output_handles=(("result", late_handle),),
            )
            assert agent.endpoint is not None
            with pytest.raises(RuntimeError, match="stale_worker_generation"):
                await asyncio.to_thread(
                    report_worker_event,
                    endpoint=agent.endpoint,
                    identity=agent.identity,
                    controller_generation="controller_1",
                    runtime_generation=old_runtime_generation,
                    event=stale_worker_event,
                    timeout_seconds=2,
                )
            await _wait_reconciled(second)
            for _ in range(200):
                if agent.message_ack(stale_message_id) is not None:
                    break
                await asyncio.sleep(0.01)
            assert agent.message_ack(stale_message_id) == "stale"
            assert agent.controller_generation == "controller_2"
            assert second.ray_data_store.state_of(late_handle) == "released"

            recovered = second.snapshot(run_id)
            assert recovered.status is RunStatus.INTERRUPTED
            assert recovered.task(loaded.task_id).status is TaskStatus.SUCCEEDED
            attempt = recovered.task(slow.task_id).attempts[-1]
            assert attempt.dispatch_id == old_attempt.dispatch_id
            assert attempt.status is AttemptStatus.CANCELLED
            assert (
                second.placement.lease_snapshot(old_attempt.lease_id).status
                is LeaseStatus.INVALIDATED
            )
            assert second.placement.snapshot().nodes[0].status is NodeStatus.HEALTHY
            assert second.ray_runtime.active_dispatch_count() == 0
            assert second.result(run_id, loaded.task_id) == {"text": "payload"}

            replay_client = InMemoryRuntimeClient(second)
            replay = replay_client.get_submission_status(
                "submission_ray_controller_recovery"
            )
            assert replay is not None
            assert replay.run_id == run_id
            assert second.ray_runtime.active_dispatch_count() == 0

            destroyed = await second.destroy_run(run_id)
            assert destroyed.tombstone.destroy_succeeded
            assert second.ray_data_store.active_count == 0
        finally:
            await agent.close(grace_seconds=0)
            await second.close()
            second.ray_data_store.close(kill_owner=True)

    asyncio.run(scenario())


def test_destroyed_checkpoint_rotates_missing_data_owner_and_accepts_new_run(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        @task
        def echo(value: str):
            return {"result": value}

        recovery_path = tmp_path / "destroyed-owner-rotation.sqlite3"
        first = _controller(
            ray_namespace=ray_namespace,
            recovery_path=recovery_path,
            generation="controller_rotation_1",
        )
        await first.start()
        agent = NodeAgent(
            identity=NodeAgentIdentity(
                cluster_id="cluster_controller_recovery",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_rotation_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_rotation_1",
            ),
            authorization_token=b"recovery-token",
            heartbeat_interval_ms=25,
        )
        await agent.start(controller_endpoint=first.node_rpc_endpoint)

        workflow = Workflow("destroyed-owner-rotation")
        value = workflow.input("value")
        echoed = workflow.add_task(echo, inputs={"value": value})
        first_outcome = await InMemoryRuntimeClient(first).submit(
            workflow,
            inputs={"value": "before-rotation"},
            submission_id="submission_before_owner_rotation",
        )
        assert first_outcome.run_id is not None
        old_run_id = first_outcome.run_id
        completed = await first.wait_run(old_run_id, timeout_seconds=10)
        assert completed.status is RunStatus.SUCCEEDED
        destroyed = await first.destroy_run(old_run_id)
        assert destroyed.tombstone.destroy_succeeded
        old_descriptor = first.ray_data_store.descriptor
        await first.close()
        first.ray_data_store.close(kill_owner=True)

        second = _controller(
            ray_namespace=ray_namespace,
            recovery_path=recovery_path,
            generation="controller_rotation_2",
        )
        try:
            await second.start()
            await agent.reconnect(second.node_rpc_endpoint)
            await _wait_reconciled(second)
            assert second.ray_data_store.descriptor != old_descriptor
            assert (
                second.data_owner_generation
                == second.ray_data_store.descriptor.owner_generation
            )
            assert second.ray_data_store.active_count == 0
            assert second.snapshot(old_run_id).status is RunStatus.SUCCEEDED
            with pytest.raises(RunDataIndexError, match="destroyed"):
                second.result(old_run_id, echoed.task_id)
            assert await second.destroy_run(old_run_id) == destroyed
            replay = InMemoryRuntimeClient(second).get_submission_status(
                "submission_before_owner_rotation"
            )
            assert replay is not None
            assert replay.run_id == old_run_id

            checkpoint = second.recovery_store.load()
            assert checkpoint is not None
            assert checkpoint.data_owner_generation == second.data_owner_generation
            assert checkpoint.data_store_descriptor == second.ray_data_store.descriptor
            assert all(not item.workflow_inputs for item in checkpoint.submissions)

            new_outcome = await InMemoryRuntimeClient(second).submit(
                workflow,
                inputs={"value": "after-rotation"},
                submission_id="submission_after_owner_rotation",
            )
            assert new_outcome.run_id is not None
            new_run_id = new_outcome.run_id
            new_completed = await second.wait_run(new_run_id, timeout_seconds=10)
            assert new_completed.status is RunStatus.SUCCEEDED
            assert second.result(new_run_id, echoed.task_id) == {
                "result": "after-rotation"
            }
            new_destroyed = await second.destroy_run(new_run_id)
            assert new_destroyed.tombstone.destroy_succeeded
            assert second.ray_data_store.active_count == 0
        finally:
            await agent.close(grace_seconds=0)
            await second.close()
            second.ray_data_store.close(kill_owner=True)

    asyncio.run(scenario())


def test_missing_data_owner_does_not_rotate_live_checkpoint(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        @task
        def slow(value: str):
            import time

            time.sleep(30)
            return {"result": value}

        recovery_path = tmp_path / "live-owner-rotation.sqlite3"
        first = _controller(
            ray_namespace=ray_namespace,
            recovery_path=recovery_path,
            generation="controller_live_rotation_1",
        )
        await first.start()
        agent = NodeAgent(
            identity=NodeAgentIdentity(
                cluster_id="cluster_controller_recovery",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_live_rotation_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_live_rotation_1",
            ),
            authorization_token=b"recovery-token",
            heartbeat_interval_ms=25,
        )
        await agent.start(controller_endpoint=first.node_rpc_endpoint)
        workflow = Workflow("live-owner-rotation")
        value = workflow.input("value")
        pending = workflow.add_task(slow, inputs={"value": value})
        outcome = await InMemoryRuntimeClient(first).submit(
            workflow,
            inputs={"value": "live"},
            submission_id="submission_live_owner_rotation",
        )
        assert outcome.run_id is not None
        await _wait_task_running(first, outcome.run_id, pending.task_id)
        await first.crash()
        first.ray_data_store.close(kill_owner=True)

        with pytest.raises(
            StateTransitionError,
            match="checkpoint Run is not successfully destroyed",
        ):
            _controller(
                ray_namespace=ray_namespace,
                recovery_path=recovery_path,
                generation="controller_live_rotation_2",
            )
        await agent.close(grace_seconds=0)

    asyncio.run(scenario())


def test_controller_generation_retires_and_replaces_idle_standby_worker(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def wait_idle(controller: RayHostController):
        for _ in range(400):
            idle = tuple(
                worker
                for worker in controller.worker_broker.snapshot().workers
                if worker.state is StandbyWorkerState.IDLE
            )
            if idle:
                return idle[0]
            await asyncio.sleep(0.025)
        raise AssertionError(controller.worker_broker.snapshot())

    async def process_exited(process_id: int) -> bool:
        for _ in range(400):
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                return True
            await asyncio.sleep(0.025)
        return False

    async def scenario() -> None:
        recovery_path = tmp_path / "standby-controller-recovery.sqlite3"
        config = _pool_config()
        first = _controller(
            ray_namespace=ray_namespace,
            recovery_path=recovery_path,
            generation="controller_1",
            worker_pool_config=config,
        )
        await first.start()
        agent = NodeAgent(
            identity=NodeAgentIdentity(
                cluster_id="cluster_controller_recovery",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            ),
            authorization_token=b"recovery-token",
            heartbeat_interval_ms=25,
        )
        await agent.start(controller_endpoint=first.node_rpc_endpoint)
        old_worker = await wait_idle(first)
        assert old_worker.process_id is not None
        assert old_worker.standby_lease_id is not None
        old_process_id = old_worker.process_id
        old_lease_id = old_worker.standby_lease_id
        await first.crash()
        assert await process_exited(old_process_id)

        second = _controller(
            ray_namespace=ray_namespace,
            recovery_path=recovery_path,
            generation="controller_2",
            worker_pool_config=config,
        )
        try:
            await second.start()
            assert (
                second.placement.lease_snapshot(old_lease_id).status
                is LeaseStatus.INVALIDATED
            )
            assert second.recovery_pending_nodes == ("node_a",)
            await agent.reconnect(second.node_rpc_endpoint)
            await _wait_reconciled(second)
            replacement = await wait_idle(second)
            assert replacement.worker_id != old_worker.worker_id
            assert replacement.standby_lease_id != old_lease_id
            assert replacement.process_id != old_process_id
            assert second.placement.active_lease_count() == 1
        finally:
            await agent.close(grace_seconds=0)
            await second.close()
            second.ray_data_store.close(kill_owner=True)

    asyncio.run(scenario())
