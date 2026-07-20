from __future__ import annotations

import asyncio
import os
from pathlib import Path

import ray

from ascend_maze import Workflow, task
from ascend_maze.control.client import InMemoryRuntimeClient
from ascend_maze.control.local_rpc import UdsRuntimeClient
from ascend_maze.control.node_rpc import NodeAgent, NodeAgentIdentity
from ascend_maze.control.ray_controller import RayHostController
from ascend_maze.contracts.resources import ReservationVector
from ascend_maze.contracts.worker import (
    StandbyWorkerState,
    WarmupManifest,
    WorkerPoolConfig,
    WorkerPoolProfileConfig,
    WorkerProfile,
)
from ascend_maze.lifecycle import RunStatus
from ascend_maze.placement import NodeCapacity
from ascend_maze.runtime.ray_node_registry import RuntimeNodeStatus
from tests_ray_host.standby_task_fixtures import (
    drain_slow_cpu_task,
    leak_file_descriptor,
    leak_child_process,
    slow_cpu_task,
)


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
        capabilities={"environment_fingerprint": ENVIRONMENT},
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


def test_control_socket_opens_only_after_worker_broker_is_ready(
    ray_namespace: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        controller = RayHostController(
            cluster_id="cluster_start_order",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
            worker_pool_config=_pool_config(),
            control_socket_path=(tmp_path / "runtime" / "control.sock").resolve(),
        )
        assert controller.local_rpc is not None
        events: list[str] = []
        original_broker_start = controller.worker_broker.start
        original_rpc_start = controller.local_rpc.start

        async def broker_start() -> None:
            events.append("worker_broker_ready")
            await original_broker_start()

        async def rpc_start() -> None:
            events.append("control_socket_open")
            await original_rpc_start()

        monkeypatch.setattr(controller.worker_broker, "start", broker_start)
        monkeypatch.setattr(controller.local_rpc, "start", rpc_start)
        try:
            await controller.start()
            client = UdsRuntimeClient(
                (tmp_path / "runtime" / "control.sock").resolve()
            )
            await client.get_controller_status()
            pools = await client.query("GetWorkerPools")
            assert pools["worker_pool"]["mode"] == "zero_hbm_standby"
            assert pools["worker_pools"] == [pools["worker_pool"]]
            assert events == ["worker_broker_ready", "control_socket_open"]
        finally:
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_ray_standby_actor_reuses_pid_and_common_execution_path(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task(resources={"cpu_num": 1, "mem": 64})
        def first(value: str):
            return {"text": value + ":first"}

        @task(resources={"cpu_num": 1, "mem": 64})
        def second(text: str):
            return {"result": text + ":second"}

        controller = RayHostController(
            cluster_id="cluster_standby",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
            worker_pool_config=_pool_config(),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_standby",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(
                identity=identity,
                authorization_token=b"test-token",
                heartbeat_interval_ms=50,
            )
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            for _ in range(400):
                workers = controller.worker_broker.snapshot().workers
                if any(worker.state is StandbyWorkerState.IDLE for worker in workers):
                    break
                await asyncio.sleep(0.025)
            else:
                raise AssertionError(controller.worker_broker.snapshot())

            workflow = Workflow("ray-standby-reuse")
            value = workflow.input("value")
            loaded = workflow.add_task(first, inputs={"value": value})
            finished = workflow.add_task(
                second, inputs={"text": loaded.outputs["text"]}
            )
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={"value": "payload"},
                submission_id="submission_ray_standby_reuse",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=30)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, finished.task_id) == {
                "result": "payload:first:second"
            }
            task_pids: list[int] = []
            for task_id in (loaded.task_id, finished.task_id):
                attempt = terminal.task(task_id).attempts[0]
                worker_outcome = controller.ray_runtime.worker_outcome(
                    attempt.dispatch_id
                )
                assert worker_outcome is not None
                assert worker_outcome.reuse_safe
                task_pids.append(worker_outcome.worker_pid)
            assert task_pids[0] == task_pids[1]
            pool = controller.worker_broker.snapshot()
            assert pool.standby_hits == 2
            assert pool.cold_starts == 0
            assert pool.sanitize_failures == 0
            idle = [
                worker
                for worker in pool.workers
                if worker.state is StandbyWorkerState.IDLE
            ]
            assert len(idle) == 1
            assert idle[0].process_id == task_pids[0]
            assert idle[0].tasks_completed == 2
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            assert controller.placement.active_lease_count() == 1
            assert any(event.event_type == "standby_hit" for event in controller.pool_events)

            await controller.destroy_run(outcome.run_id)
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()
        assert controller.placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_node_drain_allows_active_standby_task_then_retires_and_resumes(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        controller = RayHostController(
            cluster_id="cluster_node_drain",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
            worker_pool_config=_pool_config(),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            agent = NodeAgent(
                identity=NodeAgentIdentity(
                    cluster_id="cluster_node_drain",
                    node_id="node_a",
                    boot_id="boot_1",
                    ray_node_id=ray.get_runtime_context().get_node_id(),
                    agent_generation="agent_1",
                    environment_fingerprint=ENVIRONMENT,
                    producer_id="node_agent:node_a:agent_1",
                ),
                authorization_token=b"test-token",
                heartbeat_interval_ms=25,
            )
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            for _ in range(400):
                idle = [
                    worker
                    for worker in controller.worker_broker.snapshot().workers
                    if worker.state is StandbyWorkerState.IDLE
                ]
                if idle:
                    break
                await asyncio.sleep(0.025)
            else:
                raise AssertionError(controller.worker_broker.snapshot())
            original_pid = idle[0].process_id

            workflow = Workflow("ray-node-drain-active")
            value = workflow.input("value")
            workflow.add_task(drain_slow_cpu_task, inputs={"value": value})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={"value": "completed"},
                submission_id="submission_ray_node_drain_active",
            )
            assert outcome.run_id is not None
            for _ in range(400):
                if any(
                    item.lease.run_id == outcome.run_id
                    and item.status.value == "bound"
                    for item in controller.placement.lease_snapshots()
                ):
                    break
                await asyncio.sleep(0.005)
            else:
                raise AssertionError("Task did not reach the running state")

            draining = asyncio.create_task(
                controller.drain_node(
                    "node_a",
                    boot_id="boot_1",
                    timeout_ms=5_000,
                )
            )
            await asyncio.sleep(0.05)
            assert not draining.done()
            assert (
                controller.node_registry.status("node_a")
                is RuntimeNodeStatus.DRAINING
            )
            assert controller.worker_broker.active_count("node_a") == 1

            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=5
            )
            assert terminal.status is RunStatus.SUCCEEDED
            result = await draining
            assert result.status == "drained"
            assert result.cleanup_confirmed
            assert controller.worker_broker.active_count("node_a") == 0
            assert controller.worker_broker.live_count("node_a", "boot_1") == 0
            assert controller.placement.active_lease_count() == 0
            assert (
                controller.node_registry.status("node_a")
                is RuntimeNodeStatus.DRAINED
            )
            try:
                assert original_pid is not None
                os.kill(original_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("drained Standby Worker process is still alive")

            repeated = await controller.drain_node(
                "node_a", boot_id="boot_1", timeout_ms=1_000
            )
            assert repeated.status == "drained"
            resumed = await controller.resume_node("node_a", boot_id="boot_1")
            assert resumed.status == "healthy"
            for _ in range(400):
                replacement = [
                    worker
                    for worker in controller.worker_broker.snapshot().workers
                    if worker.state is StandbyWorkerState.IDLE
                ]
                if replacement:
                    break
                await asyncio.sleep(0.025)
            else:
                raise AssertionError(controller.worker_broker.snapshot())
            assert replacement[0].process_id != original_pid
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_cpu_sanitize_failure_discards_worker_and_replenishes(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        controller = RayHostController(
            cluster_id="cluster_standby_sanitize",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
            worker_pool_config=_pool_config(),
        )
        agent: NodeAgent | None = None
        original_pid: int | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_standby_sanitize",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(
                identity=identity,
                authorization_token=b"test-token",
                heartbeat_interval_ms=50,
            )
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            for _ in range(400):
                idle = [
                    worker
                    for worker in controller.worker_broker.snapshot().workers
                    if worker.state is StandbyWorkerState.IDLE
                ]
                if idle:
                    original_pid = idle[0].process_id
                    break
                await asyncio.sleep(0.025)
            assert original_pid is not None

            workflow = Workflow("ray-standby-sanitize-failure")
            node = workflow.add_task(
                leak_file_descriptor,
                inputs={"path": str(tmp_path / "leaked.txt")},
            )
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_ray_standby_sanitize_failure",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=30)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, node.task_id) == {
                "result": "published"
            }
            attempt = terminal.task(node.task_id).attempts[0]
            worker_outcome = controller.ray_runtime.worker_outcome(
                attempt.dispatch_id
            )
            assert worker_outcome is not None
            assert not worker_outcome.reuse_safe
            assert worker_outcome.cleanup_reason == "file_descriptor_leaked"
            for _ in range(400):
                replacement = [
                    worker
                    for worker in controller.worker_broker.snapshot().workers
                    if worker.state is StandbyWorkerState.IDLE
                    and worker.process_id != original_pid
                ]
                if replacement:
                    break
                await asyncio.sleep(0.025)
            assert len(replacement) == 1
            try:
                os.kill(original_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("sanitization-rejected Worker is still alive")
            pool = controller.worker_broker.snapshot()
            assert pool.sanitize_failures == 1
            assert any(
                event.event_type == "worker_cleanup_rejected"
                and event.reason == "file_descriptor_leaked"
                and event.run_id == outcome.run_id
                and event.placement_lease_id == attempt.lease_id
                for event in controller.pool_events
            )
            await controller.destroy_run(outcome.run_id)
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_standby_timeout_kills_actor_releases_task_and_replenishes(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        controller = RayHostController(
            cluster_id="cluster_standby_timeout",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
            worker_pool_config=_pool_config(),
        )
        agent: NodeAgent | None = None
        original_pid: int | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_standby_timeout",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(
                identity=identity,
                authorization_token=b"test-token",
                heartbeat_interval_ms=50,
            )
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            for _ in range(400):
                idle = [
                    worker
                    for worker in controller.worker_broker.snapshot().workers
                    if worker.state is StandbyWorkerState.IDLE
                ]
                if idle:
                    original_pid = idle[0].process_id
                    break
                await asyncio.sleep(0.025)
            assert original_pid is not None

            workflow = Workflow("ray-standby-timeout")
            node = workflow.add_task(slow_cpu_task, inputs={"value": "late"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_ray_standby_timeout",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=30)
            assert terminal.status is RunStatus.FAILED
            task = terminal.task(node.task_id)
            assert task.last_error is not None
            assert task.last_error.error_code == "task_timeout"
            for _ in range(400):
                replacement = [
                    worker
                    for worker in controller.worker_broker.snapshot().workers
                    if worker.state is StandbyWorkerState.IDLE
                    and worker.process_id != original_pid
                ]
                if replacement:
                    break
                await asyncio.sleep(0.025)
            assert len(replacement) == 1
            try:
                os.kill(original_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("timed-out Standby Worker is still alive")
            assert controller.worker_broker.active_count() == 0
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            assert controller.placement.active_lease_count() == 1
            await controller.destroy_run(outcome.run_id)
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_child_process_leak_is_killed_before_worker_replacement(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        controller = RayHostController(
            cluster_id="cluster_standby_child_cleanup",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
            worker_pool_config=_pool_config(),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_standby_child_cleanup",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(
                identity=identity,
                authorization_token=b"test-token",
                heartbeat_interval_ms=50,
            )
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            for _ in range(400):
                if any(
                    worker.state is StandbyWorkerState.IDLE
                    for worker in controller.worker_broker.snapshot().workers
                ):
                    break
                await asyncio.sleep(0.025)
            workflow = Workflow("ray-standby-child-cleanup")
            node = workflow.add_task(leak_child_process, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_ray_standby_child_cleanup",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=30)
            assert terminal.status is RunStatus.SUCCEEDED
            child_pid = controller.result(outcome.run_id, node.task_id)["child_pid"]
            assert isinstance(child_pid, int)
            attempt = terminal.task(node.task_id).attempts[0]
            worker = controller.ray_runtime.worker_outcome(attempt.dispatch_id)
            assert worker is not None
            assert worker.cleanup_reason == "child_process_leaked"
            for _ in range(400):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.025)
            else:
                raise AssertionError("leaked child process is still alive")
            assert controller.worker_broker.snapshot().sanitize_failures == 1
            await controller.destroy_run(outcome.run_id)
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())
