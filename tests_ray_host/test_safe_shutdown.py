from __future__ import annotations

import asyncio
import grpc
import ray

from ascend_maze import Workflow, task
from ascend_maze.control import (
    ControllerLifecycleState,
    InMemoryRuntimeClient,
    ShutdownMode,
)
from ascend_maze.control.node_rpc import NodeAgent, NodeAgentIdentity
from ascend_maze.control.ray_controller import RayHostController
from ascend_maze.lifecycle import RunStatus, TaskStatus
from ascend_maze.placement import NodeCapacity


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


def _identity(generation: str) -> NodeAgentIdentity:
    return NodeAgentIdentity(
        cluster_id="cluster_shutdown",
        node_id="node_a",
        boot_id="boot_1",
        ray_node_id=ray.get_runtime_context().get_node_id(),
        agent_generation=generation,
        environment_fingerprint=ENVIRONMENT,
        producer_id=f"node_agent:node_a:{generation}",
    )


def test_ray_host_graceful_shutdown_drains_worker_rejects_node_and_orders_cleanup(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task
        def slow_shutdown_task(value: str):
            import time

            time.sleep(0.2)
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_shutdown",
            authorization_token=b"shutdown-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="stage7b_test",
            node_capacities=(_node(),),
            shutdown_drain_timeout_ms=2_000,
        )
        agent = NodeAgent(
            identity=_identity("agent_1"),
            authorization_token=b"shutdown-token",
            heartbeat_interval_ms=20,
        )
        rejected_agent: NodeAgent | None = None
        try:
            await controller.start()
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            workflow = Workflow("ray-safe-shutdown")
            node = workflow.add_task(
                slow_shutdown_task,
                inputs={"value": "done"},
            )
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="ray_safe_shutdown",
            )
            assert outcome.run_id is not None
            for _ in range(1_000):
                if controller.snapshot(outcome.run_id).task(node.task_id).status is (
                    TaskStatus.RUNNING
                ):
                    break
                await asyncio.sleep(0.002)
            else:
                raise AssertionError("Ray task did not start before shutdown")

            shutdown_task = asyncio.create_task(controller.shutdown())
            for _ in range(1_000):
                if controller.lifecycle_state is ControllerLifecycleState.DRAINING:
                    break
                await asyncio.sleep(0.001)
            else:
                raise AssertionError("Controller did not enter draining")

            rejected_agent = NodeAgent(
                identity=_identity("agent_2"),
                authorization_token=b"shutdown-token",
                heartbeat_interval_ms=20,
            )
            try:
                await rejected_agent.start(
                    controller_endpoint=controller.node_rpc_endpoint
                )
            except (grpc.aio.AioRpcError, RuntimeError):
                pass
            else:
                raise AssertionError("draining Controller accepted a new NodeAgent")

            result = await shutdown_task
            snapshot = controller.snapshot(outcome.run_id)
            assert result.mode is ShutdownMode.GRACEFUL
            assert result.drained_run_ids == (outcome.run_id,)
            assert result.terminated_run_ids == ()
            assert result.cleanup_confirmed
            assert result.recording_complete
            assert result.exit_code == 0
            assert snapshot.status is RunStatus.SUCCEEDED
            assert controller.ray_runtime.active_dispatch_count() == 0
            assert controller.worker_broker.active_count() == 0
            assert controller.placement.active_lease_count() == 0
            assert result.steps.index("models_stopped") < result.steps.index(
                "worker_pool_stopped"
            )
            assert result.steps.index("worker_pool_stopped") < result.steps.index(
                "recorder_flushed_closed"
            )
            assert result.steps.index(
                "recorder_flushed_closed"
            ) < result.steps.index("runtime_generation_stopped")
            assert result.steps.index(
                "runtime_generation_stopped"
            ) < result.steps.index("control_transports_stopped")
        finally:
            if rejected_agent is not None:
                await rejected_agent.close(grace_seconds=0)
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())
