from __future__ import annotations

import asyncio
import hashlib
import os
from uuid import uuid4

import ray
import pytest

from ascend_maze import Workflow, task
from ascend_maze.contracts.recording import RunRecordingContext
from ascend_maze.contracts.resources import (
    ExecutionTarget,
    PlacementLease,
    ReservationVector,
)
from ascend_maze.contracts.runtime import ExecutionRequest, RuntimeArgument
from ascend_maze.data.ray_store import RayDataStore
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime.events import RuntimeEventKind
from ascend_maze.runtime.packaging import build_code_packages
from ascend_maze.runtime.ray_backend import RayRuntimeBackend
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry
from ascend_maze.runtime.worker_broker import ColdWorkerBroker

from ascend_maze.control.node_rpc import (
    NodeAgent,
    NodeAgentIdentity,
    NodeControlServer,
)


ENVIRONMENT = "e" * 64


def test_ray_backend_executes_one_shot_worker_on_hard_bound_node(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task
        def host_echo(value: str):
            return {"result": value}

        generation = f"controller-{uuid4().hex}"
        store = RayDataStore.start(
            owner_generation=generation,
            namespace=ray_namespace,
        )
        registry = RayNodeRegistry()
        recorder = InMemoryRecorder()
        broker = ColdWorkerBroker(
            node_registry=registry,
            environment_fingerprint=ENVIRONMENT,
        )
        backend = RayRuntimeBackend(
            data_store=store,
            node_registry=registry,
            worker_broker=broker,
            cluster_id="cluster_1",
            owner_generation=generation,
            environment_fingerprint=ENVIRONMENT,
        )
        events = []
        backend.set_event_sink(events.append)
        controller_rpc = NodeControlServer(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            controller_generation=generation,
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=recorder,
            event_sink=backend.post_node_event,
            on_binding_replaced=backend.invalidate_binding,
            on_binding_disconnected=backend.invalidate_binding,
        )
        controller_endpoint = await controller_rpc.start()
        identity = NodeAgentIdentity(
            cluster_id="cluster_1",
            node_id="node_a",
            boot_id="boot_1",
            ray_node_id=ray.get_runtime_context().get_node_id(),
            agent_generation="agent_1",
            environment_fingerprint=ENVIRONMENT,
            producer_id="node_agent:node_a:agent_1",
        )
        agent = NodeAgent(identity=identity, authorization_token=b"test-token")
        await agent.start(controller_endpoint=controller_endpoint)
        recorder.open_run(
            RunRecordingContext(
                schema_version=1,
                experiment_id="run_1",
                run_id="run_1",
                workflow_fingerprint="w" * 64,
                config_fingerprint="c" * 64,
                environment_fingerprint=ENVIRONMENT,
                build_revision="test",
                started_wall_time_ms=1,
                initial_expected_producer_ids=(identity.producer_id,),
            )
        )
        await backend.start()
        workflow = Workflow("ray-host-worker")
        node = workflow.add_task(host_echo, inputs={"value": "unused"})
        compiled = workflow.compile()
        packages = build_code_packages(
            compiled,
            environment_fingerprint=ENVIRONMENT,
            callables_by_definition={
                compiled.tasks[node.task_id].definition_id: host_echo,
            },
        )
        code_handles = await backend.prepare(packages)
        definition_id = compiled.tasks[node.task_id].definition_id
        code_handle = next(
            item for item in code_handles if item.definition_id == definition_id
        )
        lease = PlacementLease(
            lease_id="lease_1",
            reservation_kind="task",
            run_id="run_1",
            task_id=node.task_id,
            attempt=1,
            node_id="node_a",
            boot_id="boot_1",
            npu_device_id=None,
            resources=ReservationVector(1, 64, 0, 0, 0),
            snapshot_version=1,
            created_at_ms=1,
            dispatch_deadline_ms=100_000,
        )
        request = ExecutionRequest(
            dispatch_id="dispatch_1",
            run_id="run_1",
            task_id=node.task_id,
            attempt=1,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            model_route=None,
            code_handle=code_handle,
            arguments=(RuntimeArgument("value", "literal", literal="hello"),),
            expected_outputs=("result",),
            timeout_ms=None,
            environment_fingerprint=ENVIRONMENT,
        )
        dispatch = await backend.dispatch(request, lease)
        assert await backend.dispatch(request, lease) == dispatch
        await asyncio.wait_for(backend.wait_idle(), timeout=15)
        for _ in range(1_000):
            if len(events) >= 2:
                break
            await asyncio.sleep(0.01)
        assert [event.kind for event in events] == [
            RuntimeEventKind.WORKER_STARTED,
            RuntimeEventKind.TASK_RESULT,
        ], tuple(
            (
                event.kind,
                None if event.error is None else event.error.error_code,
                None if event.error is None else event.error.message,
            )
            for event in events
        )
        outcome = backend.worker_outcome("dispatch_1")
        assert outcome is not None
        assert outcome.ray_node_id == identity.ray_node_id
        started = backend.worker_started_event("dispatch_1")
        assert started is not None
        assert started.worker_pid == outcome.worker_pid
        for _ in range(1_000):
            try:
                os.kill(outcome.worker_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        with pytest.raises(ProcessLookupError):
            os.kill(outcome.worker_pid, 0)
        result_handle = events[-1].output_handles[0][1]
        assert store.get(result_handle) == "hello"
        assert store.state_of(result_handle) == "staged"
        assert broker.active_count() == 0
        assert recorder.events("run_1")[0].producer_id == identity.producer_id

        store.release(result_handle)
        await backend.release_code(code_handles)
        assert backend.code_reference_count() == 0
        await backend.close()
        await agent.close(grace_seconds=0)
        await controller_rpc.close(grace_seconds=0)
        store.close(kill_owner=True)

    asyncio.run(scenario())


def test_hard_unavailable_ray_node_never_migrates_to_a_live_node(
    ray_namespace: str,
    ray_node_ids: tuple[str, str],
) -> None:
    async def scenario() -> None:
        @task
        def should_not_run(value: str):
            return {"result": value}

        generation = f"controller-{uuid4().hex}"
        store = RayDataStore.start(
            owner_generation=generation,
            namespace=ray_namespace,
        )
        registry = RayNodeRegistry()
        fake_ray_node_id = hashlib.sha256(b"missing-ray-node").hexdigest()[
            : len(ray_node_ids[0])
        ]
        if fake_ray_node_id in ray_node_ids:
            fake_ray_node_id = hashlib.sha256(b"other-missing-node").hexdigest()[
                : len(ray_node_ids[0])
            ]
        registry.register(
            node_id="missing_node",
            boot_id="boot_missing",
            ray_node_id=fake_ray_node_id,
            agent_generation="agent_missing",
            agent_endpoint="127.0.0.1:1",
            producer_id="node_agent:missing:1",
        )
        broker = ColdWorkerBroker(
            node_registry=registry,
            environment_fingerprint=ENVIRONMENT,
        )
        events = []
        backend = RayRuntimeBackend(
            data_store=store,
            node_registry=registry,
            worker_broker=broker,
            cluster_id="cluster_1",
            owner_generation=generation,
            environment_fingerprint=ENVIRONMENT,
            event_sink=events.append,
        )
        code_handles = ()
        try:
            await backend.start()
            workflow = Workflow("hard-unavailable")
            node = workflow.add_task(should_not_run, inputs={"value": "unused"})
            compiled = workflow.compile()
            code_handles = await backend.prepare(
                build_code_packages(
                    compiled,
                    environment_fingerprint=ENVIRONMENT,
                    callables_by_definition={
                        compiled.tasks[node.task_id].definition_id: should_not_run,
                    },
                )
            )
            lease = PlacementLease(
                lease_id="lease_missing",
                reservation_kind="task",
                run_id="run_missing",
                task_id=node.task_id,
                attempt=1,
                node_id="missing_node",
                boot_id="boot_missing",
                npu_device_id=None,
                resources=ReservationVector(1, 64, 0, 0, 0),
                snapshot_version=1,
                created_at_ms=1,
                dispatch_deadline_ms=100_000,
            )
            request = ExecutionRequest(
                dispatch_id="dispatch_missing",
                run_id="run_missing",
                task_id=node.task_id,
                attempt=1,
                task_kind="cpu",
                execution_target=ExecutionTarget.LOCAL_WORKER,
                model_route=None,
                code_handle=code_handles[0],
                arguments=(RuntimeArgument("value", "literal", literal="never"),),
                expected_outputs=("result",),
                timeout_ms=None,
                environment_fingerprint=ENVIRONMENT,
            )
            dispatch = await backend.dispatch(request, lease)
            await asyncio.sleep(0.5)
            assert backend.worker_outcome(dispatch.dispatch_id) is None
            assert not any(
                event.kind in {
                    RuntimeEventKind.WORKER_STARTED,
                    RuntimeEventKind.TASK_RESULT,
                }
                for event in events
            )
            await backend.cancel(dispatch, "test_complete")
            await asyncio.wait_for(backend.wait_idle(), timeout=10)
            assert backend.worker_outcome(dispatch.dispatch_id) is None
            assert broker.active_count() == 0
        finally:
            if code_handles:
                await backend.release_code(code_handles)
            await backend.close()
            store.close(kill_owner=True)

    asyncio.run(scenario())
