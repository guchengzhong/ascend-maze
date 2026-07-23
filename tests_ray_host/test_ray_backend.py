from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
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
from ascend_maze.inference import (
    InferenceCoordinator,
    ModelCatalog,
    ModelSpec,
)
from ascend_maze.inference.adapters.fake import FakeInferenceEngineAdapter
from ascend_maze.placement import NodeCapacity, NpuCapacity, PlacementManager
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
        code_package_puts_before = int(store.stats()["code_package_put_count"])
        canonicalize_before_prepare = int(store.stats()["canonicalize_count"])
        code_handles = await backend.prepare(packages)
        assert int(store.stats()["code_package_put_count"]) == (
            code_package_puts_before + 1
        )
        assert int(store.stats()["canonicalize_count"]) == canonicalize_before_prepare
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
        input_handle = store.put_staged("hello", generation)
        resolve_batches_before = int(store.stats()["resolve_batch_count"])
        runtime_outputs_before = int(store.stats()["runtime_output_put_count"])
        request = ExecutionRequest(
            dispatch_id="dispatch_1",
            run_id="run_1",
            task_id=node.task_id,
            attempt=1,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            model_route=None,
            code_handle=code_handle,
            arguments=(
                RuntimeArgument("value", "data_handle", data_handle=input_handle),
            ),
            expected_outputs=("result",),
            timeout_ms=None,
            environment_fingerprint=ENVIRONMENT,
        )
        dispatch = await backend.dispatch(request, lease)
        assert await backend.dispatch(request, lease) == dispatch
        assert store.local_get_count == 0
        assert int(store.stats()["resolve_batch_count"]) == resolve_batches_before + 1
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
        timings = backend.task_timing_records("run_1")
        assert len(timings) == 1
        assert timings[0]["dispatch_id"] == "dispatch_1"
        assert timings[0]["status"] == "succeeded"
        assert timings[0]["input_fetch_scope"] == (
            "ray_materialized_argument_binding"
        )
        assert timings[0]["output_put_scope"] == "ray_data_store_put_staged"
        assert timings[0]["dispatch_wait_ms"] >= timings[0]["worker_startup_ms"]
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
        assert result_handle.stable_digest is None
        assert result_handle.size_bytes is None
        assert store.get(result_handle) == "hello"
        assert store.state_of(result_handle) == "staged"
        assert int(store.stats()["runtime_output_put_count"]) == (
            runtime_outputs_before + 1
        )
        assert broker.active_count() == 0
        assert recorder.events("run_1")[0].producer_id == identity.producer_id

        store.release(result_handle)
        store.release(input_handle)
        await backend.release_code(code_handles)
        assert backend.code_reference_count() == 0
        await backend.close()
        await agent.close(grace_seconds=0)
        await controller_rpc.close(grace_seconds=0)
        store.close(kill_owner=True)

    asyncio.run(scenario())


def test_ray_backend_executes_model_service_worker_without_device_binding(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        @task(task_kind="npu", resources={"cpu_num": 1, "mem": 64})
        def service_chat(value: str) -> dict[str, object]:
            from ascend_maze.inference import chat

            first = chat([{"role": "user", "content": value}], max_tokens=8)
            second = chat([{"role": "user", "content": first.text}], max_tokens=8)
            return {"result": second.text}

        generation = f"controller-{uuid4().hex}"
        store = RayDataStore.start(
            owner_generation=generation,
            namespace=ray_namespace,
        )
        registry = RayNodeRegistry()
        placement = PlacementManager()
        capacity = NodeCapacity(
            node_id="node_a",
            boot_id="boot_1",
            node_ip="127.0.0.1",
            cpu_total=8,
            mem_total_mb=8_192,
            cpu_system_reserved=0,
            mem_system_reserved_mb=0,
            io_slots_total=4,
            npus=(
                NpuCapacity(
                    device_id="0",
                    chip_type="fake_npu",
                    total_hbm_mb=8_192,
                    system_reserved_hbm_mb=0,
                    task_slots_total=1,
                    observed_free_hbm_mb=8_192,
                ),
            ),
            observed_free_mem_mb=8_192,
            capabilities={"environment_fingerprint": ENVIRONMENT},
        )
        placement.register_node(capacity)
        artifact = tmp_path / "model"
        artifact.mkdir()
        spec = ModelSpec(
            model_id="ray_model",
            catalog_revision="catalog_v1",
            artifact_path=str(artifact),
            tokenizer_path=None,
            artifact_revision="artifact_v1",
            backend="fake",
            dtype="float16",
            quantization=None,
            tensor_parallel_size=1,
            max_model_len=2_048,
            instance_cpu_num=1,
            instance_host_mem_mb=256,
            weight_hbm_mb=512,
            runtime_hbm_mb=128,
            kv_cache_hbm_mb=128,
            instance_hbm_mb=1_024,
            npu_slots=1,
            allow_colocation=False,
            request_capacity=1,
            required_capabilities=(),
            environment_fingerprint=ENVIRONMENT,
            launch_options={"response_prefix": "ray_model"},
            warmup_request={"prompt": "warmup"},
            min_replicas=0,
            max_replicas=1,
            target_route_utilization=1.0,
            scale_up_pending_threshold=1,
            scale_up_sustain_ms=0,
            scale_down_idle_ms=0,
            scale_cooldown_ms=0,
            max_parallel_starts=1,
            startup_timeout_ms=5_000,
            drain_timeout_ms=5_000,
        )
        adapter = FakeInferenceEngineAdapter()
        inference = InferenceCoordinator(
            catalog=ModelCatalog(
                (spec,),
                adapters={"fake": adapter},
                max_single_npu_hbm_mb=8_192,
            ),
            placement=placement,
            service_backend=adapter,
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
            cluster_id="cluster_service",
            owner_generation=generation,
            environment_fingerprint=ENVIRONMENT,
            event_sink=events.append,
            inference=inference,
        )
        recorder = InMemoryRecorder()
        controller_rpc = NodeControlServer(
            cluster_id="cluster_service",
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
            cluster_id="cluster_service",
            node_id="node_a",
            boot_id="boot_1",
            ray_node_id=ray.get_runtime_context().get_node_id(),
            agent_generation="agent_service",
            environment_fingerprint=ENVIRONMENT,
            producer_id="node_agent:node_a:agent_service",
        )
        agent = NodeAgent(identity=identity, authorization_token=b"test-token")
        await agent.start(controller_endpoint=controller_endpoint)
        code_handles = ()
        result_handle = None
        route = None
        try:
            await backend.start()
            workflow = Workflow("ray-model-service")
            node = workflow.add_task(
                service_chat,
                inputs={"value": "unused"},
                model_anchor={"model": spec.model_id, "mode": "service"},
            )
            compiled = workflow.compile()
            code_handles = await backend.prepare(
                build_code_packages(
                    compiled,
                    environment_fingerprint=ENVIRONMENT,
                    callables_by_definition={
                        compiled.tasks[node.task_id].definition_id: service_chat,
                    },
                )
            )
            inference.register_demand(
                run_id="run_service",
                task_id=node.task_id,
                model_id=spec.model_id,
            )
            await inference.reconcile()
            await inference.replicas.wait_for_background()
            acquired = await inference.acquire_route(
                run_id="run_service",
                task_id=node.task_id,
                attempt=1,
                model_id=spec.model_id,
                session_key_hash=None,
                dispatch_deadline_ms=10**15,
            )
            assert acquired.lease is not None
            route = acquired.lease
            lease = PlacementLease(
                lease_id="lease_service",
                reservation_kind="model_request",
                run_id="run_service",
                task_id=node.task_id,
                attempt=1,
                node_id="node_a",
                boot_id="boot_1",
                npu_device_id=None,
                resources=ReservationVector(1, 64, 0, 0, 0),
                snapshot_version=1,
                created_at_ms=1,
                dispatch_deadline_ms=10**15,
            )
            request = ExecutionRequest(
                dispatch_id="dispatch_service",
                run_id="run_service",
                task_id=node.task_id,
                attempt=1,
                task_kind="npu",
                execution_target=ExecutionTarget.MODEL_SERVICE,
                model_route=route,
                code_handle=code_handles[0],
                arguments=(RuntimeArgument("value", "literal", literal="hello"),),
                expected_outputs=("result",),
                timeout_ms=None,
                environment_fingerprint=ENVIRONMENT,
            )
            dispatch = await backend.dispatch(request, lease)
            assert dispatch.route_lease_id == route.route_lease_id
            await asyncio.wait_for(backend.wait_idle(), timeout=15)
            for _ in range(1_000):
                if len(events) >= 2:
                    break
                await asyncio.sleep(0.01)
            worker_outcome = backend.worker_outcome("dispatch_service")
            assert [event.kind for event in events] == [
                RuntimeEventKind.WORKER_STARTED,
                RuntimeEventKind.TASK_RESULT,
            ], (
                tuple(
                    (
                        event.kind,
                        None if event.error is None else event.error.error_code,
                        None if event.error is None else event.error.message,
                    )
                    for event in events
                ),
                None
                if worker_outcome is None
                else (
                    worker_outcome.terminal_event.kind,
                    worker_outcome.terminal_event.inference_summary,
                    None
                    if worker_outcome.terminal_event.error is None
                    else worker_outcome.terminal_event.error.error_code,
                    None
                    if worker_outcome.terminal_event.error is None
                    else worker_outcome.terminal_event.error.message,
                ),
            )
            assert all(event.device_id is None for event in events)
            result_handle = events[-1].output_handles[0][1]
            assert store.get(result_handle) == "ray_model:ray_model:hello"
            records = inference.request_records(route.route_lease_id)
            assert [record.call_index for record in records] == [1, 2]
            assert all(record.status == "succeeded" for record in records)
            assert inference.attempt_summary(route.route_lease_id).context_cleared
            assert inference.model_instances()[0].actual_request_inflight == 0
        finally:
            if result_handle is not None:
                store.release(result_handle)
            if route is not None:
                await inference.release_route(route, reason="test_complete")
                await inference.replicas.wait_for_background()
            if code_handles:
                await backend.release_code(code_handles)
            await backend.close()
            await inference.close()
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
                event.kind
                in {
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
