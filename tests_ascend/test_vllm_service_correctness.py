from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from ascend_maze.ascend import (
    build_ascend_node_capacity,
    discover_atb_runtime_library_preloads,
)
from ascend_maze.control.node_rpc import NodeAgent, NodeAgentIdentity, NodeControlServer
from ascend_maze.control.service_process import (
    NodeAgentServiceProcessBackend,
    NodeServiceProcessManager,
)
from ascend_maze.inference import (
    ChatRequest,
    InferenceCallError,
    InferenceCoordinator,
    ModelCatalog,
    ModelInstanceState,
    ModelSpec,
)
from ascend_maze.inference.adapters.vllm_ascend import (
    VllmAscendInferenceEngineAdapter,
)
from ascend_maze.inference.client import chat
from ascend_maze.inference.context import install_route_session
from ascend_maze.inference.contracts import ServiceProcessExit
from ascend_maze.placement import PlacementManager
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry
from tests_ascend.conftest import AscendAdmission


MODEL_PATH = Path("/home/user2/workplace/model_weight/model_from_hf/Qwen3-4B")
EXPECTED_INDEX_SHA256 = (
    "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2"
)
ATB_LIBMKI_PATH = Path(
    "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libmki.so"
).resolve(strict=True)
EXPECTED_ATB_LIBMKI_SHA256 = (
    "41d55d3994ab35b0460a0ce12aec1a35c6a9ed515d3d6424e654465a44d0f27f"
)


def _artifact_revision(path: Path) -> str:
    digest = hashlib.sha256(
        (path / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    if digest != EXPECTED_INDEX_SHA256:
        raise RuntimeError("Qwen3-4B model index revision changed")
    return digest


def _spec(
    *,
    model_id: str,
    artifact_path: Path,
    environment: str,
    startup_timeout_ms: int = 180_000,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        catalog_revision=f"stage6b-{model_id}",
        artifact_path=str(artifact_path),
        tokenizer_path=str(artifact_path),
        artifact_revision=(
            _artifact_revision(artifact_path)
            if artifact_path == MODEL_PATH
            else hashlib.sha256(str(artifact_path).encode()).hexdigest()
        ),
        backend="vllm_ascend",
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=1,
        max_model_len=2048,
        instance_cpu_num=4,
        instance_host_mem_mb=16_384,
        weight_hbm_mb=7_500,
        runtime_hbm_mb=4_000,
        kv_cache_hbm_mb=22_000,
        instance_hbm_mb=36_000,
        npu_slots=1,
        allow_colocation=False,
        request_capacity=1,
        required_capabilities=("vllm_ascend",),
        environment_fingerprint=environment,
        launch_options={  # type: ignore[arg-type]
            "block_size": 128,
            "enable_prefix_caching": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.5,
            "log_level": "INFO",
        },
        warmup_request={  # type: ignore[arg-type]
            "messages": [{"role": "user", "content": "Reply briefly: ready."}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
        min_replicas=0,
        max_replicas=1,
        scale_down_idle_ms=600_000,
        scale_cooldown_ms=0,
        startup_timeout_ms=startup_timeout_ms,
        drain_timeout_ms=30_000,
    )


async def _wait_state(
    inference: InferenceCoordinator,
    instance_id: str,
    states: set[ModelInstanceState],
    *,
    timeout: float,
) -> ModelInstanceState:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = inference.instances.snapshot(instance_id).state
        if state in states:
            return state
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"model instance did not enter {states}: "
        f"{inference.instances.snapshot(instance_id)}"
    )


async def _wait_hbm(
    admission: AscendAdmission,
    baseline_mb: int,
    *,
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = admission.adapter.device(admission.device.physical_device_id)
        if (
            last.used_hbm_mb <= baseline_mb + admission.config.hbm_recovery_tolerance_mb
            and not last.processes
        ):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"vLLM HBM did not recover: baseline={baseline_mb}, last={last}"
    )


async def _runtime(
    admission: AscendAdmission,
    tmp_path: Path,
) -> tuple[
    NodeControlServer,
    NodeAgent,
    NodeAgentServiceProcessBackend,
    PlacementManager,
    list[InferenceCoordinator],
]:
    environment = admission.environment
    assert environment.versions["vllm"] == "0.11.0+empty"
    assert environment.versions["vllm_ascend"] == "0.11.0"
    capacity = build_ascend_node_capacity(
        node_id="node_ascend",
        boot_id="boot_stage6b",
        node_ip="127.0.0.1",
        adapter=admission.adapter,
        environment=environment,
        config=admission.config,
    )
    selected = next(
        item
        for item in capacity.npus
        if item.device_id == admission.device.physical_device_id
    )
    capacity = replace(capacity, npus=(selected,))
    placement = PlacementManager(
        host_mem_headroom_mb=admission.config.host_mem_headroom_mb,
        npu_hbm_headroom_mb=admission.config.npu_hbm_headroom_mb,
        required_environment_fingerprint=environment.environment_fingerprint,
    )
    placement.register_node(capacity)
    registry = RayNodeRegistry()
    coordinators: list[InferenceCoordinator] = []

    def process_exited(event: ServiceProcessExit) -> None:
        for inference in tuple(coordinators):
            instances = {item.instance_id: item for item in inference.model_instances()}
            if event.instance_id in instances:
                inference.report_process_exited(
                    event.instance_id,
                    event.generation,
                    reason=f"service_process_exited:{event.exit_code}",
                )

    controller = NodeControlServer(
        cluster_id="cluster_stage6b",
        authorization_token=b"stage6b-token",
        controller_generation="controller_stage6b",
        environment_fingerprint=environment.environment_fingerprint,
        registry=registry,
        recorder=InMemoryRecorder(),
        event_sink=lambda event: None,
        on_service_process_exited=process_exited,
    )
    endpoint = await controller.start()
    manager = NodeServiceProcessManager(
        node_id=capacity.node_id,
        boot_id=capacity.boot_id,
        device_monitor=admission.adapter,
        allowed_executables=(sys.executable,),
        log_directory=tmp_path / "service-logs",
        first_port=32400,
        last_port=32420,
        hbm_recovery_tolerance_mb=admission.config.hbm_recovery_tolerance_mb,
        poll_interval_ms=100,
    )
    identity = NodeAgentIdentity(
        cluster_id="cluster_stage6b",
        node_id=capacity.node_id,
        boot_id=capacity.boot_id,
        ray_node_id="local-stage6b",
        agent_generation="agent_stage6b",
        environment_fingerprint=environment.environment_fingerprint,
        producer_id="node_agent:stage6b",
    )
    agent = NodeAgent(
        identity=identity,
        authorization_token=b"stage6b-token",
        heartbeat_interval_ms=100,
        service_process_manager=manager,
    )
    await agent.start(controller_endpoint=endpoint)
    backend = NodeAgentServiceProcessBackend(
        cluster_id="cluster_stage6b",
        authorization_token=b"stage6b-token",
        controller_generation="controller_stage6b",
        node_registry=registry,
        rpc_timeout_ms=180_000,
    )
    return controller, agent, backend, placement, coordinators


def _coordinator(
    *,
    spec: ModelSpec,
    backend: NodeAgentServiceProcessBackend,
    placement: PlacementManager,
    coordinators: list[InferenceCoordinator],
    api_server_entrypoint: tuple[str, ...] | None = None,
    probe_timeout_ms: int = 180_000,
) -> InferenceCoordinator:
    runtime_library_preloads = discover_atb_runtime_library_preloads()
    assert dict(runtime_library_preloads) == {
        str(ATB_LIBMKI_PATH): EXPECTED_ATB_LIBMKI_SHA256,
    }
    adapter = VllmAscendInferenceEngineAdapter(
        process_backend=backend,
        python_executable=sys.executable,
        endpoint_host_resolver=backend.endpoint_host,
        api_server_entrypoint=(
            ("-m", "vllm.entrypoints.openai.api_server")
            if api_server_entrypoint is None
            else api_server_entrypoint
        ),
        runtime_library_preloads=runtime_library_preloads,
        request_timeout_ms=30_000,
        probe_timeout_ms=probe_timeout_ms,
        probe_interval_ms=250,
    )
    catalog = ModelCatalog(
        (spec,),
        adapters={adapter.name: adapter},
        environment_capabilities=("vllm_ascend",),
        max_single_npu_hbm_mb=60_000,
    )
    inference = InferenceCoordinator(
        catalog=catalog,
        placement=placement,
        service_backend=backend,
        port_leases=backend,
        reconcile_interval_ms=100,
    )
    coordinators.append(inference)
    return inference


def test_vllm_ascend_route_chat_drain_restart_and_crash(
    ascend_admission: AscendAdmission,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        baseline = ascend_admission.adapter.device(
            ascend_admission.device.physical_device_id
        ).used_hbm_mb
        controller, agent, backend, placement, coordinators = await _runtime(
            ascend_admission, tmp_path
        )
        spec = _spec(
            model_id="qwen3-4b",
            artifact_path=MODEL_PATH,
            environment=ascend_admission.environment.environment_fingerprint,
        )
        inference = _coordinator(
            spec=spec,
            backend=backend,
            placement=placement,
            coordinators=coordinators,
        )
        await inference.start()
        route = None
        try:
            requested = inference.instances.create_requested(spec.model_id)
            ready = await inference.instances.start_instance(requested.instance_id)
            assert ready.state is ModelInstanceState.READY
            assert ready.npu_device_id == ascend_admission.device.physical_device_id
            assert placement.active_lease_count() == 1
            warming = next(
                event
                for event in reversed(inference.events())
                if event.event_type == "model_instance_warming"
            )
            api_pid = int(warming.payload["process_id"])
            assert os.getpgid(api_pid) == api_pid
            device = ascend_admission.adapter.device(ready.npu_device_id)
            assert any(os.getpgid(item.pid) == api_pid for item in device.processes)

            inference.register_demand(
                run_id="run_1", task_id="task_1", model_id=spec.model_id
            )
            route_result = await inference.acquire_route(
                run_id="run_1",
                task_id="task_1",
                attempt=1,
                model_id=spec.model_id,
                session_key_hash="session",
                dispatch_deadline_ms=inference.clock.monotonic_ms() + 30_000,
            )
            assert route_result.lease is not None
            route = route_result.lease
            assert inference.activate_route(route.route_lease_id)
            session = inference.create_attempt_session(route)

            def invoke_sequentially() -> tuple[str, str]:
                with install_route_session(session):
                    first = chat(
                        [{"role": "user", "content": "Reply briefly: first."}],
                        max_tokens=16,
                    )
                    second = chat(
                        [{"role": "user", "content": "Reply briefly: second."}],
                        max_tokens=16,
                    )
                    return first.text, second.text

            first, second = await asyncio.to_thread(invoke_sequentially)
            assert first and second
            assert [
                record.call_index
                for record in inference.request_records(route.route_lease_id)
            ] == [1, 2]

            long_request = ChatRequest.create(
                [
                    {
                        "role": "user",
                        "content": "Write a numbered list with fifty short items.",
                    }
                ],
                max_tokens=128,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                running = executor.submit(session.invoke, long_request)
                deadline = time.monotonic() + 10
                while (
                    inference.model_instances()[0].actual_request_inflight != 1
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.01)
                assert inference.model_instances()[0].actual_request_inflight == 1
                with pytest.raises(InferenceCallError) as concurrent:
                    await asyncio.to_thread(session.invoke, long_request)
                assert (
                    concurrent.value.error_code
                    == "model_route_concurrent_call_forbidden"
                )
                assert (await asyncio.wrap_future(running)).text
            assert inference.model_instances()[0].actual_request_inflight == 0
            assert await inference.release_route(route, reason="succeeded")

            assert inference.instances.begin_drain(ready.instance_id, ready.generation)
            assert (
                inference.instances.snapshot(ready.instance_id).state
                is ModelInstanceState.DRAINING
            )
            stopped = await inference.instances.stop_if_drained(
                ready.instance_id, ready.generation
            )
            assert stopped.state is ModelInstanceState.STOPPED
            assert placement.active_lease_count() == 0
            await _wait_hbm(ascend_admission, baseline)

            restarted = inference.instances.restart_stopped(ready.instance_id)
            restarted = await inference.instances.start_instance(restarted.instance_id)
            assert restarted.state is ModelInstanceState.READY
            warming = next(
                event
                for event in reversed(inference.events())
                if event.event_type == "model_instance_warming"
                and event.instance_generation == restarted.generation
            )
            restarted_pid = int(warming.payload["process_id"])
            os.kill(restarted_pid, signal.SIGKILL)
            state = await _wait_state(
                inference,
                restarted.instance_id,
                {ModelInstanceState.FAILED, ModelInstanceState.STOPPED},
                timeout=10,
            )
            if state is ModelInstanceState.FAILED:
                restarted = await inference.instances.stop_if_drained(
                    restarted.instance_id, restarted.generation
                )
                assert restarted.state is ModelInstanceState.STOPPED
            assert any(
                event.event_type == "model_process_exited"
                and event.instance_generation == restarted.generation
                for event in inference.events()
            )
            assert placement.active_lease_count() == 0
            await _wait_hbm(ascend_admission, baseline)
        finally:
            try:
                if route is not None:
                    await inference.release_route(route, reason="test_cleanup")
                await inference.close()
            finally:
                coordinators.remove(inference)
                await agent.close(grace_seconds=0)
                await controller.close(grace_seconds=0)

    asyncio.run(scenario())


def test_vllm_ascend_startup_failure_and_timeout_release_resources(
    ascend_admission: AscendAdmission,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        baseline = ascend_admission.adapter.device(
            ascend_admission.device.physical_device_id
        ).used_hbm_mb
        controller, agent, _, placement, coordinators = await _runtime(
            ascend_admission, tmp_path
        )
        try:
            empty_model = tmp_path / "empty-model"
            empty_model.mkdir()
            for model_id, artifact, entrypoint, probe_timeout in (
                ("bad-model", empty_model, None, 60_000),
                (
                    "timeout-model",
                    MODEL_PATH,
                    ("-c", "import time; time.sleep(60)"),
                    500,
                ),
            ):
                backend = NodeAgentServiceProcessBackend(
                    cluster_id="cluster_stage6b",
                    authorization_token=b"stage6b-token",
                    controller_generation="controller_stage6b",
                    node_registry=controller.registry,
                    rpc_timeout_ms=90_000,
                )
                spec = _spec(
                    model_id=model_id,
                    artifact_path=artifact,
                    environment=ascend_admission.environment.environment_fingerprint,
                    startup_timeout_ms=90_000,
                )
                inference = _coordinator(
                    spec=spec,
                    backend=backend,
                    placement=placement,
                    coordinators=coordinators,
                    api_server_entrypoint=entrypoint,
                    probe_timeout_ms=probe_timeout,
                )
                requested = inference.instances.create_requested(model_id)
                result = await inference.instances.start_instance(requested.instance_id)
                assert result.state is ModelInstanceState.STOPPED
                assert any(
                    event.event_type == "model_instance_failed"
                    for event in inference.events()
                )
                assert placement.active_lease_count() == 0
                assert backend.active_count() == 0
                await inference.close()
                coordinators.remove(inference)
                await _wait_hbm(ascend_admission, baseline)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())
