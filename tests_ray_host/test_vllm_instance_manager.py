from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ascend_maze.control.node_rpc import NodeAgent, NodeAgentIdentity, NodeControlServer
from ascend_maze.control.service_process import (
    NodeAgentServiceProcessBackend,
    NodeServiceProcessManager,
)
from ascend_maze.inference import (
    ChatRequest,
    InferenceCoordinator,
    ModelCatalog,
    ModelInstanceState,
    ModelSpec,
)
from ascend_maze.inference.adapters.vllm_ascend import (
    VllmAscendInferenceEngineAdapter,
    VllmHttpResponse,
)
from ascend_maze.inference.context import install_route_session
from ascend_maze.inference.contracts import InferenceCallError, ServiceProcessExit
from ascend_maze.placement import NodeCapacity, NpuCapacity, PlacementManager
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry


@dataclass
class _Device:
    used_hbm_mb: int = 100


class _Monitor:
    def device(self, physical_device_id: str) -> _Device:
        assert physical_device_id == "7"
        return _Device()

    def process_hbm_mb(self, physical_device_id: str, pid: int) -> int | None:
        assert physical_device_id == "7"
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        return 16_000

    def verify_process_device(
        self,
        pid: int,
        physical_device_id: str,
        *,
        deadline_seconds: float = 2.0,
        poll_interval_seconds: float = 0.05,
    ) -> bool:
        del deadline_seconds, poll_interval_seconds
        return self.process_hbm_mb(physical_device_id, pid) is not None


class _UrllibTransport:
    async def request(
        self,
        method: str,
        url: str,
        *,
        json_body: object | None,
        timeout_ms: int,
    ) -> VllmHttpResponse:
        def execute() -> VllmHttpResponse:
            body = None if json_body is None else json.dumps(json_body).encode()
            request = Request(
                url,
                data=body,
                method=method,
                headers={} if body is None else {"content-type": "application/json"},
            )
            try:
                with urlopen(request, timeout=timeout_ms / 1_000) as response:
                    return VllmHttpResponse(
                        int(response.status),
                        response.read(),
                        dict(response.headers.items()),
                    )
            except HTTPError as exc:
                return VllmHttpResponse(exc.code, exc.read(), dict(exc.headers.items()))
            except (TimeoutError, URLError) as exc:
                raise InferenceCallError(
                    "model_service_unavailable", f"test endpoint unavailable: {exc}"
                ) from exc

        return await asyncio.to_thread(execute)

    async def close(self) -> None:
        return None


def _spec(model_path: Path, environment: str) -> ModelSpec:
    model_path.mkdir()
    return ModelSpec(
        model_id="qwen3-4b",
        catalog_revision="catalog_1",
        artifact_path=str(model_path),
        tokenizer_path=None,
        artifact_revision="a" * 64,
        backend="vllm_ascend",
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=1,
        max_model_len=2048,
        instance_cpu_num=2,
        instance_host_mem_mb=2048,
        weight_hbm_mb=8000,
        runtime_hbm_mb=4000,
        kv_cache_hbm_mb=20000,
        instance_hbm_mb=36000,
        npu_slots=1,
        allow_colocation=False,
        request_capacity=1,
        required_capabilities=("vllm_ascend",),
        environment_fingerprint=environment,
        launch_options={"gpu_memory_utilization": 0.5, "block_size": 128},  # type: ignore[arg-type]
        warmup_request={  # type: ignore[arg-type]
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
        min_replicas=0,
        max_replicas=1,
        scale_down_idle_ms=0,
        scale_cooldown_ms=0,
        startup_timeout_ms=10_000,
        drain_timeout_ms=5_000,
    )


def _node(environment: str) -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_1",
        node_ip="127.0.0.1",
        cpu_total=16,
        mem_total_mb=65536,
        cpu_system_reserved=1,
        mem_system_reserved_mb=1024,
        io_slots_total=8,
        npus=(NpuCapacity("7", "910B3", 65536, 3200, 1, 62000),),
        capabilities={"environment_fingerprint": environment},  # type: ignore[arg-type]
    )


def test_real_adapter_node_agent_and_instance_manager_lifecycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        environment = "e" * 64
        identity = NodeAgentIdentity(
            cluster_id="cluster_1",
            node_id="node_a",
            boot_id="boot_1",
            ray_node_id="ray_node_a",
            agent_generation="agent_1",
            environment_fingerprint=environment,
            producer_id="node_agent:node_a:agent_1",
        )
        registry = RayNodeRegistry()
        inference_holder: list[InferenceCoordinator] = []

        def process_exited(event: ServiceProcessExit) -> None:
            if inference_holder:
                inference_holder[0].report_process_exited(
                    event.instance_id,
                    event.generation,
                    reason=f"service_process_exited:{event.exit_code}",
                )

        controller = NodeControlServer(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=environment,
            registry=registry,
            recorder=InMemoryRecorder(),
            event_sink=lambda event: None,
            on_service_process_exited=process_exited,
        )
        endpoint = await controller.start()
        manager = NodeServiceProcessManager(
            node_id="node_a",
            boot_id="boot_1",
            device_monitor=_Monitor(),
            allowed_executables=(sys.executable,),
            log_directory=tmp_path / "logs",
            first_port=32200,
            last_port=32210,
            hbm_recovery_tolerance_mb=0,
            poll_interval_ms=10,
        )
        agent = NodeAgent(
            identity=identity,
            authorization_token=b"test-token",
            heartbeat_interval_ms=20,
            service_process_manager=manager,
        )
        await agent.start(controller_endpoint=endpoint)
        backend = NodeAgentServiceProcessBackend(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            controller_generation="controller_1",
            node_registry=registry,
            rpc_timeout_ms=10_000,
        )
        fake_server = Path(__file__).with_name("fake_vllm_server.py")
        adapter = VllmAscendInferenceEngineAdapter(
            process_backend=backend,
            python_executable=sys.executable,
            endpoint_host_resolver=backend.endpoint_host,
            transport=_UrllibTransport(),
            api_server_entrypoint=(str(fake_server),),
            request_timeout_ms=1_000,
            probe_timeout_ms=8_000,
            probe_interval_ms=20,
        )
        spec = _spec(tmp_path / "model", environment)
        catalog = ModelCatalog(
            (spec,),
            adapters={adapter.name: adapter},
            environment_capabilities=("vllm_ascend",),
            max_single_npu_hbm_mb=60_000,
        )
        placement = PlacementManager(required_environment_fingerprint=environment)
        placement.register_node(_node(environment))
        inference = InferenceCoordinator(
            catalog=catalog,
            placement=placement,
            service_backend=backend,
            port_leases=backend,
            reconcile_interval_ms=20,
        )
        inference_holder.append(inference)
        await inference.start()
        try:
            inference.register_demand(
                run_id="run_1",
                task_id="task_1",
                model_id=spec.model_id,
            )
            await inference.reconcile()
            await inference.replicas.wait_for_background()
            instance = inference.model_instances()[0]
            assert instance.state is ModelInstanceState.READY
            assert instance.npu_device_id == "7"
            assert placement.active_lease_count() == 1
            assert backend.active_count() == 1
            assert [
                event.event_type
                for event in inference.events()
                if event.event_type.startswith("model_instance_")
            ] == [
                "model_instance_requested",
                "model_instance_reserving",
                "model_instance_starting",
                "model_instance_warming",
                "model_instance_ready",
            ]

            acquired = await inference.acquire_route(
                run_id="run_1",
                task_id="task_1",
                attempt=1,
                model_id=spec.model_id,
                session_key_hash="session",
                dispatch_deadline_ms=inference.clock.monotonic_ms() + 5_000,
            )
            assert acquired.lease is not None
            route = acquired.lease
            assert inference.activate_route(route.route_lease_id)
            session = inference.create_attempt_session(route)

            def invoke_twice() -> tuple[str, str]:
                with install_route_session(session):
                    request = ChatRequest.create(
                        [{"role": "user", "content": "hello"}], max_tokens=8
                    )
                    return session.invoke(request).text, session.invoke(request).text

            assert await asyncio.to_thread(invoke_twice) == (
                "ok:hello",
                "ok:hello",
            )
            assert [
                record.call_index for record in inference.request_records(route.route_lease_id)
            ] == [1, 2]
            assert await inference.release_route(route, reason="succeeded")
            await inference.replicas.wait_for_background()
            assert inference.model_instances()[0].state is ModelInstanceState.STOPPED
            assert placement.active_lease_count() == 0
            assert backend.active_count() == 0
            assert [
                event.event_type
                for event in inference.events()
                if event.event_type.startswith("model_instance_")
            ][-3:] == [
                "model_instance_draining",
                "model_instance_stopping",
                "model_instance_stopped",
            ]
        finally:
            await inference.close()
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())
