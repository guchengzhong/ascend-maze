from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ray

from ascend_maze.control.node_rpc import NodeAgent, NodeAgentIdentity
from ascend_maze.control.ray_controller import RayHostController
from ascend_maze.control.service_process import (
    NodeAgentServiceProcessBackend,
    NodeServiceProcessManager,
)
from ascend_maze.inference import (
    InferenceCoordinator,
    ModelCatalog,
    ModelInstanceState,
    ModelSpec,
)
from ascend_maze.inference.adapters.vllm_ascend import (
    VllmAscendInferenceEngineAdapter,
    VllmHttpResponse,
)
from ascend_maze.inference.contracts import InferenceCallError
from ascend_maze.placement import NodeCapacity, NodeStatus, NpuCapacity, PlacementManager
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry


CONFIG = "c" * 64
ENVIRONMENT = "e" * 64


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


class _Transport:
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


def _node() -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_1",
        node_ip="127.0.0.1",
        cpu_total=16,
        mem_total_mb=65_536,
        cpu_system_reserved=1,
        mem_system_reserved_mb=1_024,
        io_slots_total=8,
        npus=(NpuCapacity("7", "910B3", 65_536, 3_200, 1, 62_000),),
        capabilities={"environment_fingerprint": ENVIRONMENT},  # type: ignore[arg-type]
    )


def _spec(model_path: Path) -> ModelSpec:
    model_path.mkdir(exist_ok=True)
    return ModelSpec(
        model_id="model-recovery",
        catalog_revision="catalog_1",
        artifact_path=str(model_path),
        tokenizer_path=None,
        artifact_revision="a" * 64,
        backend="vllm_ascend",
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=1,
        max_model_len=2_048,
        instance_cpu_num=2,
        instance_host_mem_mb=2_048,
        weight_hbm_mb=8_000,
        runtime_hbm_mb=4_000,
        kv_cache_hbm_mb=20_000,
        instance_hbm_mb=36_000,
        npu_slots=1,
        allow_colocation=False,
        request_capacity=1,
        required_capabilities=("vllm_ascend",),
        environment_fingerprint=ENVIRONMENT,
        launch_options={"gpu_memory_utilization": 0.5, "block_size": 128},  # type: ignore[arg-type]
        warmup_request={  # type: ignore[arg-type]
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
        min_replicas=0,
        max_replicas=1,
        startup_timeout_ms=10_000,
        drain_timeout_ms=5_000,
    )


def _inference(
    *,
    generation: str,
    registry: RayNodeRegistry,
    placement: PlacementManager,
    spec: ModelSpec,
) -> tuple[InferenceCoordinator, NodeAgentServiceProcessBackend]:
    backend = NodeAgentServiceProcessBackend(
        cluster_id="cluster_model_recovery",
        authorization_token=b"recovery-token",
        controller_generation=generation,
        node_registry=registry,
        rpc_timeout_ms=10_000,
    )
    fake_server = Path(__file__).with_name("fake_vllm_server.py")
    adapter = VllmAscendInferenceEngineAdapter(
        process_backend=backend,
        python_executable=sys.executable,
        endpoint_host_resolver=backend.endpoint_host,
        transport=_Transport(),
        api_server_entrypoint=(str(fake_server),),
        request_timeout_ms=1_000,
        probe_timeout_ms=8_000,
        probe_interval_ms=20,
    )
    catalog = ModelCatalog(
        (spec,),
        adapters={adapter.name: adapter},
        environment_capabilities=("vllm_ascend",),
        max_single_npu_hbm_mb=60_000,
    )
    return (
        InferenceCoordinator(
            catalog=catalog,
            placement=placement,
            service_backend=backend,
            port_leases=backend,
            reconcile_interval_ms=20,
        ),
        backend,
    )


async def _wait_reconciled(
    controller: RayHostController,
    inference: InferenceCoordinator,
    instance_id: str,
) -> None:
    for _ in range(1_000):
        if (
            not controller.recovery_pending_nodes
            and inference.instances.snapshot(instance_id).state
            is ModelInstanceState.STOPPED
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("model service recovery did not complete")


def test_node_agent_model_process_is_reconciled_across_controller_generation(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        recovery_path = tmp_path / "controller-recovery.sqlite3"
        spec = _spec(tmp_path / "model")
        first_registry = RayNodeRegistry()
        first_placement = PlacementManager(
            required_environment_fingerprint=ENVIRONMENT
        )
        first_inference, _ = _inference(
            generation="controller_1",
            registry=first_registry,
            placement=first_placement,
            spec=spec,
        )
        first = RayHostController(
            cluster_id="cluster_model_recovery",
            authorization_token=b"recovery-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_1",
            placement=first_placement,
            inference=first_inference,
            node_registry=first_registry,
            recovery_path=recovery_path,
        )
        await first.start()
        manager = NodeServiceProcessManager(
            node_id="node_a",
            boot_id="boot_1",
            device_monitor=_Monitor(),
            allowed_executables=(sys.executable,),
            log_directory=tmp_path / "logs",
            first_port=32500,
            last_port=32510,
            hbm_recovery_tolerance_mb=0,
            poll_interval_ms=10,
        )
        agent = NodeAgent(
            identity=NodeAgentIdentity(
                cluster_id="cluster_model_recovery",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            ),
            authorization_token=b"recovery-token",
            heartbeat_interval_ms=20,
            service_process_manager=manager,
        )
        await agent.start(controller_endpoint=first.node_rpc_endpoint)
        requested = first_inference.instances.create_requested(spec.model_id)
        ready = await first_inference.instances.start_instance(requested.instance_id)
        assert ready.state is ModelInstanceState.READY
        assert ready.service_handle_id is not None
        assert ready.placement_lease_id is not None
        assert first_placement.active_lease_count() == 1
        service_pid = next(
            int(event.payload["process_id"])
            for event in first_inference.events()
            if event.event_type == "model_instance_warming"
        )
        os.kill(service_pid, 0)
        await first.crash()

        second_registry = RayNodeRegistry()
        second_placement = PlacementManager(
            required_environment_fingerprint=ENVIRONMENT
        )
        second_inference, second_backend = _inference(
            generation="controller_2",
            registry=second_registry,
            placement=second_placement,
            spec=spec,
        )
        second = RayHostController(
            cluster_id="cluster_model_recovery",
            authorization_token=b"recovery-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_2",
            placement=second_placement,
            inference=second_inference,
            node_registry=second_registry,
            recovery_path=recovery_path,
        )
        try:
            await second.start()
            await agent.reconnect(second.node_rpc_endpoint)
            await _wait_reconciled(second, second_inference, ready.instance_id)
            restored = second_inference.instances.snapshot(ready.instance_id)
            assert restored.generation == ready.generation
            assert restored.state is ModelInstanceState.STOPPED
            assert second_placement.active_lease_count() == 0
            assert second_backend.active_count() == 0
            assert second.placement.snapshot().nodes[0].status is NodeStatus.HEALTHY
            try:
                os.kill(service_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("recovered model service process is still alive")
        finally:
            await agent.close(grace_seconds=0)
            await second.close()
            second.ray_data_store.close(kill_owner=True)

    asyncio.run(scenario())
