"""Stage-three Controller composition for the Ray Host execution path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ascend_maze.core.clock import Clock
from ascend_maze.core.identifiers import new_id
from ascend_maze.contracts.runtime import RuntimeNodeBinding
from ascend_maze.data.ray_store import RayDataStore
from ascend_maze.placement import NodeCapacity, NodeStatus
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime.ray_backend import RayRuntimeBackend
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry
from ascend_maze.runtime.worker_broker import ColdWorkerBroker

from ascend_maze.control.controller import InMemoryController
from ascend_maze.control.local_rpc import (
    ControllerStatus,
    LocalControlServer,
)
from ascend_maze.control.node_rpc import NodeControlServer


class RayHostController(InMemoryController):
    """Reuse the serial submission/scheduler authority with Ray boundaries."""

    def __init__(
        self,
        *,
        cluster_id: str,
        authorization_token: bytes,
        ray_namespace: str,
        config_fingerprint: str,
        environment_fingerprint: str,
        build_revision: str,
        node_capacities: tuple[NodeCapacity, ...],
        control_socket_path: Path | None = None,
        controller_generation: str | None = None,
        node_rpc_bind_address: str = "127.0.0.1:0",
        node_rpc_advertised_host: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        generation = controller_generation or new_id("controller")
        data_store = RayDataStore.start(
            owner_generation=generation,
            namespace=ray_namespace,
        )
        recorder = InMemoryRecorder()
        node_registry = RayNodeRegistry()
        worker_broker = ColdWorkerBroker(
            node_registry=node_registry,
            environment_fingerprint=environment_fingerprint,
        )
        runtime = RayRuntimeBackend(
            data_store=data_store,
            node_registry=node_registry,
            worker_broker=worker_broker,
            cluster_id=cluster_id,
            owner_generation=generation,
            environment_fingerprint=environment_fingerprint,
            recording_error_sink=recorder.record_writer_error,
        )
        super().__init__(
            config_fingerprint=config_fingerprint,
            environment_fingerprint=environment_fingerprint,
            build_revision=build_revision,
            node_capacities=node_capacities,
            controller_generation=generation,
            clock=clock,
            data_store=data_store,
            recorder=recorder,
            runtime=runtime,
        )
        self.cluster_id = cluster_id
        self.authorization_token = authorization_token
        self.ray_namespace = ray_namespace
        self.ray_data_store = data_store
        self.ray_recorder = recorder
        self.node_registry = node_registry
        self.worker_broker = worker_broker
        self.ray_runtime = runtime
        self._node_capacities = {
            capacity.node_id: capacity for capacity in node_capacities
        }
        self.node_rpc_bind_address = node_rpc_bind_address
        self.node_rpc_advertised_host = node_rpc_advertised_host
        self.node_rpc = NodeControlServer(
            cluster_id=cluster_id,
            authorization_token=authorization_token,
            controller_generation=generation,
            environment_fingerprint=environment_fingerprint,
            registry=node_registry,
            recorder=recorder,
            event_sink=runtime.post_node_event,
            on_binding_replaced=runtime.invalidate_binding,
            on_binding_disconnected=self._binding_disconnected,
            on_binding_registered=self._binding_registered,
            registration_validator=self._validate_node_registration,
            clock=self.clock,
        )
        self.local_rpc = (
            None
            if control_socket_path is None
            else LocalControlServer(
                socket_path=control_socket_path,
                status_provider=self._controller_status,
            )
        )
        self._ray_host_closed = False

    @property
    def node_rpc_endpoint(self) -> str:
        endpoint = self.node_rpc.endpoint
        if endpoint is None:
            raise RuntimeError("RayHostController is not started")
        return endpoint

    async def start(self) -> None:
        if self._started:
            return
        await super().start()
        try:
            await self.node_rpc.start(
                self.node_rpc_bind_address,
                advertised_host=self.node_rpc_advertised_host,
            )
            if self.local_rpc is not None:
                await self.local_rpc.start()
        except Exception:
            await super().close()
            self.ray_data_store.close(kill_owner=True)
            raise

    async def close(self) -> None:
        if self._ray_host_closed:
            return
        self._ray_host_closed = True
        if self.local_rpc is not None:
            await self.local_rpc.close()
        await self.node_rpc.close()
        await super().close()
        self.ray_data_store.close(kill_owner=True)

    def _controller_status(self) -> ControllerStatus:
        return ControllerStatus(
            controller_generation=self.controller_generation,
            build_revision=self.build_revision,
            environment_fingerprint=self.environment_fingerprint,
            healthy_node_count=len(self.node_registry.active_bindings()),
        )

    def _binding_registered(
        self,
        binding: RuntimeNodeBinding,
        previous: RuntimeNodeBinding | None,
    ) -> None:
        del previous
        capacity = self._node_capacities.get(binding.node_id)
        if capacity is None:
            raise ValueError(f"NodeAgent registered unknown node: {binding.node_id}")
        if capacity.boot_id != binding.boot_id:
            capacity = replace(capacity, boot_id=binding.boot_id)
            self._node_capacities[binding.node_id] = capacity
            self.placement.register_node(capacity)
        else:
            self.placement.set_node_status(
                binding.node_id,
                NodeStatus.HEALTHY,
                now_ms=self.clock.monotonic_ms(),
            )

    def _validate_node_registration(self, node_id: str) -> None:
        if node_id not in self._node_capacities:
            raise ValueError(f"NodeAgent registered unknown node: {node_id}")

    def _binding_disconnected(self, binding: RuntimeNodeBinding) -> None:
        self.ray_runtime.invalidate_binding(binding)
        self.placement.set_node_status(
            binding.node_id,
            NodeStatus.OFFLINE,
            now_ms=self.clock.monotonic_ms(),
        )
