"""Stage-three Controller composition for the Ray Host execution path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ascend_maze.core.clock import Clock
from ascend_maze.core.identifiers import new_id
from ascend_maze.contracts.runtime import RuntimeNodeBinding
from ascend_maze.contracts.worker import WorkerPoolConfig
from ascend_maze.data.ray_store import RayDataStore
from ascend_maze.placement import NodeCapacity, NodeObservation, NodeStatus
from ascend_maze.placement import PlacementManager
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime.ray_backend import RayRuntimeBackend
from ascend_maze.runtime.ray_node_registry import (
    RayNodeRegistry,
    RuntimeNodeStatus,
)
from ascend_maze.runtime.worker_broker import ColdWorkerBroker
from ascend_maze.runtime.worker_pool import (
    StandbyWorkerBroker,
    WorkerPoolEvent,
)
from ascend_maze.runtime.ray_worker_pool import RayWorkerEndpointFactory
from ascend_maze.resources import ResourceAnchorProvider
from ascend_maze.scheduler import QueuePartitioner, SchedulingPolicy

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
        anchors: ResourceAnchorProvider | None = None,
        placement: PlacementManager | None = None,
        policy: SchedulingPolicy | None = None,
        partitioner: QueuePartitioner | None = None,
        placement_lookahead: int = 8,
        max_bypass_count: int = 8,
        dispatch_timeout_ms: int = 5_000,
        worker_pool_config: WorkerPoolConfig | None = None,
    ) -> None:
        generation = controller_generation or new_id("controller")
        data_store = RayDataStore.start(
            owner_generation=generation,
            namespace=ray_namespace,
        )
        recorder = InMemoryRecorder()
        node_registry = RayNodeRegistry()
        effective_placement = placement or PlacementManager()
        pool_events: list[WorkerPoolEvent] = []
        worker_broker: ColdWorkerBroker | StandbyWorkerBroker
        if worker_pool_config is None:
            worker_broker = ColdWorkerBroker(
                node_registry=node_registry,
                environment_fingerprint=environment_fingerprint,
            )
        else:
            worker_broker = StandbyWorkerBroker(
                node_registry=node_registry,
                placement=effective_placement,
                environment_fingerprint=environment_fingerprint,
                config=worker_pool_config,
                endpoint_factory=RayWorkerEndpointFactory(),
                event_sink=pool_events.append,
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
            anchors=anchors,
            placement=effective_placement,
            policy=policy,
            partitioner=partitioner,
            placement_lookahead=placement_lookahead,
            max_bypass_count=max_bypass_count,
            dispatch_timeout_ms=dispatch_timeout_ms,
        )
        self.cluster_id = cluster_id
        self.authorization_token = authorization_token
        self.ray_namespace = ray_namespace
        self.ray_data_store = data_store
        self.ray_recorder = recorder
        self.node_registry = node_registry
        self.worker_broker = worker_broker
        self.worker_pool_config = worker_pool_config
        self.pool_events = pool_events
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
            on_node_observation=self._node_observation,
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
        if isinstance(worker_broker, StandbyWorkerBroker):
            worker_broker.set_resource_changed_sink(
                lambda reason: self._post_pool_resource_changed(reason)
            )

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
            if isinstance(self.worker_broker, StandbyWorkerBroker):
                await self.worker_broker.start()
        except Exception:
            if isinstance(self.worker_broker, StandbyWorkerBroker):
                await self.worker_broker.close()
            await super().close()
            self.ray_data_store.close(kill_owner=True)
            raise

    async def close(self) -> None:
        if self._ray_host_closed:
            return
        if self.local_rpc is not None:
            await self.local_rpc.close()
        await self.node_rpc.close()
        await super().close()
        if isinstance(self.worker_broker, StandbyWorkerBroker):
            await self.worker_broker.close()
        self.ray_data_store.close(kill_owner=True)
        self._ray_host_closed = True

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
        if self.node_registry.status(binding.node_id) is RuntimeNodeStatus.HEALTHY:
            self.placement.set_node_status(
                binding.node_id,
                NodeStatus.HEALTHY,
                now_ms=self.clock.monotonic_ms(),
            )
        else:
            self.placement.set_node_status(
                binding.node_id,
                NodeStatus.UNSCHEDULABLE,
                now_ms=self.clock.monotonic_ms(),
            )
        self.core.post_resource_changed(f"node_binding_registered:{binding.node_id}")
        if isinstance(self.worker_broker, StandbyWorkerBroker):
            self.worker_broker.notify_changed()

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
        self.core.post_resource_changed(f"node_binding_disconnected:{binding.node_id}")
        if isinstance(self.worker_broker, StandbyWorkerBroker):
            self.worker_broker.notify_changed()

    def _node_observation(self, observation: NodeObservation) -> bool:
        changed = self.placement.update_observation(observation)
        if changed:
            self.core.post_resource_changed(
                f"node_observation:{observation.node_id}:{observation.sequence}"
            )
        return changed

    def _post_pool_resource_changed(self, reason: str) -> None:
        self.core.post_resource_changed(reason)
