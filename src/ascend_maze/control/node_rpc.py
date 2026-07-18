"""Long-lived Controller/NodeAgent stream and node-local Worker event RPC."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
import hmac
from typing import Any

import grpc

from ascend_maze.contracts.recording import ExecutionEvent, ExecutionRecorder
from ascend_maze.contracts.runtime import RuntimeNodeBinding
from ascend_maze.core.canonical import FrozenMap, freeze_canonical
from ascend_maze.core.clock import Clock, SystemClock
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.core.identifiers import new_id
from ascend_maze.placement import NodeObservation, NpuObservation
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.runtime.ray_node_registry import (
    RayNodeRegistry,
    RuntimeNodeStatus,
)

from ascend_maze.control.proto import control_pb2 as _control_pb2
from ascend_maze.control.proto import control_pb2_grpc
from ascend_maze.control.proto_codec import decode_runtime_event, encode_runtime_event

control_pb2: Any = _control_pb2


@dataclass(frozen=True, slots=True)
class NodeAgentIdentity:
    cluster_id: str
    node_id: str
    boot_id: str
    ray_node_id: str
    agent_generation: str
    environment_fingerprint: str
    producer_id: str

    def __post_init__(self) -> None:
        for name in (
            "cluster_id",
            "node_id",
            "boot_id",
            "ray_node_id",
            "agent_generation",
            "environment_fingerprint",
            "producer_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ContractValidationError(f"{name} is required")


class _NodeControlServicer:
    def __init__(self, owner: "NodeControlServer") -> None:
        self.owner = owner

    async def Connect(
        self,
        request_iterator: AsyncIterator[Any],
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> AsyncIterator[Any]:
        try:
            first = await anext(request_iterator)
        except StopAsyncIteration:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "registration required")
            return
        if first.WhichOneof("body") != "register":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "registration must be first")
            return
        registration = first.register
        try:
            binding = self.owner._accept_registration(registration)
        except (ContractValidationError, ValueError) as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            return
        yield control_pb2.ControllerStreamMessage(
            registration=control_pb2.RegistrationAccepted(
                request_id=registration.meta.message_id,
                controller_generation=self.owner.controller_generation,
                runtime_generation=binding.runtime_generation,
                status_code="accepted",
                message="node registration accepted",
            )
        )
        try:
            async for request in request_iterator:
                ack = self.owner._handle_message(binding, request)
                yield control_pb2.ControllerStreamMessage(ack=ack)
        finally:
            changed = self.owner.registry.set_status(
                binding.node_id,
                RuntimeNodeStatus.STALE,
                boot_id=binding.boot_id,
                agent_generation=binding.agent_generation,
            )
            if changed and self.owner.on_binding_disconnected is not None:
                self.owner.on_binding_disconnected(binding)


class NodeControlServer:
    def __init__(
        self,
        *,
        cluster_id: str,
        authorization_token: bytes,
        controller_generation: str,
        environment_fingerprint: str,
        registry: RayNodeRegistry,
        recorder: ExecutionRecorder,
        event_sink: Callable[[RuntimeEvent], None],
        on_binding_replaced: Callable[[RuntimeNodeBinding], None] | None = None,
        on_binding_disconnected: Callable[[RuntimeNodeBinding], None] | None = None,
        on_binding_registered: (
            Callable[[RuntimeNodeBinding, RuntimeNodeBinding | None], None] | None
        ) = None,
        registration_validator: Callable[[str], None] | None = None,
        on_node_observation: Callable[[NodeObservation], object] | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not cluster_id or not authorization_token or not controller_generation:
            raise ValueError("cluster, token and controller generation are required")
        self.cluster_id = cluster_id
        self.authorization_token = authorization_token
        self.controller_generation = controller_generation
        self.environment_fingerprint = environment_fingerprint
        self.registry = registry
        self.recorder = recorder
        self.event_sink = event_sink
        self.on_binding_replaced = on_binding_replaced
        self.on_binding_disconnected = on_binding_disconnected
        self.on_binding_registered = on_binding_registered
        self.registration_validator = registration_validator
        self.on_node_observation = on_node_observation
        self.clock = clock or SystemClock()
        self._server: grpc.aio.Server | None = None
        self.endpoint: str | None = None

    async def start(
        self,
        bind_address: str = "127.0.0.1:0",
        *,
        advertised_host: str | None = None,
    ) -> str:
        if self._server is not None:
            assert self.endpoint is not None
            return self.endpoint
        host = advertised_host or bind_address.rsplit(":", 1)[0]
        if host in {"0.0.0.0", "::", "[::]"}:
            raise ValueError("wildcard RPC bind requires an advertised_host")
        server = grpc.aio.server()
        control_pb2_grpc.add_NodeControlServicer_to_server(
            _NodeControlServicer(self), server
        )
        port = server.add_insecure_port(bind_address)
        if port == 0:
            raise RuntimeError(f"failed to bind NodeControl RPC: {bind_address}")
        self.endpoint = f"{host}:{port}"
        await server.start()
        self._server = server
        return self.endpoint

    async def close(self, grace_seconds: float = 1.0) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        await server.stop(grace_seconds)

    def _accept_registration(self, registration: Any) -> RuntimeNodeBinding:
        meta = registration.meta
        if int(meta.schema_version) != 1:
            raise ContractValidationError("unsupported NodeAgent schema version")
        if str(meta.cluster_id) != self.cluster_id:
            raise ContractValidationError("NodeAgent cluster_id mismatch")
        if not hmac.compare_digest(
            bytes(registration.authorization_token), self.authorization_token
        ):
            raise ContractValidationError("NodeAgent authorization failed")
        if self.registration_validator is not None:
            self.registration_validator(str(meta.node_id))
        status = (
            RuntimeNodeStatus.HEALTHY
            if str(registration.environment_fingerprint)
            == self.environment_fingerprint
            else RuntimeNodeStatus.UNSCHEDULABLE
        )
        binding, previous = self.registry.register(
            node_id=str(meta.node_id),
            boot_id=str(meta.boot_id),
            ray_node_id=str(registration.ray_node_id),
            agent_generation=str(meta.agent_generation),
            agent_endpoint=str(registration.agent_endpoint),
            producer_id=str(registration.producer_id),
            status=status,
        )
        if previous is not None and self.on_binding_replaced is not None:
            self.on_binding_replaced(previous)
        if self.on_binding_registered is not None:
            self.on_binding_registered(binding, previous)
        return binding

    def _handle_message(self, binding: RuntimeNodeBinding, request: Any) -> Any:
        body = request.WhichOneof("body")
        if body not in {"heartbeat", "runtime_event"}:
            return self._ack("", "rejected", "unsupported stream message")
        value = getattr(request, body)
        meta = value.meta
        if not self._meta_matches(binding, meta):
            return self._ack(str(meta.message_id), "stale", "node generation mismatch")
        accepted = self.registry.accept_message(
            node_id=binding.node_id,
            boot_id=binding.boot_id,
            agent_generation=binding.agent_generation,
            sequence=int(meta.sequence),
        )
        if not accepted:
            return self._ack(str(meta.message_id), "duplicate", "old message sequence")
        if body == "runtime_event":
            try:
                event = decode_runtime_event(value.event)
                self._record_node_event(
                    binding, int(value.producer_sequence), event
                )
                self.event_sink(event)
            except Exception as exc:
                self.recorder.record_writer_error(
                    str(value.event.run_id), f"{type(exc).__name__}: {exc}"
                )
                return self._ack(str(meta.message_id), "rejected", str(exc))
        elif body == "heartbeat" and value.has_observation:
            try:
                observation = NodeObservation(
                    node_id=binding.node_id,
                    boot_id=binding.boot_id,
                    sequence=int(meta.sequence),
                    received_at_ms=self.clock.monotonic_ms(),
                    observed_free_mem_mb=int(value.observed_free_mem_mb),
                    npus=tuple(
                        NpuObservation(
                            device_id=str(item.device_id),
                            health=str(item.health),
                            observed_free_hbm_mb=int(item.observed_free_hbm_mb),
                            utilization=(
                                float(item.utilization)
                                if item.has_utilization
                                else None
                            ),
                        )
                        for item in value.npus
                    ),
                )
                if self.on_node_observation is not None:
                    self.on_node_observation(observation)
            except Exception as exc:
                return self._ack(str(meta.message_id), "rejected", str(exc))
        return self._ack(str(meta.message_id), "accepted", "")

    def _record_node_event(
        self,
        binding: RuntimeNodeBinding,
        producer_sequence: int,
        event: RuntimeEvent,
    ) -> None:
        payload_items: list[tuple[str, object]] = []
        if event.worker_pid is not None:
            payload_items.append(("worker_pid", event.worker_pid))
        if event.device_id is not None:
            payload_items.extend(
                (
                    ("physical_device_id", event.device_id),
                    ("binding_verified", event.binding_verified),
                )
            )
        observation = event.resource_observation
        if observation is not None:
            payload_items.extend(
                (
                    ("peak_host_rss_mb", observation.peak_host_rss_mb),
                    (
                        "peak_npu_allocated_mb",
                        observation.peak_npu_allocated_mb,
                    ),
                    ("peak_npu_reserved_mb", observation.peak_npu_reserved_mb),
                    (
                        "peak_npu_process_hbm_mb",
                        observation.peak_npu_process_hbm_mb,
                    ),
                    ("npu_metric_source", observation.npu_metric_source),
                    ("npu_metric_quality", observation.npu_metric_quality),
                )
            )
        payload = freeze_canonical(dict(payload_items))
        if not isinstance(payload, FrozenMap):
            raise AssertionError("node event payload must be a mapping")
        accepted = self.recorder.emit(
            ExecutionEvent(
                schema_version=1,
                event_id=f"node_record:{event.event_id}",
                experiment_id=event.run_id,
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                lease_id=event.lease_id,
                route_lease_id=event.route_lease_id,
                model_instance_id=None,
                event_type=event.kind.value,
                producer_id=binding.producer_id,
                producer_sequence=producer_sequence,
                node_id=binding.node_id,
                device_id=event.device_id,
                monotonic_time_ms=event.occurred_at_ms,
                wall_time_ms=self.clock.wall_ms(),
                duration_ms=None,
                payload=payload,
            )
        )
        if not accepted:
            self.recorder.record_writer_error(
                event.run_id, "NodeAgent recorder rejected a control event"
            )

    def _meta_matches(self, binding: RuntimeNodeBinding, meta: Any) -> bool:
        return (
            int(meta.schema_version) == 1
            and str(meta.cluster_id) == self.cluster_id
            and str(meta.node_id) == binding.node_id
            and str(meta.boot_id) == binding.boot_id
            and str(meta.agent_generation) == binding.agent_generation
        )

    def _ack(self, message_id: str, status_code: str, message: str) -> Any:
        return control_pb2.NodeMessageAck(
            message_id=message_id,
            controller_generation=self.controller_generation,
            status_code=status_code,
            message=message,
        )


class _WorkerEventServicer:
    def __init__(self, owner: "NodeAgent") -> None:
        self.owner = owner

    async def Report(
        self,
        request: Any,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> Any:
        del context
        return self.owner._accept_worker_event(request)


class NodeAgent:
    def __init__(
        self,
        *,
        identity: NodeAgentIdentity,
        authorization_token: bytes,
        heartbeat_interval_ms: int = 1_000,
        event_queue_capacity: int = 1_024,
        worker_device_verifier: Callable[[int, str], bool] | None = None,
        node_observation_provider: (
            Callable[[int, int], NodeObservation] | None
        ) = None,
        clock: Clock | None = None,
    ) -> None:
        if not authorization_token:
            raise ValueError("NodeAgent authorization token is required")
        if heartbeat_interval_ms <= 0 or event_queue_capacity <= 0:
            raise ValueError("NodeAgent intervals and capacities must be positive")
        self.identity = identity
        self.authorization_token = authorization_token
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.clock = clock or SystemClock()
        self.worker_device_verifier = worker_device_verifier
        self.node_observation_provider = node_observation_provider
        self._queue: asyncio.Queue[Any] = asyncio.Queue(event_queue_capacity)
        self._sequence = 0
        self._producer_sequence = 0
        self._event_ids: set[str] = set()
        self._event_id_order: deque[str] = deque()
        self._event_dedup_capacity = max(1_024, event_queue_capacity * 4)
        self._server: grpc.aio.Server | None = None
        self._channel: grpc.aio.Channel | None = None
        self._call: Any = None
        self._response_task: asyncio.Task[None] | None = None
        self._registered = asyncio.Event()
        self._closed = False
        self.endpoint: str | None = None
        self.runtime_generation: int | None = None

    async def start(
        self,
        *,
        controller_endpoint: str,
        worker_bind_address: str = "127.0.0.1:0",
        worker_advertised_host: str | None = None,
    ) -> str:
        if self._server is not None:
            assert self.endpoint is not None
            return self.endpoint
        host = worker_advertised_host or worker_bind_address.rsplit(":", 1)[0]
        if host in {"0.0.0.0", "::", "[::]"}:
            raise ValueError("wildcard Worker RPC bind requires an advertised_host")
        server = grpc.aio.server()
        control_pb2_grpc.add_WorkerEventSinkServicer_to_server(
            _WorkerEventServicer(self), server
        )
        port = server.add_insecure_port(worker_bind_address)
        if port == 0:
            raise RuntimeError(f"failed to bind WorkerEvent RPC: {worker_bind_address}")
        self.endpoint = f"{host}:{port}"
        await server.start()
        self._server = server
        self._channel = grpc.aio.insecure_channel(controller_endpoint)
        stub = control_pb2_grpc.NodeControlStub(self._channel)
        self._call = stub.Connect(self._request_stream())
        self._response_task = asyncio.create_task(self._consume_responses())
        registration_waiter = asyncio.create_task(self._registered.wait())
        try:
            done, _ = await asyncio.wait(
                {registration_waiter, self._response_task},
                timeout=5,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if registration_waiter in done and self._registered.is_set():
                return self.endpoint
            if self._response_task in done:
                await self._response_task
                raise RuntimeError("NodeControl stream ended before registration")
            raise TimeoutError("NodeAgent registration timed out")
        except Exception:
            await self.close()
            raise
        finally:
            registration_waiter.cancel()
            await asyncio.gather(registration_waiter, return_exceptions=True)

    async def close(self, grace_seconds: float = 1.0) -> None:
        if self._closed:
            return
        self._closed = True
        call = self._call
        if call is not None:
            call.cancel()
        task = self._response_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._channel is not None:
            await self._channel.close()
        if self._server is not None:
            await self._server.stop(grace_seconds)
        self._server = None

    async def stop_worker_event_server(self, grace_seconds: float = 0) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        await server.stop(grace_seconds)

    async def _request_stream(self) -> AsyncIterator[Any]:
        assert self.endpoint is not None
        yield control_pb2.AgentStreamMessage(
            register=control_pb2.RegisterNode(
                meta=self._next_meta(),
                ray_node_id=self.identity.ray_node_id,
                agent_endpoint=self.endpoint,
                producer_id=self.identity.producer_id,
                environment_fingerprint=self.identity.environment_fingerprint,
                authorization_token=self.authorization_token,
            )
        )
        await self._registered.wait()
        while not self._closed:
            try:
                message = await asyncio.wait_for(
                    self._queue.get(), self.heartbeat_interval_ms / 1_000
                )
            except asyncio.TimeoutError:
                meta = self._next_meta()
                heartbeat = control_pb2.NodeHeartbeat(meta=meta)
                provider = self.node_observation_provider
                if provider is not None:
                    try:
                        observation = provider(
                            int(meta.sequence), self.clock.monotonic_ms()
                        )
                    except Exception:
                        observation = None
                    if observation is not None:
                        heartbeat.has_observation = True
                        heartbeat.observed_free_mem_mb = (
                            observation.observed_free_mem_mb
                        )
                        for item in observation.npus:
                            heartbeat.npus.add(
                                device_id=item.device_id,
                                health=item.health,
                                observed_free_hbm_mb=item.observed_free_hbm_mb,
                                utilization=item.utilization or 0.0,
                                has_utilization=item.utilization is not None,
                            )
                message = control_pb2.AgentStreamMessage(heartbeat=heartbeat)
            yield message

    async def _consume_responses(self) -> None:
        assert self._call is not None
        async for response in self._call:
            body = response.WhichOneof("body")
            if body == "registration":
                if response.registration.status_code != "accepted":
                    raise RuntimeError(response.registration.message)
                self.runtime_generation = int(response.registration.runtime_generation)
                self._registered.set()

    def _accept_worker_event(self, request: Any) -> Any:
        event_id = str(request.event.event_id)
        if (
            int(request.schema_version) != 1
            or str(request.cluster_id) != self.identity.cluster_id
            or str(request.node_id) != self.identity.node_id
            or str(request.boot_id) != self.identity.boot_id
            or str(request.agent_generation) != self.identity.agent_generation
        ):
            return control_pb2.WorkerEventAck(
                event_id=event_id,
                accepted=False,
                error_code="stale_worker_generation",
                message="Worker event identity does not match NodeAgent",
            )
        if event_id in self._event_ids:
            return control_pb2.WorkerEventAck(event_id=event_id, accepted=True)
        if str(request.event.kind) == RuntimeEventKind.WORKER_STARTED.value and str(
            request.event.device_id
        ):
            verifier = self.worker_device_verifier
            if (
                verifier is None
                or not request.event.has_worker_pid
                or not request.event.binding_verified
                or not verifier(
                    int(request.event.worker_pid),
                    str(request.event.device_id),
                )
            ):
                return control_pb2.WorkerEventAck(
                    event_id=event_id,
                    accepted=False,
                    error_code="device_bind_failed",
                    message=(
                        "NodeAgent could not verify Worker PID on the leased "
                        "physical NPU"
                    ),
                )
        message = control_pb2.AgentStreamMessage(
            runtime_event=control_pb2.NodeRuntimeEvent(
                meta=self._next_meta(),
                event=request.event,
                producer_sequence=self._next_producer_sequence(),
            )
        )
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            return control_pb2.WorkerEventAck(
                event_id=event_id,
                accepted=False,
                error_code="node_event_queue_full",
                message="NodeAgent control event queue is full",
            )
        self._event_ids.add(event_id)
        self._event_id_order.append(event_id)
        if len(self._event_id_order) > self._event_dedup_capacity:
            expired = self._event_id_order.popleft()
            self._event_ids.discard(expired)
        return control_pb2.WorkerEventAck(event_id=event_id, accepted=True)

    def _next_meta(self) -> Any:
        self._sequence += 1
        return control_pb2.AgentMeta(
            schema_version=1,
            cluster_id=self.identity.cluster_id,
            node_id=self.identity.node_id,
            boot_id=self.identity.boot_id,
            agent_generation=self.identity.agent_generation,
            sequence=self._sequence,
            message_id=new_id("node_message"),
            sent_at_ms=self.clock.wall_ms(),
        )

    def _next_producer_sequence(self) -> int:
        self._producer_sequence += 1
        return self._producer_sequence


def report_worker_event(
    *,
    endpoint: str,
    identity: NodeAgentIdentity,
    event: RuntimeEvent,
    timeout_seconds: float,
) -> None:
    request = control_pb2.WorkerEventRequest(
        schema_version=1,
        cluster_id=identity.cluster_id,
        node_id=identity.node_id,
        boot_id=identity.boot_id,
        agent_generation=identity.agent_generation,
        event=encode_runtime_event(event),
    )
    with grpc.insecure_channel(endpoint) as channel:
        stub = control_pb2_grpc.WorkerEventSinkStub(channel)
        response = stub.Report(request, timeout=timeout_seconds)
    if not response.accepted:
        raise RuntimeError(f"{response.error_code}: {response.message}")
