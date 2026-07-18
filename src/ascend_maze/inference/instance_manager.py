"""Serial model-instance state and global PlacementLease ownership."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Callable

from ascend_maze.contracts.resources import PlacementLease
from ascend_maze.core.clock import Clock, SystemClock
from ascend_maze.core.canonical import FrozenMap, freeze_canonical
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.core.identifiers import new_id
from ascend_maze.inference.catalog import ModelCatalog
from ascend_maze.inference.contracts import (
    EngineProbe,
    ModelControlEvent,
    ModelInstance,
    ModelInstanceState,
    ModelSpec,
    PortLease,
    ServiceHandle,
    ServiceProcessBackend,
)
from ascend_maze.placement import PlacementManager


@dataclass(slots=True)
class _InstanceRecord:
    instance_id: str
    spec: ModelSpec
    generation: int
    state: ModelInstanceState
    created_at_ms: int
    state_changed_at_ms: int
    ready_at_ms: int | None = None
    placement_lease: PlacementLease | None = None
    port_lease: PortLease | None = None
    service_handle: ServiceHandle | None = None
    route_occupancy: int = 0
    actual_request_inflight: int = 0
    last_used_at_ms: int = 0
    failure_reason: str | None = None


class ModelInstanceManager:
    def __init__(
        self,
        *,
        catalog: ModelCatalog,
        placement: PlacementManager,
        service_backend: ServiceProcessBackend,
        event_sink: Callable[[ModelControlEvent], None] | None = None,
        clock: Clock | None = None,
        first_port: int = 25_000,
    ) -> None:
        if first_port < 1 or first_port > 65_535:
            raise ValueError("first_port is invalid")
        self.catalog = catalog
        self.placement = placement
        self.service_backend = service_backend
        self.event_sink = event_sink
        self.clock = clock or SystemClock()
        self._records: dict[str, _InstanceRecord] = {}
        self._next_port = first_port
        self._lock = RLock()

    def create_requested(self, model_id: str) -> ModelInstance:
        spec = self.catalog.get(model_id)
        now = self.clock.monotonic_ms()
        record = _InstanceRecord(
            instance_id=new_id("model_instance"),
            spec=spec,
            generation=1,
            state=ModelInstanceState.REQUESTED,
            created_at_ms=now,
            state_changed_at_ms=now,
            last_used_at_ms=now,
        )
        with self._lock:
            self._records[record.instance_id] = record
        self._emit(record, "model_instance_requested")
        return self._snapshot(record)

    async def start_instance(self, instance_id: str) -> ModelInstance:
        with self._lock:
            record = self._require(instance_id)
            if record.state is ModelInstanceState.READY:
                return self._snapshot(record)
            if record.state not in {
                ModelInstanceState.REQUESTED,
                ModelInstanceState.RESERVING,
            }:
                raise StateTransitionError(
                    f"cannot start instance from {record.state.value}"
                )
            self._transition(record, ModelInstanceState.RESERVING)
            now = self.clock.monotonic_ms()
            placement = self.placement.reserve_model_instance(
                instance_id=record.instance_id,
                generation=record.generation,
                resources=record.spec.reservation,
                allow_colocation=record.spec.allow_colocation,
                now_ms=now,
                startup_deadline_ms=now + record.spec.startup_timeout_ms,
            )
            if not placement.selected:
                self._emit(
                    record,
                    "model_placement_pending",
                    {"reason": placement.rejection_reason},
                )
                return self._snapshot(record)
            assert placement.lease is not None
            record.placement_lease = placement.lease
            self._transition(record, ModelInstanceState.STARTING)
            port = self._allocate_port(record)
            adapter = self.catalog.adapter(record.spec.model_id)
            request = adapter.build_launch_request(record.spec, placement.lease, port)
        try:
            startup_deadline = monotonic() + record.spec.startup_timeout_ms / 1_000
            handle = await asyncio.wait_for(
                self.service_backend.launch(request, placement.lease),
                timeout=self._remaining(startup_deadline),
            )
            attach = getattr(self.service_backend, "attach_spec", None)
            if callable(attach):
                attach(handle, record.spec)
            with self._lock:
                current = self._require_generation(instance_id, record.generation)
                current.service_handle = handle
                if not self.placement.bind_lease(
                    placement.lease.lease_id,
                    now_ms=self.clock.monotonic_ms(),
                ):
                    raise RuntimeError("model PlacementLease could not be bound")
                self._transition(current, ModelInstanceState.WARMING)
            probe = await asyncio.wait_for(
                adapter.probe(handle, record.spec),
                timeout=self._remaining(startup_deadline),
            )
            self._validate_probe(record.spec, placement.lease, probe)
            warmup = await asyncio.wait_for(
                adapter.warmup(handle, record.spec),
                timeout=self._remaining(startup_deadline),
            )
            if not warmup.succeeded or not warmup.response_digest:
                raise RuntimeError("model warmup did not produce a valid response")
            metrics = await asyncio.wait_for(
                adapter.read_metrics(handle),
                timeout=self._remaining(startup_deadline),
            )
            if metrics.actual_request_inflight != 0:
                raise RuntimeError("new model instance reported active requests")
            with self._lock:
                current = self._require_generation(instance_id, record.generation)
                now = self.clock.monotonic_ms()
                current.ready_at_ms = now
                current.last_used_at_ms = now
                self._transition(current, ModelInstanceState.READY)
                return self._snapshot(current)
        except Exception as exc:
            with self._lock:
                current = self._require_generation(instance_id, record.generation)
                current.failure_reason = f"{type(exc).__name__}: {exc}"
                self._transition(
                    current,
                    ModelInstanceState.FAILED,
                    {"reason": current.failure_reason},
                )
            await self._cleanup_failed(instance_id, record.generation)
            return self.snapshot(instance_id)

    def begin_drain(self, instance_id: str, generation: int) -> bool:
        with self._lock:
            record = self._require_generation(instance_id, generation)
            if record.state is ModelInstanceState.DRAINING:
                return False
            if record.state is not ModelInstanceState.READY:
                return False
            self._transition(record, ModelInstanceState.DRAINING)
            return True

    def cancel_drain(self, instance_id: str, generation: int) -> bool:
        with self._lock:
            record = self._require_generation(instance_id, generation)
            if record.state is not ModelInstanceState.DRAINING:
                return False
            self._transition(record, ModelInstanceState.READY)
            return True

    async def stop_if_drained(
        self, instance_id: str, generation: int
    ) -> ModelInstance:
        with self._lock:
            record = self._require_generation(instance_id, generation)
            if record.state is ModelInstanceState.STOPPED:
                return self._snapshot(record)
            if record.state not in {
                ModelInstanceState.DRAINING,
                ModelInstanceState.FAILED,
            }:
                return self._snapshot(record)
            if record.route_occupancy or record.actual_request_inflight:
                return self._snapshot(record)
            self._transition(record, ModelInstanceState.STOPPING)
            handle = record.service_handle
            lease = record.placement_lease
            timeout_ms = record.spec.drain_timeout_ms
        try:
            if handle is not None:
                result = await asyncio.wait_for(
                    self.service_backend.stop(handle, timeout_ms=timeout_ms),
                    timeout=timeout_ms / 1_000,
                )
                if not (
                    result.process_exited
                    and result.port_released
                    and result.hbm_recovered
                ):
                    raise RuntimeError(
                        "service stop did not confirm process, port and HBM recovery"
                    )
            if lease is not None:
                self.placement.release_lease(
                    lease.lease_id,
                    now_ms=self.clock.monotonic_ms(),
                    reason="model_instance_stopped",
                )
            with self._lock:
                current = self._require_generation(instance_id, generation)
                self._transition(current, ModelInstanceState.STOPPED)
                return self._snapshot(current)
        except Exception as exc:
            with self._lock:
                current = self._require_generation(instance_id, generation)
                current.failure_reason = f"{type(exc).__name__}: {exc}"
                self._transition(current, ModelInstanceState.FAILED)
                self._emit(
                    current,
                    "model_resource_release_blocked",
                    {"reason": current.failure_reason},
                )
                return self._snapshot(current)

    def reserve_route(self, instance_id: str, generation: int) -> None:
        with self._lock:
            record = self._require_generation(instance_id, generation)
            if record.state is not ModelInstanceState.READY:
                raise StateTransitionError("model instance is not ready")
            if record.route_occupancy >= record.spec.request_capacity:
                raise StateTransitionError("model route capacity is full")
            record.route_occupancy += 1
            record.last_used_at_ms = self.clock.monotonic_ms()

    def release_route(self, instance_id: str, generation: int) -> None:
        with self._lock:
            record = self._require_generation(instance_id, generation)
            if record.route_occupancy < 1:
                raise StateTransitionError("model route occupancy underflow")
            record.route_occupancy -= 1
            record.last_used_at_ms = self.clock.monotonic_ms()

    def request_started(self, instance_id: str, generation: int) -> None:
        with self._lock:
            record = self._require_generation(instance_id, generation)
            if record.state not in {
                ModelInstanceState.READY,
                ModelInstanceState.DRAINING,
            }:
                raise StateTransitionError("model instance cannot accept requests")
            if record.route_occupancy < 1:
                raise StateTransitionError("model request requires route occupancy")
            record.actual_request_inflight += 1
            record.last_used_at_ms = self.clock.monotonic_ms()

    def request_finished(self, instance_id: str, generation: int) -> None:
        with self._lock:
            record = self._require_generation(instance_id, generation)
            if record.actual_request_inflight < 1:
                raise StateTransitionError("actual request inflight underflow")
            record.actual_request_inflight -= 1
            record.last_used_at_ms = self.clock.monotonic_ms()

    def snapshot(self, instance_id: str) -> ModelInstance:
        with self._lock:
            return self._snapshot(self._require(instance_id))

    def instances(
        self,
        *,
        model_id: str | None = None,
        states: frozenset[ModelInstanceState] | None = None,
    ) -> tuple[ModelInstance, ...]:
        with self._lock:
            return tuple(
                self._snapshot(record)
                for record in sorted(
                    self._records.values(), key=lambda item: item.instance_id
                )
                if (model_id is None or record.spec.model_id == model_id)
                and (states is None or record.state in states)
            )

    def spec_for_instance(self, instance_id: str) -> ModelSpec:
        with self._lock:
            return self._require(instance_id).spec

    async def close(self) -> None:
        with self._lock:
            for record in self._records.values():
                if record.state in {
                    ModelInstanceState.REQUESTED,
                    ModelInstanceState.RESERVING,
                }:
                    record.failure_reason = "inference_coordinator_closed"
                    self._transition(record, ModelInstanceState.FAILED)
        for instance in self.instances():
            if instance.state is ModelInstanceState.READY:
                self.begin_drain(instance.instance_id, instance.generation)
        for instance in self.instances():
            await self.stop_if_drained(instance.instance_id, instance.generation)

    async def _cleanup_failed(self, instance_id: str, generation: int) -> None:
        await self.stop_if_drained(instance_id, generation)

    def _allocate_port(self, record: _InstanceRecord) -> PortLease:
        port = self._next_port
        self._next_port += 1
        if self._next_port > 65_535:
            self._next_port = 25_000
        lease = record.placement_lease
        assert lease is not None
        result = PortLease(
            port_lease_id=new_id("port"),
            node_id=lease.node_id,
            boot_id=lease.boot_id,
            port=port,
            owner_instance_id=record.instance_id,
            generation=record.generation,
        )
        record.port_lease = result
        return result

    @staticmethod
    def _validate_probe(
        spec: ModelSpec, lease: PlacementLease, probe: EngineProbe
    ) -> None:
        expected = (
            True,
            spec.model_id,
            spec.artifact_revision,
            spec.environment_fingerprint,
            spec.dtype,
            spec.quantization,
            lease.npu_device_id,
            spec.request_capacity,
        )
        actual = (
            probe.process_alive,
            probe.model_id,
            probe.artifact_revision,
            probe.environment_fingerprint,
            probe.dtype,
            probe.quantization,
            probe.physical_device_id,
            probe.request_capacity,
        )
        if actual != expected:
            raise RuntimeError("model probe identity or capacity mismatch")
        if not spec.weight_hbm_mb <= probe.process_hbm_mb <= spec.instance_hbm_mb:
            raise RuntimeError("model process HBM is outside its Lease budget")

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.001, deadline - monotonic())

    def _transition(
        self,
        record: _InstanceRecord,
        target: ModelInstanceState,
        payload: dict[str, object] | None = None,
    ) -> None:
        if record.state is target:
            return
        allowed = {
            ModelInstanceState.REQUESTED: {
                ModelInstanceState.RESERVING,
                ModelInstanceState.FAILED,
            },
            ModelInstanceState.RESERVING: {
                ModelInstanceState.STARTING,
                ModelInstanceState.FAILED,
            },
            ModelInstanceState.STARTING: {
                ModelInstanceState.WARMING,
                ModelInstanceState.FAILED,
            },
            ModelInstanceState.WARMING: {
                ModelInstanceState.READY,
                ModelInstanceState.FAILED,
            },
            ModelInstanceState.READY: {
                ModelInstanceState.DRAINING,
                ModelInstanceState.FAILED,
            },
            ModelInstanceState.DRAINING: {
                ModelInstanceState.READY,
                ModelInstanceState.STOPPING,
                ModelInstanceState.FAILED,
            },
            ModelInstanceState.FAILED: {ModelInstanceState.STOPPING},
            ModelInstanceState.STOPPING: {
                ModelInstanceState.STOPPED,
                ModelInstanceState.FAILED,
            },
            ModelInstanceState.STOPPED: set(),
        }
        if target not in allowed[record.state]:
            raise StateTransitionError(
                f"invalid model instance transition {record.state.value}->{target.value}"
            )
        record.state = target
        record.state_changed_at_ms = self.clock.monotonic_ms()
        self._emit(record, f"model_instance_{target.value}", payload)

    def _emit(
        self,
        record: _InstanceRecord,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink(
            ModelControlEvent(
                event_type=event_type,
                occurred_at_ms=self.clock.monotonic_ms(),
                model_id=record.spec.model_id,
                instance_id=record.instance_id,
                instance_generation=record.generation,
                payload=self._event_payload(payload),
            )
        )

    @staticmethod
    def _event_payload(payload: dict[str, object] | None) -> FrozenMap:
        frozen = freeze_canonical(payload or {})
        assert isinstance(frozen, FrozenMap)
        return frozen

    def _require(self, instance_id: str) -> _InstanceRecord:
        try:
            return self._records[instance_id]
        except KeyError as exc:
            raise KeyError(f"unknown model instance: {instance_id}") from exc

    def _require_generation(
        self, instance_id: str, generation: int
    ) -> _InstanceRecord:
        record = self._require(instance_id)
        if record.generation != generation:
            raise StateTransitionError("model instance generation is stale")
        return record

    @staticmethod
    def _snapshot(record: _InstanceRecord) -> ModelInstance:
        lease = record.placement_lease
        handle = record.service_handle
        return ModelInstance(
            instance_id=record.instance_id,
            model_id=record.spec.model_id,
            catalog_revision=record.spec.catalog_revision,
            state=record.state,
            placement_lease_id=None if lease is None else lease.lease_id,
            service_handle_id=None if handle is None else handle.service_handle_id,
            node_id=None if lease is None else lease.node_id,
            boot_id=None if lease is None else lease.boot_id,
            npu_device_id=None if lease is None else lease.npu_device_id,
            endpoint_id=None if handle is None else handle.endpoint_id,
            generation=record.generation,
            created_at_ms=record.created_at_ms,
            ready_at_ms=record.ready_at_ms,
            state_changed_at_ms=record.state_changed_at_ms,
            route_occupancy=record.route_occupancy,
            actual_request_inflight=record.actual_request_inflight,
            last_used_at_ms=record.last_used_at_ms,
            failure_reason=record.failure_reason,
        )
