"""C11 composition consumed by SchedulerCore and service client Workers."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from ascend_maze.compiler.ir import CompiledWorkflow
from ascend_maze.contracts.runtime import ModelRouteLease
from ascend_maze.core.clock import Clock, SystemClock
from ascend_maze.inference.catalog import ModelCatalog
from ascend_maze.inference.context import AttemptInferenceSession
from ascend_maze.inference.contracts import (
    AttemptInferenceSummary,
    InferenceRequestRecord,
    ModelControlEvent,
    ModelInstance,
    ModelRouteAcquireResult,
    ModelRouteLeaseSnapshot,
    ServiceProcessBackend,
)
from ascend_maze.inference.instance_manager import ModelInstanceManager
from ascend_maze.inference.replica_controller import ReplicaController
from ascend_maze.inference.router import InferenceRouter
from ascend_maze.placement import PlacementManager


class InferenceCoordinator:
    def __init__(
        self,
        *,
        catalog: ModelCatalog,
        placement: PlacementManager,
        service_backend: ServiceProcessBackend,
        clock: Clock | None = None,
        affinity_ttl_ms: int = 300_000,
        affinity_capacity: int = 10_000,
    ) -> None:
        self.catalog = catalog
        self.clock = clock or SystemClock()
        self._events: list[ModelControlEvent] = []
        self._request_records: dict[str, list[InferenceRequestRecord]] = {}
        self._sessions: dict[str, AttemptInferenceSession] = {}
        self._capacity_sink: Callable[[str], object] | None = None
        self._lock = RLock()
        self.instances = ModelInstanceManager(
            catalog=catalog,
            placement=placement,
            service_backend=service_backend,
            event_sink=self._record_event,
            clock=self.clock,
        )
        self.router = InferenceRouter(
            instances=self.instances,
            event_sink=self._record_event,
            clock=self.clock,
            affinity_ttl_ms=affinity_ttl_ms,
            affinity_capacity=affinity_capacity,
        )
        self.replicas = ReplicaController(
            catalog=catalog,
            instances=self.instances,
            router=self.router,
            event_sink=self._record_event,
            clock=self.clock,
        )

    async def start(self) -> None:
        await self.replicas.reconcile()

    def set_capacity_sink(self, sink: Callable[[str], object] | None) -> None:
        self._capacity_sink = sink

    def validate_workflow(self, compiled: CompiledWorkflow) -> None:
        self.catalog.validate_workflow(compiled)

    def register_demand(
        self, *, run_id: str, task_id: str, model_id: str
    ) -> None:
        self.replicas.register_demand(
            run_id=run_id,
            task_id=task_id,
            model_id=model_id,
        )

    async def acquire_route(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt: int,
        model_id: str,
        session_key_hash: str | None,
        dispatch_deadline_ms: int,
    ) -> ModelRouteAcquireResult:
        await self.replicas.reconcile()
        return self.router.acquire(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            model_id=model_id,
            session_key_hash=session_key_hash,
            dispatch_deadline_ms=dispatch_deadline_ms,
        )

    def activate_route(self, route_lease_id: str) -> bool:
        return self.router.activate(route_lease_id)

    async def release_route(
        self,
        lease: ModelRouteLease,
        *,
        reason: str,
    ) -> bool:
        released = self.router.release(
            lease.route_lease_id,
            run_id=lease.run_id,
            task_id=lease.task_id,
            attempt=lease.attempt,
            instance_generation=lease.instance_generation,
            reason=reason,
        )
        if released:
            self.replicas.remove_demand(
                run_id=lease.run_id,
                task_id=lease.task_id,
                model_id=lease.model_id,
            )
            await self.replicas.reconcile()
            self._notify_capacity("model_route_released")
        return released

    def abandon_route(self, lease: ModelRouteLease, *, reason: str) -> bool:
        return self.router.abandon_reserved(
            lease.route_lease_id,
            reason=reason,
        )

    def create_attempt_session(
        self, lease: ModelRouteLease
    ) -> AttemptInferenceSession:
        with self._lock:
            existing = self._sessions.get(lease.route_lease_id)
            if existing is not None:
                return existing
            adapter = self.catalog.adapter(lease.model_id)
            session = AttemptInferenceSession(
                lease=lease,
                router=self.router,
                adapter=adapter,
                record_sink=self._record_request,
                clock=self.clock,
            )
            self._sessions[lease.route_lease_id] = session
            return session

    def attempt_summary(
        self, route_lease_id: str
    ) -> AttemptInferenceSummary | None:
        with self._lock:
            session = self._sessions.get(route_lease_id)
            return None if session is None else session.summary()

    def request_records(
        self, route_lease_id: str | None = None
    ) -> tuple[InferenceRequestRecord, ...]:
        with self._lock:
            if route_lease_id is not None:
                return tuple(self._request_records.get(route_lease_id, ()))
            return tuple(
                record
                for route_id in sorted(self._request_records)
                for record in self._request_records[route_id]
            )

    def route_snapshot(self, route_lease_id: str) -> ModelRouteLeaseSnapshot:
        return self.router.snapshot(route_lease_id)

    def model_instances(self, model_id: str | None = None) -> tuple[ModelInstance, ...]:
        return self.instances.instances(model_id=model_id)

    def events(self) -> tuple[ModelControlEvent, ...]:
        with self._lock:
            return tuple(self._events)

    async def reconcile(self) -> None:
        await self.replicas.reconcile()

    async def close(self) -> None:
        await self.replicas.wait_for_background()
        if self.router.active_count() != 0:
            raise RuntimeError("cannot close C11 while RouteLeases are active")
        await self.instances.close()

    def destroy_run(self, run_id: str) -> int:
        self.replicas.remove_run(run_id)
        with self._lock:
            route_ids = {
                route_id
                for route_id, session in self._sessions.items()
                if session.lease.run_id == run_id
            }
            route_ids.update(
                route_id
                for route_id, records in self._request_records.items()
                if records and records[0].run_id == run_id
            )
            for route_id in route_ids:
                self._sessions.pop(route_id, None)
                self._request_records.pop(route_id, None)
            self._events = [
                event for event in self._events if event.run_id != run_id
            ]
        return self.router.destroy_run(run_id)

    def _record_event(self, event: ModelControlEvent) -> None:
        with self._lock:
            self._events.append(event)
        if event.event_type in {
            "model_instance_ready",
            "model_instance_stopped",
            "model_resource_release_blocked",
        }:
            self._notify_capacity(event.event_type)

    def _record_request(self, record: InferenceRequestRecord) -> None:
        with self._lock:
            self._request_records.setdefault(record.route_lease_id, []).append(record)

    def _notify_capacity(self, reason: str) -> None:
        sink = self._capacity_sink
        if sink is not None:
            sink(reason)
