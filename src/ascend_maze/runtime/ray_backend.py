"""Ray RuntimeBackend with C6 hard placement and NodeAgent event delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ascend_maze.contracts.data import DataHandle, DataOwner
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.contracts.resources import ExecutionTarget, PlacementLease
from ascend_maze.contracts.runtime import (
    CodeHandle,
    CodePackage,
    DeviceBinding,
    DispatchHandle,
    ExecutionRequest,
    RuntimeNodeBinding,
)
from ascend_maze.contracts.worker import WorkerLease
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.core.identifiers import stable_id
from ascend_maze.core.time import monotonic_time_ms
from ascend_maze.data.ray_store import RayDataStore
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.runtime.code_loader import load_code_package
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry
from ascend_maze.runtime.ray_cluster import validate_ray_version
from ascend_maze.runtime.ray_worker import RAY_ONE_SHOT_WORKER, RayWorkerOutcome
from ascend_maze.runtime.worker_broker import ColdWorkerBroker

from ascend_maze.control.node_rpc import NodeAgentIdentity, report_worker_event


@dataclass(slots=True)
class _CodeRecord:
    handle: CodeHandle
    package_handle: DataHandle
    reference_count: int


@dataclass(slots=True)
class _DispatchRecord:
    request: ExecutionRequest
    lease: PlacementLease
    binding: RuntimeNodeBinding
    worker_lease: WorkerLease
    handle: DispatchHandle
    object_ref: Any
    monitor: asyncio.Task[None] | None
    cancel_requested: bool = False
    invalidated: bool = False
    terminal: bool = False
    outcome: RayWorkerOutcome | None = None
    node_terminal_event: RuntimeEvent | None = None
    node_terminal_received: asyncio.Event = field(default_factory=asyncio.Event)


class RayRuntimeBackend:
    backend_name = "ray"

    def __init__(
        self,
        *,
        data_store: RayDataStore,
        node_registry: RayNodeRegistry,
        worker_broker: ColdWorkerBroker,
        cluster_id: str,
        owner_generation: str,
        environment_fingerprint: str,
        event_timeout_seconds: float = 2.0,
        event_sink: Callable[[RuntimeEvent], None] | None = None,
        recording_error_sink: Callable[[str, str], None] | None = None,
    ) -> None:
        if event_timeout_seconds <= 0:
            raise ValueError("event_timeout_seconds must be positive")
        self.data_store = data_store
        self.node_registry = node_registry
        self.worker_broker = worker_broker
        self.cluster_id = cluster_id
        self.owner_generation = owner_generation
        self.environment_fingerprint = environment_fingerprint
        self.event_timeout_seconds = event_timeout_seconds
        self._event_sink = event_sink
        self._recording_error_sink = recording_error_sink
        self._code: dict[str, _CodeRecord] = {}
        self._dispatches: dict[str, _DispatchRecord] = {}
        self._attempt_dispatches: dict[tuple[str, str, int], str] = {}
        self._emitted_events: dict[str, str] = {}
        self._retired_runs: set[str] = set()
        self._started = False
        self._closed = False

    def set_event_sink(self, sink: Callable[[RuntimeEvent], None]) -> None:
        self._event_sink = sink

    def post_node_event(self, event: RuntimeEvent) -> None:
        record = self._dispatches.get(event.dispatch_id)
        if (
            event.kind is not RuntimeEventKind.WORKER_STARTED
            and record is not None
            and not record.terminal
        ):
            record.node_terminal_event = event
            record.node_terminal_received.set()
            return
        self._emit(event)

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("runtime backend is closed")
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before RayRuntimeBackend")
        validate_ray_version()
        self._started = True

    async def prepare(
        self, definitions: tuple[CodePackage, ...]
    ) -> tuple[CodeHandle, ...]:
        self._require_running()
        definition_ids = [package.definition_id for package in definitions]
        if len(definition_ids) != len(set(definition_ids)):
            raise ContractValidationError("CodePackage definitions must be unique")
        prepared: list[tuple[CodePackage, CodeHandle, DataHandle | None]] = []
        staged: list[DataHandle] = []
        try:
            for package in definitions:
                if package.environment_fingerprint != self.environment_fingerprint:
                    raise ContractValidationError("code package environment mismatch")
                existing = self._code.get(package.definition_id)
                if existing is not None:
                    if (
                        existing.handle.code_hash != package.code_hash
                        or existing.handle
                        != self._code_handle_for_package(package)
                    ):
                        raise ContractValidationError("definition code hash conflict")
                    prepared.append((package, existing.handle, None))
                    continue
                await asyncio.to_thread(load_code_package, package)
                code_handle = self._code_handle_for_package(package)
                package_handle = await asyncio.to_thread(
                    self.data_store.put_staged, package, self.owner_generation
                )
                staged.append(package_handle)
                prepared.append((package, code_handle, package_handle))
            if staged:
                await asyncio.to_thread(
                    self.data_store.adopt,
                    tuple(staged),
                    DataOwner(
                        owner_kind="code_registry",
                        owner_id=self.owner_generation,
                        owner_generation=self.owner_generation,
                    ),
                )
        except Exception:
            if staged:
                await asyncio.to_thread(self.data_store.release_many, tuple(staged))
            raise

        handles: list[CodeHandle] = []
        for package, code_handle, prepared_package_handle in prepared:
            existing = self._code.get(package.definition_id)
            if existing is None:
                assert prepared_package_handle is not None
                self._code[package.definition_id] = _CodeRecord(
                    code_handle, prepared_package_handle, 1
                )
            else:
                existing.reference_count += 1
            handles.append(code_handle)
        return tuple(handles)

    async def dispatch(
        self,
        request: ExecutionRequest,
        lease: PlacementLease,
    ) -> DispatchHandle:
        self._require_running()
        existing = self._dispatches.get(request.dispatch_id)
        if existing is not None:
            if existing.request != request or existing.lease != lease:
                raise ContractValidationError("dispatch_id payload conflict")
            return existing.handle
        attempt_key = (request.run_id, request.task_id, request.attempt)
        conflicting = self._attempt_dispatches.get(attempt_key)
        if conflicting is not None and conflicting != request.dispatch_id:
            raise ContractValidationError("attempt already has another dispatch_id")
        if (
            lease.run_id != request.run_id
            or lease.task_id != request.task_id
            or lease.attempt != request.attempt
        ):
            raise ContractValidationError("PlacementLease does not match request")
        if request.environment_fingerprint != self.environment_fingerprint:
            raise ContractValidationError("execution environment mismatch")
        if request.execution_target is not ExecutionTarget.LOCAL_WORKER:
            raise ContractValidationError("Ray Host backend only supports local Worker tasks")
        code = self._code.get(request.code_handle.definition_id)
        if code is None or code.handle != request.code_handle:
            raise ContractValidationError("CodeHandle is not prepared")
        binding = self.node_registry.resolve_lease(lease)
        worker_lease = self.worker_broker.acquire(
            placement_lease=lease,
            task_kind=request.task_kind,
            execution_target=request.execution_target,
            now_ms=monotonic_time_ms(),
        )
        device_binding: DeviceBinding | None = None
        if request.task_kind == "npu":
            device_binding = DeviceBinding.from_lease(lease, binding)
            if worker_lease.bound_device_id != device_binding.physical_device_id:
                self.worker_broker.release(
                    worker_lease.worker_lease_id, disposition="discard"
                )
                raise ContractValidationError(
                    "WorkerLease device does not match PlacementLease"
                )
        elif lease.npu_device_id is not None or lease.resources.npu_slots != 0:
            self.worker_broker.release(
                worker_lease.worker_lease_id, disposition="discard"
            )
            raise ContractValidationError(
                "CPU/I/O Worker cannot receive an NPU PlacementLease"
            )
        handle = DispatchHandle(
            dispatch_id=request.dispatch_id,
            backend_name=self.backend_name,
            run_id=request.run_id,
            task_id=request.task_id,
            attempt=request.attempt,
            lease_id=lease.lease_id,
            route_lease_id=None,
            worker_endpoint_id=worker_lease.worker_endpoint_id,
        )
        identity = NodeAgentIdentity(
            cluster_id=self.cluster_id,
            node_id=binding.node_id,
            boot_id=binding.boot_id,
            ray_node_id=binding.ray_node_id,
            agent_generation=binding.agent_generation,
            environment_fingerprint=self.environment_fingerprint,
            producer_id=binding.producer_id,
        )
        try:
            object_ref = RAY_ONE_SHOT_WORKER.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    binding.ray_node_id, soft=False
                ),
                name=f"maze:{request.dispatch_id}",
            ).remote(
                request=request,
                placement_lease=lease,
                worker_lease=worker_lease,
                binding=binding,
                agent_identity=identity,
                data_store_descriptor=self.data_store.descriptor,
                code_package_handle=code.package_handle,
                event_timeout_seconds=self.event_timeout_seconds,
                device_binding=device_binding,
            )
        except Exception:
            self.worker_broker.release(
                worker_lease.worker_lease_id, disposition="discard"
            )
            raise
        record = _DispatchRecord(
            request=request,
            lease=lease,
            binding=binding,
            worker_lease=worker_lease,
            handle=handle,
            object_ref=object_ref,
            monitor=None,
        )
        self._dispatches[request.dispatch_id] = record
        self._attempt_dispatches[attempt_key] = request.dispatch_id
        record.monitor = asyncio.create_task(self._monitor(record))
        return handle

    async def cancel(self, handle: DispatchHandle, reason: str) -> None:
        del reason
        record = self._dispatches.get(handle.dispatch_id)
        if record is None:
            return
        if record.handle != handle:
            raise ContractValidationError("DispatchHandle payload conflict")
        if record.cancel_requested:
            return
        record.cancel_requested = True
        if not record.terminal:
            ray.cancel(record.object_ref, force=True, recursive=True)
            if record.monitor is not None:
                await asyncio.gather(record.monitor, return_exceptions=True)
        elif record.outcome is not None:
            await self._release_staged_outputs(record.outcome.terminal_event)

    async def release_code(self, handles: tuple[CodeHandle, ...]) -> None:
        for handle in handles:
            record = self._code.get(handle.definition_id)
            if record is None:
                continue
            if record.handle != handle:
                raise ContractValidationError("CodeHandle payload conflict")
            if record.reference_count > 0:
                record.reference_count -= 1
            if record.reference_count == 0:
                await asyncio.to_thread(
                    self.data_store.release, record.package_handle
                )
                del self._code[handle.definition_id]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for record in self._dispatches.values():
            if not record.terminal:
                record.cancel_requested = True
                ray.cancel(record.object_ref, force=True, recursive=True)
        monitors = [
            record.monitor
            for record in self._dispatches.values()
            if record.monitor is not None and not record.monitor.done()
        ]
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)
        for code_record in tuple(self._code.values()):
            await asyncio.to_thread(
                self.data_store.release, code_record.package_handle
            )
        self._code.clear()
        self._started = False

    async def wait_idle(self) -> None:
        monitors = [
            record.monitor
            for record in self._dispatches.values()
            if record.monitor is not None and not record.monitor.done()
        ]
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)

    def producer_for_lease(self, lease: PlacementLease) -> str | None:
        return self.node_registry.producer_for_lease(lease)

    def dispatch_invalidated(self, dispatch_id: str) -> bool:
        record = self._dispatches.get(dispatch_id)
        return (
            record is None
            or record.cancel_requested
            or record.invalidated
            or record.terminal
        )

    def worker_released(self, dispatch_id: str) -> bool:
        record = self._dispatches.get(dispatch_id)
        return record is None or record.terminal

    async def release_run(self, run_id: str) -> int:
        self._retired_runs.add(run_id)
        dispatch_ids = [
            dispatch_id
            for dispatch_id, record in self._dispatches.items()
            if record.request.run_id == run_id and record.terminal
        ]
        for dispatch_id in dispatch_ids:
            record = self._dispatches[dispatch_id]
            if record.outcome is not None:
                await self._release_staged_outputs(record.outcome.terminal_event)
            self._drop_dispatch(dispatch_id)
        if not any(
            record.request.run_id == run_id for record in self._dispatches.values()
        ):
            self._retired_runs.discard(run_id)
        for event_id in [
            event_id
            for event_id, event_run_id in self._emitted_events.items()
            if event_run_id == run_id
        ]:
            del self._emitted_events[event_id]
        return len(dispatch_ids)

    def invalidate_binding(self, binding: RuntimeNodeBinding) -> None:
        self.data_store.release_staged_for_runtime_node(
            node_id=binding.node_id,
            boot_id=binding.boot_id,
            runtime_generation=binding.runtime_generation,
        )
        self.worker_broker.invalidate_node(binding.node_id, binding.boot_id)
        for record in self._dispatches.values():
            if (
                not record.terminal
                and record.binding.node_id == binding.node_id
                and record.binding.boot_id == binding.boot_id
                and record.binding.runtime_generation == binding.runtime_generation
            ):
                record.invalidated = True
                self._record_delivery_error(
                    record.request.run_id,
                    "NodeAgent binding disconnected during an active Attempt",
                )
                ray.cancel(record.object_ref, force=True, recursive=True)

    def active_dispatch_count(self, run_id: str | None = None) -> int:
        return sum(
            not record.terminal and (run_id is None or record.request.run_id == run_id)
            for record in self._dispatches.values()
        )

    def code_reference_count(self) -> int:
        return sum(record.reference_count for record in self._code.values())

    def worker_outcome(self, dispatch_id: str) -> RayWorkerOutcome | None:
        record = self._dispatches.get(dispatch_id)
        return None if record is None else record.outcome

    async def _monitor(self, record: _DispatchRecord) -> None:
        terminal_to_publish: RuntimeEvent | None = None
        try:
            outcome = await asyncio.to_thread(ray.get, record.object_ref)
            if not isinstance(outcome, RayWorkerOutcome):
                raise TypeError("Ray Worker returned an invalid control outcome")
            if (
                outcome.dispatch_id != record.request.dispatch_id
                or outcome.ray_node_id != record.binding.ray_node_id
            ):
                raise RuntimeError("Ray Worker outcome identity mismatch")
            record.outcome = outcome
            if record.cancel_requested or record.invalidated:
                await self._release_staged_outputs(outcome.terminal_event)
                terminal_to_publish = self._cancelled_or_lost_event(
                    record, lost=record.invalidated
                )
                await self._deliver_synthesized_event(
                    record,
                    terminal_to_publish,
                )
            else:
                terminal_to_publish = outcome.terminal_event
                if not outcome.terminal_event_delivered:
                    self._record_delivery_error(
                        record.request.run_id,
                        "NodeAgent did not accept the terminal Worker event",
                    )
                    await self._deliver_synthesized_event(
                        record, outcome.terminal_event
                    )
        except Exception as exc:
            terminal_to_publish = self._cancelled_or_lost_event(
                record,
                lost=not record.cancel_requested,
                message=f"{type(exc).__name__}: {exc}",
            )
            await self._deliver_synthesized_event(
                record,
                terminal_to_publish,
            )
        finally:
            if not record.invalidated and not record.node_terminal_received.is_set():
                try:
                    await asyncio.wait_for(
                        record.node_terminal_received.wait(),
                        timeout=self.event_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    self._record_delivery_error(
                        record.request.run_id,
                        "Controller did not receive the NodeAgent terminal event",
                    )
            record.terminal = True
            self.worker_broker.release(
                record.worker_lease.worker_lease_id, disposition="discard"
            )
            publish = terminal_to_publish or record.node_terminal_event
            if publish is not None:
                self._emit(publish)
            if record.request.run_id in self._retired_runs:
                self._drop_dispatch(record.request.dispatch_id)

    def _cancelled_or_lost_event(
        self,
        record: _DispatchRecord,
        *,
        lost: bool,
        message: str | None = None,
    ) -> RuntimeEvent:
        kind = RuntimeEventKind.TASK_FAILED if lost else RuntimeEventKind.TASK_CANCELLED
        error = None
        if lost:
            error = ErrorInfo(
                schema_version=1,
                error_code="worker_lost",
                category="worker",
                origin="runtime",
                message=message or "Ray Worker or NodeAgent generation was lost",
                retryable_hint=True,
                classification_confidence="exact",
                execution_phase="running",
                run_id=record.request.run_id,
                task_id=record.request.task_id,
                attempt=record.request.attempt,
                dispatch_id=record.request.dispatch_id,
                lease_id=record.lease.lease_id,
                node_id=record.binding.node_id,
                boot_id=record.binding.boot_id,
                worker_id=record.worker_lease.worker_id,
                occurred_at_ms=monotonic_time_ms(),
            )
        return RuntimeEvent.create(
            kind=kind,
            dispatch_id=record.request.dispatch_id,
            run_id=record.request.run_id,
            task_id=record.request.task_id,
            attempt=record.request.attempt,
            lease_id=record.lease.lease_id,
            route_lease_id=None,
            occurred_at_ms=monotonic_time_ms(),
            error=error,
        )

    async def _release_staged_outputs(self, event: RuntimeEvent) -> None:
        def release() -> None:
            for _, handle in event.output_handles:
                try:
                    if self.data_store.state_of(handle) == "staged":
                        self.data_store.release(handle)
                except Exception:
                    pass

        await asyncio.to_thread(release)

    async def _deliver_synthesized_event(
        self,
        record: _DispatchRecord,
        event: RuntimeEvent,
    ) -> None:
        if record.invalidated:
            return
        try:
            await asyncio.to_thread(
                report_worker_event,
                endpoint=record.binding.agent_endpoint,
                identity=self._agent_identity(record.binding),
                event=event,
                timeout_seconds=self.event_timeout_seconds,
            )
        except Exception as exc:
            self._record_delivery_error(
                record.request.run_id,
                f"NodeAgent rejected synthesized RuntimeEvent: {type(exc).__name__}: {exc}",
            )

    def _emit(self, event: RuntimeEvent) -> None:
        if self._event_sink is None:
            raise RuntimeError("runtime event sink is not configured")
        if event.event_id in self._emitted_events:
            return
        self._emitted_events[event.event_id] = event.run_id
        self._event_sink(event)

    def _record_delivery_error(self, run_id: str, message: str) -> None:
        if self._recording_error_sink is not None:
            self._recording_error_sink(run_id, message)

    def _drop_dispatch(self, dispatch_id: str) -> None:
        record = self._dispatches.pop(dispatch_id, None)
        if record is None:
            return
        self._attempt_dispatches.pop(
            (record.request.run_id, record.request.task_id, record.request.attempt),
            None,
        )
        run_id = record.request.run_id
        if not any(
            item.request.run_id == run_id for item in self._dispatches.values()
        ):
            self._retired_runs.discard(run_id)

    def _require_running(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("runtime backend is not running")

    def _agent_identity(self, binding: RuntimeNodeBinding) -> NodeAgentIdentity:
        return NodeAgentIdentity(
            cluster_id=self.cluster_id,
            node_id=binding.node_id,
            boot_id=binding.boot_id,
            ray_node_id=binding.ray_node_id,
            agent_generation=binding.agent_generation,
            environment_fingerprint=self.environment_fingerprint,
            producer_id=binding.producer_id,
        )

    @staticmethod
    def _code_handle_for_package(package: CodePackage) -> CodeHandle:
        return CodeHandle(
            code_handle_id=stable_id(
                "code",
                package.definition_id,
                package.code_hash,
                package.environment_fingerprint,
            ),
            definition_id=package.definition_id,
            code_hash=package.code_hash,
        )
