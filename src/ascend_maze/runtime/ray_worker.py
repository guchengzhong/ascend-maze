"""Hard-placed cold one-shot Ray Worker for Host tasks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import ray

from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.contracts.resources import (
    PlacementLease,
    ResourceObservation,
    ResourceSpec,
)
from ascend_maze.contracts.runtime import (
    CodePackage,
    DeviceBinding,
    ExecutionRequest,
    RuntimeNodeBinding,
)
from ascend_maze.contracts.worker import WorkerLease
from ascend_maze.core.canonical import FrozenMap
from ascend_maze.core.time import monotonic_time_ms
from ascend_maze.data.ray_store import RayDataStore, RayDataStoreDescriptor
from ascend_maze.runtime.code_loader import load_code_package
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.ascend.torch_runtime import (
    BoundTorchNpuRuntime,
    bind_torch_npu_device,
    contains_npu_tensor,
    host_peak_rss_mb,
    platform_error_code,
)

from ascend_maze.control.node_rpc import (
    NodeAgentIdentity,
    report_worker_event,
)


@dataclass(frozen=True, slots=True)
class RayWorkerOutcome:
    dispatch_id: str
    ray_node_id: str
    worker_pid: int
    worker_started_delivered: bool
    terminal_event: RuntimeEvent
    terminal_event_delivered: bool
    physical_device_id: str | None = None
    binding_verified: bool = False


def _execute_one_shot(
    *,
    request: ExecutionRequest,
    placement_lease: PlacementLease,
    worker_lease: WorkerLease,
    binding: RuntimeNodeBinding,
    agent_identity: NodeAgentIdentity,
    data_store_descriptor: RayDataStoreDescriptor,
    code_package_handle: DataHandle,
    event_timeout_seconds: float,
    device_binding: DeviceBinding | None,
) -> RayWorkerOutcome:
    worker_pid = os.getpid()
    ray_node_id = ray.get_runtime_context().get_node_id()
    store = RayDataStore.connect(data_store_descriptor)
    started_delivered = False
    if (
        ray_node_id != binding.ray_node_id
        or placement_lease.node_id != binding.node_id
        or placement_lease.boot_id != binding.boot_id
        or worker_lease.node_id != binding.node_id
        or worker_lease.boot_id != binding.boot_id
    ):
        terminal = _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.DISPATCH_FAILED,
            error_code="runtime_node_unavailable",
            category="runtime",
            phase="dispatched",
            message="Ray Worker did not start on the leased node generation",
        )
        delivered = _try_report(
            binding.agent_endpoint,
            agent_identity,
            terminal,
            event_timeout_seconds,
        )
        return RayWorkerOutcome(
            request.dispatch_id,
            ray_node_id,
            worker_pid,
            False,
            terminal,
            delivered,
            None,
            False,
        )

    npu_runtime: BoundTorchNpuRuntime | None = None
    if request.task_kind == "npu":
        if device_binding is None:
            terminal = _failure_event(
                request=request,
                lease=placement_lease,
                worker_lease=worker_lease,
                binding=binding,
                kind=RuntimeEventKind.DISPATCH_FAILED,
                error_code="device_bind_failed",
                category="node_device",
                phase="dispatched",
                message="NPU Worker did not receive a DeviceBinding",
                device_binding=None,
                npu_runtime=None,
            )
            delivered = _try_report(
                binding.agent_endpoint,
                agent_identity,
                terminal,
                event_timeout_seconds,
            )
            return RayWorkerOutcome(
                request.dispatch_id,
                ray_node_id,
                worker_pid,
                False,
                terminal,
                delivered,
            )
        try:
            npu_runtime = bind_torch_npu_device(device_binding)
        except Exception as exc:
            terminal = _failure_event(
                request=request,
                lease=placement_lease,
                worker_lease=worker_lease,
                binding=binding,
                kind=RuntimeEventKind.DISPATCH_FAILED,
                error_code="device_bind_failed",
                category="node_device",
                phase="dispatched",
                message=f"{type(exc).__name__}: {exc}",
                exception=exc,
                device_binding=device_binding,
                npu_runtime=None,
            )
            delivered = _try_report(
                binding.agent_endpoint,
                agent_identity,
                terminal,
                event_timeout_seconds,
            )
            return RayWorkerOutcome(
                request.dispatch_id,
                ray_node_id,
                worker_pid,
                False,
                terminal,
                delivered,
                device_binding.physical_device_id,
                False,
            )
    elif device_binding is not None:
        terminal = _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.DISPATCH_FAILED,
            error_code="device_bind_failed",
            category="node_device",
            phase="dispatched",
            message="CPU/I/O Worker unexpectedly received a DeviceBinding",
            device_binding=device_binding,
            npu_runtime=None,
        )
        delivered = _try_report(
            binding.agent_endpoint,
            agent_identity,
            terminal,
            event_timeout_seconds,
        )
        return RayWorkerOutcome(
            request.dispatch_id,
            ray_node_id,
            worker_pid,
            False,
            terminal,
            delivered,
            device_binding.physical_device_id,
            False,
        )

    started = RuntimeEvent.create(
        kind=RuntimeEventKind.WORKER_STARTED,
        dispatch_id=request.dispatch_id,
        run_id=request.run_id,
        task_id=request.task_id,
        attempt=request.attempt,
        lease_id=placement_lease.lease_id,
        route_lease_id=None,
        occurred_at_ms=monotonic_time_ms(),
        worker_pid=worker_pid,
        device_id=(
            None if device_binding is None else device_binding.physical_device_id
        ),
        binding_verified=npu_runtime is not None,
    )
    started_delivered = _try_report(
        binding.agent_endpoint,
        agent_identity,
        started,
        event_timeout_seconds,
    )
    if not started_delivered:
        terminal = _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.DISPATCH_FAILED,
            error_code="worker_start_failed",
            category="worker",
            phase="dispatched",
            message="WorkerStarted could not be delivered to NodeAgent",
            device_binding=device_binding,
            npu_runtime=npu_runtime,
        )
        return RayWorkerOutcome(
            request.dispatch_id,
            ray_node_id,
            worker_pid,
            False,
            terminal,
            False,
            None if device_binding is None else device_binding.physical_device_id,
            npu_runtime is not None,
        )

    terminal = _run_user_code(
        request=request,
        placement_lease=placement_lease,
        worker_lease=worker_lease,
        binding=binding,
        store=store,
        code_package_handle=code_package_handle,
        device_binding=device_binding,
        npu_runtime=npu_runtime,
    )
    delivered = _try_report(
        binding.agent_endpoint,
        agent_identity,
        terminal,
        event_timeout_seconds,
    )
    return RayWorkerOutcome(
        request.dispatch_id,
        ray_node_id,
        worker_pid,
        started_delivered,
        terminal,
        delivered,
        None if device_binding is None else device_binding.physical_device_id,
        npu_runtime is not None,
    )


def _run_user_code(
    *,
    request: ExecutionRequest,
    placement_lease: PlacementLease,
    worker_lease: WorkerLease,
    binding: RuntimeNodeBinding,
    store: RayDataStore,
    code_package_handle: DataHandle,
    device_binding: DeviceBinding | None,
    npu_runtime: BoundTorchNpuRuntime | None,
) -> RuntimeEvent:
    try:
        package = store.get(code_package_handle)
        if not isinstance(package, CodePackage):
            raise TypeError("code registry value is not CodePackage")
        if (
            package.definition_id != request.code_handle.definition_id
            or package.code_hash != request.code_handle.code_hash
            or package.environment_fingerprint != request.environment_fingerprint
        ):
            raise ValueError("CodePackage identity does not match ExecutionRequest")
        func = load_code_package(package)
        kwargs: dict[str, object] = {}
        for argument in request.arguments:
            if argument.kind == "literal":
                kwargs[argument.name] = argument.literal
            elif argument.kind == "data_handle":
                assert argument.data_handle is not None
                kwargs[argument.name] = store.get(argument.data_handle)
    except Exception as exc:
        return _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.TASK_FAILED,
            error_code="data_binding_failed",
            category="data",
            phase="binding",
            message=f"{type(exc).__name__}: {exc}",
            exception=exc,
            device_binding=device_binding,
            npu_runtime=npu_runtime,
        )
    try:
        result = func(**kwargs)
    except Exception as exc:
        oom_confidence = (
            None
            if npu_runtime is None
            else npu_runtime.oom_classification_confidence(exc)
        )
        if oom_confidence is not None:
            error_code = "npu_oom"
            category = "resource"
            confidence = oom_confidence
        elif npu_runtime is not None and platform_error_code(exc) is not None:
            error_code = "npu_async_error"
            category = "node_device"
            confidence = "mapped"
        else:
            error_code = "user_code_failed"
            category = "user"
            confidence = "exact"
        return _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.TASK_FAILED,
            error_code=error_code,
            category=category,
            phase="user_code",
            message=f"{type(exc).__name__}: {exc}",
            exception=exc,
            classification_confidence=confidence,
            device_binding=device_binding,
            npu_runtime=npu_runtime,
        )
    if npu_runtime is not None:
        try:
            npu_runtime.synchronize()
        except Exception as exc:
            return _failure_event(
                request=request,
                lease=placement_lease,
                worker_lease=worker_lease,
                binding=binding,
                kind=RuntimeEventKind.TASK_FAILED,
                error_code="npu_async_error",
                category="node_device",
                phase="npu_synchronize",
                message=f"{type(exc).__name__}: {exc}",
                exception=exc,
                classification_confidence="mapped",
                device_binding=device_binding,
                npu_runtime=npu_runtime,
            )
    if not isinstance(result, dict) or tuple(sorted(result)) != tuple(
        sorted(request.expected_outputs)
    ):
        return _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.TASK_FAILED,
            error_code="invalid_task_output",
            category="data",
            phase="publishing",
            message="Task returned keys that do not match its output contract",
            device_binding=device_binding,
            npu_runtime=npu_runtime,
        )
    if npu_runtime is not None and contains_npu_tensor(result):
        return _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.TASK_FAILED,
            error_code="invalid_task_output",
            category="data",
            phase="publishing",
            message="NPU Tensor outputs must be moved to host memory before return",
            device_binding=device_binding,
            npu_runtime=npu_runtime,
        )
    output_handles: list[tuple[str, DataHandle]] = []
    try:
        for output_name in request.expected_outputs:
            output_handles.append(
                (
                    output_name,
                    store.put_staged_for_runtime_node(
                        result[output_name],
                        data_store_descriptor_generation(store),
                        node_id=binding.node_id,
                        boot_id=binding.boot_id,
                        runtime_generation=binding.runtime_generation,
                    ),
                )
            )
    except Exception as exc:
        store.release_many(tuple(handle for _, handle in output_handles))
        return _failure_event(
            request=request,
            lease=placement_lease,
            worker_lease=worker_lease,
            binding=binding,
            kind=RuntimeEventKind.TASK_FAILED,
            error_code="result_publish_failed",
            category="data",
            phase="publishing",
            message=f"{type(exc).__name__}: {exc}",
            exception=exc,
            device_binding=device_binding,
            npu_runtime=npu_runtime,
        )
    observation = _resource_observation(
        request=request,
        lease=placement_lease,
        status="succeeded",
        error_type=None,
        device_binding=device_binding,
        npu_runtime=npu_runtime,
    )
    return RuntimeEvent.create(
        kind=RuntimeEventKind.TASK_RESULT,
        dispatch_id=request.dispatch_id,
        run_id=request.run_id,
        task_id=request.task_id,
        attempt=request.attempt,
        lease_id=placement_lease.lease_id,
        route_lease_id=None,
        occurred_at_ms=monotonic_time_ms(),
        output_handles=tuple(output_handles),
        worker_pid=os.getpid(),
        device_id=(
            None if device_binding is None else device_binding.physical_device_id
        ),
        binding_verified=npu_runtime is not None,
        resource_observation=observation,
    )


def data_store_descriptor_generation(store: RayDataStore) -> str:
    return store.descriptor.owner_generation


def _failure_event(
    *,
    request: ExecutionRequest,
    lease: PlacementLease,
    worker_lease: WorkerLease,
    binding: RuntimeNodeBinding,
    kind: RuntimeEventKind,
    error_code: str,
    category: str,
    phase: str,
    message: str,
    exception: BaseException | None = None,
    classification_confidence: str = "exact",
    device_binding: DeviceBinding | None = None,
    npu_runtime: BoundTorchNpuRuntime | None = None,
) -> RuntimeEvent:
    observation = _resource_observation(
        request=request,
        lease=lease,
        status="failed",
        error_type=None if exception is None else type(exception).__name__,
        device_binding=device_binding,
        npu_runtime=npu_runtime,
    )
    error = ErrorInfo(
        schema_version=1,
        error_code=error_code,
        category=category,
        origin="worker" if kind is RuntimeEventKind.TASK_FAILED else "runtime",
        message=message,
        retryable_hint=error_code
        in {"runtime_node_unavailable", "worker_start_failed"},
        classification_confidence=classification_confidence,
        execution_phase=phase,
        run_id=request.run_id,
        task_id=request.task_id,
        attempt=request.attempt,
        dispatch_id=request.dispatch_id,
        lease_id=lease.lease_id,
        node_id=binding.node_id,
        boot_id=binding.boot_id,
        worker_id=worker_lease.worker_id,
        device_id=(
            None if device_binding is None else device_binding.physical_device_id
        ),
        exception_type=None if exception is None else type(exception).__name__,
        platform_error_code=(
            None if exception is None else platform_error_code(exception)
        ),
        occurred_at_ms=monotonic_time_ms(),
        details=FrozenMap(
            (
                ("binding_verified", npu_runtime is not None),
                ("worker_pid", os.getpid()),
            )
        ),
    )
    return RuntimeEvent.create(
        kind=kind,
        dispatch_id=request.dispatch_id,
        run_id=request.run_id,
        task_id=request.task_id,
        attempt=request.attempt,
        lease_id=lease.lease_id,
        route_lease_id=None,
        occurred_at_ms=error.occurred_at_ms,
        error=error,
        worker_pid=os.getpid(),
        device_id=(
            None if device_binding is None else device_binding.physical_device_id
        ),
        binding_verified=npu_runtime is not None,
        resource_observation=observation,
    )


def _resource_observation(
    *,
    request: ExecutionRequest,
    lease: PlacementLease,
    status: str,
    error_type: str | None,
    device_binding: DeviceBinding | None,
    npu_runtime: BoundTorchNpuRuntime | None,
) -> ResourceObservation:
    peak_allocated: int | None = None
    peak_reserved: int | None = None
    process_hbm: int | None = None
    metric_quality: str | None = None
    if npu_runtime is not None:
        try:
            peak_allocated = npu_runtime.peak_allocated_mb()
            peak_reserved = npu_runtime.peak_reserved_mb()
        except Exception:
            pass
        try:
            sampled = npu_runtime.process_hbm_mb()
            process_hbm = max(
                npu_runtime.initial_process_hbm_mb,
                sampled if sampled is not None else 0,
            )
            metric_quality = "sampled"
        except Exception:
            process_hbm = None
    return ResourceObservation(
        run_id=request.run_id,
        task_id=request.task_id,
        definition_id=request.code_handle.definition_id,
        attempt=request.attempt,
        code_hash=request.code_handle.code_hash,
        environment_fingerprint=request.environment_fingerprint,
        requested=ResourceSpec(
            cpu_num=lease.resources.cpu_num,
            mem_mb=lease.resources.host_mem_mb,
            npu_mem_mb=lease.resources.npu_hbm_mb,
            io_num=lease.resources.io_slots,
        ),
        status=status,
        peak_host_rss_mb=host_peak_rss_mb(),
        peak_npu_allocated_mb=peak_allocated,
        peak_npu_reserved_mb=peak_reserved,
        peak_npu_process_hbm_mb=process_hbm,
        npu_metric_source=(
            "torch_npu_allocator+dcmi_process" if npu_runtime is not None else None
        ),
        npu_metric_quality=metric_quality,
        error_type=error_type,
        device_id=(
            None if device_binding is None else device_binding.physical_device_id
        ),
        worker_pid=os.getpid(),
        binding_verified=npu_runtime is not None,
    )


def _try_report(
    endpoint: str,
    identity: NodeAgentIdentity,
    event: RuntimeEvent,
    timeout_seconds: float,
) -> bool:
    try:
        report_worker_event(
            endpoint=endpoint,
            identity=identity,
            event=event,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        return False
    return True


_RAY_REMOTE: Any = ray.remote(
    num_cpus=0,
    max_retries=0,
    max_calls=1,
)
RAY_ONE_SHOT_WORKER: Any = _RAY_REMOTE(_execute_one_shot)
