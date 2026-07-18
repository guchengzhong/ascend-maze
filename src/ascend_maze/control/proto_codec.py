"""Translate generated Protobuf messages at the control boundary."""

from __future__ import annotations

from typing import Any

from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.contracts.resources import ResourceObservation, ResourceSpec
from ascend_maze.core.canonical import (
    CanonicalValue,
    FrozenMap,
    canonical_bytes,
    decode_canonical_bytes,
)
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind

from ascend_maze.control.proto import control_pb2 as _control_pb2

control_pb2: Any = _control_pb2


def encode_data_handle(handle: DataHandle) -> Any:
    backend = handle.metadata.get("backend")
    if not isinstance(backend, str):
        raise ContractValidationError("DataHandle backend metadata is required")
    source_node_id = handle.metadata.get("source_node_id")
    source_boot_id = handle.metadata.get("source_boot_id")
    source_generation = handle.metadata.get("source_runtime_generation")
    if source_node_id is not None and not isinstance(source_node_id, str):
        raise ContractValidationError("source_node_id metadata must be a string")
    if source_boot_id is not None and not isinstance(source_boot_id, str):
        raise ContractValidationError("source_boot_id metadata must be a string")
    if source_generation is not None and (
        isinstance(source_generation, bool) or not isinstance(source_generation, int)
    ):
        raise ContractValidationError(
            "source_runtime_generation metadata must be an integer"
        )
    return control_pb2.DataHandleMessage(
        owner_generation=handle.owner_generation,
        staged_handle_id=handle.staged_handle_id,
        stable_digest=handle.stable_digest or "",
        has_stable_digest=handle.stable_digest is not None,
        size_bytes=handle.size_bytes or 0,
        has_size_bytes=handle.size_bytes is not None,
        backend=backend,
        source_node_id=source_node_id or "",
        source_boot_id=source_boot_id or "",
        source_runtime_generation=source_generation or 0,
    )


def decode_data_handle(message: Any) -> DataHandle:
    metadata: list[tuple[CanonicalValue, CanonicalValue]] = [
        ("backend", str(message.backend))
    ]
    if message.source_node_id:
        metadata.extend(
            (
                ("source_node_id", str(message.source_node_id)),
                ("source_boot_id", str(message.source_boot_id)),
                (
                    "source_runtime_generation",
                    int(message.source_runtime_generation),
                ),
            )
        )
    return DataHandle(
        owner_generation=str(message.owner_generation),
        staged_handle_id=str(message.staged_handle_id),
        stable_digest=(str(message.stable_digest) if message.has_stable_digest else None),
        size_bytes=(int(message.size_bytes) if message.has_size_bytes else None),
        metadata=FrozenMap(tuple(metadata)),
    )


def encode_error(error: ErrorInfo) -> Any:
    return control_pb2.ErrorMessage(
        schema_version=error.schema_version,
        error_code=error.error_code,
        category=error.category,
        origin=error.origin,
        message=error.message,
        retryable_hint=error.retryable_hint,
        classification_confidence=error.classification_confidence,
        execution_phase=error.execution_phase,
        run_id=error.run_id,
        task_id=error.task_id,
        attempt=error.attempt,
        dispatch_id=error.dispatch_id or "",
        lease_id=error.lease_id or "",
        route_lease_id=error.route_lease_id or "",
        node_id=error.node_id or "",
        boot_id=error.boot_id or "",
        worker_id=error.worker_id or "",
        exception_type=error.exception_type or "",
        occurred_at_ms=error.occurred_at_ms,
        device_id=error.device_id or "",
        platform_error_code=error.platform_error_code or "",
        canonical_details=canonical_bytes(error.details),
        traceback_ref=error.traceback_ref or "",
        model_instance_id=error.model_instance_id or "",
    )


def decode_error(message: Any) -> ErrorInfo:
    details = decode_canonical_bytes(bytes(message.canonical_details))
    if not isinstance(details, FrozenMap):
        raise ContractValidationError("ErrorInfo details must decode to a mapping")
    return ErrorInfo(
        schema_version=int(message.schema_version),
        error_code=str(message.error_code),
        category=str(message.category),
        origin=str(message.origin),
        message=str(message.message),
        retryable_hint=bool(message.retryable_hint),
        classification_confidence=str(message.classification_confidence),
        execution_phase=str(message.execution_phase),
        run_id=str(message.run_id),
        task_id=str(message.task_id),
        attempt=int(message.attempt),
        dispatch_id=str(message.dispatch_id) or None,
        lease_id=str(message.lease_id) or None,
        route_lease_id=str(message.route_lease_id) or None,
        node_id=str(message.node_id) or None,
        boot_id=str(message.boot_id) or None,
        worker_id=str(message.worker_id) or None,
        exception_type=str(message.exception_type) or None,
        device_id=str(message.device_id) or None,
        platform_error_code=str(message.platform_error_code) or None,
        occurred_at_ms=int(message.occurred_at_ms),
        details=details,
        traceback_ref=str(message.traceback_ref) or None,
        model_instance_id=str(message.model_instance_id) or None,
    )


def encode_resource_observation(observation: ResourceObservation) -> Any:
    return control_pb2.ResourceObservationMessage(
        run_id=observation.run_id,
        task_id=observation.task_id,
        definition_id=observation.definition_id,
        attempt=observation.attempt,
        code_hash=observation.code_hash,
        environment_fingerprint=observation.environment_fingerprint,
        requested=control_pb2.ResourceSpecMessage(
            cpu_num=observation.requested.cpu_num,
            mem_mb=observation.requested.mem_mb,
            npu_mem_mb=observation.requested.npu_mem_mb,
            io_num=observation.requested.io_num,
        ),
        status=observation.status,
        canonical_input_features=canonical_bytes(observation.input_features),
        peak_host_rss_mb=observation.peak_host_rss_mb or 0,
        has_peak_host_rss_mb=observation.peak_host_rss_mb is not None,
        peak_npu_allocated_mb=observation.peak_npu_allocated_mb or 0,
        has_peak_npu_allocated_mb=observation.peak_npu_allocated_mb is not None,
        peak_npu_reserved_mb=observation.peak_npu_reserved_mb or 0,
        has_peak_npu_reserved_mb=observation.peak_npu_reserved_mb is not None,
        peak_npu_process_hbm_mb=observation.peak_npu_process_hbm_mb or 0,
        has_peak_npu_process_hbm_mb=(
            observation.peak_npu_process_hbm_mb is not None
        ),
        npu_metric_source=observation.npu_metric_source or "",
        npu_metric_quality=observation.npu_metric_quality or "",
        error_type=observation.error_type or "",
        device_id=observation.device_id or "",
        worker_pid=observation.worker_pid or 0,
        has_worker_pid=observation.worker_pid is not None,
        binding_verified=observation.binding_verified,
    )


def decode_resource_observation(message: Any) -> ResourceObservation:
    features = decode_canonical_bytes(bytes(message.canonical_input_features))
    if not isinstance(features, FrozenMap):
        raise ContractValidationError(
            "ResourceObservation input_features must decode to a mapping"
        )
    return ResourceObservation(
        run_id=str(message.run_id),
        task_id=str(message.task_id),
        definition_id=str(message.definition_id),
        attempt=int(message.attempt),
        code_hash=str(message.code_hash),
        environment_fingerprint=str(message.environment_fingerprint),
        requested=ResourceSpec(
            cpu_num=int(message.requested.cpu_num),
            mem_mb=int(message.requested.mem_mb),
            npu_mem_mb=int(message.requested.npu_mem_mb),
            io_num=int(message.requested.io_num),
        ),
        status=str(message.status),
        input_features=features,
        peak_host_rss_mb=(
            int(message.peak_host_rss_mb)
            if message.has_peak_host_rss_mb
            else None
        ),
        peak_npu_allocated_mb=(
            int(message.peak_npu_allocated_mb)
            if message.has_peak_npu_allocated_mb
            else None
        ),
        peak_npu_reserved_mb=(
            int(message.peak_npu_reserved_mb)
            if message.has_peak_npu_reserved_mb
            else None
        ),
        peak_npu_process_hbm_mb=(
            int(message.peak_npu_process_hbm_mb)
            if message.has_peak_npu_process_hbm_mb
            else None
        ),
        npu_metric_source=str(message.npu_metric_source) or None,
        npu_metric_quality=str(message.npu_metric_quality) or None,
        error_type=str(message.error_type) or None,
        device_id=str(message.device_id) or None,
        worker_pid=int(message.worker_pid) if message.has_worker_pid else None,
        binding_verified=bool(message.binding_verified),
    )


def encode_runtime_event(event: RuntimeEvent) -> Any:
    message = control_pb2.RuntimeEventMessage(
        event_id=event.event_id,
        kind=event.kind.value,
        dispatch_id=event.dispatch_id,
        run_id=event.run_id,
        task_id=event.task_id,
        attempt=event.attempt,
        lease_id=event.lease_id,
        route_lease_id=event.route_lease_id or "",
        occurred_at_ms=event.occurred_at_ms,
        has_error=event.error is not None,
        worker_pid=event.worker_pid or 0,
        has_worker_pid=event.worker_pid is not None,
        device_id=event.device_id or "",
        binding_verified=event.binding_verified,
        has_resource_observation=event.resource_observation is not None,
    )
    for name, handle in event.output_handles:
        message.output_handles.add(name=name, handle=encode_data_handle(handle))
    if event.error is not None:
        message.error.CopyFrom(encode_error(event.error))
    if event.resource_observation is not None:
        message.resource_observation.CopyFrom(
            encode_resource_observation(event.resource_observation)
        )
    return message


def decode_runtime_event(message: Any) -> RuntimeEvent:
    try:
        kind = RuntimeEventKind(str(message.kind))
    except ValueError as exc:
        raise ContractValidationError(
            f"unsupported runtime event kind: {message.kind}"
        ) from exc
    return RuntimeEvent(
        event_id=str(message.event_id),
        kind=kind,
        dispatch_id=str(message.dispatch_id),
        run_id=str(message.run_id),
        task_id=str(message.task_id),
        attempt=int(message.attempt),
        lease_id=str(message.lease_id),
        route_lease_id=str(message.route_lease_id) or None,
        occurred_at_ms=int(message.occurred_at_ms),
        output_handles=tuple(
            (str(item.name), decode_data_handle(item.handle))
            for item in message.output_handles
        ),
        error=decode_error(message.error) if message.has_error else None,
        worker_pid=int(message.worker_pid) if message.has_worker_pid else None,
        device_id=str(message.device_id) or None,
        binding_verified=bool(message.binding_verified),
        resource_observation=(
            decode_resource_observation(message.resource_observation)
            if message.has_resource_observation
            else None
        ),
    )
