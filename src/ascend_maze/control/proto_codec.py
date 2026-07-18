"""Translate generated Protobuf messages at the control boundary."""

from __future__ import annotations

from typing import Any

from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.core.canonical import CanonicalValue, FrozenMap
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
    )


def decode_error(message: Any) -> ErrorInfo:
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
        occurred_at_ms=int(message.occurred_at_ms),
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
    )
    for name, handle in event.output_handles:
        message.output_handles.add(name=name, handle=encode_data_handle(handle))
    if event.error is not None:
        message.error.CopyFrom(encode_error(event.error))
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
    )
