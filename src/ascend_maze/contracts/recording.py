"""Lightweight recording sink contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ascend_maze.core.canonical import CanonicalValue, FrozenMap, freeze_canonical
from ascend_maze.core.errors import ContractValidationError


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    schema_version: int
    event_id: str
    experiment_id: str
    run_id: str | None
    task_id: str | None
    attempt: int | None
    lease_id: str | None
    route_lease_id: str | None
    model_instance_id: str | None
    event_type: str
    producer_id: str
    producer_sequence: int
    node_id: str | None
    device_id: str | None
    monotonic_time_ms: int
    wall_time_ms: int
    duration_ms: int | None
    payload: FrozenMap[CanonicalValue, CanonicalValue] = field(
        default_factory=FrozenMap
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ContractValidationError("schema_version must be a positive integer")
        for name in ("event_id", "experiment_id", "event_type", "producer_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ContractValidationError(f"{name} is required")
        if self.attempt is not None and (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 0
        ):
            raise ContractValidationError("attempt must be non-negative")
        for name in (
            "producer_sequence",
            "monotonic_time_ms",
            "wall_time_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractValidationError(f"{name} must be non-negative")
        if self.duration_ms is not None and (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ContractValidationError("duration_ms must be non-negative")
        frozen = freeze_canonical(self.payload)
        if not isinstance(frozen, FrozenMap):
            raise ContractValidationError("payload must be a mapping")
        object.__setattr__(self, "payload", frozen)


@runtime_checkable
class RecorderSink(Protocol):
    def emit(self, event: ExecutionEvent) -> bool: ...
