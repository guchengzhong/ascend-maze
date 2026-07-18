"""Small immutable events emitted by RuntimeBackend implementations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.core.identifiers import new_id


class RuntimeEventKind(str, Enum):
    WORKER_STARTED = "worker_started"
    TASK_RESULT = "task_result"
    TASK_FAILED = "task_failed"
    DISPATCH_FAILED = "dispatch_failed"
    TASK_CANCELLED = "task_cancelled"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event_id: str
    kind: RuntimeEventKind
    dispatch_id: str
    run_id: str
    task_id: str
    attempt: int
    lease_id: str
    route_lease_id: str | None
    occurred_at_ms: int
    output_handles: tuple[tuple[str, DataHandle], ...] = ()
    error: ErrorInfo | None = None

    @classmethod
    def create(
        cls,
        *,
        kind: RuntimeEventKind,
        dispatch_id: str,
        run_id: str,
        task_id: str,
        attempt: int,
        lease_id: str,
        route_lease_id: str | None,
        occurred_at_ms: int,
        output_handles: tuple[tuple[str, DataHandle], ...] = (),
        error: ErrorInfo | None = None,
    ) -> "RuntimeEvent":
        return cls(
            event_id=new_id("event"),
            kind=kind,
            dispatch_id=dispatch_id,
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            lease_id=lease_id,
            route_lease_id=route_lease_id,
            occurred_at_ms=occurred_at_ms,
            output_handles=output_handles,
            error=error,
        )
