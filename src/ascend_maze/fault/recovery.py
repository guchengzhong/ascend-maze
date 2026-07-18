"""Idempotent retry/fail decisions after explicit resource cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock

from ascend_maze.compiler.ir import TaskDefinition
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.core.identifiers import stable_id


class RecoveryAction(str, Enum):
    RETRY = "retry"
    FAIL_TASK = "fail_task"
    CANCEL_RUN = "cancel_run"
    INTERRUPT_RUN = "interrupt_run"


@dataclass(frozen=True, slots=True)
class CleanupBarrier:
    dispatch_invalidated: bool
    worker_released: bool
    unpublished_data_released: bool
    route_released: bool
    placement_released: bool
    node_or_device_quarantined: bool = False

    @property
    def confirmed(self) -> bool:
        return (
            self.dispatch_invalidated
            and self.worker_released
            and self.unpublished_data_released
            and self.route_released
            and self.placement_released
        )

    @property
    def quarantined(self) -> bool:
        return (
            self.dispatch_invalidated
            and self.unpublished_data_released
            and self.route_released
            and self.node_or_device_quarantined
        )

    @property
    def satisfied(self) -> bool:
        return self.confirmed or self.quarantined

    @classmethod
    def confirmed_cleanup(cls) -> "CleanupBarrier":
        return cls(True, True, True, True, True)


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    decision_id: str
    run_id: str
    task_id: str
    attempt: int
    error: ErrorInfo
    action: RecoveryAction
    reason: str
    next_attempt: int | None
    eligible_at_ms: int | None
    cleanup_requirements: tuple[str, ...]
    retry_budget_before: int
    retry_budget_after: int


_PERMANENT_ERRORS = frozenset(
    {
        "data_binding_failed",
        "environment_mismatch",
        "invalid_task_output",
        "model_catalog_invalid",
        "serialization_failed",
        "task_definition_invalid",
        "user_code_failed",
    }
)


class RecoveryPolicy:
    """Return one stable decision for each failed run/task/attempt identity."""

    def __init__(self) -> None:
        self._decisions: dict[tuple[str, str, int], RecoveryDecision] = {}
        self._lock = RLock()

    def decide(
        self,
        *,
        definition: TaskDefinition,
        error: ErrorInfo,
        attempt_count: int,
        cleanup: CleanupBarrier,
        now_ms: int,
        run_deadline_at_ms: int | None,
    ) -> RecoveryDecision:
        key = (error.run_id, error.task_id, error.attempt)
        with self._lock:
            existing = self._decisions.get(key)
            if existing is not None:
                if existing.error != error:
                    raise StateTransitionError(
                        "one attempt produced conflicting recovery errors"
                    )
                return existing

            retries_used = max(0, attempt_count - 1)
            budget_before = max(0, definition.max_retries - retries_used)
            cleanup_requirements = (
                "dispatch_invalidated",
                "worker_released_or_quarantined",
                "unpublished_data_released",
                "route_released",
                "placement_released_or_invalidated",
            )
            action = RecoveryAction.RETRY
            reason = "retry_eligible"
            if not cleanup.satisfied:
                action = RecoveryAction.FAIL_TASK
                reason = "cleanup_barrier_incomplete"
            elif error.error_code in _PERMANENT_ERRORS:
                action = RecoveryAction.FAIL_TASK
                reason = "permanent_error"
            elif error.error_code not in definition.retry_on:
                action = RecoveryAction.FAIL_TASK
                reason = "error_not_in_retry_on"
            elif budget_before <= 0:
                action = RecoveryAction.FAIL_TASK
                reason = "retry_budget_exhausted"
            elif run_deadline_at_ms is not None and now_ms >= run_deadline_at_ms:
                action = RecoveryAction.FAIL_TASK
                reason = "run_deadline_exhausted"

            retry = action is RecoveryAction.RETRY
            eligible_at = now_ms + definition.retry_backoff_ms if retry else None
            decision = RecoveryDecision(
                decision_id=stable_id(
                    "decision",
                    error.run_id,
                    error.task_id,
                    str(error.attempt),
                    error.error_code,
                ),
                run_id=error.run_id,
                task_id=error.task_id,
                attempt=error.attempt,
                error=error,
                action=action,
                reason=reason,
                next_attempt=attempt_count + 1 if retry else None,
                eligible_at_ms=eligible_at,
                cleanup_requirements=cleanup_requirements,
                retry_budget_before=budget_before,
                retry_budget_after=budget_before - 1 if retry else budget_before,
            )
            self._decisions[key] = decision
            return decision

    def decision(
        self,
        run_id: str,
        task_id: str,
        attempt: int,
    ) -> RecoveryDecision | None:
        with self._lock:
            return self._decisions.get((run_id, task_id, attempt))

    def count_for_run(self, run_id: str) -> int:
        with self._lock:
            return sum(key[0] == run_id for key in self._decisions)
