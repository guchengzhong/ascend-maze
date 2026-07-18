from __future__ import annotations

from ascend_maze import Workflow
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.fault import CleanupBarrier, RecoveryAction, RecoveryPolicy
from task_fixtures import finish


def _definition():
    workflow = Workflow("recovery-definition")
    task = workflow.add_task(finish, inputs={"summary": "value"})
    compiled = workflow.compile()
    return compiled.definitions[compiled.tasks[task.task_id].definition_id], task


def _error(run_id: str, task_id: str, attempt: int, code: str) -> ErrorInfo:
    return ErrorInfo(
        schema_version=1,
        error_code=code,
        category="worker",
        origin="runtime",
        message=code,
        retryable_hint=True,
        classification_confidence="exact",
        execution_phase="dispatched",
        run_id=run_id,
        task_id=task_id,
        attempt=attempt,
        occurred_at_ms=10,
    )


def test_recovery_decision_is_idempotent_and_consumes_budget_once() -> None:
    definition, task = _definition()
    error = _error("run_1", task.task_id, 1, "worker_start_failed")
    policy = RecoveryPolicy()
    decision = policy.decide(
        definition=definition,
        error=error,
        attempt_count=1,
        cleanup=CleanupBarrier.confirmed_cleanup(),
        now_ms=100,
        run_deadline_at_ms=1_000,
    )
    duplicate = policy.decide(
        definition=definition,
        error=error,
        attempt_count=1,
        cleanup=CleanupBarrier.confirmed_cleanup(),
        now_ms=200,
        run_deadline_at_ms=1_000,
    )
    assert duplicate is decision
    assert decision.action is RecoveryAction.RETRY
    assert decision.next_attempt == 2
    assert decision.retry_budget_before == 1
    assert decision.retry_budget_after == 0
    assert policy.count_for_run("run_1") == 1


def test_recovery_requires_cleanup_and_rejects_permanent_or_expired_cases() -> None:
    definition, task = _definition()
    policy = RecoveryPolicy()
    incomplete = policy.decide(
        definition=definition,
        error=_error("run_cleanup", task.task_id, 1, "worker_start_failed"),
        attempt_count=1,
        cleanup=CleanupBarrier(False, False, True, True, False),
        now_ms=100,
        run_deadline_at_ms=None,
    )
    assert incomplete.action is RecoveryAction.FAIL_TASK
    assert incomplete.reason == "cleanup_barrier_incomplete"

    permanent = policy.decide(
        definition=definition,
        error=_error("run_permanent", task.task_id, 1, "user_code_failed"),
        attempt_count=1,
        cleanup=CleanupBarrier.confirmed_cleanup(),
        now_ms=100,
        run_deadline_at_ms=None,
    )
    assert permanent.action is RecoveryAction.FAIL_TASK
    assert permanent.reason == "permanent_error"

    expired = policy.decide(
        definition=definition,
        error=_error("run_expired", task.task_id, 1, "worker_start_failed"),
        attempt_count=1,
        cleanup=CleanupBarrier.confirmed_cleanup(),
        now_ms=100,
        run_deadline_at_ms=100,
    )
    assert expired.action is RecoveryAction.FAIL_TASK
    assert expired.reason == "run_deadline_exhausted"
