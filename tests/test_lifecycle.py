from __future__ import annotations

import pytest

from ascend_maze import Workflow
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.lifecycle import (
    AttemptStatus,
    DeadlineKind,
    DeadlineManager,
    RunStateManager,
    RunStatus,
    TaskStatus,
)
from task_fixtures import barrier


def _compiled_chain():
    workflow = Workflow("lifecycle-chain")
    first = workflow.add_task(barrier, task_name="first")
    second = workflow.add_task(barrier, task_name="second")
    workflow.add_edge(first, second)
    return workflow.compile(), first, second


def _error(run_id: str, task_id: str, attempt: int, code: str) -> ErrorInfo:
    return ErrorInfo(
        schema_version=1,
        error_code=code,
        category="worker",
        origin="worker",
        message=code,
        retryable_hint=True,
        classification_confidence="exact",
        execution_phase="user_code",
        run_id=run_id,
        task_id=task_id,
        attempt=attempt,
        occurred_at_ms=10,
    )


def test_run_state_manager_completes_a_chain_and_ignores_duplicate_success() -> None:
    compiled, first, second = _compiled_chain()
    state = RunStateManager()
    state.create_run(
        run_id="run_1",
        compiled=compiled,
        routing_session_key_hash=None,
        submitted_at_ms=1,
        deadline_at_ms=100,
    )
    started = state.start_run("run_1", 2)
    assert started.ready_task_ids == (first.task_id,)
    assert state.mark_queued("run_1", first.task_id, 3)
    attempt = state.create_attempt(
        run_id="run_1",
        task_id=first.task_id,
        dispatch_id="dispatch_1",
        lease_id="lease_1",
        node_id="node_a",
        device_ids=(),
        anchor_revision=1,
        now_ms=4,
    )
    assert attempt.attempt == 1
    assert state.snapshot("run_1").task(first.task_id).status is TaskStatus.STARTING
    assert state.worker_started(
        run_id="run_1",
        task_id=first.task_id,
        attempt=1,
        dispatch_id="dispatch_1",
        now_ms=5,
    ).accepted
    completed = state.attempt_succeeded(
        run_id="run_1",
        task_id=first.task_id,
        attempt=1,
        dispatch_id="dispatch_1",
        now_ms=6,
    )
    assert completed.ready_task_ids == (second.task_id,)
    assert not state.attempt_succeeded(
        run_id="run_1",
        task_id=first.task_id,
        attempt=1,
        dispatch_id="dispatch_1",
        now_ms=7,
    ).accepted

    state.mark_queued("run_1", second.task_id, 8)
    state.create_attempt(
        run_id="run_1",
        task_id=second.task_id,
        dispatch_id="dispatch_2",
        lease_id="lease_2",
        node_id="node_b",
        device_ids=(),
        anchor_revision=1,
        now_ms=9,
    )
    state.worker_started(
        run_id="run_1",
        task_id=second.task_id,
        attempt=1,
        dispatch_id="dispatch_2",
        now_ms=10,
    )
    final = state.attempt_succeeded(
        run_id="run_1",
        task_id=second.task_id,
        attempt=1,
        dispatch_id="dispatch_2",
        now_ms=11,
    )
    assert final.run_terminal
    snapshot = state.snapshot("run_1")
    assert snapshot.status is RunStatus.SUCCEEDED
    assert all(task.status is TaskStatus.SUCCEEDED for task in snapshot.task_states)


def test_retry_uses_a_new_attempt_and_rejects_conflicting_dispatch_id() -> None:
    compiled, first, _ = _compiled_chain()
    state = RunStateManager()
    state.create_run(
        run_id="run_retry",
        compiled=compiled,
        routing_session_key_hash=None,
        submitted_at_ms=1,
        deadline_at_ms=None,
    )
    state.start_run("run_retry", 2)
    state.mark_queued("run_retry", first.task_id, 3)
    state.create_attempt(
        run_id="run_retry",
        task_id=first.task_id,
        dispatch_id="dispatch_1",
        lease_id="lease_1",
        node_id="node_a",
        device_ids=(),
        anchor_revision=1,
        now_ms=4,
    )
    with pytest.raises(StateTransitionError, match="different dispatch"):
        state.worker_started(
            run_id="run_retry",
            task_id=first.task_id,
            attempt=1,
            dispatch_id="dispatch_conflict",
            now_ms=5,
        )
    state.worker_started(
        run_id="run_retry",
        task_id=first.task_id,
        attempt=1,
        dispatch_id="dispatch_1",
        now_ms=5,
    )
    error = _error("run_retry", first.task_id, 1, "worker_lost")
    assert state.attempt_retry_wait(
        run_id="run_retry",
        task_id=first.task_id,
        attempt=1,
        dispatch_id="dispatch_1",
        attempt_status=AttemptStatus.FAILED,
        error=error,
        now_ms=6,
    ).accepted
    eligible = state.retry_eligible(
        run_id="run_retry",
        task_id=first.task_id,
        attempt=1,
        now_ms=7,
    )
    assert eligible.ready_task_ids == (first.task_id,)
    state.mark_queued("run_retry", first.task_id, 8)
    second_attempt = state.create_attempt(
        run_id="run_retry",
        task_id=first.task_id,
        dispatch_id="dispatch_2",
        lease_id="lease_2",
        node_id="node_b",
        device_ids=(),
        anchor_revision=1,
        now_ms=9,
    )
    assert second_attempt.attempt == 2
    assert not state.worker_started(
        run_id="run_retry",
        task_id=first.task_id,
        attempt=1,
        dispatch_id="dispatch_1",
        now_ms=10,
    ).accepted


def test_final_failure_is_fail_fast_and_late_start_cannot_revive_run() -> None:
    workflow = Workflow("parallel-failure")
    left = workflow.add_task(barrier, task_name="left")
    right = workflow.add_task(barrier, task_name="right")
    compiled = workflow.compile()
    state = RunStateManager()
    state.create_run(
        run_id="run_failed",
        compiled=compiled,
        routing_session_key_hash=None,
        submitted_at_ms=1,
        deadline_at_ms=None,
    )
    for task_id in state.start_run("run_failed", 2).ready_task_ids:
        state.mark_queued("run_failed", task_id, 3)
    for number, task in enumerate((left, right), start=1):
        state.create_attempt(
            run_id="run_failed",
            task_id=task.task_id,
            dispatch_id=f"dispatch_{number}",
            lease_id=f"lease_{number}",
            node_id=f"node_{number}",
            device_ids=(),
            anchor_revision=1,
            now_ms=4,
        )
        state.worker_started(
            run_id="run_failed",
            task_id=task.task_id,
            attempt=1,
            dispatch_id=f"dispatch_{number}",
            now_ms=5,
        )
    result = state.attempt_final_failure(
        run_id="run_failed",
        task_id=left.task_id,
        attempt=1,
        dispatch_id="dispatch_1",
        attempt_status=AttemptStatus.FAILED,
        task_status=TaskStatus.FAILED,
        error=_error("run_failed", left.task_id, 1, "user_code_failed"),
        now_ms=6,
    )
    assert result.run_terminal
    assert result.cancelled_attempts[0].dispatch_id == "dispatch_2"
    snapshot = state.snapshot("run_failed")
    assert snapshot.status is RunStatus.FAILED
    assert snapshot.task(right.task_id).status is TaskStatus.CANCELLED
    assert not state.worker_started(
        run_id="run_failed",
        task_id=right.task_id,
        attempt=1,
        dispatch_id="dispatch_2",
        now_ms=7,
    ).accepted


def test_deadline_manager_invalidates_replaced_and_cancelled_timers() -> None:
    deadlines = DeadlineManager()
    deadlines.register(
        kind=DeadlineKind.RUN,
        run_id="run",
        due_at_ms=100,
    )
    deadlines.register(
        kind=DeadlineKind.RUN,
        run_id="run",
        due_at_ms=200,
    )
    deadlines.register(
        kind=DeadlineKind.LEASE,
        run_id="run",
        task_id="task",
        attempt=1,
        due_at_ms=125,
    )
    deadlines.register(
        kind=DeadlineKind.TASK,
        run_id="run",
        task_id="task",
        attempt=1,
        due_at_ms=150,
    )
    assert deadlines.pop_due(100) == ()
    lease_due = deadlines.pop_due(125)
    assert len(lease_due) == 1
    assert lease_due[0].kind is DeadlineKind.LEASE
    assert deadlines.cancel(
        kind=DeadlineKind.TASK,
        run_id="run",
        task_id="task",
        attempt=1,
    )
    due = deadlines.pop_due(200)
    assert len(due) == 1
    assert due[0].kind is DeadlineKind.RUN
    assert deadlines.active_count == 0
