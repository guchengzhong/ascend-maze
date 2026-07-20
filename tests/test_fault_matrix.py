from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ascend_maze import Workflow, task
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.core.canonical import FrozenMap
from ascend_maze.core.clock import ManualClock
from ascend_maze.data import InMemoryDataStore
from ascend_maze.fault import (
    CleanupBarrier,
    ErrorNormalizer,
    FaultIdentity,
    RecoveryAction,
    RecoveryPolicy,
    ReplayabilityChecker,
)
from ascend_maze.lifecycle import RunStatus, TaskStatus
from ascend_maze.placement import NodeCapacity
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime import (
    FakeExecutionPlan,
    FakeRuntimeBackend,
    RuntimeEvent,
    RuntimeEventKind,
)
from task_fixtures import finish, timeout_task, user_failure_task


CONFIG_FINGERPRINT = "c" * 64
ENVIRONMENT_FINGERPRINT = "e" * 64


@task
def business_failed_output():
    return {"status": "failed"}


@task(
    max_retries=1,
    retry_backoff_seconds=0.25,
    retry_on=["worker_start_failed"],
)
def backoff_retry_task(value: str):
    return {"result": value}


@task(max_retries=1, retry_on=["worker_lost"])
def worker_lost_retry_task(value: str):
    return {"result": value}


@task(max_retries=1, retry_on=["model_service_timeout"])
def model_timeout_retry_task(value: str):
    return {"result": value}


def _node() -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_a",
        node_ip="127.0.0.1",
        cpu_total=2,
        mem_total_mb=512,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=2,
        observed_free_mem_mb=512,
    )


def _controller(
    *,
    node: NodeCapacity | None = None,
    clock: ManualClock | None = None,
) -> InMemoryController:
    return InMemoryController(
        config_fingerprint=CONFIG_FINGERPRINT,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        build_revision="stage7b_test",
        node_capacities=(node or _node(),),
        clock=clock,
    )


def _error(
    *,
    error_code: str = "unknown_error",
    category: str = "control",
    message: str = "unknown",
    exception_type: str | None = None,
    platform_error_code: str | None = None,
    confidence: str = "exact",
) -> ErrorInfo:
    return ErrorInfo(
        schema_version=1,
        error_code=error_code,
        category=category,
        origin="worker",
        message=message,
        retryable_hint=True,
        classification_confidence=confidence,
        execution_phase="user_code",
        run_id="run_a",
        task_id="task_a",
        attempt=1,
        dispatch_id="dispatch_a",
        lease_id="lease_a",
        exception_type=exception_type,
        platform_error_code=platform_error_code,
        occurred_at_ms=10,
    )


def _identity() -> FaultIdentity:
    return FaultIdentity(
        run_id="run_a",
        task_id="task_a",
        attempt=1,
        dispatch_id="dispatch_a",
        lease_id="lease_a",
    )


def test_error_normalizer_prefers_structured_facts_and_exposes_fallback() -> None:
    normalizer = ErrorNormalizer(
        exception_types={"TypedOom": ("npu_oom", "resource")},
        platform_error_codes={"507011": ("device_unhealthy", "node_device")},
    )
    typed = normalizer.normalize(
        _error(
            message="not an OOM string",
            exception_type="TypedOom",
            platform_error_code="507011",
        ),
        identity=_identity(),
    )
    assert typed.error_code == "npu_oom"
    assert typed.classification_confidence == "mapped"
    assert typed.details["classification_source"] == "exception_type"

    coded = normalizer.normalize(
        _error(
            message="NPU out of memory.",
            platform_error_code="507011",
        ),
        identity=_identity(),
    )
    assert coded.error_code == "device_unhealthy"
    assert coded.details["classification_source"] == "platform_error_code"

    fallback = ErrorNormalizer().normalize(
        _error(message="NPU out of memory. allocation failed"),
        identity=_identity(),
    )
    assert fallback.error_code == "npu_oom"
    assert fallback.classification_confidence == "fallback"
    assert fallback.details["classification_source"] == "message"

    unknown = ErrorNormalizer().normalize(
        _error(message="unmapped platform failure"),
        identity=_identity(),
    )
    assert unknown.error_code == "unknown_error"
    assert not unknown.retryable_hint

    redacted = ErrorNormalizer().normalize(
        _error(message="Authorization: Bearer top-secret\nsecond line"),
        identity=_identity(),
    )
    assert "top-secret" not in redacted.message
    assert "\n" not in redacted.message


def test_error_normalizer_rejects_conflicting_attempt_identity() -> None:
    mismatched = replace(_error(), run_id="another_run", lease_id=None)
    normalized = ErrorNormalizer().normalize(mismatched, identity=_identity())
    assert normalized.error_code == "backend_internal_error"
    assert normalized.run_id == "run_a"
    assert normalized.dispatch_id == "dispatch_a"
    assert normalized.details["identity_mismatches"] == ("run_id",)


class _StateOnlyStore:
    def __init__(self, states: dict[str, str]) -> None:
        self.states = states

    def state_of(self, handle: DataHandle) -> str:
        return self.states[handle.staged_handle_id]


def test_replayability_requires_adopted_handles_and_rejects_node_local_paths() -> None:
    adopted = DataHandle("owner", "adopted")
    staged = DataHandle("owner", "staged")
    node_local = DataHandle(
        "owner",
        "local",
        metadata=FrozenMap((("node_local_node_id", "node_a"),)),
    )
    checker = ReplayabilityChecker(
        _StateOnlyStore(
            {"adopted": "adopted", "staged": "staged", "local": "adopted"}
        )
    )
    assert checker.check(
        code_available=True,
        environment_matches=True,
        handles=(adopted,),
    ).replayable
    assert checker.check(
        code_available=True,
        environment_matches=True,
        handles=(staged,),
    ).reason == "data_handle_not_adopted"
    local = checker.check(
        code_available=True,
        environment_matches=True,
        handles=(node_local,),
    )
    assert not local.replayable
    assert local.reason == "node_local_input_not_replayable"
    assert local.required_node_id == "node_a"


@pytest.mark.parametrize(
    "error_code",
    [
        "unknown_error",
        "user_code_failed",
        "invalid_task_output",
        "serialization_failed",
        "environment_mismatch",
        "data_handle_invalid",
    ],
)
def test_permanent_and_unknown_errors_never_retry_when_explicitly_listed(
    error_code: str,
) -> None:
    workflow = Workflow(f"permanent-{error_code}")
    node = workflow.add_task(finish, inputs={"summary": "x"})
    compiled = workflow.compile()
    definition = compiled.definitions[compiled.tasks[node.task_id].definition_id]
    definition = replace(definition, max_retries=3, retry_on=(error_code,))
    error = replace(
        _error(error_code=error_code),
        task_id=node.task_id,
    )
    decision = RecoveryPolicy().decide(
        definition=definition,
        error=error,
        attempt_count=1,
        cleanup=CleanupBarrier.confirmed_cleanup(),
        now_ms=100,
        run_deadline_at_ms=None,
    )
    assert decision.action is RecoveryAction.FAIL_TASK
    assert decision.reason == "permanent_error"
    assert decision.retry_budget_before == 3
    assert decision.retry_budget_after == 3


def test_c8_links_error_decision_cleanup_attempt_lease_and_recovery_result() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        workflow = Workflow("stage7b-c8-recovery-links")
        node = workflow.add_task(finish, inputs={"summary": "ok"})
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="stage7b_c8_recovery_links",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert isinstance(controller.recorder, InMemoryRecorder)
        events = controller.recorder.events(outcome.run_id)
        error_event = next(
            item for item in events if item.event_type == "error_normalized"
        )
        decision_event = next(
            item for item in events if item.event_type == "recovery_decision"
        )
        recovered_event = next(
            item for item in events if item.event_type == "recovery_succeeded"
        )
        first_attempt = terminal.task(node.task_id).attempts[0]
        assert error_event.task_id == node.task_id
        assert error_event.attempt == 1
        assert error_event.lease_id == first_attempt.lease_id
        assert error_event.payload["dispatch_id"] == first_attempt.dispatch_id
        assert decision_event.payload["cleanup"]["satisfied"] is True
        assert decision_event.payload["retry_budget_before"] == 1
        assert decision_event.payload["retry_budget_after"] == 0
        assert recovered_event.payload["decision_id"] == decision_event.payload[
            "decision_id"
        ]
        assert recovered_event.payload["recovered_attempt"] == 2
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_business_failed_status_is_ordinary_successful_output() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        workflow = Workflow("business-failed-output")
        node = workflow.add_task(business_failed_output, inputs={})
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="business_failed_output",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, node.task_id) == {"status": "failed"}
        assert controller.core.recovery.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_capacity_wait_does_not_create_attempt_or_consume_retry_budget() -> None:
    async def scenario() -> None:
        blocked_node = replace(
            _node(),
            cpu_total=0,
            observed_free_mem_mb=512,
        )
        controller = _controller(node=blocked_node)
        await controller.start()
        workflow = Workflow("capacity-wait-is-not-failure")
        node = workflow.add_task(finish, inputs={"summary": "ok"})
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="capacity_wait_is_not_failure",
        )
        assert outcome.run_id is not None
        for _ in range(500):
            task_state = controller.snapshot(outcome.run_id).task(node.task_id)
            if task_state.status is TaskStatus.QUEUED and task_state.pending_reason:
                break
            await asyncio.sleep(0.002)
        assert task_state.attempt_count == 0
        assert controller.core.recovery.count_for_run(outcome.run_id) == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0

        controller.placement.register_node(_node())
        controller.core.post_resource_changed("test_capacity_added")
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert terminal.task(node.task_id).attempt_count == 1
        assert controller.core.recovery.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error_code", "retry_task"),
    [
        ("worker_lost", worker_lost_retry_task),
        ("model_service_timeout", model_timeout_retry_task),
    ],
)
def test_post_start_uncertain_failures_default_to_no_retry_but_opt_in_is_at_least_once(
    error_code: str,
    retry_task,
) -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)

        default_workflow = Workflow(f"default-no-retry-{error_code}")
        default_node = default_workflow.add_task(
            finish,
            inputs={"summary": "default"},
        )
        controller.runtime.set_plan(
            default_node.task_id,
            1,
            FakeExecutionPlan(fail_after_start=error_code),
        )
        default_outcome = await client.submit(
            default_workflow,
            inputs={},
            submission_id=f"default_no_retry_{error_code}",
        )
        assert default_outcome.run_id is not None
        default_terminal = await controller.wait_run(
            default_outcome.run_id,
            timeout_seconds=2,
        )
        assert default_terminal.status is RunStatus.FAILED
        assert default_terminal.task(default_node.task_id).attempt_count == 1

        retry_workflow = Workflow(f"explicit-retry-{error_code}")
        retry_node = retry_workflow.add_task(retry_task, inputs={"value": "retry"})
        controller.runtime.set_plan(
            retry_node.task_id,
            1,
            FakeExecutionPlan(fail_after_start=error_code),
        )
        retry_outcome = await client.submit(
            retry_workflow,
            inputs={},
            submission_id=f"explicit_retry_{error_code}",
        )
        assert retry_outcome.run_id is not None
        retry_terminal = await controller.wait_run(
            retry_outcome.run_id,
            timeout_seconds=2,
        )
        assert retry_terminal.status is RunStatus.SUCCEEDED
        assert retry_terminal.task(retry_node.task_id).attempt_count == 2
        decision = controller.core.recovery.decision(
            retry_outcome.run_id,
            retry_node.task_id,
            1,
        )
        assert decision is not None
        assert decision.action is RecoveryAction.RETRY
        assert decision.retry_budget_after == 0
        await controller.destroy_run(default_outcome.run_id)
        await controller.destroy_run(retry_outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_retry_backoff_uses_deadline_heap_and_owns_no_old_lease() -> None:
    async def scenario() -> None:
        clock = ManualClock()
        controller = _controller(clock=clock)
        await controller.start()
        workflow = Workflow("retry-backoff-heap")
        node = workflow.add_task(backoff_retry_task, inputs={"value": "ok"})
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="retry_backoff_heap",
        )
        assert outcome.run_id is not None
        for _ in range(500):
            task_state = controller.snapshot(outcome.run_id).task(node.task_id)
            if task_state.status is TaskStatus.RETRY_WAIT:
                break
            await asyncio.sleep(0.002)
        assert task_state.status is TaskStatus.RETRY_WAIT
        assert task_state.attempt_count == 1
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.runtime.active_dispatch_count(outcome.run_id) == 0
        assert controller.core.active_route_lease_ids() == ()
        assert controller.deadlines.next_due_at_ms() == 250

        clock.advance(249)
        await controller.core.wake_deadlines()
        assert controller.snapshot(outcome.run_id).task(node.task_id).status is (
            TaskStatus.RETRY_WAIT
        )
        clock.advance(1)
        await controller.core.wake_deadlines()
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert terminal.task(node.task_id).attempt_count == 2
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_user_cancel_preempts_retry_backoff_without_creating_new_attempt() -> None:
    async def scenario() -> None:
        clock = ManualClock()
        controller = _controller(clock=clock)
        await controller.start()
        workflow = Workflow("cancel-preempts-backoff")
        node = workflow.add_task(backoff_retry_task, inputs={"value": "never"})
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="cancel_preempts_backoff",
        )
        assert outcome.run_id is not None
        for _ in range(500):
            task_state = controller.snapshot(outcome.run_id).task(node.task_id)
            if task_state.status is TaskStatus.RETRY_WAIT:
                break
            await asyncio.sleep(0.002)
        assert task_state.status is TaskStatus.RETRY_WAIT

        cancelled = await controller.cancel_run(outcome.run_id)
        assert cancelled.status is RunStatus.CANCELLED
        assert cancelled.task(node.task_id).attempt_count == 1
        clock.advance(1_000)
        await controller.core.wake_deadlines()
        after_deadline = controller.snapshot(outcome.run_id)
        assert after_deadline.status is RunStatus.CANCELLED
        assert after_deadline.task(node.task_id).attempt_count == 1
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.runtime.active_dispatch_count(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_run_deadline_preempts_retry_backoff_without_creating_new_attempt() -> None:
    async def scenario() -> None:
        clock = ManualClock()
        controller = _controller(clock=clock)
        await controller.start()
        workflow = Workflow("deadline-preempts-backoff")
        node = workflow.add_task(backoff_retry_task, inputs={"value": "never"})
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="deadline_preempts_backoff",
            run_deadline_ms=100,
        )
        assert outcome.run_id is not None
        for _ in range(500):
            task_state = controller.snapshot(outcome.run_id).task(node.task_id)
            if task_state.status is TaskStatus.RETRY_WAIT:
                break
            await asyncio.sleep(0.002)
        assert task_state.status is TaskStatus.RETRY_WAIT
        clock.advance(100)
        await controller.core.wake_deadlines()
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.TIMED_OUT
        assert terminal.task(node.task_id).attempt_count == 1
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_conflicting_dispatch_id_fails_attempt_and_isolates_original_worker() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        workflow = Workflow("conflicting-dispatch")
        node = workflow.add_task(finish, inputs={"summary": "late"})
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=10_000),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="conflicting_dispatch",
        )
        assert outcome.run_id is not None
        for _ in range(500):
            task_state = controller.snapshot(outcome.run_id).task(node.task_id)
            if task_state.status is TaskStatus.RUNNING:
                break
            await asyncio.sleep(0.002)
        active = task_state.attempts[0]
        controller.core.post_runtime_event(
            RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_FAILED,
                dispatch_id="conflicting_dispatch_id",
                run_id=outcome.run_id,
                task_id=node.task_id,
                attempt=1,
                lease_id=active.lease_id,
                route_lease_id=None,
                occurred_at_ms=controller.clock.monotonic_ms(),
            )
        )
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.FAILED
        failed = terminal.task(node.task_id)
        assert failed.attempt_count == 1
        assert failed.last_error is not None
        assert failed.last_error.error_code == "backend_internal_error"
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.runtime.worker_released(active.dispatch_id)
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_task_timeout_starts_only_after_worker_started() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        workflow = Workflow("timeout-starts-at-worker-started")
        node = workflow.add_task(timeout_task, inputs={"value": "ok"})
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(start_delay_ms=50),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="timeout_starts_at_worker_started",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        attempt = terminal.task(node.task_id).attempts[0]
        assert attempt.worker_started_at_ms is not None
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


class _ObserveRunBeforeCancel(FakeRuntimeBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.controller: InMemoryController | None = None
        self.run_statuses_seen_by_cancel: list[RunStatus] = []

    async def cancel(self, handle, reason: str) -> None:
        assert self.controller is not None
        self.run_statuses_seen_by_cancel.append(
            self.controller.snapshot(handle.run_id).status
        )
        await super().cancel(handle, reason)


def test_fail_fast_sets_run_terminal_before_sibling_cancel_and_never_unlocks_successor() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        runtime = _ObserveRunBeforeCancel(
            data_store=store,
            owner_generation="controller_fail_fast",
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        )
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="stage7b_test",
            node_capacities=(_node(),),
            controller_generation="controller_fail_fast",
            data_owner_generation="controller_fail_fast",
            data_store=store,
            runtime=runtime,
        )
        runtime.controller = controller
        await controller.start()
        workflow = Workflow("fail-fast-order")
        failing = workflow.add_task(
            user_failure_task,
            inputs={"should_fail": True},
        )
        sibling = workflow.add_task(finish, inputs={"summary": "slow"})
        successor = workflow.add_task(
            finish,
            inputs={"summary": failing.outputs["result"]},
        )
        runtime.set_plan(
            failing.task_id,
            1,
            FakeExecutionPlan(start_delay_ms=50),
        )
        runtime.set_plan(
            sibling.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=10_000),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="fail_fast_order",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.FAILED
        assert terminal.task(failing.task_id).attempt_count == 1
        assert terminal.task(sibling.task_id).attempt_count == 1
        assert runtime.run_statuses_seen_by_cancel
        assert set(runtime.run_statuses_seen_by_cancel) == {RunStatus.FAILED}
        successor_state = terminal.task(successor.task_id)
        assert successor_state.status is TaskStatus.CANCELLED
        assert successor_state.attempt_count == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


class _FirstAttemptCleanupUnconfirmed(FakeRuntimeBackend):
    def worker_released(self, dispatch_id: str) -> bool:
        record = self._dispatches.get(dispatch_id)
        if record is not None and record.request.attempt == 1:
            return False
        return super().worker_released(dispatch_id)


def test_unconfirmed_worker_cleanup_quarantines_node_before_retry_reuses_resources() -> None:
    async def scenario() -> None:
        store = InMemoryDataStore()
        runtime = _FirstAttemptCleanupUnconfirmed(
            data_store=store,
            owner_generation="controller_quarantine",
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        )
        node_b = replace(
            _node(),
            node_id="node_b",
            boot_id="boot_b",
            node_ip="127.0.0.2",
        )
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="stage7b_test",
            node_capacities=(_node(), node_b),
            controller_generation="controller_quarantine",
            data_owner_generation="controller_quarantine",
            data_store=store,
            runtime=runtime,
        )
        await controller.start()
        workflow = Workflow("cleanup-quarantine-before-retry")
        node = workflow.add_task(finish, inputs={"summary": "ok"})
        runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="cleanup_quarantine_before_retry",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        attempts = terminal.task(node.task_id).attempts
        assert [item.node_id for item in attempts] == ["node_a", "node_b"]
        assert controller.placement.lease_snapshot(
            attempts[0].lease_id
        ).status.value == "invalidated"
        decision = controller.core.recovery.decision(
            outcome.run_id,
            node.task_id,
            1,
        )
        assert decision is not None and decision.action is RecoveryAction.RETRY
        assert isinstance(controller.recorder, InMemoryRecorder)
        decision_event = next(
            event
            for event in controller.recorder.events(outcome.run_id)
            if event.event_type == "recovery_decision"
        )
        assert decision_event.payload["cleanup"][
            "node_or_device_quarantined"
        ] is True
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())
