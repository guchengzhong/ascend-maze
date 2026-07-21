from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ascend_maze import Workflow, task
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.contracts.submission import SubmissionState
from ascend_maze.core.errors import (
    DataHandleInvalidError,
    ResponseLostError,
    RunDataIndexError,
    SubmissionConflictError,
)
from ascend_maze.fault import RecoveryAction
from ascend_maze.lifecycle import AttemptStatus, RunStatus, TaskStatus
from ascend_maze.placement import LeaseStatus, NodeCapacity
from ascend_maze.runtime import FakeExecutionPlan, RuntimeEvent, RuntimeEventKind
from task_fixtures import (
    barrier,
    finish,
    load_text,
    summarize,
    timeout_task,
    user_failure_task,
)

CONFIG_FINGERPRINT = "c" * 64
ENVIRONMENT_FINGERPRINT = "e" * 64


def _node(node_id: str, *, cpu: int = 2, memory: int = 512, io: int = 2):
    return NodeCapacity(
        node_id=node_id,
        boot_id=f"boot_{node_id}",
        node_ip=f"10.0.0.{1 if node_id == 'node_a' else 2}",
        cpu_total=cpu,
        mem_total_mb=memory,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=io,
        observed_free_mem_mb=memory,
    )


def _controller(*, nodes: tuple[NodeCapacity, ...] | None = None):
    return InMemoryController(
        config_fingerprint=CONFIG_FINGERPRINT,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        build_revision="test_build",
        node_capacities=nodes or (_node("node_a"), _node("node_b")),
    )


def test_cpu_io_multinode_dag_reaches_both_resource_checkpoints() -> None:
    async def scenario() -> None:
        controller = _controller(
            nodes=(
                _node("node_a", cpu=1, memory=128, io=1),
                _node("node_b", cpu=1, memory=128, io=1),
            )
        )
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("stage2-multinode")
        path = workflow.input("path")
        left = workflow.add_task(
            load_text,
            inputs={"path": path},
            task_name="left_load",
        )
        right = workflow.add_task(
            load_text,
            inputs={"path": "side"},
            task_name="right_load",
        )
        result_task = workflow.add_task(
            finish,
            inputs={"summary": left.outputs["text"]},
            task_name="result",
        )
        workflow.add_edge(right, result_task)

        outcome = await client.submit(
            workflow,
            inputs={"path": "payload"},
            submission_id="submission_multinode",
        )
        assert outcome.state is SubmissionState.COMMITTED
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, result_task.task_id) == {
            "result": "payload"
        }
        attempt_nodes = {
            task.attempts[0].node_id
            for task in terminal.task_states
            if task.attempts
        }
        assert attempt_nodes == {"node_a", "node_b"}

        index = controller.indexes.get(outcome.run_id)
        handle_count = index.handle_count()
        assert handle_count == 4
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        assert controller.data_store.staged_count == 0
        assert controller.data_store.active_count == handle_count
        assert controller.runtime.code_reference_count() == 2

        destroyed = await controller.destroy_run(outcome.run_id)
        repeated = await controller.destroy_run(outcome.run_id)
        assert repeated is destroyed
        assert destroyed.tombstone.released_handle_count == handle_count
        assert destroyed.tombstone.destroy_succeeded
        assert destroyed.flush_result.recording_complete
        assert controller.data_store.active_count == 0
        assert controller.runtime.code_reference_count() == 0
        assert controller.runtime.dispatch_record_count() == 0
        assert controller.placement.lease_record_count() == 0
        assert controller.indexes.active_count == 0
        await controller.close()

    asyncio.run(scenario())


def test_post_flush_destroy_request_does_not_gap_next_run_recording() -> None:
    async def scenario() -> None:
        controller = _controller(nodes=(_node("node_a"),))
        await controller.start()
        client = InMemoryRuntimeClient(controller)

        async def execute(submission_id: str) -> str:
            workflow = Workflow(submission_id)
            result = workflow.add_task(
                finish,
                inputs={"summary": submission_id},
                task_name="result",
            )
            outcome = await client.submit(
                workflow,
                inputs={},
                submission_id=submission_id,
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, result.task_id) == {
                "result": submission_id
            }
            return outcome.run_id

        try:
            first_run_id = await execute("submission_before_flush")
            first_flush = await controller.flush_run(first_run_id)
            assert first_flush.recording_complete

            controller.record_control_request(
                first_run_id,
                request_id="destroy_after_flush",
                operation="destroy_run",
            )
            await controller.destroy_run(first_run_id)

            second_run_id = await execute("submission_after_flush")
            second_flush = await controller.flush_run(second_run_id)
            assert second_flush.recording_complete
            assert second_flush.sequence_gap_count == 0
            await controller.destroy_run(second_run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())


def test_submission_response_loss_replays_original_run_and_conflict_releases_upload() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        first_client = InMemoryRuntimeClient(controller)
        workflow = Workflow("submission-idempotency")
        value = workflow.input("value")
        result_task = workflow.add_task(finish, inputs={"summary": value})
        prepared = first_client.prepare_submission(
            workflow,
            inputs={"value": "first"},
            submission_id="submission_stable",
        )
        input_handle = prepared.request.workflow_inputs[0][1]
        with pytest.raises(ResponseLostError):
            await first_client.submit_prepared(
                prepared,
                lose_response_after_commit=True,
            )
        assert controller.data_store.state_of(input_handle) == "adopted"
        committed = controller.submission_outcome("submission_stable")
        assert committed.state is SubmissionState.COMMITTED
        assert committed.run_id is not None

        replayed = await first_client.submit_prepared(prepared)
        assert replayed.replayed
        assert replayed.run_id == committed.run_id
        assert controller.data_store.state_of(input_handle) == "adopted"

        same_content_client = InMemoryRuntimeClient(controller)
        same_content = same_content_client.prepare_submission(
            workflow,
            inputs={"value": "first"},
            submission_id="submission_stable",
        )
        redundant_handle = same_content.request.workflow_inputs[0][1]
        same_replay = await same_content_client.submit_prepared(same_content)
        assert same_replay.run_id == committed.run_id
        assert same_replay.replayed
        with pytest.raises(DataHandleInvalidError, match="released"):
            controller.data_store.get(redundant_handle)

        second_client = InMemoryRuntimeClient(controller)
        conflicting = second_client.prepare_submission(
            workflow,
            inputs={"value": "different"},
            submission_id="submission_stable",
        )
        conflicting_handle = conflicting.request.workflow_inputs[0][1]
        before = controller.data_store.active_count
        with pytest.raises(SubmissionConflictError):
            await second_client.submit_prepared(conflicting)
        assert controller.data_store.active_count == before - 1
        with pytest.raises(DataHandleInvalidError, match="released"):
            controller.data_store.get(conflicting_handle)

        terminal = await controller.wait_run(committed.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(committed.run_id, result_task.task_id) == {
            "result": "first"
        }
        await controller.destroy_run(committed.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_concurrent_identical_submission_commits_exactly_one_run() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("concurrent-submit")
        value = workflow.input("value")
        workflow.add_task(finish, inputs={"summary": value})
        prepared = client.prepare_submission(
            workflow,
            inputs={"value": "same"},
            submission_id="submission_concurrent",
        )
        first, second = await asyncio.gather(
            controller.submit(prepared.request),
            controller.submit(prepared.request),
        )
        assert first.run_id == second.run_id
        assert {first.replayed, second.replayed} == {False, True}
        assert first.run_id is not None
        await controller.wait_run(first.run_id, timeout_seconds=2)
        assert controller.indexes.active_count == 1
        await controller.destroy_run(first.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_no_digest_reupload_conflicts_even_when_values_are_equivalent() -> None:
    class NonCanonicalValue:
        def __init__(self, value: str) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, NonCanonicalValue) and other.value == self.value
            )

    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        first_client = InMemoryRuntimeClient(controller)
        workflow = Workflow("no-digest-submit")
        value = workflow.input("value")
        workflow.add_task(finish, inputs={"summary": value})
        first = first_client.prepare_submission(
            workflow,
            inputs={"value": NonCanonicalValue("same")},
            submission_id="submission_no_digest",
        )
        first_outcome = await first_client.submit_prepared(first)
        assert first_outcome.run_id is not None
        first_handle = first.request.workflow_inputs[0][1]
        assert first_handle.stable_digest is None

        second_client = InMemoryRuntimeClient(controller)
        second = second_client.prepare_submission(
            workflow,
            inputs={"value": NonCanonicalValue("same")},
            submission_id="submission_no_digest",
        )
        second_handle = second.request.workflow_inputs[0][1]
        assert second_handle.stable_digest is None
        with pytest.raises(SubmissionConflictError):
            await second_client.submit_prepared(second)
        with pytest.raises(DataHandleInvalidError, match="released"):
            controller.data_store.get(second_handle)

        await controller.wait_run(first_outcome.run_id, timeout_seconds=2)
        await controller.destroy_run(first_outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_tampered_compiled_ir_aborts_before_input_adoption() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("tampered-ir")
        value = workflow.input("value")
        workflow.add_task(finish, inputs={"summary": value})
        prepared = client.prepare_submission(
            workflow,
            inputs={"value": "payload"},
            submission_id="submission_tampered_ir",
        )
        tampered = replace(
            prepared.request.compiled,
            canonical_ir_bytes=prepared.request.compiled.canonical_ir_bytes + b"x",
        )
        request = replace(prepared.request, compiled=tampered)
        outcome = await controller.submit(request)
        assert outcome.state is SubmissionState.ABORTED
        handle = prepared.request.workflow_inputs[0][1]
        assert controller.data_store.state_of(handle) == "staged"
        assert controller.runtime.code_reference_count() == 0
        replayed = await client.submit_prepared(prepared)
        assert replayed.state is SubmissionState.ABORTED
        await controller.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_point",
    ["after_prepare", "after_open_run", "before_commit"],
)
def test_aborted_submission_leaves_input_staged_until_client_observes_abort(
    failure_point: str,
) -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow(f"abort-{failure_point}")
        value = workflow.input("value")
        workflow.add_task(finish, inputs={"summary": value})
        prepared = client.prepare_submission(
            workflow,
            inputs={"value": "payload"},
            submission_id=f"submission_{failure_point}",
        )
        handle = prepared.request.workflow_inputs[0][1]
        controller.inject_submit_failure(failure_point)

        outcome = await controller.submit(prepared.request)
        assert outcome.state is SubmissionState.ABORTED
        assert controller.data_store.state_of(handle) == "staged"
        assert controller.runtime.code_reference_count() == 0
        assert controller.recorder.active_run_count == 0
        assert controller.indexes.active_count == 0
        assert controller.deadlines.active_count == 0

        replayed = await client.submit_prepared(prepared)
        assert replayed.state is SubmissionState.ABORTED
        with pytest.raises(DataHandleInvalidError, match="released"):
            controller.data_store.get(handle)
        assert controller.data_store.active_count == 0
        await controller.close()

    asyncio.run(scenario())


def test_partial_output_failure_publishes_nothing_and_empty_task_puts_nothing() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)

        failing = Workflow("partial-output-e2e")
        task = failing.add_task(
            summarize,
            inputs={"text": "hello", "options": {}},
        )
        successor = failing.add_task(
            finish,
            inputs={"summary": task.outputs["summary"]},
        )
        controller.data_store.fail_on_put_number(controller.data_store.put_count + 2)
        outcome = await client.submit(
            failing,
            inputs={},
            submission_id="submission_partial_output",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.FAILED
        task_state = terminal.task(task.task_id)
        assert task_state.status is TaskStatus.FAILED
        assert task_state.last_error is not None
        assert task_state.last_error.error_code == "result_publish_failed"
        successor_state = terminal.task(successor.task_id)
        assert successor_state.status is TaskStatus.CANCELLED
        assert successor_state.attempt_count == 0
        assert controller.indexes.get(outcome.run_id).handle_count() == 0
        assert controller.data_store.active_count == 0
        assert controller.data_store.staged_count == 0
        with pytest.raises(RunDataIndexError, match="unpublished"):
            controller.result(outcome.run_id, task.task_id)
        await controller.destroy_run(outcome.run_id)

        control = Workflow("empty-output-e2e")
        empty = control.add_task(barrier)
        puts_before = controller.data_store.put_count
        control_outcome = await client.submit(
            control,
            inputs={},
            submission_id="submission_empty_output",
        )
        assert control_outcome.run_id is not None
        control_terminal = await controller.wait_run(
            control_outcome.run_id, timeout_seconds=2
        )
        assert control_terminal.status is RunStatus.SUCCEEDED
        assert controller.result(control_outcome.run_id, empty.task_id) == {}
        assert controller.data_store.put_count == puts_before
        assert controller.indexes.get(control_outcome.run_id).handle_count() == 0
        await controller.destroy_run(control_outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_prestart_failure_retries_after_cleanup_with_new_lease() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("retry-e2e")
        task = workflow.add_task(finish, inputs={"summary": "ok"})
        compiled = workflow.compile()
        controller.runtime.set_plan(
            task.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        outcome = await client.submit(
            compiled,
            inputs={},
            submission_id="submission_retry",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        task_state = terminal.task(task.task_id)
        assert task_state.attempt_count == 2
        assert [item.status for item in task_state.attempts] == [
            AttemptStatus.FAILED,
            AttemptStatus.SUCCEEDED,
        ]
        first_lease = controller.placement.lease_snapshot(
            task_state.attempts[0].lease_id
        )
        second_lease = controller.placement.lease_snapshot(
            task_state.attempts[1].lease_id
        )
        assert first_lease.status is LeaseStatus.RELEASED
        assert second_lease.status is LeaseStatus.RELEASED
        assert first_lease.lease.lease_id != second_lease.lease.lease_id
        decision = controller.core.recovery.decision(
            outcome.run_id,
            task.task_id,
            1,
        )
        assert decision is not None
        assert decision.action is RecoveryAction.RETRY
        assert decision.retry_budget_after == 0
        assert controller.result(outcome.run_id, task.task_id) == {"result": "ok"}
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_dispatch_deadline_expires_lease_and_ignores_late_worker_events() -> None:
    async def scenario() -> None:
        controller = _controller(nodes=(_node("node_a"),))
        controller.core.dispatch_timeout_ms = 10
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("dispatch-deadline-e2e")
        task = workflow.add_task(finish, inputs={"summary": "on-time-result"})
        compiled = workflow.compile()
        controller.runtime.set_plan(
            task.task_id,
            1,
            FakeExecutionPlan(start_delay_ms=60),
        )
        outcome = await client.submit(
            compiled,
            inputs={},
            submission_id="submission_dispatch_deadline",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        task_state = terminal.task(task.task_id)
        assert [item.status for item in task_state.attempts] == [
            AttemptStatus.FAILED,
            AttemptStatus.SUCCEEDED,
        ]
        assert task_state.attempts[0].error is not None
        assert task_state.attempts[0].error.error_code == "worker_start_failed"
        first_lease = controller.placement.lease_snapshot(
            task_state.attempts[0].lease_id
        )
        assert first_lease.status is LeaseStatus.EXPIRED

        first_attempt = task_state.attempts[0]
        late_handle = controller.data_store.put_staged(
            "late-result", controller.controller_generation
        )
        controller.core.post_runtime_event(
            RuntimeEvent.create(
                kind=RuntimeEventKind.WORKER_STARTED,
                dispatch_id=first_attempt.dispatch_id,
                run_id=outcome.run_id,
                task_id=task.task_id,
                attempt=first_attempt.attempt,
                lease_id=first_attempt.lease_id,
                route_lease_id=None,
                occurred_at_ms=1,
            )
        )
        controller.core.post_runtime_event(
            RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_RESULT,
                dispatch_id=first_attempt.dispatch_id,
                run_id=outcome.run_id,
                task_id=task.task_id,
                attempt=first_attempt.attempt,
                lease_id=first_attempt.lease_id,
                route_lease_id=None,
                occurred_at_ms=2,
                output_handles=(("result", late_handle),),
            )
        )
        await controller.core.wake_deadlines()
        assert controller.snapshot(outcome.run_id).status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, task.task_id) == {
            "result": "on-time-result"
        }
        assert controller.data_store.staged_count == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_boot_change_before_worker_started_retries_with_a_new_lease() -> None:
    async def scenario() -> None:
        controller = _controller(nodes=(_node("node_a"),))
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("prestart-boot-change-e2e")
        task = workflow.add_task(finish, inputs={"summary": "new-generation"})
        compiled = workflow.compile()
        controller.runtime.set_plan(
            task.task_id,
            1,
            FakeExecutionPlan(
                start_delay_ms=60,
                execution_delay_ms=20,
            ),
        )
        outcome = await client.submit(
            compiled,
            inputs={},
            submission_id="submission_prestart_boot_change",
        )
        assert outcome.run_id is not None
        for _ in range(200):
            current = controller.snapshot(outcome.run_id).task(task.task_id)
            if current.attempt_count == 1:
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("first attempt was not dispatched")

        controller.placement.register_node(
            replace(_node("node_a"), boot_id="boot_node_a_restarted")
        )
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        task_state = terminal.task(task.task_id)
        assert [item.status for item in task_state.attempts] == [
            AttemptStatus.FAILED,
            AttemptStatus.SUCCEEDED,
        ]
        first_lease = controller.placement.lease_snapshot(
            task_state.attempts[0].lease_id
        )
        second_lease = controller.placement.lease_snapshot(
            task_state.attempts[1].lease_id
        )
        assert first_lease.status is LeaseStatus.INVALIDATED
        assert first_lease.lease.boot_id == "boot_node_a"
        assert second_lease.lease.boot_id == "boot_node_a_restarted"
        assert second_lease.status is LeaseStatus.RELEASED

        await controller.runtime.wait_idle()
        await controller.core.wake_deadlines()
        assert controller.result(outcome.run_id, task.task_id) == {
            "result": "new-generation"
        }
        assert controller.data_store.staged_count == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_stale_event_cannot_release_the_current_attempt_lease() -> None:
    async def scenario() -> None:
        controller = _controller(nodes=(_node("node_a"),))
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("stale-event-lease-identity")
        task = workflow.add_task(finish, inputs={"summary": "current-attempt"})
        compiled = workflow.compile()
        controller.runtime.set_plan(
            task.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        controller.runtime.set_plan(
            task.task_id,
            2,
            FakeExecutionPlan(start_delay_ms=200),
        )
        outcome = await client.submit(
            compiled,
            inputs={},
            submission_id="submission_stale_event_lease_identity",
        )
        assert outcome.run_id is not None
        for _ in range(200):
            current = controller.snapshot(outcome.run_id).task(task.task_id)
            if current.attempt_count == 2:
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("second attempt was not dispatched")

        first_attempt, second_attempt = current.attempts
        second_before = controller.placement.lease_snapshot(second_attempt.lease_id)
        assert second_before.status is LeaseStatus.RESERVED
        controller.core.post_runtime_event(
            RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_RESULT,
                dispatch_id=first_attempt.dispatch_id,
                run_id=outcome.run_id,
                task_id=task.task_id,
                attempt=first_attempt.attempt,
                lease_id=second_attempt.lease_id,
                route_lease_id=None,
                occurred_at_ms=1,
            )
        )
        await controller.core.wake_deadlines()
        second_after = controller.placement.lease_snapshot(second_attempt.lease_id)
        assert second_after.status is LeaseStatus.RESERVED

        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, task.task_id) == {
            "result": "current-attempt"
        }
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_user_code_failure_is_permanent_and_does_not_publish_output() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("user-failure-e2e")
        task = workflow.add_task(
            user_failure_task,
            inputs={"should_fail": True},
        )
        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="submission_user_failure",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.FAILED
        task_state = terminal.task(task.task_id)
        assert task_state.attempt_count == 1
        assert task_state.last_error is not None
        assert task_state.last_error.error_code == "user_code_failed"
        assert controller.indexes.get(outcome.run_id).handle_count() == 0
        assert controller.data_store.staged_count == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_timeout_and_cancel_ignore_late_results_and_restore_resources() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)

        timeout_workflow = Workflow("timeout-e2e")
        timeout_node = timeout_workflow.add_task(
            timeout_task,
            inputs={"value": "late-timeout"},
        )
        timeout_compiled = timeout_workflow.compile()
        controller.runtime.set_plan(
            timeout_node.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=60, ignore_cancel=True),
        )
        timeout_outcome = await client.submit(
            timeout_compiled,
            inputs={},
            submission_id="submission_timeout",
        )
        assert timeout_outcome.run_id is not None
        timed_out = await controller.wait_run(
            timeout_outcome.run_id, timeout_seconds=2
        )
        assert timed_out.status is RunStatus.FAILED
        assert timed_out.task(timeout_node.task_id).status is TaskStatus.TIMED_OUT
        await controller.runtime.wait_idle()
        await controller.core.wake_deadlines()
        assert controller.snapshot(timeout_outcome.run_id).status is RunStatus.FAILED
        assert controller.indexes.get(timeout_outcome.run_id).handle_count() == 0
        assert controller.data_store.staged_count == 0
        assert controller.placement.active_lease_count(timeout_outcome.run_id) == 0
        assert controller.deadlines.count_for_run(timeout_outcome.run_id) == 0
        await controller.destroy_run(timeout_outcome.run_id)

        cancel_workflow = Workflow("cancel-e2e")
        cancel_node = cancel_workflow.add_task(
            finish,
            inputs={"summary": "late-cancel"},
        )
        cancel_compiled = cancel_workflow.compile()
        controller.runtime.set_plan(
            cancel_node.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=60, ignore_cancel=True),
        )
        cancel_outcome = await client.submit(
            cancel_compiled,
            inputs={},
            submission_id="submission_cancel",
        )
        assert cancel_outcome.run_id is not None
        for _ in range(200):
            current = controller.snapshot(cancel_outcome.run_id)
            if current.task(cancel_node.task_id).status is TaskStatus.RUNNING:
                break
            await asyncio.sleep(0.001)
        cancelled = await controller.cancel_run(cancel_outcome.run_id)
        assert cancelled.status is RunStatus.CANCELLED
        await controller.runtime.wait_idle()
        await controller.core.wake_deadlines()
        after_late = controller.snapshot(cancel_outcome.run_id)
        assert after_late.status is RunStatus.CANCELLED
        assert after_late.task(cancel_node.task_id).status is TaskStatus.CANCELLED
        assert controller.indexes.get(cancel_outcome.run_id).handle_count() == 0
        assert controller.data_store.staged_count == 0
        assert controller.placement.active_lease_count(cancel_outcome.run_id) == 0
        assert controller.deadlines.count_for_run(cancel_outcome.run_id) == 0
        await controller.destroy_run(cancel_outcome.run_id)

        assert controller.data_store.active_count == 0
        assert controller.runtime.code_reference_count() == 0
        await controller.close()

    asyncio.run(scenario())


def test_run_deadline_includes_resource_queue_wait_and_cancel_is_idempotent() -> None:
    async def scenario() -> None:
        controller = _controller(
            nodes=(_node("node_a", cpu=0, memory=128, io=0),)
        )
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("run-deadline-queued")
        task = workflow.add_task(finish, inputs={"summary": "never-runs"})
        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="submission_run_deadline",
            run_deadline_ms=20,
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.TIMED_OUT
        task_state = terminal.task(task.task_id)
        assert task_state.status is TaskStatus.CANCELLED
        assert task_state.attempt_count == 0
        repeated_cancel = await controller.cancel_run(outcome.run_id)
        assert repeated_cancel.status is RunStatus.TIMED_OUT
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_duplicate_terminal_runtime_event_does_not_release_published_result() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("duplicate-runtime-event")
        task = workflow.add_task(finish, inputs={"summary": "stable"})
        compiled = workflow.compile()
        controller.runtime.set_plan(
            task.task_id,
            1,
            FakeExecutionPlan(duplicate_terminal_event=True),
        )
        outcome = await client.submit(
            compiled,
            inputs={},
            submission_id="submission_duplicate_event",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        await controller.core.wake_deadlines()
        attempt = terminal.task(task.task_id).attempts[0]
        index = controller.indexes.get(outcome.run_id)
        ref = index.reference
        output_handle = index.task_output_handle(
            task.task_id,
            "result",
            controller_generation=ref.controller_generation,
            index_generation=ref.index_generation,
        )
        controller.core.post_runtime_event(
            RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_RESULT,
                dispatch_id=attempt.dispatch_id,
                run_id=outcome.run_id,
                task_id=task.task_id,
                attempt=attempt.attempt,
                lease_id=attempt.lease_id,
                route_lease_id=None,
                occurred_at_ms=1,
                output_handles=(("result", output_handle),),
            )
        )
        await controller.core.wake_deadlines()
        assert controller.result(outcome.run_id, task.task_id) == {
            "result": "stable"
        }
        assert controller.indexes.get(outcome.run_id).handle_count() == 1
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_repeated_run_destroy_cycles_do_not_grow_live_registries() -> None:
    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("repeat-cycles")
        task = workflow.add_task(finish, inputs={"summary": "ok"})
        compiled = workflow.compile()
        for index in range(12):
            outcome = await client.submit(
                compiled,
                inputs={},
                submission_id=f"submission_cycle_{index}",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, task.task_id) == {"result": "ok"}
            await controller.destroy_run(outcome.run_id)
            assert controller.data_store.active_count == 0
            assert controller.indexes.active_count == 0
            assert controller.placement.active_lease_count() == 0
            assert controller.placement.lease_record_count() == 0
            assert controller.deadlines.active_count == 0
            assert controller.runtime.active_dispatch_count() == 0
            assert controller.runtime.dispatch_record_count() == 0
            assert controller.runtime.code_reference_count() == 0
            assert controller.core.policy.active_count() == 0
            assert controller.core.policy.record_count() == 0
            assert client.prepared_submission_count == 0
        await controller.close()

    asyncio.run(scenario())


def test_post_commit_activation_failure_interrupts_a_committed_run() -> None:
    class FailEnqueueOncePolicy:
        name = "fail_enqueue_once"
        version = "test"

        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.capabilities = delegate.capabilities
            self.failed = False

        def enqueue(self, partition, task) -> None:
            if not self.failed:
                self.failed = True
                raise RuntimeError("injected enqueue failure")
            self.delegate.enqueue(partition, task)

        def depart(self, token) -> None:
            self.delegate.depart(token)

        def propose(self, partition, limit):
            return self.delegate.propose(partition, limit)

    async def scenario() -> None:
        controller = _controller()
        controller.core.policy = FailEnqueueOncePolicy(controller.core.policy)
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("post-commit-activation-failure")
        value = workflow.input("value")
        workflow.add_task(finish, inputs={"summary": value})
        prepared = client.prepare_submission(
            workflow,
            inputs={"value": "adopted"},
            submission_id="submission_post_commit_activation_failure",
        )
        input_handle = prepared.request.workflow_inputs[0][1]
        outcome = await client.submit_prepared(prepared)
        assert outcome.state is SubmissionState.COMMITTED
        assert outcome.run_id is not None
        assert controller.submission_outcome(outcome.submission_id).state is (
            SubmissionState.COMMITTED
        )
        assert controller.data_store.state_of(input_handle) == "adopted"

        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.INTERRUPTED
        assert controller.indexes.get(outcome.run_id).handle_count() == 1
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        assert controller.data_store.active_count == 0
        assert controller.runtime.code_reference_count() == 0
        await controller.close()

    asyncio.run(scenario())


def test_scheduler_policy_exception_interrupts_run_without_killing_actor() -> None:
    class FailOncePolicy:
        name = "fail_once"
        version = "test"

        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.capabilities = delegate.capabilities
            self.failed = False

        def enqueue(self, partition, task) -> None:
            self.delegate.enqueue(partition, task)

        def depart(self, token) -> None:
            self.delegate.depart(token)

        def propose(self, partition, limit):
            if not self.failed:
                self.failed = True
                raise RuntimeError("injected policy failure")
            return self.delegate.propose(partition, limit)

    async def scenario() -> None:
        controller = _controller()
        controller.core.policy = FailOncePolicy(controller.core.policy)
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("scheduler-interruption")
        workflow.add_task(finish, inputs={"summary": "never"})
        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="submission_scheduler_interrupt",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.INTERRUPTED
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_incomplete_recording_blocks_destroy_until_force_is_explicit() -> None:
    async def scenario() -> None:
        controller = _controller()
        controller.recorder.control_capacity_per_run = 1
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("recording-incomplete")
        task = workflow.add_task(finish, inputs={"summary": "result"})
        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="submission_recording_incomplete",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, task.task_id) == {
            "result": "result"
        }
        with pytest.raises(RuntimeError, match="recording is incomplete"):
            await controller.destroy_run(outcome.run_id)
        assert controller.data_store.active_count == 1
        forced = await controller.destroy_run(outcome.run_id, force=True)
        assert not forced.flush_result.recording_complete
        assert forced.flush_result.dropped_control_event_count > 0
        assert controller.data_store.active_count == 0
        await controller.close()

    asyncio.run(scenario())


def test_recorder_emit_failure_does_not_change_scheduling_result() -> None:
    async def scenario() -> None:
        controller = _controller()
        controller.recorder.inject_emit_failure()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("recorder-emit-failure")
        task = workflow.add_task(finish, inputs={"summary": "result"})
        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="submission_recorder_emit_failure",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, task.task_id) == {
            "result": "result"
        }
        with pytest.raises(RuntimeError, match="recording is incomplete"):
            await controller.destroy_run(outcome.run_id)
        forced = await controller.destroy_run(outcome.run_id, force=True)
        assert not forced.flush_result.recording_complete
        assert forced.flush_result.writer_errors == (
            "RuntimeError: injected recorder emit failure",
        )
        assert controller.data_store.active_count == 0
        await controller.close()

    asyncio.run(scenario())


def test_runtime_client_delivers_supported_non_importable_function_by_fallback() -> None:
    @task
    def local_task(value: str):
        return {"result": value}

    async def scenario() -> None:
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("cloudpickle-fallback")
        node = workflow.add_task(local_task, inputs={"value": "fallback"})
        prepared = client.prepare_submission(
            workflow,
            inputs={},
            submission_id="submission_cloudpickle_fallback",
        )
        package = prepared.request.code_packages[0]
        assert package.serialized_fallback is not None
        assert package.serialized_payload_digest is not None
        assert (
            prepared.request.compiled.workflow_fingerprint
            == workflow.compile().workflow_fingerprint
        )
        outcome = await client.submit_prepared(prepared)
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, node.task_id) == {
            "result": "fallback"
        }
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())
