from __future__ import annotations

import asyncio

from ascend_maze import Workflow
from ascend_maze.contracts.recording import ExecutionEvent, RunRecordingContext
from ascend_maze.contracts.resources import (
    ExecutionTarget,
    PlacementLease,
    ReservationVector,
)
from ascend_maze.contracts.runtime import ExecutionRequest, RuntimeArgument
from ascend_maze.data import InMemoryDataStore
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.resources import DeclaredOnlyAnchorProvider
from ascend_maze.runtime import (
    FakeExecutionPlan,
    FakeRuntimeBackend,
    RuntimeEventKind,
    build_code_packages,
)
from ascend_maze.scheduler import (
    FcfsPolicy,
    QueueToken,
    SchedulableTaskView,
    TaskKey,
)
from task_fixtures import barrier, finish, summarize

ENVIRONMENT = "e" * 64


def _lease(run_id: str, task_id: str, attempt: int) -> PlacementLease:
    return PlacementLease(
        lease_id=f"lease_{run_id}_{task_id}_{attempt}",
        reservation_kind="task",
        run_id=run_id,
        task_id=task_id,
        attempt=attempt,
        node_id="node_a",
        boot_id="boot_1",
        npu_device_id=None,
        resources=ReservationVector(1, 64, 0, 0, 0),
        snapshot_version=1,
        created_at_ms=1,
        dispatch_deadline_ms=10_000,
    )


def _single_task_runtime():
    workflow = Workflow("fake-runtime")
    node = workflow.add_task(finish, inputs={"summary": "unused"})
    compiled = workflow.compile()
    definition = compiled.definitions[compiled.tasks[node.task_id].definition_id]
    return compiled, node, definition


def test_fake_runtime_returns_staged_handles_and_releases_code() -> None:
    async def scenario() -> None:
        compiled, node, definition = _single_task_runtime()
        store = InMemoryDataStore()
        events = []
        backend = FakeRuntimeBackend(
            data_store=store,
            owner_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            event_sink=events.append,
        )
        await backend.start()
        handles = await backend.prepare(
            build_code_packages(compiled, environment_fingerprint=ENVIRONMENT)
        )
        code_handle = next(
            item for item in handles if item.definition_id == definition.definition_id
        )
        request = ExecutionRequest(
            dispatch_id="dispatch_1",
            run_id="run_1",
            task_id=node.task_id,
            attempt=1,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            model_route=None,
            code_handle=code_handle,
            arguments=(RuntimeArgument("summary", "literal", literal="hello"),),
            expected_outputs=("result",),
            timeout_ms=None,
            environment_fingerprint=ENVIRONMENT,
        )
        dispatch = await backend.dispatch(
            request, _lease("run_1", node.task_id, 1)
        )
        assert await backend.dispatch(
            request, _lease("run_1", node.task_id, 1)
        ) == dispatch
        await backend.wait_idle()
        assert [event.kind for event in events] == [
            RuntimeEventKind.WORKER_STARTED,
            RuntimeEventKind.TASK_RESULT,
        ]
        result_event = events[-1]
        assert result_event.output_handles[0][0] == "result"
        output_handle = result_event.output_handles[0][1]
        assert store.get(output_handle) == "hello"
        assert store.state_of(output_handle) == "staged"
        await backend.release_code(handles)
        assert backend.code_reference_count() == 0
        await backend.close()

    asyncio.run(scenario())


def test_fake_runtime_partial_output_put_failure_releases_every_staged_output() -> None:
    async def scenario() -> None:
        workflow = Workflow("partial-output")
        node = workflow.add_task(
            summarize,
            inputs={"text": "unused", "options": {}},
        )
        compiled = workflow.compile()
        definition = compiled.definitions[compiled.tasks[node.task_id].definition_id]
        store = InMemoryDataStore()
        events = []
        backend = FakeRuntimeBackend(
            data_store=store,
            owner_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            event_sink=events.append,
        )
        await backend.start()
        code_handles = await backend.prepare(
            build_code_packages(compiled, environment_fingerprint=ENVIRONMENT)
        )
        store.fail_on_put_number(store.put_count + 2)
        request = ExecutionRequest(
            dispatch_id="dispatch_partial",
            run_id="run_partial",
            task_id=node.task_id,
            attempt=1,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            model_route=None,
            code_handle=next(
                item
                for item in code_handles
                if item.definition_id == definition.definition_id
            ),
            arguments=(
                RuntimeArgument("text", "literal", literal="hello"),
                RuntimeArgument("options", "literal", literal={}),
                RuntimeArgument("max_length", "default_omitted"),
            ),
            expected_outputs=("size", "summary"),
            timeout_ms=100,
            environment_fingerprint=ENVIRONMENT,
        )
        await backend.dispatch(
            request,
            _lease("run_partial", node.task_id, 1),
        )
        await backend.wait_idle()
        assert events[-1].kind is RuntimeEventKind.TASK_FAILED
        assert events[-1].error is not None
        assert events[-1].error.error_code == "result_publish_failed"
        assert store.active_count == 0
        await backend.close()

    asyncio.run(scenario())


def test_fake_runtime_can_emit_a_late_result_after_cancel() -> None:
    async def scenario() -> None:
        compiled, node, definition = _single_task_runtime()
        store = InMemoryDataStore()
        events = []
        started = asyncio.Event()

        def sink(event) -> None:
            events.append(event)
            if event.kind is RuntimeEventKind.WORKER_STARTED:
                started.set()

        backend = FakeRuntimeBackend(
            data_store=store,
            owner_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            event_sink=sink,
        )
        backend.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=20, ignore_cancel=True),
        )
        await backend.start()
        code_handles = await backend.prepare(
            build_code_packages(compiled, environment_fingerprint=ENVIRONMENT)
        )
        request = ExecutionRequest(
            dispatch_id="dispatch_late",
            run_id="run_late",
            task_id=node.task_id,
            attempt=1,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            model_route=None,
            code_handle=next(
                item
                for item in code_handles
                if item.definition_id == definition.definition_id
            ),
            arguments=(RuntimeArgument("summary", "literal", literal="late"),),
            expected_outputs=("result",),
            timeout_ms=None,
            environment_fingerprint=ENVIRONMENT,
        )
        handle = await backend.dispatch(
            request,
            _lease("run_late", node.task_id, 1),
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        await backend.cancel(handle, "test_cancel")
        await backend.wait_idle()
        assert events[-1].kind is RuntimeEventKind.TASK_RESULT
        for _, output_handle in events[-1].output_handles:
            store.release(output_handle)
        assert store.active_count == 0
        await backend.close()

    asyncio.run(scenario())


def test_in_memory_recorder_tracks_sequences_missing_producers_and_flush() -> None:
    async def scenario() -> None:
        recorder = InMemoryRecorder()
        context = RunRecordingContext(
            schema_version=1,
            experiment_id="run_1",
            run_id="run_1",
            workflow_fingerprint="a" * 64,
            config_fingerprint="b" * 64,
            environment_fingerprint=ENVIRONMENT,
            build_revision="build_1",
            started_wall_time_ms=1,
            initial_expected_producer_ids=("controller", "node_a"),
        )
        recorder.open_run(context)
        event = ExecutionEvent(
            schema_version=1,
            event_id="event_1",
            experiment_id="run_1",
            run_id="run_1",
            task_id=None,
            attempt=None,
            lease_id=None,
            route_lease_id=None,
            model_instance_id=None,
            event_type="run_submitted",
            producer_id="controller",
            producer_sequence=1,
            node_id=None,
            device_id=None,
            monotonic_time_ms=1,
            wall_time_ms=1,
            duration_ms=None,
        )
        assert recorder.emit(event)
        assert recorder.emit(event)
        result = await recorder.flush_run("run_1", 100)
        assert not result.recording_complete
        assert result.missing_producer_count == 1
        assert len(recorder.events("run_1")) == 1

    asyncio.run(scenario())


def test_recorder_accepts_global_producer_sequence_interleaved_across_runs() -> None:
    async def scenario() -> None:
        recorder = InMemoryRecorder()
        for run_id in ("run_1", "run_2"):
            recorder.open_run(
                RunRecordingContext(
                    schema_version=1,
                    experiment_id=run_id,
                    run_id=run_id,
                    workflow_fingerprint="a" * 64,
                    config_fingerprint="b" * 64,
                    environment_fingerprint=ENVIRONMENT,
                    build_revision="build_1",
                    started_wall_time_ms=1,
                    initial_expected_producer_ids=("controller",),
                )
            )

        for sequence, run_id in ((1, "run_1"), (2, "run_2"), (3, "run_1")):
            assert recorder.emit(
                ExecutionEvent(
                    schema_version=1,
                    event_id=f"event_{sequence}",
                    experiment_id=run_id,
                    run_id=run_id,
                    task_id=None,
                    attempt=None,
                    lease_id=None,
                    route_lease_id=None,
                    model_instance_id=None,
                    event_type="test",
                    producer_id="controller",
                    producer_sequence=sequence,
                    node_id=None,
                    device_id=None,
                    monotonic_time_ms=sequence,
                    wall_time_ms=sequence,
                    duration_ms=None,
                )
            )
        assert (await recorder.flush_run("run_1", 100)).recording_complete
        assert (await recorder.flush_run("run_2", 100)).recording_complete

    asyncio.run(scenario())


def test_fcfs_policy_uses_queued_time_then_stable_enqueue_sequence() -> None:
    workflow = Workflow("fcfs")
    first = workflow.add_task(barrier, task_name="first")
    second = workflow.add_task(barrier, task_name="second")
    compiled = workflow.compile()
    provider = DeclaredOnlyAnchorProvider(environment_fingerprint=ENVIRONMENT)
    policy = FcfsPolicy()
    views = []
    for sequence, node in ((2, first), (1, second)):
        token = QueueToken(TaskKey("run_1", node.task_id), 1)
        view = SchedulableTaskView(
            queue_token=token,
            task_kind="cpu",
            ready_at_ms=10,
            queued_at_ms=20,
            enqueue_sequence=sequence,
            depth_from_entry=0,
            depth_to_exit=0,
            resource_anchor=provider.resolve(
                run_id="run_1",
                compiled=compiled,
                task_id=node.task_id,
            ),
        )
        views.append(view)
        policy.enqueue("cpu", view)
    proposals = policy.propose("cpu", 2)
    assert [item.task_key for item in proposals] == [
        views[1].queue_token.task_key,
        views[0].queue_token.task_key,
    ]
    policy.depart(views[1].queue_token)
    assert policy.propose("cpu", 2)[0].task_key == views[0].queue_token.task_key
