from __future__ import annotations

import asyncio

from ascend_maze import Workflow
from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.inference import ModelInstanceState, ModelRouteLeaseStatus
from ascend_maze.inference.adapters.fake import FakeAdapterPlan
from ascend_maze.lifecycle import AttemptStatus, RunStatus, TaskStatus
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime import FakeExecutionPlan, RuntimeEvent, RuntimeEventKind
from inference_helpers import make_controller, make_node, make_spec
from task_fixtures import (
    inference_retry_task,
    inference_timeout_task,
    inference_twice_task,
    inference_zero_call_task,
    service_task,
)


def _workflow(name: str, task_func, *, model_id: str = "model_a"):
    workflow = Workflow(name)
    node = workflow.add_task(
        task_func,
        inputs={"prompt": "hello"},
        model_anchor={"model": model_id, "mode": "service"},
    )
    return workflow, node


def test_service_task_runs_two_calls_with_one_route_and_c8_links(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        workflow, node = _workflow("service-two-calls", inference_twice_task)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="service_two_calls",
            session_key="session_a",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)

        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, node.task_id) == {
            "answer": "model_a:hello|model_a:model_a:hello"
        }
        assert adapter.launch_count == 1
        records = inference.request_records()
        assert [record.call_index for record in records] == [1, 2]
        assert [record.ttft_ms for record in records] == [0, 0]
        assert len({record.route_lease_id for record in records}) == 1
        route_id = records[0].route_lease_id
        route = inference.route_snapshot(route_id)
        assert route.status is ModelRouteLeaseStatus.RELEASED
        instance = inference.model_instances()[0]
        assert instance.state is ModelInstanceState.READY
        assert instance.route_occupancy == 0
        assert instance.actual_request_inflight == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        cluster = controller.placement.snapshot()
        assert cluster.active_lease_count == 1
        assert cluster.nodes[0].per_npu_reserved == (
            ("0", spec.instance_hbm_mb, spec.npu_slots),
        )

        assert isinstance(controller.recorder, InMemoryRecorder)
        events = controller.recorder.events(outcome.run_id)
        requests = [event for event in events if event.event_type == "inference_request"]
        summary = next(
            event
            for event in events
            if event.event_type == "attempt_inference_summary"
        )
        assert [event.payload["call_index"] for event in requests] == [1, 2]
        assert [event.payload["ttft_ms"] for event in requests] == [0, 0]
        assert all(event.route_lease_id == route_id for event in requests)
        assert all(event.model_instance_id == instance.instance_id for event in requests)
        assert all(event.lease_id is not None for event in requests)
        assert all(
            event.payload["instance_placement_lease_id"]
            == instance.placement_lease_id
            for event in requests
        )
        assert summary.payload["request_count"] == 2
        assert summary.payload["context_cleared"] is True
        assert "hello" not in repr(tuple(event.payload for event in requests))

        await controller.destroy_run(outcome.run_id)
        assert inference.request_records() == ()
        assert all(event.run_id != outcome.run_id for event in inference.events())
        try:
            inference.route_snapshot(route_id)
        except KeyError:
            pass
        else:
            raise AssertionError("destroy must purge terminal RouteLease history")
        await controller.close()
        assert controller.placement.active_lease_count() == 0
        assert inference.model_instances()[0].state is ModelInstanceState.STOPPED

    asyncio.run(scenario())


def test_same_callable_bound_to_two_models_never_crosses_routes(tmp_path) -> None:
    async def scenario() -> None:
        first_spec = make_spec(tmp_path / "model_a", model_id="model_a")
        second_spec = make_spec(tmp_path / "model_b", model_id="model_b")
        controller, inference, adapter = make_controller(
            (first_spec, second_spec),
            nodes=(make_node(npu_count=2),),
        )
        await controller.start()
        workflow = Workflow("two-model-bindings")
        first = workflow.add_task(
            inference_twice_task,
            inputs={"prompt": "first"},
            task_name="first",
            model_anchor={"model": "model_a", "mode": "service"},
        )
        second = workflow.add_task(
            inference_twice_task,
            inputs={"prompt": "second"},
            task_name="second",
            model_anchor={"model": "model_b", "mode": "service"},
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow, inputs={}, submission_id="two_model_bindings"
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)

        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, first.task_id) == {
            "answer": "model_a:first|model_a:model_a:first"
        }
        assert controller.result(outcome.run_id, second.task_id) == {
            "answer": "model_b:second|model_b:model_b:second"
        }
        records = inference.request_records()
        assert {record.model_id for record in records} == {"model_a", "model_b"}
        instance_models = {
            instance.instance_id: instance.model_id
            for instance in inference.model_instances()
        }
        assert all(
            instance_models[record.instance_id] == record.model_id
            for record in records
        )
        assert adapter.launch_count == 2
        cluster = controller.placement.snapshot()
        assert cluster.nodes[0].per_npu_reserved == (
            ("0", 1_024, 1),
            ("1", 1_024, 1),
        )

        await controller.destroy_run(outcome.run_id)
        await controller.close()
        assert controller.placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_zero_chat_task_succeeds_and_records_zero_request_summary(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        workflow, node = _workflow("service-zero-call", inference_zero_call_task)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow, inputs={}, submission_id="service_zero_call"
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)

        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, node.task_id) == {"answer": "hello"}
        assert adapter.invoke_count == 0
        route_event = next(
            event
            for event in inference.events()
            if event.event_type == "model_route_reserved"
        )
        assert route_event.route_lease_id is not None
        summary = inference.attempt_summary(route_event.route_lease_id)
        assert summary is not None
        assert summary.request_count == 0
        assert summary.context_cleared
        assert not summary.request_inflight
        assert isinstance(controller.recorder, InMemoryRecorder)
        summary_event = next(
            event
            for event in controller.recorder.events(outcome.run_id)
            if event.event_type == "attempt_inference_summary"
        )
        assert summary_event.payload["request_count"] == 0
        assert summary_event.route_lease_id == route_event.route_lease_id

        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_adapter_failure_is_structured_and_releases_both_counters(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, adapter = make_controller(spec)
        adapter.set_plan(spec.model_id, FakeAdapterPlan(fail_invoke="engine failed"))
        await controller.start()
        workflow, node = _workflow("service-adapter-failure", inference_twice_task)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow, inputs={}, submission_id="service_adapter_failure"
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)

        assert terminal.status is RunStatus.FAILED
        task = terminal.task(node.task_id)
        assert task.status is TaskStatus.FAILED
        assert task.attempts[0].error is not None
        assert task.attempts[0].error.error_code == "model_inference_failed"
        record = inference.request_records()[0]
        assert record.status == "failed"
        assert record.error_code == "model_inference_failed"
        instance = inference.model_instances()[0]
        assert instance.route_occupancy == 0
        assert instance.actual_request_inflight == 0
        assert inference.router.active_count() == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0

        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_client_placement_failure_abandons_reserved_route(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, _ = make_controller(
            spec,
            nodes=(make_node(cpu=1),),
        )
        await controller.start()
        workflow, node = _workflow("service-client-blocked", service_task)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow, inputs={}, submission_id="service_client_blocked"
        )
        assert outcome.run_id is not None
        for _ in range(100):
            task = controller.snapshot(outcome.run_id).task(node.task_id)
            if task.pending_reason == "insufficient_cpu":
                break
            await asyncio.sleep(0.005)
        assert task.status is TaskStatus.QUEUED
        assert task.attempt_count == 0
        assert task.pending_reason == "insufficient_cpu"
        assert inference.router.active_count() == 0
        assert inference.model_instances()[0].route_occupancy == 0
        assert any(
            event.event_type == "model_route_released"
            and event.payload["reason"] == "insufficient_cpu"
            for event in inference.events()
        )

        cancelled = await controller.cancel_run(outcome.run_id)
        assert cancelled.status is RunStatus.CANCELLED
        assert inference.router.active_count() == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_model_capacity_wait_does_not_create_attempt_or_consume_retry_budget(
    tmp_path,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, adapter = make_controller(spec)
        adapter.set_plan(spec.model_id, FakeAdapterPlan(launch_delay_ms=100))
        await controller.start()
        workflow, node = _workflow("service-model-capacity-wait", service_task)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="service_model_capacity_wait",
        )
        assert outcome.run_id is not None

        for _ in range(1_000):
            task = controller.snapshot(outcome.run_id).task(node.task_id)
            if task.pending_reason == "model_route_unavailable":
                break
            await asyncio.sleep(0.001)
        assert task.status is TaskStatus.QUEUED
        assert task.attempt_count == 0
        assert controller.core.recovery.count_for_run(outcome.run_id) == 0
        assert inference.router.active_count() == 0
        assert controller.placement.active_lease_count(outcome.run_id) == 0

        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert terminal.task(node.task_id).attempt_count == 1
        assert controller.core.recovery.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_dispatch_failure_retries_with_new_route_and_late_event_is_idempotent(
    tmp_path,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, _ = make_controller(spec)
        await controller.start()
        workflow, node = _workflow("service-route-retry", inference_retry_task)
        controller.runtime.set_plan(
            node.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow, inputs={}, submission_id="service_route_retry"
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)

        assert terminal.status is RunStatus.SUCCEEDED
        task = terminal.task(node.task_id)
        assert [attempt.status for attempt in task.attempts] == [
            AttemptStatus.FAILED,
            AttemptStatus.SUCCEEDED,
        ]
        routes = [
            event
            for event in inference.events()
            if event.event_type == "model_route_reserved"
        ]
        assert len(routes) == 2
        assert routes[0].route_lease_id != routes[1].route_lease_id
        assert inference.router.active_count() == 0
        first_route_id = routes[0].route_lease_id
        assert first_route_id is not None
        first_route = inference.route_snapshot(first_route_id).lease
        first_attempt = task.attempts[0]
        controller.core.post_runtime_event(
            RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_FAILED,
                dispatch_id=first_attempt.dispatch_id,
                run_id=outcome.run_id,
                task_id=node.task_id,
                attempt=1,
                lease_id=first_attempt.lease_id,
                route_lease_id=first_route.route_lease_id,
                occurred_at_ms=controller.clock.monotonic_ms(),
            )
        )
        await controller.core.wake_deadlines()
        assert controller.snapshot(outcome.run_id).status is RunStatus.SUCCEEDED
        assert inference.router.active_count() == 0
        assert inference.model_instances()[0].route_occupancy == 0

        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_timeout_and_cancel_wait_for_inflight_cleanup_before_route_release(
    tmp_path,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, adapter = make_controller(spec)
        adapter.set_plan(spec.model_id, FakeAdapterPlan(invoke_delay_ms=80))
        await controller.start()
        client = InMemoryRuntimeClient(controller)

        timeout_workflow, timeout_node = _workflow(
            "service-timeout", inference_timeout_task
        )
        timeout_outcome = await client.submit(
            timeout_workflow, inputs={}, submission_id="service_timeout"
        )
        assert timeout_outcome.run_id is not None
        timed_out = await controller.wait_run(
            timeout_outcome.run_id, timeout_seconds=2
        )
        assert timed_out.status is RunStatus.FAILED
        assert timed_out.task(timeout_node.task_id).attempts[0].status is AttemptStatus.TIMED_OUT
        assert inference.router.active_count() == 0
        assert all(
            instance.actual_request_inflight == 0
            and instance.route_occupancy == 0
            for instance in inference.model_instances()
        )

        cancel_workflow, cancel_node = _workflow(
            "service-cancel", inference_twice_task
        )
        cancel_outcome = await client.submit(
            cancel_workflow, inputs={}, submission_id="service_cancel"
        )
        assert cancel_outcome.run_id is not None
        for _ in range(200):
            if any(
                instance.actual_request_inflight == 1
                for instance in inference.model_instances()
            ):
                break
            await asyncio.sleep(0.002)
        assert any(
            instance.actual_request_inflight == 1
            for instance in inference.model_instances()
        )
        cancelled = await controller.cancel_run(cancel_outcome.run_id)
        assert cancelled.status is RunStatus.CANCELLED
        assert cancelled.task(cancel_node.task_id).status is TaskStatus.CANCELLED
        assert inference.router.active_count() == 0
        assert all(
            instance.actual_request_inflight == 0
            and instance.route_occupancy == 0
            for instance in inference.model_instances()
        )
        assert controller.placement.active_lease_count(cancel_outcome.run_id) == 0

        await controller.destroy_run(timeout_outcome.run_id)
        await controller.destroy_run(cancel_outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_unknown_model_submission_aborts_before_run_commit(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        workflow, _ = _workflow(
            "service-unknown-model", service_task, model_id="missing"
        )
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow, inputs={}, submission_id="service_unknown_model"
        )
        assert outcome.run_id is None
        assert outcome.error is not None
        assert "not registered" in outcome.error
        assert controller.indexes.active_count == 0
        assert inference.model_instances() == ()
        assert adapter.launch_count == 0
        await controller.close()

    asyncio.run(scenario())


def test_model_startup_does_not_block_run_deadline_processing(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        controller, inference, adapter = make_controller(spec)
        adapter.set_plan(spec.model_id, FakeAdapterPlan(launch_delay_ms=100))
        await controller.start()
        workflow, node = _workflow("startup-deadline", inference_zero_call_task)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="startup_deadline",
            run_deadline_ms=20,
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=1)

        assert terminal.status is RunStatus.TIMED_OUT
        assert terminal.task(node.task_id).status is TaskStatus.CANCELLED
        assert adapter.launch_count == 0
        assert inference.router.active_count() == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()
        assert adapter.launch_count == 1
        assert adapter.stop_count == 1
        assert controller.placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_ready_service_process_exit_invalidates_route_and_fails_active_attempt(
    tmp_path,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", scale_cooldown_ms=100_000)
        controller, inference, adapter = make_controller(spec)
        adapter.set_plan(spec.model_id, FakeAdapterPlan(invoke_delay_ms=200))
        await controller.start()
        workflow, node = _workflow("service-process-exit", inference_twice_task)
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id="service_process_exit",
        )
        assert outcome.run_id is not None

        for _ in range(500):
            instances = inference.model_instances()
            if instances and instances[0].actual_request_inflight == 1:
                break
            await asyncio.sleep(0.002)
        instance = inference.model_instances()[0]
        assert instance.actual_request_inflight == 1
        client_anchor_before = controller.anchors.resolve(
            run_id=outcome.run_id,
            compiled=controller.state.compiled(outcome.run_id),
            task_id=node.task_id,
        )
        adapter.crash_instance(instance.instance_id, instance.generation)
        affected = inference.report_process_exited(
            instance.instance_id,
            instance.generation,
            reason="model_instance_npu_oom",
        )
        assert len(affected) == 1

        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.FAILED
        attempt = terminal.task(node.task_id).attempts[0]
        assert attempt.error is not None
        assert attempt.error.error_code == "model_instance_failed"
        assert isinstance(controller.recorder, InMemoryRecorder)
        failure_event = next(
            event
            for event in controller.recorder.events(outcome.run_id)
            if event.event_type == "model_instance_unhealthy"
        )
        assert failure_event.route_lease_id == affected[0].route_lease_id
        assert failure_event.model_instance_id == instance.instance_id
        assert (
            failure_event.payload["instance_placement_lease_id"]
            == instance.placement_lease_id
        )
        assert inference.router.active_count() == 0
        assert inference.model_instances()[0].route_occupancy == 0
        assert inference.model_instances()[0].actual_request_inflight == 0
        client_anchor_after = controller.anchors.resolve(
            run_id=outcome.run_id,
            compiled=controller.state.compiled(outcome.run_id),
            task_id=node.task_id,
        )
        assert client_anchor_after == client_anchor_before
        assert client_anchor_after.effective.npu_mem_mb == 0
        assert instance.placement_lease_id is not None
        assert (
            controller.placement.lease_snapshot(instance.placement_lease_id)
            .lease.resources.npu_hbm_mb
            == spec.instance_hbm_mb
        )

        for _ in range(500):
            if inference.model_instances()[0].state is ModelInstanceState.STOPPED:
                break
            await asyncio.sleep(0.002)
        assert inference.model_instances()[0].state is ModelInstanceState.STOPPED
        assert inference.instances.port_leases.active_count() == 0
        assert controller.placement.active_lease_count() == 0

        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())
