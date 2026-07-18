from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time

import pytest

from ascend_maze.core.clock import ManualClock
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.inference import (
    ChatRequest,
    InMemoryPortLeaseManager,
    InferenceCallError,
    ModelInstanceState,
    ModelRouteLeaseStatus,
)
from ascend_maze.inference.adapters.fake import (
    FakeAdapterPlan,
    FakeInferenceEngineAdapter,
)
from ascend_maze.inference.context import install_route_session
from ascend_maze.inference.client import chat, get_route_context
from inference_helpers import make_inference, make_node, make_spec


def test_public_chat_requires_attempt_context() -> None:
    with pytest.raises(InferenceCallError) as missing:
        chat([{"role": "user", "content": "outside"}])
    assert missing.value.error_code == "model_route_context_missing"
    with pytest.raises(InferenceCallError) as missing_context:
        get_route_context()
    assert missing_context.value.error_code == "model_route_context_missing"


def test_instance_route_and_sequential_request_lifecycle(tmp_path) -> None:
    async def scenario() -> None:
        clock = ManualClock(monotonic_ms=100, wall_ms=1_000)
        spec = make_spec(
            tmp_path / "model",
            scale_down_idle_ms=0,
            scale_cooldown_ms=0,
        )
        inference, placement, adapter = make_inference(spec, clock=clock)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()

        instance = inference.model_instances()[0]
        assert instance.state is ModelInstanceState.READY
        assert inference.instances.ready_instances(
            spec.model_id, spec.catalog_revision
        ) == (instance,)
        assert instance.route_occupancy == 0
        assert instance.actual_request_inflight == 0
        assert placement.active_lease_count() == 1
        assert adapter.launch_count == 1
        assert [
            event.event_type
            for event in inference.events()
            if event.event_type.startswith("model_instance_")
        ] == [
            "model_instance_requested",
            "model_instance_reserving",
            "model_instance_starting",
            "model_instance_warming",
            "model_instance_ready",
        ]
        ready_event = next(
            event
            for event in inference.events()
            if event.event_type == "model_instance_ready"
        )
        assert ready_event.payload["placement_lease_id"] == instance.placement_lease_id
        assert ready_event.payload["port_lease_id"] is not None
        assert ready_event.payload["port"] == 25_000
        assert ready_event.payload["npu_device_id"] == instance.npu_device_id
        assert ready_event.payload["warmup_response_digest"] == "fake-warmup-response"
        assert ready_event.payload["process_hbm_mb"] == spec.weight_hbm_mb
        assert ready_event.payload["request_capacity"] == spec.request_capacity

        acquired = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="session_hash",
            dispatch_deadline_ms=200,
        )
        assert acquired.lease is not None
        route = acquired.lease
        assert inference.model_instances()[0].route_occupancy == 1
        reserved_event = next(
            event
            for event in reversed(inference.events())
            if event.event_type == "model_route_reserved"
        )
        assert reserved_event.route_lease_id == route.route_lease_id
        assert reserved_event.payload["affinity_hit"] is False
        assert reserved_event.payload["acquire_duration_ms"] >= 0
        assert reserved_event.payload["route_occupancy"] == 1
        assert reserved_event.payload["request_capacity"] == spec.request_capacity
        assert inference.activate_route(route.route_lease_id)
        assert not inference.activate_route(route.route_lease_id)

        request = ChatRequest.create(
            [{"role": "user", "content": "hello"}], max_tokens=8
        )
        session = inference.create_attempt_session(route)

        def invoke_twice() -> tuple[str, str]:
            with install_route_session(session):
                first = session.invoke(request)
                second = session.invoke(request)
                return first.text, second.text

        first_text, second_text = await asyncio.to_thread(invoke_twice)
        assert (first_text, second_text) == ("model_a:hello", "model_a:hello")
        assert [record.call_index for record in inference.request_records()] == [1, 2]
        assert session.summary().request_count == 2
        assert session.summary().context_cleared
        assert inference.model_instances()[0].actual_request_inflight == 0

        with pytest.raises(StateTransitionError, match="stale"):
            inference.router.release(
                route.route_lease_id,
                run_id=route.run_id,
                task_id=route.task_id,
                attempt=route.attempt,
                instance_generation=route.instance_generation + 1,
                reason="stale",
            )
        assert await inference.release_route(route, reason="succeeded")
        await inference.replicas.wait_for_background()
        assert (
            inference.route_snapshot(route.route_lease_id).status
            is ModelRouteLeaseStatus.RELEASED
        )
        assert inference.model_instances()[0].state is ModelInstanceState.STOPPED
        assert placement.active_lease_count() == 0
        assert adapter.stop_count == 1

    asyncio.run(scenario())


def test_route_capacity_expiry_affinity_and_attempt_idempotency(tmp_path) -> None:
    async def scenario() -> None:
        clock = ManualClock(monotonic_ms=10, wall_ms=10)
        spec = make_spec(tmp_path / "model", scale_cooldown_ms=100_000)
        inference, placement, _ = make_inference(spec, clock=clock)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        inference.register_demand(
            run_id="run_1", task_id="task_2", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()

        first = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="same",
            dispatch_deadline_ms=20,
        )
        assert first.lease is not None
        duplicate = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="same",
            dispatch_deadline_ms=20,
        )
        assert duplicate.lease is first.lease
        blocked = await inference.acquire_route(
            run_id="run_1",
            task_id="task_2",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="same",
            dispatch_deadline_ms=20,
        )
        assert blocked.lease is None
        assert blocked.rejection_reason == "model_route_unavailable"

        clock.advance(10)
        assert inference.router.expire_reserved() == (first.lease,)
        assert (
            inference.route_snapshot(first.lease.route_lease_id).status
            is ModelRouteLeaseStatus.EXPIRED
        )
        replacement = await inference.acquire_route(
            run_id="run_1",
            task_id="task_2",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="same",
            dispatch_deadline_ms=40,
        )
        assert replacement.lease is not None
        assert replacement.affinity_hit
        assert inference.abandon_route(replacement.lease, reason="placement_failed")
        reacquired = await inference.acquire_route(
            run_id="run_1",
            task_id="task_2",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="same",
            dispatch_deadline_ms=40,
        )
        assert reacquired.lease is not None
        assert reacquired.lease.route_lease_id != replacement.lease.route_lease_id
        assert await inference.release_route(reacquired.lease, reason="done")
        inference.replicas.remove_run("run_1")
        await inference.close()

    asyncio.run(scenario())


def test_route_release_remains_successful_when_reconcile_fails(
    tmp_path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", scale_cooldown_ms=100_000)
        inference, placement, _ = make_inference(spec)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        acquired = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash=None,
            dispatch_deadline_ms=inference.clock.monotonic_ms() + 1_000,
        )
        assert acquired.lease is not None

        async def fail_reconcile(model_id: str) -> None:
            del model_id
            raise RuntimeError("injected reconcile failure")

        monkeypatch.setattr(inference.replicas, "_reconcile_model", fail_reconcile)
        assert await inference.release_route(acquired.lease, reason="done")
        assert (
            inference.route_snapshot(acquired.lease.route_lease_id).status
            is ModelRouteLeaseStatus.RELEASED
        )
        failure = next(
            event
            for event in inference.events()
            if event.event_type == "model_reconcile_failed"
        )
        assert failure.route_lease_id == acquired.lease.route_lease_id
        assert failure.payload["message"] == "injected reconcile failure"
        await inference.close()
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_concurrent_chat_is_rejected_without_inflight_leak(tmp_path) -> None:
    spec = make_spec(tmp_path / "model", scale_cooldown_ms=100_000)
    adapter = FakeInferenceEngineAdapter()
    adapter.set_plan(spec.model_id, FakeAdapterPlan(invoke_delay_ms=80))
    inference, placement, _ = make_inference(spec, adapter=adapter)
    placement.register_node(make_node())

    async def prepare():
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        result = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash=None,
            dispatch_deadline_ms=inference.clock.monotonic_ms() + 1_000,
        )
        assert result.lease is not None
        assert inference.activate_route(result.lease.route_lease_id)
        return result.lease

    route = asyncio.run(prepare())
    session = inference.create_attempt_session(route)
    request = ChatRequest.create([{"role": "user", "content": "hello"}])

    def first_call():
        with install_route_session(session):
            return session.invoke(request)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first_call)
        deadline = time.monotonic() + 1
        while (
            inference.model_instances()[0].actual_request_inflight != 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert inference.model_instances()[0].route_occupancy == 1
        assert inference.model_instances()[0].actual_request_inflight == 1
        with pytest.raises(
            InferenceCallError,
            match="cannot execute concurrent",
        ) as caught:
            session.invoke(request)
        assert caught.value.error_code == "model_route_concurrent_call_forbidden"
        assert future.result(timeout=1).text == "model_a:hello"

    assert inference.model_instances()[0].actual_request_inflight == 0
    assert session.summary().request_count == 1
    assert [record.call_index for record in inference.request_records()] == [1]
    asyncio.run(inference.release_route(route, reason="done"))
    asyncio.run(inference.close())


@pytest.mark.parametrize(
    "plan, expected_fragment",
    [
        (FakeAdapterPlan(wrong_model_id="wrong"), "identity or capacity"),
        (FakeAdapterPlan(wrong_device_id="7"), "identity or capacity"),
        (FakeAdapterPlan(process_hbm_mb=2_000), "outside its Lease"),
        (FakeAdapterPlan(fail_warmup="warmup failed"), "warmup failed"),
    ],
)
def test_instance_never_becomes_ready_when_probe_or_warmup_is_invalid(
    tmp_path, plan: FakeAdapterPlan, expected_fragment: str
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / expected_fragment.replace(" ", "_"))
        adapter = FakeInferenceEngineAdapter()
        adapter.set_plan(spec.model_id, plan)
        inference, placement, _ = make_inference(spec, adapter=adapter)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()

        instance = inference.model_instances()[0]
        assert instance.state is ModelInstanceState.STOPPED
        assert instance.ready_at_ms is None
        assert instance.failure_reason is not None
        assert expected_fragment in instance.failure_reason
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "plan, expected_fragment",
    [
        (
            FakeAdapterPlan(fail_build_launch="launch request failed"),
            "launch request failed",
        ),
        (FakeAdapterPlan(fail_attach="attach failed"), "attach failed"),
    ],
)
def test_start_transaction_rolls_back_request_and_attach_failures(
    tmp_path,
    plan: FakeAdapterPlan,
    expected_fragment: str,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / expected_fragment.replace(" ", "_"))
        adapter = FakeInferenceEngineAdapter()
        adapter.set_plan(spec.model_id, plan)
        inference, placement, _ = make_inference(spec, adapter=adapter)
        placement.register_node(make_node())
        requested = inference.instances.create_requested(spec.model_id)

        stopped = await inference.instances.start_instance(requested.instance_id)

        assert stopped.state is ModelInstanceState.STOPPED
        assert stopped.failure_reason is not None
        assert expected_fragment in stopped.failure_reason
        assert stopped.placement_lease_id is None
        assert stopped.service_handle_id is None
        assert inference.instances.port_leases.active_count() == 0
        assert placement.active_lease_count() == 0
        if plan.fail_attach is not None:
            assert adapter.launch_count == 1
            assert adapter.stop_count == 1
        else:
            assert adapter.launch_count == 0
            assert adapter.stop_count == 0

    asyncio.run(scenario())


def test_start_transaction_rolls_back_placement_exception(
    tmp_path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model")
        inference, placement, adapter = make_inference(spec)
        placement.register_node(make_node())

        def fail_placement(**kwargs):
            del kwargs
            raise RuntimeError("injected placement failure")

        monkeypatch.setattr(
            placement,
            "reserve_model_instance",
            fail_placement,
        )
        requested = inference.instances.create_requested(spec.model_id)
        stopped = await inference.instances.start_instance(requested.instance_id)

        assert stopped.state is ModelInstanceState.STOPPED
        assert stopped.failure_reason is not None
        assert "injected placement failure" in stopped.failure_reason
        assert placement.active_lease_count() == 0
        assert inference.instances.port_leases.active_count() == 0
        assert adapter.launch_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "wrong_field",
    [
        "instance_id",
        "generation",
        "endpoint_id",
        "node_id",
        "boot_id",
        "npu_device_id",
    ],
)
def test_start_transaction_rejects_mismatched_service_handle(
    tmp_path,
    wrong_field: str,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / wrong_field)
        adapter = FakeInferenceEngineAdapter()
        adapter.set_plan(
            spec.model_id,
            FakeAdapterPlan(wrong_service_handle_field=wrong_field),
        )
        inference, placement, _ = make_inference(spec, adapter=adapter)
        placement.register_node(make_node())
        requested = inference.instances.create_requested(spec.model_id)

        stopped = await inference.instances.start_instance(requested.instance_id)

        assert stopped.state is ModelInstanceState.STOPPED
        assert stopped.failure_reason is not None
        assert "ServiceHandle identity" in stopped.failure_reason
        assert stopped.placement_lease_id is None
        assert stopped.service_handle_id is None
        assert inference.instances.port_leases.active_count() == 0
        assert placement.active_lease_count() == 0
        assert adapter.launch_count == 1
        assert adapter.stop_count == 1

    asyncio.run(scenario())


def test_port_lease_is_node_scoped_atomic_and_released_after_stop(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", max_replicas=2)
        ports = InMemoryPortLeaseManager(first_port=25_000, last_port=25_000)
        inference, placement, _ = make_inference(spec, port_leases=ports)
        placement.register_node(make_node(npu_count=2))
        first = inference.instances.create_requested(spec.model_id)
        second = inference.instances.create_requested(spec.model_id)

        results = await asyncio.gather(
            inference.instances.start_instance(first.instance_id),
            inference.instances.start_instance(second.instance_id),
        )

        assert sorted(item.state.value for item in results) == ["ready", "stopped"]
        failed = next(item for item in results if item.state is ModelInstanceState.STOPPED)
        assert failed.failure_reason is not None
        assert "no service port is available" in failed.failure_reason
        assert ports.active_count() == 1
        assert len(ports.leases()) == 1
        assert placement.active_lease_count() == 1

        await inference.close()
        assert ports.active_count() == 0
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_starting_is_not_ready_or_duplicated_and_stop_barrier_keeps_lease(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(
            tmp_path / "model",
            scale_down_idle_ms=0,
            scale_cooldown_ms=0,
        )
        adapter = FakeInferenceEngineAdapter()
        adapter.set_plan(spec.model_id, FakeAdapterPlan(launch_delay_ms=50))
        inference, placement, _ = make_inference(spec, adapter=adapter)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await asyncio.sleep(0.01)
        assert inference.model_instances()[0].state is ModelInstanceState.STARTING
        await inference.reconcile()
        assert len(inference.model_instances()) == 1
        assert adapter.launch_count == 0
        await inference.replicas.wait_for_background()
        instance = inference.model_instances()[0]
        assert instance.state is ModelInstanceState.READY
        assert adapter.launch_count == 1

        adapter.set_plan(
            spec.model_id,
            FakeAdapterPlan(stop_hbm_recovered=False),
        )
        assert inference.instances.begin_drain(
            instance.instance_id, instance.generation
        )
        blocked = await inference.instances.stop_if_drained(
            instance.instance_id, instance.generation
        )
        assert blocked.state is ModelInstanceState.FAILED
        assert placement.active_lease_count() == 1
        assert inference.instances.port_leases.active_count() == 1
        adapter.set_plan(spec.model_id, FakeAdapterPlan())
        stopped = await inference.instances.stop_if_drained(
            instance.instance_id, instance.generation
        )
        assert stopped.state is ModelInstanceState.STOPPED
        assert placement.active_lease_count() == 0
        assert inference.instances.port_leases.active_count() == 0

    asyncio.run(scenario())


def test_startup_timeout_cancels_fake_launch_and_releases_instance_lease(
    tmp_path,
) -> None:
    async def scenario() -> None:
        spec = replace(
            make_spec(tmp_path / "model"),
            startup_timeout_ms=10,
        )
        adapter = FakeInferenceEngineAdapter()
        adapter.set_plan(spec.model_id, FakeAdapterPlan(launch_delay_ms=50))
        inference, placement, _ = make_inference(spec, adapter=adapter)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()

        instance = inference.model_instances()[0]
        assert instance.state is ModelInstanceState.STOPPED
        assert instance.ready_at_ms is None
        assert instance.failure_reason is not None
        assert "TimeoutError" in instance.failure_reason
        assert adapter.launch_count == 0
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_replica_controller_scales_in_bounded_steps_without_counting_starting_ready(
    tmp_path,
) -> None:
    async def scenario() -> None:
        spec = make_spec(
            tmp_path / "model",
            max_replicas=2,
            scale_cooldown_ms=0,
        )
        adapter = FakeInferenceEngineAdapter()
        adapter.set_plan(spec.model_id, FakeAdapterPlan(launch_delay_ms=40))
        inference, placement, _ = make_inference(spec, adapter=adapter)
        placement.register_node(make_node(npu_count=2))
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        inference.register_demand(
            run_id="run_1", task_id="task_2", model_id=spec.model_id
        )

        await inference.reconcile()
        await asyncio.sleep(0.01)
        first_wave = inference.model_instances()
        assert len(first_wave) == 1
        assert first_wave[0].state is ModelInstanceState.STARTING
        rejected = inference.router.acquire(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash=None,
            dispatch_deadline_ms=inference.clock.monotonic_ms() + 1_000,
        )
        assert rejected.lease is None
        assert rejected.rejection_reason == "model_route_unavailable"
        rejection_event = next(
            event
            for event in reversed(inference.events())
            if event.event_type == "model_route_rejected"
        )
        assert rejection_event.payload["reason"] == "model_route_unavailable"
        assert rejection_event.payload["acquire_duration_ms"] >= 0
        first_scale = next(
            event
            for event in inference.events()
            if event.event_type == "model_scale_up"
        )
        assert first_scale.payload["reason"] == "pending_or_utilization"
        assert first_scale.payload["target_replicas"] == 2
        assert first_scale.payload["actual_replicas"] == 0
        assert first_scale.payload["started"] == 1
        assert first_scale.payload["pending_demand"] == 2
        assert first_scale.payload["route_occupancy"] == 0
        assert first_scale.payload["cooldown_until_ms"] >= 0
        await inference.reconcile()
        assert len(inference.model_instances()) == 1
        await inference.replicas.wait_for_background()

        await inference.reconcile()
        await asyncio.sleep(0.01)
        states = [instance.state for instance in inference.model_instances()]
        assert states.count(ModelInstanceState.READY) == 1
        assert states.count(ModelInstanceState.STARTING) == 1
        await inference.replicas.wait_for_background()
        assert [
            instance.state for instance in inference.model_instances()
        ] == [ModelInstanceState.READY, ModelInstanceState.READY]
        assert adapter.launch_count == 2
        await inference.close()
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_periodic_reconcile_advances_sustain_without_new_events(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(
            tmp_path / "model",
            scale_up_sustain_ms=30,
            scale_cooldown_ms=0,
        )
        inference, placement, _ = make_inference(
            spec,
            reconcile_interval_ms=5,
        )
        placement.register_node(make_node())
        await inference.start()
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )

        await asyncio.sleep(0.015)
        assert inference.model_instances() == ()
        deadline = time.monotonic() + 1
        while not inference.model_instances() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert inference.model_instances()[0].state is ModelInstanceState.READY

        inference.replicas.remove_run("run_1")
        await inference.close()
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_periodic_reconcile_retries_after_start_cooldown(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(
            tmp_path / "model",
            scale_cooldown_ms=40,
        )
        adapter = FakeInferenceEngineAdapter()
        adapter.set_plan(spec.model_id, FakeAdapterPlan(fail_launch="first failed"))
        inference, placement, _ = make_inference(
            spec,
            adapter=adapter,
            reconcile_interval_ms=5,
        )
        placement.register_node(make_node())
        await inference.start()
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )

        deadline = time.monotonic() + 1
        while (
            not inference.model_instances()
            or inference.model_instances()[-1].state is not ModelInstanceState.STOPPED
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert inference.model_instances()[-1].state is ModelInstanceState.STOPPED
        adapter.set_plan(spec.model_id, FakeAdapterPlan())

        while not any(
            item.state is ModelInstanceState.READY
            for item in inference.model_instances()
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert any(
            item.state is ModelInstanceState.READY
            for item in inference.model_instances()
        )
        assert len(inference.model_instances()) == 1
        assert inference.model_instances()[0].generation == 2
        assert adapter.launch_count == 1
        assert inference.instances.port_leases.active_count() == 1
        assert any(
            event.event_type == "model_instance_restarted"
            and event.instance_generation == 2
            and event.payload["previous_generation"] == 1
            for event in inference.events()
        )

        inference.replicas.remove_run("run_1")
        await inference.close()
        assert inference.instances.port_leases.active_count() == 0
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_periodic_reconcile_continues_after_one_runner_failure(
    tmp_path, monkeypatch
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", scale_cooldown_ms=0)
        inference, placement, _ = make_inference(
            spec,
            reconcile_interval_ms=5,
        )
        placement.register_node(make_node())
        await inference.start()
        original = inference.replicas._reconcile_model
        calls = 0

        async def fail_once(model_id: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient reconcile failure")
            await original(model_id)

        monkeypatch.setattr(inference.replicas, "_reconcile_model", fail_once)
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )

        deadline = time.monotonic() + 1
        while not any(
            item.state is ModelInstanceState.READY
            for item in inference.model_instances()
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)

        assert calls >= 2
        assert any(
            event.event_type == "model_reconcile_failed"
            and event.payload["exception_type"] == "RuntimeError"
            and event.payload["message"] == "transient reconcile failure"
            for event in inference.events()
        )
        assert inference.model_instances()[0].state is ModelInstanceState.READY

        inference.replicas.remove_run("run_1")
        await inference.close()
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_periodic_reconcile_does_not_hot_retry_pending_placement(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", scale_cooldown_ms=0)
        inference, placement, _ = make_inference(
            spec,
            reconcile_interval_ms=10_000,
        )
        await inference.start()
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )

        deadline = time.monotonic() + 1
        while not any(
            event.event_type == "model_placement_pending"
            for event in inference.events()
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        first_count = sum(
            event.event_type == "model_placement_pending"
            for event in inference.events()
        )
        assert first_count == 1
        await asyncio.sleep(0.02)
        assert sum(
            event.event_type == "model_placement_pending"
            for event in inference.events()
        ) == first_count

        inference.replicas.remove_run("run_1")
        await inference.close()
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_periodic_reconcile_does_not_hot_retry_blocked_cleanup(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(
            tmp_path / "model",
            scale_down_idle_ms=0,
            scale_cooldown_ms=0,
        )
        adapter = FakeInferenceEngineAdapter()
        inference, placement, _ = make_inference(
            spec,
            adapter=adapter,
            reconcile_interval_ms=10_000,
        )
        placement.register_node(make_node())
        await inference.start()
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )

        deadline = time.monotonic() + 1
        while not any(
            item.state is ModelInstanceState.READY
            for item in inference.model_instances()
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        adapter.set_plan(
            spec.model_id,
            FakeAdapterPlan(stop_hbm_recovered=False),
        )
        assert inference.replicas.remove_run("run_1") == 1
        while not any(
            event.event_type == "model_resource_release_blocked"
            for event in inference.events()
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        first_count = sum(
            event.event_type == "model_resource_release_blocked"
            for event in inference.events()
        )
        assert first_count == 1
        await asyncio.sleep(0.02)
        assert sum(
            event.event_type == "model_resource_release_blocked"
            for event in inference.events()
        ) == first_count

        adapter.set_plan(spec.model_id, FakeAdapterPlan())
        await inference.close()
        assert inference.instances.port_leases.active_count() == 0
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_periodic_reconcile_stops_idle_replica_without_new_events(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(
            tmp_path / "model",
            scale_down_idle_ms=20,
            scale_cooldown_ms=0,
        )
        inference, placement, _ = make_inference(
            spec,
            reconcile_interval_ms=5,
        )
        placement.register_node(make_node())
        await inference.start()
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )

        deadline = time.monotonic() + 1
        while not any(
            item.state is ModelInstanceState.READY
            for item in inference.model_instances()
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert inference.replicas.remove_run("run_1") == 1

        while not any(
            item.state is ModelInstanceState.STOPPED
            for item in inference.model_instances()
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert inference.model_instances()[-1].state is ModelInstanceState.STOPPED
        assert inference.instances.port_leases.active_count() == 0
        assert placement.active_lease_count() == 0
        scale_down = next(
            event
            for event in inference.events()
            if event.event_type == "model_scale_down"
        )
        assert scale_down.payload["reason"] == "idle_capacity"
        assert scale_down.payload["target_replicas"] == 0
        assert scale_down.payload["actual_replicas"] == 1
        assert scale_down.payload["drained"] == 1
        assert scale_down.payload["cooldown_until_ms"] >= 0
        await inference.close()

    asyncio.run(scenario())


def test_scale_down_prefers_instance_without_affinity_mapping(tmp_path) -> None:
    async def scenario() -> None:
        clock = ManualClock(monotonic_ms=10, wall_ms=10)
        spec = make_spec(
            tmp_path / "model",
            request_capacity=1,
            max_replicas=2,
            scale_down_idle_ms=0,
            scale_cooldown_ms=0,
        )
        inference, placement, _ = make_inference(spec, clock=clock)
        placement.register_node(make_node(npu_count=2))
        for task_id in ("task_1", "task_2"):
            inference.register_demand(
                run_id="run_1", task_id=task_id, model_id=spec.model_id
            )
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        assert len(inference.instances.ready_instances(spec.model_id, "catalog_v1")) == 2

        acquired = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="sticky",
            dispatch_deadline_ms=1_000,
        )
        assert acquired.lease is not None
        sticky = acquired.lease
        assert inference.router.release(
            sticky.route_lease_id,
            run_id=sticky.run_id,
            task_id=sticky.task_id,
            attempt=sticky.attempt,
            instance_generation=sticky.instance_generation,
            reason="test",
        )
        other = next(
            item
            for item in inference.model_instances()
            if item.instance_id != sticky.instance_id
        )
        clock.advance(10)
        inference.instances.reserve_route(other.instance_id, other.generation)
        inference.instances.release_route(other.instance_id, other.generation)
        assert other.last_used_at_ms < inference.instances.snapshot(
            other.instance_id
        ).last_used_at_ms
        assert inference.replicas.remove_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )

        await inference.reconcile()
        await inference.replicas.wait_for_background()
        states = {
            item.instance_id: item.state for item in inference.model_instances()
        }
        assert states[sticky.instance_id] is ModelInstanceState.READY
        assert states[other.instance_id] is ModelInstanceState.STOPPED

        inference.replicas.remove_run("run_1")
        await inference.close()
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_draining_waits_for_route_occupancy_and_actual_request_inflight(
    tmp_path,
) -> None:
    spec = make_spec(tmp_path / "model", scale_cooldown_ms=100_000)
    adapter = FakeInferenceEngineAdapter()
    adapter.set_plan(spec.model_id, FakeAdapterPlan(invoke_delay_ms=60))
    inference, placement, _ = make_inference(spec, adapter=adapter)
    placement.register_node(make_node())

    async def prepare():
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        result = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash=None,
            dispatch_deadline_ms=inference.clock.monotonic_ms() + 1_000,
        )
        assert result.lease is not None
        assert inference.activate_route(result.lease.route_lease_id)
        return result.lease

    route = asyncio.run(prepare())
    session = inference.create_attempt_session(route)
    request = ChatRequest.create([{"role": "user", "content": "hold"}])

    def invoke():
        with install_route_session(session):
            return session.invoke(request)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(invoke)
        deadline = time.monotonic() + 1
        while (
            inference.model_instances()[0].actual_request_inflight != 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        instance = inference.model_instances()[0]
        assert inference.instances.begin_drain(
            instance.instance_id, instance.generation
        )
        blocked = asyncio.run(
            inference.instances.stop_if_drained(
                instance.instance_id, instance.generation
            )
        )
        assert blocked.state is ModelInstanceState.DRAINING
        assert placement.active_lease_count() == 1
        future.result(timeout=1)

    no_inflight = asyncio.run(
        inference.instances.stop_if_drained(
            instance.instance_id, instance.generation
        )
    )
    assert no_inflight.state is ModelInstanceState.DRAINING
    assert no_inflight.route_occupancy == 1
    assert inference.router.release(
        route.route_lease_id,
        run_id=route.run_id,
        task_id=route.task_id,
        attempt=route.attempt,
        instance_generation=route.instance_generation,
        reason="done",
    )
    stopped = asyncio.run(
        inference.instances.stop_if_drained(
            instance.instance_id, instance.generation
        )
    )
    assert stopped.state is ModelInstanceState.STOPPED
    assert placement.active_lease_count() == 0


def test_instance_invalidation_is_generation_exact_and_releases_route_capacity(
    tmp_path,
) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", scale_cooldown_ms=100_000)
        inference, placement, _ = make_inference(spec)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        result = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash="affinity",
            dispatch_deadline_ms=inference.clock.monotonic_ms() + 1_000,
        )
        assert result.lease is not None
        route = result.lease
        assert inference.router.invalidate_instance(
            route.instance_id,
            route.instance_generation + 1,
            reason="stale",
        ) == ()
        assert inference.model_instances()[0].route_occupancy == 1
        assert inference.router.invalidate_instance(
            route.instance_id,
            route.instance_generation,
            reason="process_exited",
        ) == (route,)
        assert (
            inference.route_snapshot(route.route_lease_id).status
            is ModelRouteLeaseStatus.INVALIDATED
        )
        assert inference.model_instances()[0].route_occupancy == 0
        inference.replicas.remove_run("run_1")
        await inference.close()

    asyncio.run(scenario())


def test_ready_process_failure_defers_inflight_route_release_and_reports_timeout(
    tmp_path,
) -> None:
    clock = ManualClock(monotonic_ms=10, wall_ms=10)
    spec = replace(
        make_spec(tmp_path / "model", scale_cooldown_ms=100_000),
        drain_timeout_ms=10,
    )
    adapter = FakeInferenceEngineAdapter()
    adapter.set_plan(spec.model_id, FakeAdapterPlan(invoke_delay_ms=80))
    inference, placement, _ = make_inference(spec, adapter=adapter, clock=clock)
    placement.register_node(make_node())

    async def prepare():
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        acquired = await inference.acquire_route(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash=None,
            dispatch_deadline_ms=1_000,
        )
        assert acquired.lease is not None
        assert inference.activate_route(acquired.lease.route_lease_id)
        return acquired.lease

    route = asyncio.run(prepare())
    session = inference.create_attempt_session(route)
    request = ChatRequest.create([{"role": "user", "content": "crash"}])

    def invoke():
        with install_route_session(session):
            return session.invoke(request)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(invoke)
        deadline = time.monotonic() + 1
        while (
            inference.model_instances()[0].actual_request_inflight != 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        instance = inference.model_instances()[0]
        adapter.crash_instance(instance.instance_id, instance.generation)
        affected = inference.report_instance_failure(
            instance.instance_id,
            instance.generation,
            reason="service_process_exited",
        )

        assert affected == (route,)
        assert inference.model_instances()[0].state is ModelInstanceState.FAILED
        assert (
            inference.route_snapshot(route.route_lease_id).status
            is ModelRouteLeaseStatus.ACTIVE
        )
        assert inference.model_instances()[0].route_occupancy == 1
        assert inference.model_instances()[0].actual_request_inflight == 1
        clock.advance(10)
        assert inference.instances.check_cleanup_timeout(
            instance.instance_id, instance.generation
        )
        assert not inference.instances.check_cleanup_timeout(
            instance.instance_id, instance.generation
        )
        with pytest.raises(InferenceCallError):
            future.result(timeout=1)

    assert (
        inference.route_snapshot(route.route_lease_id).status
        is ModelRouteLeaseStatus.INVALIDATED
    )
    assert inference.model_instances()[0].route_occupancy == 0
    assert inference.model_instances()[0].actual_request_inflight == 0
    assert len(
        [
            event
            for event in inference.events()
            if event.event_type == "model_drain_timed_out"
        ]
    ) == 1
    inference.replicas.remove_run("run_1")
    asyncio.run(inference.reconcile())
    asyncio.run(inference.replicas.wait_for_background())
    assert inference.model_instances()[0].state is ModelInstanceState.STOPPED
    assert inference.instances.port_leases.active_count() == 0
    assert placement.active_lease_count() == 0


def test_node_generation_loss_is_exact_and_cleans_ready_instance(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "model", scale_cooldown_ms=100_000)
        inference, placement, adapter = make_inference(spec)
        placement.register_node(make_node())
        inference.register_demand(
            run_id="run_1", task_id="task_1", model_id=spec.model_id
        )
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        instance = inference.model_instances()[0]
        assert inference.report_node_generation_lost(
            instance.node_id or "",
            "stale_boot",
        ) == ()
        assert inference.model_instances()[0].state is ModelInstanceState.READY

        adapter.crash_instance(instance.instance_id, instance.generation)
        assert inference.report_node_generation_lost(
            instance.node_id or "",
            instance.boot_id or "",
        ) == ()
        assert inference.model_instances()[0].state is ModelInstanceState.FAILED
        inference.replicas.remove_run("run_1")
        await inference.reconcile()
        await inference.replicas.wait_for_background()
        assert inference.model_instances()[0].state is ModelInstanceState.STOPPED
        assert inference.instances.port_leases.active_count() == 0
        assert placement.active_lease_count() == 0

        restarted = inference.instances.restart_stopped(instance.instance_id)
        assert restarted.generation == instance.generation + 1
        with pytest.raises(RuntimeError, match="generation is stale"):
            inference.report_process_exited(
                instance.instance_id,
                instance.generation,
                reason="late_old_process_event",
            )
        await inference.close()
        assert inference.model_instances()[0].state is ModelInstanceState.STOPPED

    asyncio.run(scenario())
