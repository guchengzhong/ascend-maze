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
        adapter.set_plan(spec.model_id, FakeAdapterPlan())
        stopped = await inference.instances.stop_if_drained(
            instance.instance_id, instance.generation
        )
        assert stopped.state is ModelInstanceState.STOPPED
        assert placement.active_lease_count() == 0

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
        assert not inference.router.acquire(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash=None,
            dispatch_deadline_ms=inference.clock.monotonic_ms() + 1_000,
        ).lease
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
