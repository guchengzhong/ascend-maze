from __future__ import annotations

import asyncio
from dataclasses import replace
import os
import time

import pytest
import ray

from ascend_maze import Workflow
from ascend_maze.ascend import (
    build_ascend_node_capacity,
    build_ascend_node_observation,
)
from ascend_maze.control.client import InMemoryRuntimeClient
from ascend_maze.control.node_rpc import NodeAgent, NodeAgentIdentity
from ascend_maze.control.node_rpc import report_worker_event
from ascend_maze.control.ray_controller import RayHostController
from ascend_maze.data.index import RunDataState
from ascend_maze.lifecycle import AttemptStatus, RunStatus
from ascend_maze.core.errors import DataHandleInvalidError
from ascend_maze.placement import NpuCapacity, PlacementManager
from ascend_maze.resources import DeclaredOnlyAnchorProvider
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from tests_ascend.conftest import AscendAdmission
from tests_ascend.task_fixtures import (
    cpu_visible_device,
    impossible_multi_card_npu,
    npu_add,
    npu_long_running,
    npu_oom,
    npu_sync_device_error,
    npu_tensor_output,
    npu_timeout,
    npu_user_error,
)


async def _start_controller(
    admission: AscendAdmission,
    ray_namespace: str,
    *,
    device_id: str | None = None,
    use_all_devices: bool = False,
) -> tuple[RayHostController, NodeAgent]:
    environment = admission.environment
    capacity = build_ascend_node_capacity(
        node_id="node_ascend",
        boot_id="boot_stage4",
        node_ip="127.0.0.1",
        adapter=admission.adapter,
        environment=environment,
        config=admission.config,
    )
    selected_id = admission.device.physical_device_id if device_id is None else device_id
    if use_all_devices:
        selected_ids = frozenset(item.device_id for item in capacity.npus)
    elif device_id is None:
        selected = next(item for item in capacity.npus if item.device_id == selected_id)
        selected_ids = frozenset((selected.device_id,))
    else:
        selected = NpuCapacity(
            device_id=device_id,
            chip_type=admission.device.chip_type,
            total_hbm_mb=admission.device.total_hbm_mb,
            system_reserved_hbm_mb=admission.config.npu_system_reserved_hbm_mb,
            task_slots_total=1,
            observed_free_hbm_mb=admission.device.free_hbm_mb,
        )
        selected_ids = frozenset((selected.device_id,))
    if not use_all_devices:
        capacity = replace(capacity, npus=(selected,))
    placement = PlacementManager(
        host_mem_headroom_mb=admission.config.host_mem_headroom_mb,
        npu_hbm_headroom_mb=admission.config.npu_hbm_headroom_mb,
        required_environment_fingerprint=environment.environment_fingerprint,
    )
    anchors = DeclaredOnlyAnchorProvider(
        environment_fingerprint=environment.environment_fingerprint
    )
    controller = RayHostController(
        cluster_id="cluster_stage4",
        authorization_token=b"stage4-token",
        ray_namespace=ray_namespace,
        config_fingerprint=admission.config_snapshot.config_fingerprint,
        environment_fingerprint=environment.environment_fingerprint,
        build_revision="stage4-test",
        node_capacities=(capacity,),
        anchors=anchors,
        placement=placement,
        dispatch_timeout_ms=admission.config.worker_binding_deadline_ms,
    )
    await controller.start()
    identity = NodeAgentIdentity(
        cluster_id="cluster_stage4",
        node_id=capacity.node_id,
        boot_id=capacity.boot_id,
        ray_node_id=ray.get_runtime_context().get_node_id(),
        agent_generation="agent_stage4",
        environment_fingerprint=environment.environment_fingerprint,
        producer_id="node_agent:stage4",
    )

    def observation_provider(sequence: int, received_at_ms: int):
        observation = build_ascend_node_observation(
            node_id=capacity.node_id,
            boot_id=capacity.boot_id,
            sequence=sequence,
            received_at_ms=received_at_ms,
            adapter=admission.adapter,
        )
        return replace(
            observation,
            npus=tuple(
                item for item in observation.npus if item.device_id in selected_ids
            ),
        )

    agent = NodeAgent(
        identity=identity,
        authorization_token=b"stage4-token",
        heartbeat_interval_ms=50,
        worker_device_verifier=lambda pid, physical: admission.adapter.verify_process_device(
            pid, physical, deadline_seconds=2
        ),
        node_observation_provider=observation_provider,
    )
    try:
        await agent.start(controller_endpoint=controller.node_rpc_endpoint)
    except Exception:
        await controller.close()
        raise
    return controller, agent


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _worker_pid(controller: RayHostController, dispatch_id: str) -> int:
    event = controller.ray_runtime.worker_started_event(dispatch_id)
    assert event is not None
    assert event.worker_pid is not None
    return event.worker_pid


async def _wait_hbm_recovery(
    admission: AscendAdmission,
    baseline_used_mb: int,
    *,
    worker_pids: tuple[int, ...] = (),
) -> None:
    deadline = time.monotonic() + admission.config.hbm_recovery_deadline_ms / 1_000
    last = None
    while time.monotonic() < deadline:
        last = admission.adapter.device(admission.device.physical_device_id)
        pids = {item.pid for item in last.processes}
        live_worker_pids = {pid for pid in worker_pids if _pid_exists(pid)}
        if (
            last.used_hbm_mb
            <= baseline_used_mb + admission.config.hbm_recovery_tolerance_mb
            and not pids.intersection(worker_pids)
            and not live_worker_pids
        ):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(
        "NPU Worker/HBM did not recover: "
        f"baseline={baseline_used_mb}, last={last}, "
        f"live_worker_pids={[pid for pid in worker_pids if _pid_exists(pid)]}"
    )


def _assert_run_terminal_checkpoint(
    controller: RayHostController,
    run_id: str,
    *,
    expected_handle_count: int,
) -> None:
    assert controller.placement.active_lease_count(run_id) == 0
    assert controller.worker_broker.active_count() == 0
    assert controller.deadlines.count_for_run(run_id) == 0
    assert controller.ray_runtime.active_dispatch_count(run_id) == 0
    assert controller.ray_data_store.staged_count == 0
    index = controller.indexes.get(run_id)
    assert index.state is RunDataState.ACTIVE
    assert index.handle_count() == expected_handle_count


async def _destroy_and_assert(
    controller: RayHostController,
    run_id: str,
    *,
    force: bool = False,
) -> None:
    handle_count = controller.indexes.get(run_id).handle_count()
    destroyed = await controller.destroy_run(run_id, force=force)
    repeated = await controller.destroy_run(run_id, force=force)
    assert repeated == destroyed
    assert destroyed.tombstone.destroy_succeeded
    assert destroyed.tombstone.released_handle_count == handle_count
    assert destroyed.code_handles_released > 0
    assert controller.indexes.get(run_id).state is RunDataState.DESTROYED
    assert controller.placement.active_lease_count(run_id) == 0
    assert controller.placement.lease_record_count(run_id) == 0
    assert controller.deadlines.count_for_run(run_id) == 0
    assert controller.anchors.count_for_run(run_id) == 0
    assert controller.ray_runtime.active_dispatch_count(run_id) == 0
    assert controller.ray_data_store.staged_count == 0


def test_real_dcmi_inventory_and_frozen_environment(
    ascend_admission: AscendAdmission,
) -> None:
    admission = ascend_admission
    devices = admission.adapter.devices()
    assert len(devices) >= 1
    assert all(item.health == "healthy" for item in devices)
    assert admission.device.chip_type == "910B3"
    assert admission.device.total_hbm_mb == 65_536
    assert admission.environment.chip_types == ("910B3",)
    assert admission.config.task_slots_total == 1
    assert not admission.config.allow_colocation
    assert admission.config.max_tasks_per_worker == 1
    assert admission.config.standby_min_idle == 0


def test_real_npu_worker_binding_result_and_cpu_isolation(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        baseline = admission.adapter.device(admission.device.physical_device_id).used_hbm_mb
        controller, agent = await _start_controller(admission, ascend_ray)
        try:
            assert controller.core.dispatch_timeout_ms == 30_000
            assert controller.core.policy.name == "fcfs"
            assert controller.anchors.strategy == "declared_only"
            assert controller.config_fingerprint == (
                admission.config_snapshot.config_fingerprint
            )
            workflow = Workflow("stage4-real-success")
            npu_node = workflow.add_task(npu_add, inputs={"megabytes": 64})
            cpu_node = workflow.add_task(cpu_visible_device, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_real_success",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=60)
            assert terminal.status is RunStatus.SUCCEEDED, [
                (
                    "npu" if task.task_id == npu_node.task_id else "cpu",
                    task.status.value,
                    None if task.last_error is None else task.last_error.error_code,
                    None if task.last_error is None else task.last_error.message,
                    tuple(
                        (
                            item.status.value,
                            None if item.error is None else item.error.error_code,
                            None if item.error is None else item.error.message,
                        )
                        for item in task.attempts
                    ),
                )
                for task in terminal.task_states
            ]
            assert controller.result(outcome.run_id, npu_node.task_id) == {
                "result": 1024
            }
            assert controller.result(outcome.run_id, cpu_node.task_id) == {
                "visible": None
            }
            npu_attempt = terminal.task(npu_node.task_id).attempts[0]
            npu_lease = controller.placement.lease_snapshot(
                npu_attempt.lease_id
            ).lease
            assert npu_lease.npu_device_id == admission.device.physical_device_id
            assert npu_lease.resources.npu_hbm_mb == 1_024
            assert npu_lease.resources.npu_slots == 1
            worker = controller.ray_runtime.worker_outcome(npu_attempt.dispatch_id)
            assert worker is not None
            assert worker.ray_node_id == ray.get_runtime_context().get_node_id()
            assert worker.physical_device_id == admission.device.physical_device_id
            assert worker.binding_verified
            observation = worker.terminal_event.resource_observation
            assert observation is not None
            assert observation.binding_verified
            assert observation.device_id == admission.device.physical_device_id
            assert observation.peak_npu_process_hbm_mb is not None
            assert observation.peak_npu_process_hbm_mb > 0
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            assert controller.worker_broker.active_count() == 0
            placement_node = controller.placement.snapshot().nodes[0]
            assert placement_node.observation_sequence > 0
            assert placement_node.capacity.npus[0].observed_free_hbm_mb is not None
            await _wait_hbm_recovery(
                admission, baseline, worker_pids=(worker.worker_pid,)
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=2,
            )
            await _destroy_and_assert(controller, outcome.run_id)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("task_func", "error_code"),
    [
        (npu_tensor_output, "invalid_task_output"),
        (npu_sync_device_error, "npu_async_error"),
        (npu_user_error, "user_code_failed"),
    ],
)
def test_real_npu_failures_are_structured(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
    task_func,
    error_code: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        baseline = admission.adapter.device(admission.device.physical_device_id).used_hbm_mb
        controller, agent = await _start_controller(admission, ascend_ray)
        try:
            workflow = Workflow(f"stage4-{error_code}")
            node = workflow.add_task(task_func, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id=f"stage4_{error_code}",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=60)
            assert terminal.status is RunStatus.FAILED
            task = terminal.task(node.task_id)
            assert task.last_error is not None
            assert task.last_error.error_code == error_code
            if error_code == "npu_async_error":
                assert task.last_error.execution_phase == "npu_synchronize"
                assert task.last_error.platform_error_code == "107001"
            elif error_code == "user_code_failed":
                assert task.last_error.execution_phase == "user_code"
                assert task.last_error.exception_type == "RuntimeError"
                assert task.last_error.platform_error_code is None
            attempt = task.attempts[0]
            worker = controller.ray_runtime.worker_outcome(attempt.dispatch_id)
            assert worker is not None
            await _wait_hbm_recovery(
                admission, baseline, worker_pids=(worker.worker_pid,)
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=0,
            )
            await _destroy_and_assert(controller, outcome.run_id)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_wrong_physical_device_binding_is_rejected_without_fallback(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        baseline = admission.adapter.device(admission.device.physical_device_id).used_hbm_mb
        controller, agent = await _start_controller(
            admission, ascend_ray, device_id="999"
        )
        try:
            workflow = Workflow("stage4-invalid-device")
            node = workflow.add_task(npu_add, inputs={"megabytes": 1})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_invalid_device",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=60)
            assert terminal.status is RunStatus.FAILED
            task = terminal.task(node.task_id)
            assert task.last_error is not None
            assert task.last_error.error_code == "device_bind_failed"
            assert task.attempt_count == 1
            assert task.attempts[0].device_ids == ("999",)
            worker = controller.ray_runtime.worker_outcome(
                task.attempts[0].dispatch_id
            )
            assert worker is not None
            assert worker.physical_device_id == "999"
            assert not worker.binding_verified
            await _wait_hbm_recovery(
                admission,
                baseline,
                worker_pids=(worker.worker_pid,),
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=0,
            )
            await _destroy_and_assert(controller, outcome.run_id)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_multi_card_hbm_is_not_aggregated_for_one_task(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        controller, agent = await _start_controller(
            admission,
            ascend_ray,
            use_all_devices=True,
        )
        try:
            placement_node = controller.placement.snapshot().nodes[0]
            assert len(placement_node.capacity.npus) == 8
            assert sum(
                item.total_hbm_mb for item in placement_node.capacity.npus
            ) > 70_000

            workflow = Workflow("stage4-multi-card-not-aggregated")
            node = workflow.add_task(impossible_multi_card_npu, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_multi_card_not_aggregated",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=10)
            task = terminal.task(node.task_id)
            assert terminal.status is RunStatus.FAILED
            assert task.attempt_count == 0
            assert task.last_error is not None
            assert task.last_error.error_code == "resource_request_unsatisfiable"
            assert all(
                controller.ray_runtime.worker_started_event(attempt.dispatch_id)
                is None
                for attempt in task.attempts
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=0,
            )
            await _destroy_and_assert(controller, outcome.run_id)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_real_npu_timeout_kills_worker_and_recovers_hbm(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        baseline = admission.adapter.device(admission.device.physical_device_id).used_hbm_mb
        controller, agent = await _start_controller(admission, ascend_ray)
        try:
            workflow = Workflow("stage4-timeout")
            node = workflow.add_task(npu_timeout, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_timeout",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=60)
            task = terminal.task(node.task_id)
            assert terminal.status is RunStatus.FAILED
            assert task.last_error is not None
            assert task.last_error.error_code == "task_timeout"
            assert task.attempts[0].status is AttemptStatus.TIMED_OUT
            worker_pid = _worker_pid(
                controller,
                task.attempts[0].dispatch_id,
            )
            worker = controller.ray_runtime.worker_outcome(
                task.attempts[0].dispatch_id
            )
            assert worker is None or worker.physical_device_id == admission.device.physical_device_id
            await _wait_hbm_recovery(
                admission,
                baseline,
                worker_pids=(worker_pid,),
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=0,
            )
            await _destroy_and_assert(controller, outcome.run_id)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_real_npu_cancel_releases_late_output_and_hbm(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        baseline = admission.adapter.device(admission.device.physical_device_id).used_hbm_mb
        controller, agent = await _start_controller(admission, ascend_ray)
        try:
            workflow = Workflow("stage4-cancel")
            node = workflow.add_task(npu_long_running, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_cancel",
            )
            assert outcome.run_id is not None
            attempt = None
            for _ in range(1_000):
                task = controller.snapshot(outcome.run_id).task(node.task_id)
                if task.attempts and task.attempts[0].worker_started_at_ms is not None:
                    attempt = task.attempts[0]
                    break
                await asyncio.sleep(0.01)
            assert attempt is not None
            worker_pid = _worker_pid(controller, attempt.dispatch_id)
            cancelled = await controller.cancel_run(outcome.run_id)
            assert cancelled.status is RunStatus.CANCELLED
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            assert controller.worker_broker.active_count() == 0

            late_handle = await asyncio.to_thread(
                controller.ray_data_store.put_staged,
                "late-npu-output",
                controller.controller_generation,
            )
            late = RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_RESULT,
                dispatch_id=attempt.dispatch_id,
                run_id=outcome.run_id,
                task_id=node.task_id,
                attempt=attempt.attempt,
                lease_id=attempt.lease_id,
                route_lease_id=None,
                occurred_at_ms=controller.clock.monotonic_ms(),
                output_handles=(("result", late_handle),),
                worker_pid=1,
                device_id=admission.device.physical_device_id,
                binding_verified=True,
            )
            assert agent.endpoint is not None
            await asyncio.to_thread(
                report_worker_event,
                endpoint=agent.endpoint,
                identity=agent.identity,
                event=late,
                timeout_seconds=2,
            )
            for _ in range(200):
                try:
                    controller.ray_data_store.state_of(late_handle)
                except DataHandleInvalidError:
                    break
                await asyncio.sleep(0.01)
            with pytest.raises(DataHandleInvalidError):
                controller.ray_data_store.get(late_handle)
            await _wait_hbm_recovery(
                admission,
                baseline,
                worker_pids=(worker_pid,),
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=0,
            )
            await _destroy_and_assert(controller, outcome.run_id)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_real_npu_nodeagent_loss_invalidates_worker_and_device_lease(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        baseline = admission.adapter.device(admission.device.physical_device_id).used_hbm_mb
        controller, agent = await _start_controller(admission, ascend_ray)
        try:
            workflow = Workflow("stage4-node-loss")
            node = workflow.add_task(npu_long_running, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_node_loss",
            )
            assert outcome.run_id is not None
            attempt = None
            for _ in range(1_000):
                task = controller.snapshot(outcome.run_id).task(node.task_id)
                if task.attempts and task.attempts[0].worker_started_at_ms is not None:
                    attempt = task.attempts[0]
                    break
                await asyncio.sleep(0.01)
            assert attempt is not None
            worker_pid = _worker_pid(controller, attempt.dispatch_id)
            await agent.close(grace_seconds=0)
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=60)
            task = terminal.task(node.task_id)
            assert terminal.status is RunStatus.FAILED
            assert task.last_error is not None
            assert task.last_error.error_code == "worker_lost"
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            assert controller.worker_broker.active_count() == 0
            await _wait_hbm_recovery(
                admission,
                baseline,
                worker_pids=(worker_pid,),
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=0,
            )
            await _destroy_and_assert(controller, outcome.run_id, force=True)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_real_npu_oom_creates_exactly_one_reanchored_attempt(
    ascend_admission: AscendAdmission,
    ascend_ray: str,
) -> None:
    async def scenario() -> None:
        admission = ascend_admission
        baseline = admission.adapter.device(admission.device.physical_device_id).used_hbm_mb
        controller, agent = await _start_controller(admission, ascend_ray)
        try:
            workflow = Workflow("stage4-oom")
            node = workflow.add_task(npu_oom, inputs={})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_oom",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=120)
            task = terminal.task(node.task_id)
            assert terminal.status is RunStatus.FAILED
            assert task.attempt_count == 2, (
                None
                if task.last_error is None
                else (
                    task.last_error.error_code,
                    task.last_error.exception_type,
                    task.last_error.platform_error_code,
                    task.last_error.message,
                ),
                controller.core.recovery.decision(outcome.run_id, node.task_id, 1),
                controller.anchors.resolve(
                    run_id=outcome.run_id,
                    compiled=controller.state.compiled(outcome.run_id),
                    task_id=node.task_id,
                ),
                controller.placement.max_single_npu_allocatable_hbm_mb(),
            )
            assert tuple(item.anchor_revision for item in task.attempts) == (1, 2)
            assert task.last_error is not None
            assert task.last_error.error_code == "npu_oom"
            assert task.last_error.classification_confidence == "fallback"
            anchor = controller.anchors.resolve(
                run_id=outcome.run_id,
                compiled=controller.state.compiled(outcome.run_id),
                task_id=node.task_id,
            )
            assert anchor.revision == 2
            anchor_events = [
                event
                for event in controller.ray_recorder.events(outcome.run_id)
                if event.event_type == "resource_anchor_oom"
            ]
            assert len(anchor_events) == 2
            assert sum(bool(event.payload["created"]) for event in anchor_events) == 1
            worker_pids = tuple(
                _worker_pid(controller, attempt.dispatch_id)
                for attempt in task.attempts
            )
            assert len(set(worker_pids)) == 2
            await _wait_hbm_recovery(
                admission, baseline, worker_pids=worker_pids
            )
            _assert_run_terminal_checkpoint(
                controller,
                outcome.run_id,
                expected_handle_count=0,
            )
            await _destroy_and_assert(controller, outcome.run_id)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())
