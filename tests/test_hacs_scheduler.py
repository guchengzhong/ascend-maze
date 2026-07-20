from __future__ import annotations

import asyncio
import math

import pytest

from ascend_maze import Workflow
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.contracts.resources import ExecutionTarget, ResourceSpec
from ascend_maze.core.clock import ManualClock
from ascend_maze.lifecycle import RunStatus, TaskStatus
from ascend_maze.placement import LeaseStatus, NodeCapacity, NodeStatus, NpuCapacity
from ascend_maze.resources import ResourceAnchor
from ascend_maze.runtime import FakeExecutionPlan
from ascend_maze.scheduler import (
    FcfsPolicy,
    HacsConfig,
    HacsNoTpStaticPolicy,
    HeterogeneousPartitioner,
    QueueToken,
    SchedulableTaskView,
    TaskKey,
    UnifiedPartitioner,
)
from ascend_maze.scheduler.policies.hacs import LinearScanReferenceQueue
from stage5_task_fixtures import one_cpu_task, two_cpu_task
from task_fixtures import local_npu_task


CONFIG_FINGERPRINT = "5" * 64
ENVIRONMENT_FINGERPRINT = "e" * 64
ZERO_RESOURCES = ResourceSpec(cpu_num=0, mem_mb=0, npu_mem_mb=0, io_num=0)


def _view(
    run_id: str,
    task_id: str,
    *,
    sequence: int,
    depth_to_exit: int,
    task_kind: str = "npu",
    queue_generation: int = 1,
) -> SchedulableTaskView:
    token = QueueToken(TaskKey(run_id, task_id), queue_generation)
    anchor = ResourceAnchor(
        definition_id=f"definition_{task_id}",
        task_kind=task_kind,
        execution_target=ExecutionTarget.LOCAL_WORKER,
        declared=ZERO_RESOURCES,
        static_inferred=ZERO_RESOURCES,
        learned=None,
        effective=ZERO_RESOURCES,
        model_id=None,
        profile_key=f"profile_{task_id}",
        revision=1,
        strategy="test",
    )
    return SchedulableTaskView(
        queue_token=token,
        task_kind=task_kind,
        ready_at_ms=0,
        queued_at_ms=0,
        enqueue_sequence=sequence,
        depth_from_entry=0,
        depth_to_exit=depth_to_exit,
        resource_anchor=anchor,
    )


def _host_node(*, cpu: int = 2) -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_a",
        node_ip="10.0.0.1",
        cpu_total=cpu,
        mem_total_mb=4096,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=4,
        observed_free_mem_mb=4096,
    )


def _npu_node() -> NodeCapacity:
    return NodeCapacity(
        node_id="node_npu",
        boot_id="boot_npu",
        node_ip="10.0.0.2",
        cpu_total=4,
        mem_total_mb=4096,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=4,
        npus=(
            NpuCapacity(
                device_id="0",
                chip_type="910B3",
                total_hbm_mb=65_536,
                system_reserved_hbm_mb=3_200,
                task_slots_total=1,
                observed_free_hbm_mb=62_000,
            ),
        ),
        observed_free_mem_mb=4096,
    )


def _selected_task_ids(controller: InMemoryController, run_ids: tuple[str, ...]) -> list[str]:
    events = [
        event
        for run_id in run_ids
        for event in controller.recorder.events(run_id)
        if event.event_type == "scheduling_decision"
        and event.payload["placement_selected"] is True
    ]
    events.sort(key=lambda event: event.producer_sequence)
    return [event.task_id for event in events if event.task_id is not None]


def test_hacs_formula_matches_hand_calculation_and_raw_score_order() -> None:
    clock = ManualClock(monotonic_ms=130_000)
    policy = HacsNoTpStaticPolicy(clock=clock, scheduler_epoch_ms=0)
    policy.register_run(run_id="run", submitted_at_ms=10_000, total_value_tasks=2)
    view = _view("run", "task", sequence=1, depth_to_exit=3)
    policy.enqueue("npu", view)

    score = policy.score_for(view)
    expected_omega = math.log2(2 + 2 * 3)
    expected_phi = 120 / (2 * 60) - 2
    expected_raw_score = expected_omega * 5**expected_phi
    assert score.omega == pytest.approx(expected_omega)
    assert score.phi == pytest.approx(expected_phi)
    assert score.log_score == pytest.approx(math.log(expected_raw_score))

    candidates = []
    for index, (submitted, remaining, depth) in enumerate(
        ((0, 1, 0), (15_000, 2, 5), (90_000, 0, 1)),
        start=1,
    ):
        run_id = f"run_{index}"
        task = _view(run_id, f"task_{index}", sequence=index + 1, depth_to_exit=depth)
        policy.register_run(
            run_id=run_id,
            submitted_at_ms=submitted,
            total_value_tasks=remaining,
        )
        policy.enqueue("npu", task)
        candidate_score = policy.score_for(task)
        raw_score = candidate_score.omega * 5**candidate_score.phi
        candidates.append((raw_score, task.queue_token.task_key))

    raw_order = [key for _, key in sorted(candidates, key=lambda item: -item[0])]
    heap_order = [
        proposal.task_key
        for proposal in policy.propose("npu", 10)
        if proposal.task_key.run_id != "run"
    ]
    assert heap_order == raw_order


def test_hacs_tie_break_and_partitioners_are_orthogonal() -> None:
    policy = HacsNoTpStaticPolicy(clock=ManualClock(), scheduler_epoch_ms=0)
    policy.register_run(run_id="run_a", submitted_at_ms=0, total_value_tasks=1)
    policy.register_run(run_id="run_b", submitted_at_ms=0, total_value_tasks=1)
    later_sequence = _view("run_a", "task_a", sequence=2, depth_to_exit=0)
    earlier_sequence = _view("run_b", "task_b", sequence=1, depth_to_exit=0)

    heterogeneous = HeterogeneousPartitioner()
    unified = UnifiedPartitioner()
    assert heterogeneous.partition(later_sequence) == "npu"
    assert unified.partition(later_sequence) == "default"
    policy.enqueue("npu", later_sequence)
    policy.enqueue("npu", earlier_sequence)
    assert [item.task_key for item in policy.propose("npu", 2)] == [
        earlier_sequence.queue_token.task_key,
        later_sequence.queue_token.task_key,
    ]


@pytest.mark.parametrize("policy_name", ("fcfs", "hacs_no_tp"))
@pytest.mark.parametrize("partitioner_name", ("heterogeneous", "unified"))
def test_policy_and_partitioner_combinations_share_the_core_execution_path(
    policy_name: str,
    partitioner_name: str,
) -> None:
    async def scenario() -> None:
        clock = ManualClock()
        policy = (
            FcfsPolicy()
            if policy_name == "fcfs"
            else HacsNoTpStaticPolicy(clock=clock, scheduler_epoch_ms=0)
        )
        partitioner = (
            HeterogeneousPartitioner()
            if partitioner_name == "heterogeneous"
            else UnifiedPartitioner()
        )
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="stage5a_test",
            node_capacities=(_npu_node(),),
            clock=clock,
            policy=policy,
            partitioner=partitioner,
        )
        await controller.start()
        workflow = Workflow(f"{policy_name}-{partitioner_name}")
        cpu_node = workflow.add_task(one_cpu_task, inputs={"value": "cpu"})
        npu_node = workflow.add_task(local_npu_task, inputs={"value": "npu"})
        outcome = await InMemoryRuntimeClient(controller).submit(
            workflow,
            inputs={},
            submission_id=f"stage5a_{policy_name}_{partitioner_name}",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, cpu_node.task_id) == {"value": "cpu"}
        assert controller.result(outcome.run_id, npu_node.task_id) == {"value": "npu"}

        selected = [
            event
            for event in controller.recorder.events(outcome.run_id)
            if event.event_type == "scheduling_decision"
            and event.payload["placement_selected"] is True
        ]
        assert {event.payload["policy_name"] for event in selected} == {policy_name}
        expected_partitions = (
            {"cpu", "npu"}
            if partitioner_name == "heterogeneous"
            else {"default"}
        )
        assert {event.payload["partition"] for event in selected} == expected_partitions
        assert all(event.payload["queue_length"] >= 1 for event in selected)
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        assert controller.deadlines.count_for_run(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        assert controller.data_store.active_count == 0
        assert controller.runtime.code_reference_count() == 0
        await controller.close()

    asyncio.run(scenario())


def test_hacs_heap_matches_linear_reference_across_lifecycle_events() -> None:
    clock = ManualClock(monotonic_ms=300_000)
    heap = HacsNoTpStaticPolicy(clock=clock, scheduler_epoch_ms=0)
    reference = LinearScanReferenceQueue(clock=clock, scheduler_epoch_ms=0)
    policies = (heap, reference)
    registrations = (
        ("run_a", 0, 2),
        ("run_b", 25_000, 1),
        ("run_c", 60_000, 0),
        ("completed", 0, 0),
    )
    for policy in policies:
        for run_id, submitted_at_ms, total_value_tasks in registrations:
            policy.register_run(
                run_id=run_id,
                submitted_at_ms=submitted_at_ms,
                total_value_tasks=total_value_tasks,
            )

    views = (
        _view("run_a", "a_1", sequence=4, depth_to_exit=4),
        _view("run_a", "a_2", sequence=2, depth_to_exit=0),
        _view("run_b", "b_1", sequence=1, depth_to_exit=2),
        _view("run_c", "c_1", sequence=3, depth_to_exit=1, task_kind="cpu"),
    )
    for policy in policies:
        for view in views:
            policy.enqueue(view.task_kind, view)

    def assert_same() -> None:
        for partition in ("npu", "cpu"):
            heap_proposals = heap.propose(partition, 10)
            reference_proposals = reference.propose(partition, 10)
            assert [item.task_key for item in heap_proposals] == [
                item.task_key for item in reference_proposals
            ]

    assert_same()
    for policy in policies:
        policy.task_succeeded(run_id="run_a", task_id="cpu", task_kind="cpu")
        policy.task_succeeded(run_id="run_a", task_id="value_1", task_kind="npu")
        policy.task_succeeded(run_id="run_a", task_id="value_1", task_kind="npu")
    assert heap.run_state("run_a").remaining_value_tasks == 1
    assert heap.run_state("run_a").priority_generation == 1
    assert_same()

    for policy in policies:
        policy.depart(views[2].queue_token)
        policy.run_terminal(
            run_id="completed",
            status="succeeded",
            finished_at_ms=120_000,
        )
    assert heap.global_state.avg_dct_seconds == pytest.approx(66.0)
    assert heap.global_state.dct_generation == 1
    assert heap.global_state.last_rebuild_ms >= 0
    assert heap.global_state.last_rebuild_task_count == heap.active_count()
    assert heap.heap_record_count() == heap.active_count()
    assert_same()

    before = heap.global_state
    for policy in policies:
        policy.run_terminal(run_id="run_b", status="failed", finished_at_ms=400_000)
    assert heap.global_state == before
    assert_same()


def test_successful_dct_update_rebuilds_heap_and_can_change_order() -> None:
    policy = HacsNoTpStaticPolicy(clock=ManualClock(), scheduler_epoch_ms=0)
    policy.register_run(run_id="old", submitted_at_ms=0, total_value_tasks=0)
    policy.register_run(run_id="new", submitted_at_ms=120_000, total_value_tasks=0)
    policy.register_run(run_id="completed", submitted_at_ms=0, total_value_tasks=0)
    old = _view("old", "old_task", sequence=1, depth_to_exit=0)
    new = _view("new", "new_task", sequence=2, depth_to_exit=3)
    policy.enqueue("npu", old)
    policy.enqueue("npu", new)
    assert policy.propose("npu", 2)[0].task_key == old.queue_token.task_key

    policy.run_terminal(
        run_id="completed",
        status="succeeded",
        finished_at_ms=660_000,
    )
    assert policy.global_state.avg_dct_seconds == pytest.approx(120.0)
    assert policy.global_state.completed_run_count == 1
    assert policy.global_state.dct_generation == 1
    assert policy.propose("npu", 2)[0].task_key == new.queue_token.task_key


def test_hacs_fake_runtime_closes_n_val_dct_and_recording_path() -> None:
    async def scenario() -> None:
        clock = ManualClock()
        policy = HacsNoTpStaticPolicy(clock=clock, scheduler_epoch_ms=0)
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="stage5a_test",
            node_capacities=(_npu_node(),),
            clock=clock,
            policy=policy,
            partitioner=HeterogeneousPartitioner(),
        )
        controller.placement.set_node_status(
            "node_npu", NodeStatus.UNSCHEDULABLE, now_ms=0
        )
        await controller.start()
        client = InMemoryRuntimeClient(controller)

        chain = Workflow("hacs-chain")
        first = chain.add_task(local_npu_task, inputs={"value": "chain"})
        second = chain.add_task(
            local_npu_task,
            inputs={"value": first.outputs["value"]},
        )
        single = Workflow("hacs-single")
        only = single.add_task(local_npu_task, inputs={"value": "single"})

        chain_outcome = await client.submit(
            chain,
            inputs={},
            submission_id="stage5a_chain",
        )
        single_outcome = await client.submit(
            single,
            inputs={},
            submission_id="stage5a_single",
        )
        assert chain_outcome.run_id is not None
        assert single_outcome.run_id is not None
        assert policy.run_state(chain_outcome.run_id).remaining_value_tasks == 2
        assert policy.run_state(single_outcome.run_id).remaining_value_tasks == 1

        controller.placement.set_node_status(
            "node_npu", NodeStatus.HEALTHY, now_ms=clock.monotonic_ms()
        )
        await controller.core.wake_deadlines()
        chain_terminal, single_terminal = await asyncio.gather(
            controller.wait_run(chain_outcome.run_id, timeout_seconds=2),
            controller.wait_run(single_outcome.run_id, timeout_seconds=2),
        )
        assert chain_terminal.status is RunStatus.SUCCEEDED
        assert single_terminal.status is RunStatus.SUCCEEDED
        assert _selected_task_ids(
            controller,
            (chain_outcome.run_id, single_outcome.run_id),
        ) == [only.task_id, first.task_id, second.task_id]
        decisions = [
            event
            for run_id in (chain_outcome.run_id, single_outcome.run_id)
            for event in controller.recorder.events(run_id)
            if event.event_type == "scheduling_decision"
        ]
        selected_decisions = [
            event for event in decisions if event.payload["placement_selected"] is True
        ]
        assert selected_decisions
        for event in decisions:
            for field in ("score_compute_ms", "policy_select_ms", "placement_ms"):
                value = event.payload[field]
                assert isinstance(value, float)
                assert value >= 0
            metadata = event.payload["policy_metadata"]
            assert metadata["last_rebuild_ms"] >= 0
            assert metadata["last_rebuild_task_count"] >= 0
        assert max(
            event.payload["policy_metadata"]["last_rebuild_task_count"]
            for event in selected_decisions
        ) >= 1
        assert policy.active_count() == 0
        assert policy.global_state.completed_run_count == 2
        assert policy.global_state.dct_generation == 2
        with pytest.raises(KeyError):
            policy.run_state(chain_outcome.run_id)

        for run_id in (chain_outcome.run_id, single_outcome.run_id):
            assert controller.placement.active_lease_count(run_id) == 0
            await controller.destroy_run(run_id)
        assert controller.data_store.active_count == 0
        assert controller.runtime.code_reference_count() == 0
        await controller.close()

    asyncio.run(scenario())


def test_resource_changed_event_wakes_a_blocked_queue() -> None:
    async def scenario() -> None:
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="stage5a_test",
            node_capacities=(_host_node(cpu=1),),
        )
        controller.placement.set_node_status(
            "node_a", NodeStatus.UNSCHEDULABLE, now_ms=0
        )
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("resource-change-wakeup")
        pending = workflow.add_task(one_cpu_task, inputs={"value": "ready"})

        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="stage5a_resource_change",
        )
        assert outcome.run_id is not None
        await controller.core.wake_deadlines()
        assert controller.snapshot(outcome.run_id).task(pending.task_id).status is (
            TaskStatus.QUEUED
        )

        controller.placement.set_node_status(
            "node_a", NodeStatus.HEALTHY, now_ms=controller.clock.monotonic_ms()
        )
        assert controller.core.post_resource_changed("node_became_healthy")
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert any(
            event.event_type == "resource_changed"
            and event.payload["reason"] == "node_became_healthy"
            for event in controller.recorder.events(outcome.run_id)
        )
        assert controller.placement.active_lease_count(outcome.run_id) == 0
        await controller.destroy_run(outcome.run_id)
        await controller.close()

    asyncio.run(scenario())


def test_bounded_bypass_protects_blocked_large_task_until_it_runs() -> None:
    async def scenario() -> None:
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="stage5a_test",
            node_capacities=(_host_node(cpu=2),),
            placement_lookahead=8,
            max_bypass_count=1,
        )
        await controller.start()
        client = InMemoryRuntimeClient(controller)

        blocker_workflow = Workflow("blocker")
        blocker = blocker_workflow.add_task(one_cpu_task, inputs={"value": "blocker"})
        large_workflow = Workflow("large")
        large = large_workflow.add_task(two_cpu_task, inputs={"value": "large"})
        first_small_workflow = Workflow("first-small")
        first_small = first_small_workflow.add_task(
            one_cpu_task, inputs={"value": "small-1"}
        )
        second_small_workflow = Workflow("second-small")
        second_small = second_small_workflow.add_task(
            one_cpu_task, inputs={"value": "small-2"}
        )
        controller.runtime.set_plan(
            blocker.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=200),
        )
        controller.runtime.set_plan(
            first_small.task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=200),
        )

        outcomes = []
        for submission_id, workflow in (
            ("stage5a_blocker", blocker_workflow),
            ("stage5a_large", large_workflow),
            ("stage5a_small_1", first_small_workflow),
            ("stage5a_small_2", second_small_workflow),
        ):
            outcome = await client.submit(
                workflow,
                inputs={},
                submission_id=submission_id,
            )
            assert outcome.run_id is not None
            outcomes.append(outcome.run_id)

        await controller.core.wake_deadlines()
        assert controller.snapshot(outcomes[3]).task(second_small.task_id).status is (
            TaskStatus.QUEUED
        )
        terminals = await asyncio.gather(
            *(controller.wait_run(run_id, timeout_seconds=3) for run_id in outcomes)
        )
        assert all(item.status is RunStatus.SUCCEEDED for item in terminals)
        assert _selected_task_ids(controller, tuple(outcomes)) == [
            blocker.task_id,
            first_small.task_id,
            large.task_id,
            second_small.task_id,
        ]
        for run_id in outcomes:
            assert controller.placement.active_lease_count(run_id) == 0
            await controller.destroy_run(run_id)
        assert controller.placement.lease_record_count() == 0
        await controller.close()

    asyncio.run(scenario())


def test_failed_npu_attempt_keeps_n_val_until_retry_succeeds() -> None:
    class ObservedHacsPolicy(HacsNoTpStaticPolicy):
        def __init__(self, *, clock: ManualClock) -> None:
            super().__init__(clock=clock, scheduler_epoch_ms=0)
            self.success_notifications: list[tuple[str, str]] = []
            self.remaining_at_terminal: dict[str, int] = {}

        def task_succeeded(
            self,
            *,
            run_id: str,
            task_id: str,
            task_kind: str,
        ) -> None:
            self.success_notifications.append((run_id, task_id))
            super().task_succeeded(
                run_id=run_id,
                task_id=task_id,
                task_kind=task_kind,
            )

        def run_terminal(
            self,
            *,
            run_id: str,
            status: str,
            finished_at_ms: int,
        ) -> None:
            self.remaining_at_terminal[run_id] = self.run_state(
                run_id
            ).remaining_value_tasks
            super().run_terminal(
                run_id=run_id,
                status=status,
                finished_at_ms=finished_at_ms,
            )

    async def scenario() -> None:
        clock = ManualClock()
        policy = ObservedHacsPolicy(clock=clock)
        controller = InMemoryController(
            config_fingerprint=CONFIG_FINGERPRINT,
            environment_fingerprint=ENVIRONMENT_FINGERPRINT,
            build_revision="stage5a_test",
            node_capacities=(_npu_node(),),
            clock=clock,
            policy=policy,
        )
        await controller.start()
        client = InMemoryRuntimeClient(controller)
        workflow = Workflow("hacs-retry")
        retrying = workflow.add_task(local_npu_task, inputs={"value": "retry"})
        controller.runtime.set_plan(
            retrying.task_id,
            1,
            FakeExecutionPlan(fail_before_start="worker_start_failed"),
        )

        outcome = await client.submit(
            workflow,
            inputs={},
            submission_id="stage5a_retry",
        )
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        task = terminal.task(retrying.task_id)
        assert task.attempt_count == 2
        assert policy.success_notifications == [(outcome.run_id, retrying.task_id)]
        assert policy.remaining_at_terminal[outcome.run_id] == 0
        assert policy.global_state.completed_run_count == 1
        assert [
            controller.placement.lease_snapshot(attempt.lease_id).status
            for attempt in task.attempts
        ] == [LeaseStatus.RELEASED, LeaseStatus.RELEASED]
        assert len({attempt.lease_id for attempt in task.attempts}) == 2
        assert controller.placement.active_lease_count(outcome.run_id) == 0

        await controller.destroy_run(outcome.run_id)
        assert controller.placement.lease_record_count() == 0

        failing_workflow = Workflow("hacs-final-failure")
        failing = failing_workflow.add_task(
            local_npu_task,
            inputs={"value": "failure"},
        )
        for attempt in (1, 2):
            controller.runtime.set_plan(
                failing.task_id,
                attempt,
                FakeExecutionPlan(fail_before_start="worker_start_failed"),
            )
        failed_outcome = await client.submit(
            failing_workflow,
            inputs={},
            submission_id="stage5a_final_failure",
        )
        assert failed_outcome.run_id is not None
        failed_terminal = await controller.wait_run(
            failed_outcome.run_id,
            timeout_seconds=2,
        )
        assert failed_terminal.status is RunStatus.FAILED
        assert failed_terminal.task(failing.task_id).attempt_count == 2
        assert policy.remaining_at_terminal[failed_outcome.run_id] == 1
        assert (failed_outcome.run_id, failing.task_id) not in (
            policy.success_notifications
        )
        await controller.destroy_run(failed_outcome.run_id)
        assert controller.placement.lease_record_count() == 0
        await controller.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("alpha", 0.0),
        ("beta", 1.0),
        ("initial_avg_dct_seconds", 0.0),
        ("dct_ema_gamma", 0.0),
        ("dct_ema_gamma", 1.1),
        ("t_pred", 2.0),
    ),
)
def test_hacs_config_rejects_invalid_static_parameters(field: str, value: float) -> None:
    values = {
        "alpha": 2.0,
        "beta": 5.0,
        "initial_avg_dct_seconds": 60.0,
        "dct_ema_gamma": 0.1,
        "t_pred": 1.0,
    }
    values[field] = value
    with pytest.raises(ValueError):
        HacsConfig(**values)
