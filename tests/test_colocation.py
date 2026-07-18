from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ascend_maze import Workflow
from ascend_maze.ascend import (
    AscendColocationConfig,
    AscendCorrectnessConfig,
    AscendEnvironmentSnapshot,
    create_ascend_colocation_config_snapshot,
)
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.contracts.resources import ExecutionTarget, ResourceSpec
from ascend_maze.core.canonical import FrozenMap
from ascend_maze.core.errors import ContractValidationError, StateTransitionError
from ascend_maze.placement import (
    LeaseStatus,
    NodeCapacity,
    NpuCapacity,
    PlacementManager,
)
from ascend_maze.resources import ResourceAnchor
from ascend_maze.runtime import FakeExecutionPlan, RuntimeEvent, RuntimeEventKind
from ascend_maze.lifecycle import RunStatus, TaskStatus
from stage4_task_fixtures import no_retry_npu
from task_fixtures import local_npu_task


def _anchor(*, npu_hbm_mb: int = 1_024) -> ResourceAnchor:
    resources = ResourceSpec(1, 256, npu_hbm_mb, 0)
    return ResourceAnchor(
        definition_id="definition_npu",
        task_kind="npu",
        execution_target=ExecutionTarget.LOCAL_WORKER,
        declared=resources,
        static_inferred=ResourceSpec(0, 0, 0, 0),
        learned=None,
        effective=resources,
        model_id=None,
        profile_key="profile_npu",
        revision=1,
        strategy="declared_only",
    )


def _placement(*, hbm_mb: int = 8_192, slots: int = 2) -> PlacementManager:
    placement = PlacementManager()
    placement.register_node(
        NodeCapacity(
            node_id="node_a",
            boot_id="boot_a",
            node_ip="127.0.0.1",
            cpu_total=8,
            mem_total_mb=8_192,
            cpu_system_reserved=0,
            mem_system_reserved_mb=0,
            io_slots_total=8,
            npus=(
                NpuCapacity(
                    device_id="0",
                    chip_type="910B3",
                    total_hbm_mb=hbm_mb,
                    system_reserved_hbm_mb=0,
                    task_slots_total=slots,
                    observed_free_hbm_mb=hbm_mb,
                ),
            ),
            observed_free_mem_mb=8_192,
        )
    )
    return placement


def _reserve(
    placement: PlacementManager,
    *,
    run_id: str,
    task_id: str,
    npu_hbm_mb: int = 1_024,
):
    result = placement.try_reserve(
        run_id=run_id,
        task_id=task_id,
        attempt=1,
        anchor=_anchor(npu_hbm_mb=npu_hbm_mb),
        now_ms=1,
        dispatch_deadline_ms=100,
    )
    assert result.selected
    assert result.lease is not None
    return result.lease


def test_colocation_profile_is_explicit_and_does_not_relax_stage_four() -> None:
    config = AscendColocationConfig()
    assert config.task_slots_total == 2
    assert config.allow_colocation
    assert config.max_tasks_per_worker == 1
    assert config.standby_min_idle == 2

    with pytest.raises(ContractValidationError, match="at least two"):
        AscendColocationConfig(task_slots_total=1)
    with pytest.raises(ContractValidationError, match="explicitly enabled"):
        AscendColocationConfig(allow_colocation=False)
    with pytest.raises(ContractValidationError, match="one Attempt"):
        AscendColocationConfig(max_tasks_per_worker=2)
    with pytest.raises(ContractValidationError, match="one slot"):
        AscendCorrectnessConfig(task_slots_total=2)


def test_colocation_snapshot_covers_capacity_and_metric_attribution() -> None:
    environment = AscendEnvironmentSnapshot.create(
        machine="aarch64",
        chip_types=("910B3",),
        versions={"cann": "9.0", "torch_npu": "2.7.1"},
    )
    config = AscendColocationConfig(
        scheduler_policy="hacs_no_tp",
        task_slots_total=3,
        standby_min_idle=3,
    )
    snapshot = create_ascend_colocation_config_snapshot(
        config,
        environment,
        source_path="/etc/ascend-maze/colocation-correctness.toml",
        build_revision="stage5c-test-build",
        created_at_ms=1,
    )

    assert snapshot.resolved["profile"] == "colocation_correctness"
    assert snapshot.resolved["scheduler"]["policy"] == "hacs_no_tp"
    assert snapshot.resolved["placement"] == FrozenMap(
        (
            ("allow_colocation", True),
            ("host_mem_headroom_mb", config.host_mem_headroom_mb),
            ("io_slots_total", config.io_slots_total),
            ("npu_hbm_headroom_mb", config.npu_hbm_headroom_mb),
            (
                "npu_system_reserved_hbm_mb",
                config.npu_system_reserved_hbm_mb,
            ),
            ("task_slots_total", 3),
        )
    )
    assert snapshot.resolved["worker"]["standby"]["min_idle"] == 3
    assert snapshot.resolved["worker"]["max_tasks_per_worker"] == 1
    assert snapshot.resolved["observation"] == FrozenMap(
        (
            ("device_metrics_attribution", "device_only"),
            ("process_hbm_attribution", "attempt"),
        )
    )


def test_two_attempts_share_one_npu_with_independent_hbm_and_slot_debits() -> None:
    placement = _placement()
    first = _reserve(placement, run_id="run_a", task_id="task_a", npu_hbm_mb=1_024)
    second = _reserve(placement, run_id="run_b", task_id="task_b", npu_hbm_mb=2_048)

    assert first.lease_id != second.lease_id
    assert first.npu_device_id == second.npu_device_id == "0"
    assert first.resources.npu_slots == second.resources.npu_slots == 1
    assert placement.snapshot().nodes[0].per_npu_reserved == (("0", 3_072, 2),)
    assert placement.active_lease_count() == 2

    blocked = placement.try_reserve(
        run_id="run_c",
        task_id="task_c",
        attempt=1,
        anchor=_anchor(npu_hbm_mb=1),
        now_ms=2,
        dispatch_deadline_ms=100,
    )
    assert not blocked.selected
    assert blocked.rejection_reason == "npu_task_slots_full"
    assert placement.snapshot().nodes[0].per_npu_reserved == (("0", 3_072, 2),)


def test_hbm_limit_blocks_third_attempt_even_when_a_slot_is_free() -> None:
    placement = _placement(hbm_mb=3_000, slots=3)
    _reserve(placement, run_id="run_a", task_id="task_a", npu_hbm_mb=1_500)
    _reserve(placement, run_id="run_b", task_id="task_b", npu_hbm_mb=1_500)

    blocked = placement.try_reserve(
        run_id="run_c",
        task_id="task_c",
        attempt=1,
        anchor=_anchor(npu_hbm_mb=1),
        now_ms=2,
        dispatch_deadline_ms=100,
    )
    assert not blocked.selected
    assert blocked.rejection_reason == "insufficient_npu_hbm"
    assert placement.snapshot().nodes[0].per_npu_reserved == (("0", 3_000, 2),)


def test_release_and_late_duplicate_are_scoped_to_the_exact_attempt() -> None:
    placement = _placement()
    first = _reserve(placement, run_id="run_a", task_id="task_a")
    second = _reserve(placement, run_id="run_b", task_id="task_b", npu_hbm_mb=2_048)
    assert placement.bind_lease(first.lease_id, now_ms=2)
    assert placement.bind_lease(second.lease_id, now_ms=2)

    with pytest.raises(StateTransitionError, match="run_id"):
        placement.release_lease(
            first.lease_id,
            now_ms=3,
            run_id="run_b",
            task_id="task_a",
            attempt=1,
        )
    assert placement.snapshot().nodes[0].per_npu_reserved == (("0", 3_072, 2),)

    assert placement.release_lease(
        first.lease_id,
        now_ms=4,
        run_id="run_a",
        task_id="task_a",
        attempt=1,
        reason="cancelled",
    )
    assert not placement.release_lease(
        first.lease_id,
        now_ms=5,
        run_id="run_a",
        task_id="task_a",
        attempt=1,
        reason="late_duplicate",
    )
    assert placement.lease_snapshot(first.lease_id).status is LeaseStatus.RELEASED
    assert placement.lease_snapshot(second.lease_id).status is LeaseStatus.BOUND
    assert placement.snapshot().nodes[0].per_npu_reserved == (("0", 2_048, 1),)
    assert placement.active_lease_count("run_a") == 0
    assert placement.active_lease_count("run_b") == 1

    # A late terminal transition for A cannot alter B's immutable reservation.
    assert not placement.invalidate_lease(
        first.lease_id,
        now_ms=6,
        reason="late_worker_lost",
    )
    assert placement.lease_snapshot(second.lease_id).lease == second
    assert placement.snapshot().nodes[0].per_npu_reserved == (("0", 2_048, 1),)

    # Replacing the observed free value affects admission, never another Lease's debit.
    placement.register_node(
        replace(
            placement.snapshot().nodes[0].capacity,
            observed_free_mem_mb=8_000,
        )
    )
    assert placement.snapshot().nodes[0].per_npu_reserved == (("0", 2_048, 1),)


def test_fake_runtime_failure_and_late_event_do_not_release_another_run() -> None:
    async def scenario() -> None:
        placement = _placement()
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="stage5c-test",
            node_capacities=(),
            placement=placement,
        )
        await controller.start()
        try:
            failed_workflow = Workflow("stage5c-failed-run")
            failed_node = failed_workflow.add_task(
                no_retry_npu,
                inputs={"value": 1},
            )
            survivor_workflow = Workflow("stage5c-surviving-run")
            survivor_node = survivor_workflow.add_task(
                local_npu_task,
                inputs={"value": "survived"},
            )
            controller.runtime.set_plan(
                failed_node.task_id,
                1,
                FakeExecutionPlan(
                    execution_delay_ms=100,
                    fail_after_start="npu_oom",
                ),
            )
            controller.runtime.set_plan(
                survivor_node.task_id,
                1,
                FakeExecutionPlan(execution_delay_ms=300),
            )
            client = InMemoryRuntimeClient(controller)
            failed_outcome = await client.submit(
                failed_workflow,
                inputs={},
                submission_id="stage5c_failed_run",
            )
            survivor_outcome = await client.submit(
                survivor_workflow,
                inputs={},
                submission_id="stage5c_surviving_run",
            )
            assert failed_outcome.run_id is not None
            assert survivor_outcome.run_id is not None

            for _ in range(500):
                failed = controller.snapshot(failed_outcome.run_id)
                survivor = controller.snapshot(survivor_outcome.run_id)
                if (
                    failed.task(failed_node.task_id).status is TaskStatus.RUNNING
                    and survivor.task(survivor_node.task_id).status
                    is TaskStatus.RUNNING
                ):
                    break
                await asyncio.sleep(0.001)
            assert controller.placement.snapshot().nodes[0].per_npu_reserved == (
                ("0", 2_048, 2),
            )

            failed_terminal = await controller.wait_run(
                failed_outcome.run_id,
                timeout_seconds=2,
            )
            assert failed_terminal.status is RunStatus.FAILED
            assert (
                failed_terminal.task(failed_node.task_id).last_error.error_code
                == "npu_oom"
            )
            survivor_running = controller.snapshot(survivor_outcome.run_id)
            assert survivor_running.status is RunStatus.RUNNING
            survivor_attempt = survivor_running.task(survivor_node.task_id).attempts[0]
            assert (
                controller.placement.lease_snapshot(survivor_attempt.lease_id).status
                is LeaseStatus.BOUND
            )
            assert controller.placement.active_lease_count(failed_outcome.run_id) == 0
            assert controller.placement.active_lease_count(survivor_outcome.run_id) == 1
            assert controller.placement.snapshot().nodes[0].per_npu_reserved == (
                ("0", 1_024, 1),
            )

            failed_attempt = failed_terminal.task(failed_node.task_id).attempts[0]
            controller.core.post_runtime_event(
                RuntimeEvent.create(
                    kind=RuntimeEventKind.TASK_CANCELLED,
                    dispatch_id=failed_attempt.dispatch_id,
                    run_id=failed_outcome.run_id,
                    task_id=failed_node.task_id,
                    attempt=failed_attempt.attempt,
                    lease_id=failed_attempt.lease_id,
                    route_lease_id=None,
                    occurred_at_ms=controller.clock.monotonic_ms(),
                )
            )
            await asyncio.sleep(0)
            assert (
                controller.placement.lease_snapshot(survivor_attempt.lease_id).status
                is LeaseStatus.BOUND
            )
            assert controller.placement.snapshot().nodes[0].per_npu_reserved == (
                ("0", 1_024, 1),
            )

            survivor_terminal = await controller.wait_run(
                survivor_outcome.run_id,
                timeout_seconds=2,
            )
            assert survivor_terminal.status is RunStatus.SUCCEEDED
            assert controller.result(
                survivor_outcome.run_id,
                survivor_node.task_id,
            ) == {"value": "survived"}
            assert controller.placement.active_lease_count() == 0

            await controller.destroy_run(failed_outcome.run_id)
            await controller.destroy_run(survivor_outcome.run_id)
            assert controller.placement.lease_record_count() == 0
            assert controller.data_store.active_count == 0
        finally:
            await controller.close()

    asyncio.run(scenario())
