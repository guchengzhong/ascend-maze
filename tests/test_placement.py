from __future__ import annotations

import pytest

from ascend_maze import Workflow
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.placement import (
    LeaseStatus,
    NodeCapacity,
    NodeStatus,
    NpuCapacity,
    PlacementManager,
)
from ascend_maze.resources import DeclaredOnlyAnchorProvider
from task_fixtures import local_npu_task, service_task, summarize


def _node(
    node_id: str,
    *,
    boot_id: str = "boot_1",
    cpu: int = 4,
    memory: int = 1024,
    io_slots: int = 2,
    npus: tuple[NpuCapacity, ...] = (),
) -> NodeCapacity:
    return NodeCapacity(
        node_id=node_id,
        boot_id=boot_id,
        node_ip=f"10.0.0.{1 if node_id == 'node_a' else 2}",
        cpu_total=cpu,
        mem_total_mb=memory,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=io_slots,
        npus=npus,
        observed_free_mem_mb=memory,
        capabilities={"environment": "test"},
    )


def test_declared_only_anchor_uses_compiled_resources_and_is_per_run() -> None:
    workflow = Workflow("anchor")
    node = workflow.add_task(
        summarize,
        inputs={"text": "hello", "options": {}},
    )
    compiled = workflow.compile()
    provider = DeclaredOnlyAnchorProvider(environment_fingerprint="env_1")
    first = provider.resolve(run_id="run_1", compiled=compiled, task_id=node.task_id)
    repeated = provider.resolve(
        run_id="run_1", compiled=compiled, task_id=node.task_id
    )
    another_run = provider.resolve(
        run_id="run_2", compiled=compiled, task_id=node.task_id
    )
    assert first is repeated
    assert first == another_run
    assert first.strategy == "declared_only"
    assert first.effective == compiled.definitions[first.definition_id].resources
    assert first.static_inferred.cpu_num == 0
    assert first.learned is None
    assert provider.count_for_run("run_1") == 1
    assert provider.destroy_run("run_1") == 1
    assert provider.count_for_run("run_1") == 0


def test_placement_is_deterministic_atomic_and_release_is_idempotent() -> None:
    workflow = Workflow("placement")
    task = workflow.add_task(
        summarize,
        inputs={"text": "hello", "options": {}},
    )
    compiled = workflow.compile()
    anchor = DeclaredOnlyAnchorProvider(
        environment_fingerprint="env_1"
    ).resolve(run_id="run_1", compiled=compiled, task_id=task.task_id)
    placement = PlacementManager()
    placement.register_node(_node("node_b", cpu=2, memory=128))
    placement.register_node(_node("node_a", cpu=2, memory=128))

    first = placement.try_reserve(
        run_id="run_1",
        task_id=task.task_id,
        attempt=1,
        anchor=anchor,
        now_ms=1,
        dispatch_deadline_ms=100,
    )
    assert first.selected
    assert first.lease is not None
    assert first.lease.node_id == "node_a"
    assert placement.active_lease_count("run_1") == 1

    second = placement.try_reserve(
        run_id="run_2",
        task_id=task.task_id,
        attempt=1,
        anchor=anchor,
        now_ms=2,
        dispatch_deadline_ms=100,
    )
    assert second.selected
    assert second.lease is not None
    assert second.lease.node_id == "node_b"
    blocked = placement.try_reserve(
        run_id="run_3",
        task_id=task.task_id,
        attempt=1,
        anchor=anchor,
        now_ms=3,
        dispatch_deadline_ms=100,
    )
    assert not blocked.selected
    assert blocked.rejection_reason == "insufficient_cpu"

    with pytest.raises(StateTransitionError, match="attempt"):
        placement.release_lease(
            first.lease.lease_id,
            now_ms=4,
            run_id="run_1",
            task_id=task.task_id,
            attempt=2,
        )
    assert placement.release_lease(first.lease.lease_id, now_ms=4)
    assert not placement.release_lease(first.lease.lease_id, now_ms=5)
    assert placement.lease_snapshot(first.lease.lease_id).status is LeaseStatus.RELEASED
    assert placement.active_lease_count("run_1") == 0


def test_npu_best_fit_and_model_service_client_have_distinct_reservations() -> None:
    workflow = Workflow("placement-targets")
    local = workflow.add_task(local_npu_task, inputs={"value": "x"})
    service = workflow.add_task(
        service_task,
        inputs={"prompt": "x"},
        model_anchor={"model": "model_1", "mode": "service"},
    )
    compiled = workflow.compile()
    provider = DeclaredOnlyAnchorProvider(environment_fingerprint="env_1")
    local_anchor = provider.resolve(
        run_id="run_1", compiled=compiled, task_id=local.task_id
    )
    service_anchor = provider.resolve(
        run_id="run_1", compiled=compiled, task_id=service.task_id
    )
    placement = PlacementManager(npu_hbm_headroom_mb=512)
    placement.register_node(
        _node(
            "node_a",
            npus=(
                NpuCapacity("0", "910B3", 20_000, 0, 1, 20_000),
                NpuCapacity("1", "910B3", 40_000, 0, 1, 40_000),
            ),
        )
    )
    local_result = placement.try_reserve(
        run_id="run_1",
        task_id=local.task_id,
        attempt=1,
        anchor=local_anchor,
        now_ms=1,
        dispatch_deadline_ms=100,
    )
    assert local_result.lease is not None
    assert local_result.lease.npu_device_id == "0"
    assert local_result.lease.resources.npu_hbm_mb == 1024
    assert local_result.lease.resources.npu_slots == 1

    service_result = placement.try_reserve(
        run_id="run_2",
        task_id=service.task_id,
        attempt=1,
        anchor=service_anchor,
        now_ms=2,
        dispatch_deadline_ms=100,
    )
    assert service_result.lease is not None
    assert service_result.lease.reservation_kind == "model_request"
    assert service_result.lease.npu_device_id is None
    assert service_result.lease.resources.npu_hbm_mb == 0
    assert service_result.lease.resources.npu_slots == 0


def test_boot_generation_change_invalidates_old_leases() -> None:
    workflow = Workflow("boot-change")
    task = workflow.add_task(
        summarize,
        inputs={"text": "hello", "options": {}},
    )
    compiled = workflow.compile()
    anchor = DeclaredOnlyAnchorProvider(
        environment_fingerprint="env_1"
    ).resolve(run_id="run_1", compiled=compiled, task_id=task.task_id)
    placement = PlacementManager()
    placement.register_node(_node("node_a"))
    result = placement.try_reserve(
        run_id="run_1",
        task_id=task.task_id,
        attempt=1,
        anchor=anchor,
        now_ms=1,
        dispatch_deadline_ms=100,
    )
    assert result.lease is not None
    placement.register_node(_node("node_a", boot_id="boot_2"))
    assert placement.lease_snapshot(result.lease.lease_id).status is LeaseStatus.INVALIDATED
    assert placement.active_lease_count("run_1") == 0
    replacement = placement.try_reserve(
        run_id="run_1",
        task_id=task.task_id,
        attempt=2,
        anchor=anchor,
        now_ms=2,
        dispatch_deadline_ms=100,
    )
    assert replacement.lease is not None
    assert replacement.lease.boot_id == "boot_2"
    context = placement.run_snapshot("run_1")
    assert context.affinity_boot_id == "boot_2"
    assert context.affinity_epoch == 1


def test_reserved_lease_expires_and_cannot_bind_at_its_deadline() -> None:
    workflow = Workflow("lease-expiry")
    task = workflow.add_task(summarize, inputs={"text": "hello", "options": {}})
    compiled = workflow.compile()
    anchor = DeclaredOnlyAnchorProvider(
        environment_fingerprint="env_1"
    ).resolve(run_id="run_expiry", compiled=compiled, task_id=task.task_id)
    placement = PlacementManager()
    placement.register_node(_node("node_a"))
    result = placement.try_reserve(
        run_id="run_expiry",
        task_id=task.task_id,
        attempt=1,
        anchor=anchor,
        now_ms=10,
        dispatch_deadline_ms=20,
    )
    assert result.lease is not None
    assert not placement.bind_lease(result.lease.lease_id, now_ms=20)
    snapshot = placement.lease_snapshot(result.lease.lease_id)
    assert snapshot.status is LeaseStatus.EXPIRED
    assert snapshot.finished_at_ms == 20
    assert snapshot.finish_reason == "dispatch_deadline"
    assert placement.active_lease_count("run_expiry") == 0


def test_offline_node_invalidates_its_reserved_lease() -> None:
    workflow = Workflow("node-offline")
    task = workflow.add_task(summarize, inputs={"text": "hello", "options": {}})
    compiled = workflow.compile()
    anchor = DeclaredOnlyAnchorProvider(
        environment_fingerprint="env_1"
    ).resolve(run_id="run_offline", compiled=compiled, task_id=task.task_id)
    placement = PlacementManager()
    placement.register_node(_node("node_a"))
    result = placement.try_reserve(
        run_id="run_offline",
        task_id=task.task_id,
        attempt=1,
        anchor=anchor,
        now_ms=10,
        dispatch_deadline_ms=100,
    )
    assert result.lease is not None
    invalidated = placement.set_node_status(
        "node_a", NodeStatus.OFFLINE, now_ms=20
    )
    assert invalidated == (result.lease,)
    snapshot = placement.lease_snapshot(result.lease.lease_id)
    assert snapshot.status is LeaseStatus.INVALIDATED
    assert snapshot.finished_at_ms == 20
    assert snapshot.finish_reason == "node_offline"
    assert placement.active_lease_count("run_offline") == 0
