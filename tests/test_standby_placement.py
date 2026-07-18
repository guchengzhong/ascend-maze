from __future__ import annotations

from ascend_maze.contracts.resources import (
    ExecutionTarget,
    ReservationVector,
    ResourceSpec,
)
from ascend_maze.placement import (
    LeaseStatus,
    NodeCapacity,
    NodeStatus,
    NpuCapacity,
    PlacementManager,
    StandbyReservationStatus,
)
from ascend_maze.resources import ResourceAnchor


def _node(*, cpu: int = 1, memory: int = 128, with_npu: bool = False) -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_1",
        node_ip="127.0.0.1",
        cpu_total=cpu,
        mem_total_mb=memory,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=1,
        npus=(NpuCapacity("0", "910B3", 65_536, 4_096, 1, 61_440),)
        if with_npu
        else (),
        observed_free_mem_mb=memory,
    )


def _anchor(
    *,
    task_kind: str = "cpu",
    cpu: int = 1,
    memory: int = 64,
    npu_hbm: int = 0,
) -> ResourceAnchor:
    resources = ResourceSpec(cpu, memory, npu_hbm, 0)
    return ResourceAnchor(
        definition_id="definition_1",
        task_kind=task_kind,
        execution_target=ExecutionTarget.LOCAL_WORKER,
        declared=resources,
        static_inferred=ResourceSpec(0, 0, 0, 0),
        learned=None,
        effective=resources,
        model_id=None,
        profile_key="profile_1",
        revision=1,
        strategy="declared_only",
    )


def _ready_standby(
    placement: PlacementManager,
    *,
    worker_id: str = "worker_1",
    profile: str = "cpu",
    resources: ReservationVector = ReservationVector(1, 64, 0, 0, 0),
) -> str:
    lease = placement.reserve_standby(
        worker_id=worker_id,
        worker_generation=1,
        profile=profile,
        node_id="node_a",
        boot_id="boot_1",
        resources=resources,
        now_ms=1,
        startup_deadline_ms=100,
    )
    assert lease is not None
    assert placement.activate_standby(
        worker_id=worker_id,
        worker_generation=1,
        lease_id=lease.lease_id,
        now_ms=2,
    )
    return lease.lease_id


def test_standby_to_task_is_one_atomic_positive_difference_conversion() -> None:
    placement = PlacementManager()
    placement.register_node(_node(cpu=1, memory=64))
    standby_lease_id = _ready_standby(placement)

    before = placement.snapshot().nodes[0]
    assert before.reserved == ReservationVector(1, 64, 0, 0, 0)
    result = placement.try_reserve(
        run_id="run_1",
        task_id="task_1",
        attempt=1,
        anchor=_anchor(),
        now_ms=3,
        dispatch_deadline_ms=100,
    )

    assert result.selected
    assert result.lease is not None
    assert result.standby_worker_id == "worker_1"
    assert result.converted_standby_lease_id == standby_lease_id
    assert result.lease.standby_worker_id == "worker_1"
    assert result.lease.converted_standby_lease_id == standby_lease_id
    assert (
        placement.lease_snapshot(standby_lease_id).status is LeaseStatus.CONVERTED
    )
    standby = placement.standby_snapshot("worker_1")
    assert standby.status is StandbyReservationStatus.CONVERTED
    assert standby.converted_task_lease_id == result.lease.lease_id
    after = placement.snapshot().nodes[0]
    assert after.reserved == ReservationVector(1, 64, 0, 0, 0)
    assert placement.active_lease_count() == 1


def test_failed_conversion_preserves_ready_standby_and_its_reservation() -> None:
    placement = PlacementManager()
    placement.register_node(_node(cpu=1, memory=64, with_npu=True))
    standby_lease_id = _ready_standby(placement, profile="npu_host")

    result = placement.try_reserve(
        run_id="run_1",
        task_id="task_1",
        attempt=1,
        anchor=_anchor(task_kind="npu", npu_hbm=62_000),
        now_ms=3,
        dispatch_deadline_ms=100,
    )

    assert not result.selected
    assert result.rejection_reason == "resource_request_unsatisfiable"
    standby = placement.standby_snapshot("worker_1")
    assert standby.status is StandbyReservationStatus.READY
    assert standby.lease.lease_id == standby_lease_id
    assert placement.lease_snapshot(standby_lease_id).status is LeaseStatus.BOUND
    assert placement.snapshot().nodes[0].reserved == ReservationVector(1, 64, 0, 0, 0)


def test_sanitized_cpu_task_can_atomically_return_to_standby() -> None:
    placement = PlacementManager()
    placement.register_node(_node(cpu=1, memory=64))
    original_standby = _ready_standby(placement)
    result = placement.try_reserve(
        run_id="run_1",
        task_id="task_1",
        attempt=1,
        anchor=_anchor(),
        now_ms=3,
        dispatch_deadline_ms=100,
    )
    assert result.lease is not None
    assert placement.bind_lease(result.lease.lease_id, now_ms=4)

    restored = placement.restore_task_to_standby(
        task_lease_id=result.lease.lease_id,
        worker_id="worker_1",
        worker_generation=1,
        profile="cpu",
        resources=ReservationVector(1, 64, 0, 0, 0),
        now_ms=5,
        idle_deadline_ms=1_000,
    )

    assert restored is not None
    assert restored.lease_id != original_standby
    assert placement.lease_snapshot(result.lease.lease_id).status is LeaseStatus.CONVERTED
    assert placement.lease_snapshot(restored.lease_id).status is LeaseStatus.BOUND
    assert placement.standby_snapshot("worker_1").status is StandbyReservationStatus.READY
    assert placement.active_lease_count("run_1") == 0
    assert placement.active_lease_count() == 1


def test_draining_node_retires_idle_standby_without_touching_other_nodes() -> None:
    placement = PlacementManager()
    placement.register_node(_node())
    lease_id = _ready_standby(placement)

    invalidated = placement.set_node_status(
        "node_a", NodeStatus.DRAINING, now_ms=10
    )

    assert invalidated == (placement.lease_snapshot(lease_id).lease,)
    assert placement.lease_snapshot(lease_id).status is LeaseStatus.INVALIDATED
    assert placement.standby_snapshot("worker_1").status is StandbyReservationStatus.RETIRED
    assert placement.active_lease_count() == 0
