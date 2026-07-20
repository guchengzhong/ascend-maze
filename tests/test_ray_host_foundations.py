from __future__ import annotations

import pytest

from ascend_maze.contracts.resources import (
    ExecutionTarget,
    PlacementLease,
    ReservationVector,
)
from ascend_maze.core.errors import StateTransitionError
from ascend_maze.runtime.ray_node_registry import (
    RayNodeRegistry,
    RuntimeNodeStatus,
)
from ascend_maze.runtime.worker_broker import ColdWorkerBroker
from ascend_maze.contracts.worker import WorkerProfile


def _lease(*, node_id: str = "node_a", boot_id: str = "boot_1") -> PlacementLease:
    return PlacementLease(
        lease_id=f"lease_{node_id}_{boot_id}",
        reservation_kind="task",
        run_id="run_1",
        task_id="task_1",
        attempt=1,
        node_id=node_id,
        boot_id=boot_id,
        npu_device_id=None,
        resources=ReservationVector(1, 64, 0, 0, 0),
        snapshot_version=1,
        created_at_ms=1,
        dispatch_deadline_ms=10_000,
    )


def _register(
    registry: RayNodeRegistry,
    *,
    boot_id: str,
    ray_node_id: str,
    agent_generation: str,
):
    return registry.register(
        node_id="node_a",
        boot_id=boot_id,
        ray_node_id=ray_node_id,
        agent_generation=agent_generation,
        agent_endpoint="127.0.0.1:41001",
        producer_id=f"node_agent:node_a:{agent_generation}",
    )


def test_node_boot_replacement_increments_generation_and_rejects_old_lease() -> None:
    registry = RayNodeRegistry()
    first, replaced = _register(
        registry,
        boot_id="boot_1",
        ray_node_id="ray_1",
        agent_generation="agent_1",
    )
    assert replaced is None
    assert first.runtime_generation == 1
    assert registry.resolve_lease(_lease()) == first

    second, replaced = _register(
        registry,
        boot_id="boot_2",
        ray_node_id="ray_2",
        agent_generation="agent_2",
    )
    assert replaced == first
    assert second.runtime_generation == 2
    assert second.boot_id == "boot_2"
    with pytest.raises(StateTransitionError, match="boot generation is stale"):
        registry.resolve_lease(_lease())
    assert registry.resolve_lease(_lease(boot_id="boot_2")) == second


def test_old_heartbeat_is_ignored_after_agent_replacement() -> None:
    registry = RayNodeRegistry()
    _register(
        registry,
        boot_id="boot_1",
        ray_node_id="ray_1",
        agent_generation="agent_1",
    )
    assert registry.heartbeat(
        node_id="node_a",
        boot_id="boot_1",
        agent_generation="agent_1",
        sequence=1,
    )
    replacement, _ = _register(
        registry,
        boot_id="boot_1",
        ray_node_id="ray_1",
        agent_generation="agent_2",
    )
    assert replacement.runtime_generation == 2
    assert not registry.heartbeat(
        node_id="node_a",
        boot_id="boot_1",
        agent_generation="agent_1",
        sequence=2,
    )
    assert registry.heartbeat(
        node_id="node_a",
        boot_id="boot_1",
        agent_generation="agent_2",
        sequence=1,
    )


def test_unhealthy_runtime_binding_rejects_placement_lease() -> None:
    registry = RayNodeRegistry()
    _register(
        registry,
        boot_id="boot_1",
        ray_node_id="ray_1",
        agent_generation="agent_1",
    )
    assert registry.set_status("node_a", RuntimeNodeStatus.OFFLINE)
    with pytest.raises(StateTransitionError, match="node is offline"):
        registry.resolve_lease(_lease())


def test_registration_and_heartbeat_cannot_clear_administrative_drain() -> None:
    registry = RayNodeRegistry()
    _register(
        registry,
        boot_id="boot_1",
        ray_node_id="ray_1",
        agent_generation="agent_1",
    )
    assert registry.set_status("node_a", RuntimeNodeStatus.DRAINING)
    _register(
        registry,
        boot_id="boot_1",
        ray_node_id="ray_1",
        agent_generation="agent_1",
    )
    assert registry.status("node_a") is RuntimeNodeStatus.DRAINING
    assert registry.heartbeat(
        node_id="node_a",
        boot_id="boot_1",
        agent_generation="agent_1",
        sequence=1,
    )
    assert registry.status("node_a") is RuntimeNodeStatus.DRAINING
    assert registry.set_status("node_a", RuntimeNodeStatus.DRAINED)
    assert registry.heartbeat(
        node_id="node_a",
        boot_id="boot_1",
        agent_generation="agent_1",
        sequence=2,
    )
    assert registry.status("node_a") is RuntimeNodeStatus.DRAINED


def test_cold_worker_lease_release_and_node_invalidation_are_idempotent() -> None:
    registry = RayNodeRegistry()
    _register(
        registry,
        boot_id="boot_1",
        ray_node_id="ray_1",
        agent_generation="agent_1",
    )
    broker = ColdWorkerBroker(
        node_registry=registry,
        environment_fingerprint="e" * 64,
    )
    first = broker.acquire(
        placement_lease=_lease(),
        task_kind="cpu",
        execution_target=ExecutionTarget.LOCAL_WORKER,
        now_ms=10,
    )
    second = broker.acquire(
        placement_lease=_lease(),
        task_kind="io",
        execution_target=ExecutionTarget.LOCAL_WORKER,
        now_ms=11,
    )
    assert broker.active_count() == 2
    assert first.profile is WorkerProfile.CPU
    assert second.profile is WorkerProfile.IO
    assert broker.active_count("node_a") == 2
    assert broker.release(first.worker_lease_id, disposition="discard")
    assert not broker.release(first.worker_lease_id, disposition="discard")
    assert broker.invalidate_node("node_a", "boot_1") == (second,)
    assert broker.invalidate_node("node_a", "boot_1") == ()
    assert broker.active_count() == 0
    assert broker.purge_released() == 2
    assert broker.purge_released() == 0
