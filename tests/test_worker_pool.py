from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

import pytest

from ascend_maze.contracts.resources import (
    ExecutionTarget,
    ReservationVector,
    ResourceSpec,
)
from ascend_maze.contracts.runtime import RuntimeNodeBinding
from ascend_maze.contracts.worker import (
    StandbyWorkerState,
    StandbyWarmupReport,
    WarmupManifest,
    WorkerPoolConfig,
    WorkerPoolProfileConfig,
    WorkerProfile,
)
from ascend_maze.core.time import monotonic_time_ms
from ascend_maze.core.errors import ContractValidationError, StateTransitionError
from ascend_maze.placement import NodeCapacity, NpuCapacity, PlacementManager
from ascend_maze.resources import ResourceAnchor
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry
from ascend_maze.runtime.worker_pool import StandbyWorkerBroker


@dataclass
class _Endpoint:
    worker_id: str
    node_id: str
    terminated: bool = False


class _EndpointFactory:
    def __init__(
        self,
        *,
        fail_starts: int = 0,
        fail_terminations: int = 0,
    ) -> None:
        self.started: list[_Endpoint] = []
        self.submissions: list[dict[str, object]] = []
        self.fail_starts = fail_starts
        self.fail_terminations = fail_terminations
        self.termination_timeouts: list[int] = []

    async def start(
        self,
        *,
        worker_id: str,
        worker_generation: int,
        binding: RuntimeNodeBinding,
        config: WorkerPoolProfileConfig,
        deadline_ms: int,
    ) -> tuple[Any, StandbyWarmupReport]:
        assert deadline_ms > monotonic_time_ms()
        if self.fail_starts:
            self.fail_starts -= 1
            raise RuntimeError("injected endpoint start failure")
        endpoint = _Endpoint(worker_id, binding.node_id)
        self.started.append(endpoint)
        return endpoint, StandbyWarmupReport(
            worker_id=worker_id,
            worker_generation=worker_generation,
            ray_node_id=binding.ray_node_id,
            worker_pid=10_000 + len(self.started),
            imported_modules=config.warmup_manifest.modules,
            forbidden_device_modules=(),
            host_rss_mb=32,
            host_warmup_ms=2,
        )

    def submit(self, endpoint: Any, kwargs: dict[str, object]) -> Any:
        assert isinstance(endpoint, _Endpoint)
        assert not endpoint.terminated
        self.submissions.append(kwargs)
        return kwargs

    async def terminate(
        self,
        endpoint: Any,
        *,
        force: bool = False,
        timeout_ms: int = 10_000,
    ) -> None:
        del force
        assert isinstance(endpoint, _Endpoint)
        self.termination_timeouts.append(timeout_ms)
        if self.fail_terminations:
            self.fail_terminations -= 1
            raise RuntimeError("injected endpoint termination failure")
        endpoint.terminated = True


class _GatedEndpointFactory(_EndpointFactory):
    def __init__(self) -> None:
        super().__init__()
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()

    async def start(
        self,
        *,
        worker_id: str,
        worker_generation: int,
        binding: RuntimeNodeBinding,
        config: WorkerPoolProfileConfig,
        deadline_ms: int,
    ) -> tuple[Any, StandbyWarmupReport]:
        self.start_entered.set()
        await self.release_start.wait()
        return await super().start(
            worker_id=worker_id,
            worker_generation=worker_generation,
            binding=binding,
            config=config,
            deadline_ms=deadline_ms,
        )


def _pool_config(profile: WorkerProfile, *, min_idle: int = 1) -> WorkerPoolConfig:
    resources = ReservationVector(
        cpu_num=1,
        host_mem_mb=64,
        io_slots=1 if profile is WorkerProfile.IO else 0,
        npu_hbm_mb=0,
        npu_slots=0,
    )
    return WorkerPoolConfig(
        mode="zero_hbm_standby",
        profiles=(
            WorkerPoolProfileConfig(
                profile=profile,
                min_idle=min_idle,
                max_idle=max(1, min_idle),
                max_total=2,
                replenish_concurrency=1,
                idle_ttl_ms=10_000,
                acquire_timeout_ms=1_000,
                max_tasks_per_worker=3,
                max_worker_lifetime_ms=60_000,
                max_rss_growth_mb=64,
                standby_resources=resources,
                termination_timeout_ms=12_345,
                warmup_manifest=WarmupManifest(("json",)),
            ),
        ),
        reconcile_interval_ms=10,
    )


def _components(
    profile: WorkerProfile,
    *,
    cpu: int = 2,
    with_npu: bool = False,
) -> tuple[PlacementManager, RayNodeRegistry, _EndpointFactory, StandbyWorkerBroker]:
    placement = PlacementManager()
    placement.register_node(
        NodeCapacity(
            node_id="node_a",
            boot_id="boot_1",
            node_ip="127.0.0.1",
            cpu_total=cpu,
            mem_total_mb=256,
            cpu_system_reserved=0,
            mem_system_reserved_mb=0,
            io_slots_total=2,
            npus=(NpuCapacity("0", "910B3", 65_536, 4_096, 1, 61_440),)
            if with_npu
            else (),
            observed_free_mem_mb=256,
        )
    )
    registry = RayNodeRegistry()
    registry.register(
        node_id="node_a",
        boot_id="boot_1",
        ray_node_id="ray_a",
        agent_generation="agent_1",
        agent_endpoint="127.0.0.1:1",
        producer_id="node_agent:1",
    )
    factory = _EndpointFactory()
    broker = StandbyWorkerBroker(
        node_registry=registry,
        placement=placement,
        environment_fingerprint="e" * 64,
        config=_pool_config(profile),
        endpoint_factory=factory,
    )
    return placement, registry, factory, broker


def _anchor(profile: WorkerProfile) -> ResourceAnchor:
    task_kind = {
        WorkerProfile.CPU: "cpu",
        WorkerProfile.IO: "io",
        WorkerProfile.NPU_HOST: "npu",
    }[profile]
    resources = ResourceSpec(
        cpu_num=1,
        mem_mb=64,
        npu_mem_mb=1024 if profile is WorkerProfile.NPU_HOST else 0,
        io_num=1 if profile is WorkerProfile.IO else 0,
    )
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


def test_reconciler_never_spawns_without_a_standby_reservation() -> None:
    async def scenario() -> None:
        placement, _, factory, broker = _components(WorkerProfile.CPU, cpu=0)
        await broker.reconcile_once()
        snapshot = broker.snapshot()
        assert snapshot.reservation_failures == 1
        assert snapshot.workers == ()
        assert factory.started == []
        assert placement.active_lease_count() == 0
        await broker.close()

    asyncio.run(scenario())


def test_standby_hit_and_sanitized_cpu_reuse_keep_one_reservation() -> None:
    async def scenario() -> None:
        placement, _, factory, broker = _components(WorkerProfile.CPU)
        await broker.reconcile_once()
        idle = broker.snapshot().workers[0]
        assert idle.state is StandbyWorkerState.IDLE
        assert placement.ready_standby_count(profile="cpu") == 1
        result = placement.try_reserve(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            anchor=_anchor(WorkerProfile.CPU),
            now_ms=monotonic_time_ms(),
            dispatch_deadline_ms=monotonic_time_ms() + 1_000,
        )
        assert result.lease is not None
        worker_lease = await broker.acquire(
            placement_lease=result.lease,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            now_ms=monotonic_time_ms(),
        )
        assert worker_lease.source == "standby"
        assert worker_lease.cold_start_ms == 0
        assert worker_lease.host_warmup_ms == 2
        assert worker_lease.worker_id == idle.worker_id
        active_snapshot = broker.snapshot().worker_leases
        assert broker.snapshot().active_worker_lease_count == 1
        assert len(active_snapshot) == 1
        assert active_snapshot[0].lease == worker_lease
        assert active_snapshot[0].placement_lease_id == result.lease.lease_id
        assert not active_snapshot[0].released
        assert not active_snapshot[0].releasing
        assert active_snapshot[0].disposition is None
        assert broker.submit(worker_lease.worker_lease_id, {"request": "same"}) == {
            "request": "same"
        }
        assert await broker.release(worker_lease.worker_lease_id, disposition="reuse")
        assert not await broker.release(
            worker_lease.worker_lease_id, disposition="reuse"
        )
        returned = broker.snapshot()
        assert returned.standby_hits == 1
        assert returned.cold_starts == 0
        assert returned.workers[0].state is StandbyWorkerState.IDLE
        assert returned.workers[0].tasks_completed == 1
        assert returned.worker_leases[0].released
        assert returned.worker_leases[0].disposition == "reuse"
        assert returned.active_worker_lease_count == 0
        assert len(factory.started) == 1
        assert placement.active_lease_count("run_1") == 0
        assert placement.active_lease_count() == 1
        await broker.close()
        assert factory.started[0].terminated
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())


def test_npu_worker_is_one_shot_and_reconciler_replenishes_it() -> None:
    async def scenario() -> None:
        placement, _, factory, broker = _components(
            WorkerProfile.NPU_HOST, with_npu=True
        )
        await broker.reconcile_once()
        original = broker.snapshot().workers[0]
        result = placement.try_reserve(
            run_id="run_npu",
            task_id="task_npu",
            attempt=1,
            anchor=_anchor(WorkerProfile.NPU_HOST),
            now_ms=monotonic_time_ms(),
            dispatch_deadline_ms=monotonic_time_ms() + 1_000,
        )
        assert result.lease is not None
        worker_lease = await broker.acquire(
            placement_lease=result.lease,
            task_kind="npu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            now_ms=monotonic_time_ms(),
        )
        assert await broker.release(worker_lease.worker_lease_id, disposition="reuse")
        assert factory.started[0].terminated
        assert broker.snapshot().workers[0].state is StandbyWorkerState.DEAD
        assert placement.release_lease(result.lease.lease_id, now_ms=monotonic_time_ms())
        await broker.reconcile_once()
        current = [
            worker
            for worker in broker.snapshot().workers
            if worker.state is StandbyWorkerState.IDLE
        ]
        assert len(current) == 1
        assert current[0].worker_id != original.worker_id
        assert len(factory.started) == 2
        assert placement.ready_standby_count(profile="npu_host") == 1
        await broker.close()

    asyncio.run(scenario())


def test_cold_mode_uses_the_same_endpoint_protocol_on_the_leased_node() -> None:
    async def scenario() -> None:
        placement, registry, factory, _ = _components(WorkerProfile.CPU)
        profile_config = _pool_config(WorkerProfile.CPU).profiles[0]
        cold = WorkerPoolConfig(
            mode="cold_start",
            profiles=(
                WorkerPoolProfileConfig(
                    profile=profile_config.profile,
                    min_idle=0,
                    max_idle=0,
                    max_total=1,
                    replenish_concurrency=profile_config.replenish_concurrency,
                    idle_ttl_ms=profile_config.idle_ttl_ms,
                    acquire_timeout_ms=profile_config.acquire_timeout_ms,
                    max_tasks_per_worker=profile_config.max_tasks_per_worker,
                    max_worker_lifetime_ms=profile_config.max_worker_lifetime_ms,
                    max_rss_growth_mb=profile_config.max_rss_growth_mb,
                    standby_resources=profile_config.standby_resources,
                    termination_timeout_ms=profile_config.termination_timeout_ms,
                    warmup_manifest=profile_config.warmup_manifest,
                ),
            ),
        )
        broker = StandbyWorkerBroker(
            node_registry=registry,
            placement=placement,
            environment_fingerprint="e" * 64,
            config=cold,
            endpoint_factory=factory,
        )
        await broker.reconcile_once()
        assert factory.started == []
        result = placement.try_reserve(
            run_id="run_cold",
            task_id="task_cold",
            attempt=1,
            anchor=_anchor(WorkerProfile.CPU),
            now_ms=monotonic_time_ms(),
            dispatch_deadline_ms=monotonic_time_ms() + 1_000,
        )
        assert result.lease is not None
        worker_lease = await broker.acquire(
            placement_lease=result.lease,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            now_ms=monotonic_time_ms(),
        )
        assert worker_lease.source == "cold_start"
        assert worker_lease.worker_acquire_ms >= worker_lease.cold_start_ms
        assert worker_lease.host_warmup_ms == 2
        assert factory.started[0].node_id == result.lease.node_id
        assert broker.submit(worker_lease.worker_lease_id, {"protocol": "shared"}) == {
            "protocol": "shared"
        }
        await broker.release(worker_lease.worker_lease_id, disposition="reuse")
        assert factory.started[0].terminated
        assert factory.termination_timeouts == [12_345]
        assert broker.snapshot().workers[0].state is StandbyWorkerState.DEAD
        assert placement.release_lease(result.lease.lease_id, now_ms=monotonic_time_ms())
        await broker.close()

    asyncio.run(scenario())


def test_concurrent_cold_acquire_reserves_max_total_before_endpoint_start() -> None:
    async def scenario() -> None:
        placement, registry, _, _ = _components(WorkerProfile.CPU, cpu=2)
        profile = replace(
            _pool_config(WorkerProfile.CPU, min_idle=0).profiles[0],
            max_idle=0,
            max_total=1,
        )
        factory = _GatedEndpointFactory()
        broker = StandbyWorkerBroker(
            node_registry=registry,
            placement=placement,
            environment_fingerprint="e" * 64,
            config=WorkerPoolConfig(mode="cold_start", profiles=(profile,)),
            endpoint_factory=factory,
        )
        first = placement.try_reserve(
            run_id="run_1",
            task_id="task_1",
            attempt=1,
            anchor=_anchor(WorkerProfile.CPU),
            now_ms=monotonic_time_ms(),
            dispatch_deadline_ms=monotonic_time_ms() + 1_000,
        )
        second = placement.try_reserve(
            run_id="run_2",
            task_id="task_2",
            attempt=1,
            anchor=_anchor(WorkerProfile.CPU),
            now_ms=monotonic_time_ms(),
            dispatch_deadline_ms=monotonic_time_ms() + 1_000,
        )
        assert first.lease is not None and second.lease is not None

        first_acquire = asyncio.create_task(
            broker.acquire(
                placement_lease=first.lease,
                task_kind="cpu",
                execution_target=ExecutionTarget.LOCAL_WORKER,
                now_ms=monotonic_time_ms(),
            )
        )
        await factory.start_entered.wait()
        assert broker.live_count() == 1
        with pytest.raises(StateTransitionError, match="max_total"):
            await broker.acquire(
                placement_lease=second.lease,
                task_kind="cpu",
                execution_target=ExecutionTarget.LOCAL_WORKER,
                now_ms=monotonic_time_ms(),
            )

        factory.release_start.set()
        worker_lease = await first_acquire
        assert broker.live_count() == 1
        await broker.release(worker_lease.worker_lease_id, disposition="discard")
        placement.release_lease(first.lease.lease_id, now_ms=monotonic_time_ms())
        placement.release_lease(second.lease.lease_id, now_ms=monotonic_time_ms())
        await broker.close()

    asyncio.run(scenario())


def test_cancelled_cold_start_releases_starting_capacity_slot() -> None:
    async def scenario() -> None:
        placement, registry, _, _ = _components(WorkerProfile.CPU, cpu=1)
        profile = replace(
            _pool_config(WorkerProfile.CPU, min_idle=0).profiles[0],
            max_idle=0,
            max_total=1,
        )
        factory = _GatedEndpointFactory()
        broker = StandbyWorkerBroker(
            node_registry=registry,
            placement=placement,
            environment_fingerprint="e" * 64,
            config=WorkerPoolConfig(mode="cold_start", profiles=(profile,)),
            endpoint_factory=factory,
        )
        result = placement.try_reserve(
            run_id="run_cancel_start",
            task_id="task_cancel_start",
            attempt=1,
            anchor=_anchor(WorkerProfile.CPU),
            now_ms=monotonic_time_ms(),
            dispatch_deadline_ms=monotonic_time_ms() + 1_000,
        )
        assert result.lease is not None
        acquire = asyncio.create_task(
            broker.acquire(
                placement_lease=result.lease,
                task_kind="cpu",
                execution_target=ExecutionTarget.LOCAL_WORKER,
                now_ms=monotonic_time_ms(),
            )
        )
        await factory.start_entered.wait()
        assert broker.live_count() == 1
        acquire.cancel()
        with pytest.raises(asyncio.CancelledError):
            await acquire
        assert broker.live_count() == 0
        assert broker.snapshot().active_worker_lease_count == 0

        factory.release_start.set()
        replacement = await broker.acquire(
            placement_lease=result.lease,
            task_kind="cpu",
            execution_target=ExecutionTarget.LOCAL_WORKER,
            now_ms=monotonic_time_ms(),
        )
        assert broker.live_count() == 1
        await broker.release(replacement.worker_lease_id, disposition="discard")
        placement.release_lease(result.lease.lease_id, now_ms=monotonic_time_ms())
        await broker.close()

    asyncio.run(scenario())


def test_worker_pool_contract_rejects_device_warmup_and_npu_reservation() -> None:
    with pytest.raises(ContractValidationError, match="device runtime modules"):
        WorkerPoolProfileConfig(
            profile=WorkerProfile.NPU_HOST,
            min_idle=1,
            max_idle=1,
            max_total=1,
            replenish_concurrency=1,
            idle_ttl_ms=1,
            acquire_timeout_ms=1,
            max_tasks_per_worker=1,
            max_worker_lifetime_ms=1,
            max_rss_growth_mb=1,
            standby_resources=ReservationVector(1, 1, 0, 0, 0),
            warmup_manifest=WarmupManifest(("torch_npu",)),
        )
    with pytest.raises(ContractValidationError, match="cannot reserve NPU"):
        WorkerPoolProfileConfig(
            profile=WorkerProfile.NPU_HOST,
            min_idle=1,
            max_idle=1,
            max_total=1,
            replenish_concurrency=1,
            idle_ttl_ms=1,
            acquire_timeout_ms=1,
            max_tasks_per_worker=1,
            max_worker_lifetime_ms=1,
            max_rss_growth_mb=1,
            standby_resources=ReservationVector(1, 1, 0, 1, 1),
        )
    with pytest.raises(ContractValidationError, match="deadlines must be positive"):
        WorkerPoolProfileConfig(
            profile=WorkerProfile.NPU_HOST,
            min_idle=1,
            max_idle=1,
            max_total=1,
            replenish_concurrency=1,
            idle_ttl_ms=1,
            acquire_timeout_ms=1,
            max_tasks_per_worker=1,
            max_worker_lifetime_ms=1,
            max_rss_growth_mb=1,
            standby_resources=ReservationVector(1, 1, 0, 0, 0),
            termination_timeout_ms=0,
        )


def test_replenish_failure_rolls_back_reservation_before_retry() -> None:
    async def scenario() -> None:
        placement, registry, _, _ = _components(WorkerProfile.CPU)
        factory = _EndpointFactory(fail_starts=1)
        broker = StandbyWorkerBroker(
            node_registry=registry,
            placement=placement,
            environment_fingerprint="e" * 64,
            config=_pool_config(WorkerProfile.CPU),
            endpoint_factory=factory,
        )
        await broker.reconcile_once()
        assert broker.snapshot().replenish_failures == 1
        assert broker.snapshot().workers == ()
        assert placement.active_lease_count() == 0
        await broker.reconcile_once()
        assert len(factory.started) == 1
        assert broker.snapshot().workers[0].state is StandbyWorkerState.IDLE
        assert placement.active_lease_count() == 1
        await broker.close()

    asyncio.run(scenario())


def test_boot_replacement_retires_old_worker_and_reserves_new_generation() -> None:
    async def scenario() -> None:
        placement, registry, factory, broker = _components(WorkerProfile.CPU)
        await broker.reconcile_once()
        old = broker.snapshot().workers[0]
        placement.register_node(
            replace(placement.snapshot().nodes[0].capacity, boot_id="boot_2")
        )
        registry.register(
            node_id="node_a",
            boot_id="boot_2",
            ray_node_id="ray_a",
            agent_generation="agent_2",
            agent_endpoint="127.0.0.1:2",
            producer_id="node_agent:2",
        )
        broker.invalidate_node("node_a", "boot_1")
        await asyncio.sleep(0)
        await broker.reconcile_once()
        current = broker.snapshot().workers
        assert len(current) == 1
        assert current[0].boot_id == "boot_2"
        assert current[0].worker_id != old.worker_id
        assert factory.started[0].terminated
        assert placement.ready_standby_count(
            node_id="node_a", boot_id="boot_2", profile="cpu"
        ) == 1
        assert placement.active_lease_count() == 1
        await broker.close()

    asyncio.run(scenario())


def test_new_pool_config_generation_retires_old_reservations() -> None:
    async def scenario() -> None:
        placement, _, factory, broker = _components(WorkerProfile.CPU)
        await broker.reconcile_once()
        assert broker.snapshot().config_generation == 1
        old_profile = broker.config.profiles[0]
        replacement = WorkerPoolConfig(
            mode="zero_hbm_standby",
            profiles=(replace(old_profile, min_idle=0, max_idle=0),),
            reconcile_interval_ms=broker.config.reconcile_interval_ms,
            config_generation=2,
        )
        broker.update_config(replacement)
        with pytest.raises(StateTransitionError, match="generation"):
            broker.update_config(replacement)
        await broker.reconcile_once()
        assert broker.snapshot().config_generation == 2
        assert broker.snapshot().workers == ()
        assert factory.started[0].terminated
        assert placement.active_lease_count() == 0
        await broker.close()

    asyncio.run(scenario())


def test_failed_process_exit_keeps_reservation_and_close_can_retry() -> None:
    async def scenario() -> None:
        placement, registry, _, _ = _components(WorkerProfile.CPU)
        factory = _EndpointFactory(fail_terminations=1)
        broker = StandbyWorkerBroker(
            node_registry=registry,
            placement=placement,
            environment_fingerprint="e" * 64,
            config=_pool_config(WorkerProfile.CPU),
            endpoint_factory=factory,
        )
        await broker.reconcile_once()
        with pytest.raises(RuntimeError, match="close failed"):
            await broker.close()
        assert placement.active_lease_count() == 1
        assert broker.snapshot().termination_failures == 1
        await broker.close()
        assert factory.started[0].terminated
        assert placement.active_lease_count() == 0

    asyncio.run(scenario())
