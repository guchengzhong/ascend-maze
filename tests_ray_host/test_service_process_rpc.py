from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import os
from pathlib import Path
import signal
import sys

import pytest

from ascend_maze.control.node_rpc import (
    NodeAgent,
    NodeAgentIdentity,
    NodeControlServer,
)
from ascend_maze.control.service_process import (
    NodeAgentServiceProcessBackend,
    NodeServiceProcessManager,
)
from ascend_maze.contracts.resources import PlacementLease, ReservationVector
from ascend_maze.contracts.runtime import RuntimeDeviceMapping
from ascend_maze.core.canonical import FrozenMap
from ascend_maze.inference.contracts import ServiceLaunchRequest, ServiceProcessExit
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime.ray_node_registry import RayNodeRegistry


@dataclass
class _Device:
    used_hbm_mb: int = 100


class _Monitor:
    def device(self, physical_device_id: str) -> _Device:
        assert physical_device_id == "7"
        return _Device()

    def process_hbm_mb(self, physical_device_id: str, pid: int) -> int | None:
        assert physical_device_id == "7"
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        return 1

    def verify_process_device(
        self,
        pid: int,
        physical_device_id: str,
        *,
        deadline_seconds: float = 2.0,
        poll_interval_seconds: float = 0.05,
    ) -> bool:
        del deadline_seconds, poll_interval_seconds
        return self.process_hbm_mb(physical_device_id, pid) is not None


def _identity() -> NodeAgentIdentity:
    return NodeAgentIdentity(
        cluster_id="cluster_1",
        node_id="node_a",
        boot_id="boot_1",
        ray_node_id="ray_node_a",
        agent_generation="agent_1",
        environment_fingerprint="e" * 64,
        producer_id="node_agent:node_a:agent_1",
    )


def _placement(instance_id: str) -> PlacementLease:
    return PlacementLease(
        lease_id=f"lease_{instance_id}",
        reservation_kind="model_instance",
        run_id=None,
        task_id=None,
        attempt=None,
        node_id="node_a",
        boot_id="boot_1",
        npu_device_id="7",
        resources=ReservationVector(1, 256, 0, 1024, 1),
        snapshot_version=1,
        created_at_ms=0,
        dispatch_deadline_ms=60_000,
        allow_npu_colocation=False,
        model_instance_id=instance_id,
    )


def _request(instance_id: str, generation: int, port_lease, tmp_path: Path):
    return ServiceLaunchRequest(
        instance_id=instance_id,
        generation=generation,
        model_id="test-model",
        artifact_revision="a" * 64,
        endpoint_id=f"http://127.0.0.1:{port_lease.port}",
        port_lease_id=port_lease.port_lease_id,
        port=port_lease.port,
        argv=(
            sys.executable,
            "-m",
            "http.server",
            str(port_lease.port),
            "--bind",
            "127.0.0.1",
        ),
        working_directory=str(tmp_path),
        environment=FrozenMap((("ASCEND_RT_VISIBLE_DEVICES", "7"),)),
    )


async def _wait_port(backend, handle) -> None:
    for _ in range(100):
        probe = await backend.probe_process(handle, timeout_ms=1000)
        if probe.process_alive and probe.port_open:
            assert probe.binding_verified
            return
        await asyncio.sleep(0.02)
    raise AssertionError("test service did not open its leased port")


def test_node_agent_service_rpc_lifecycle_and_unexpected_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        mapping = RuntimeDeviceMapping("7", "0", 0)
        identity = replace(_identity(), device_mappings=(mapping,))
        registry = RayNodeRegistry()
        exits: list[ServiceProcessExit] = []
        exit_received = asyncio.Event()

        def process_exited(event: ServiceProcessExit) -> None:
            exits.append(event)
            exit_received.set()

        controller = NodeControlServer(
            cluster_id=identity.cluster_id,
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=identity.environment_fingerprint,
            registry=registry,
            recorder=InMemoryRecorder(),
            event_sink=lambda event: None,
            on_service_process_exited=process_exited,
        )
        controller_endpoint = await controller.start()
        manager = NodeServiceProcessManager(
            node_id=identity.node_id,
            boot_id=identity.boot_id,
            device_monitor=_Monitor(),
            allowed_executables=(sys.executable,),
            log_directory=tmp_path / "logs",
            first_port=32100,
            last_port=32110,
            hbm_recovery_tolerance_mb=0,
            poll_interval_ms=10,
            device_mappings=(mapping,),
        )
        agent = NodeAgent(
            identity=identity,
            authorization_token=b"test-token",
            heartbeat_interval_ms=20,
            service_process_manager=manager,
        )
        await agent.start(controller_endpoint=controller_endpoint)
        backend = NodeAgentServiceProcessBackend(
            cluster_id=identity.cluster_id,
            authorization_token=b"test-token",
            controller_generation="controller_1",
            node_registry=registry,
            rpc_timeout_ms=5_000,
        )
        try:
            port = await backend.acquire(
                node_id="node_a",
                boot_id="boot_1",
                owner_instance_id="instance_1",
                generation=1,
            )
            handle = await backend.launch(
                _request("instance_1", 1, port, tmp_path),
                _placement("instance_1"),
            )
            await _wait_port(backend, handle)
            result = await backend.stop(handle, timeout_ms=2_000)
            assert result.process_exited
            assert result.port_released
            assert result.hbm_recovered
            assert not result.forced_termination
            assert await backend.release(port)
            assert await backend.release(port)
            assert backend.active_count() == 0
            assert exits == []

            released_elsewhere = await backend.acquire(
                node_id="node_a",
                boot_id="boot_1",
                owner_instance_id="instance_released_elsewhere",
                generation=1,
            )
            assert await manager.release_port(released_elsewhere)
            assert await backend.release(released_elsewhere)
            assert await backend.release(released_elsewhere)
            with pytest.raises(RuntimeError, match="identity is stale"):
                await backend.release(replace(released_elsewhere, generation=2))
            assert backend.active_count() == 0

            crash_port = await backend.acquire(
                node_id="node_a",
                boot_id="boot_1",
                owner_instance_id="instance_2",
                generation=1,
            )
            crashed = await backend.launch(
                _request("instance_2", 1, crash_port, tmp_path),
                _placement("instance_2"),
            )
            await _wait_port(backend, crashed)
            os.kill(crashed.process_id, signal.SIGKILL)
            await asyncio.wait_for(exit_received.wait(), timeout=3)
            assert len(exits) == 1
            assert exits[0].service_handle_id == crashed.service_handle_id
            assert exits[0].instance_id == "instance_2"
            assert exits[0].exit_code == -signal.SIGKILL
            probe = await backend.probe_process(crashed, timeout_ms=1000)
            assert not probe.process_alive
            cleanup = await backend.stop(crashed, timeout_ms=2_000)
            assert cleanup.process_exited
            assert cleanup.port_released
            assert cleanup.hbm_recovered
            assert await backend.release(crash_port)
        finally:
            await backend.close()
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())


def test_node_agent_rejects_unapproved_executable_and_forces_timeout(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        identity = _identity()
        registry = RayNodeRegistry()
        controller = NodeControlServer(
            cluster_id=identity.cluster_id,
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=identity.environment_fingerprint,
            registry=registry,
            recorder=InMemoryRecorder(),
            event_sink=lambda event: None,
        )
        endpoint = await controller.start()
        manager = NodeServiceProcessManager(
            node_id="node_a",
            boot_id="boot_1",
            device_monitor=_Monitor(),
            allowed_executables=(sys.executable,),
            log_directory=tmp_path / "logs",
            first_port=32120,
            last_port=32130,
            hbm_recovery_tolerance_mb=0,
            poll_interval_ms=10,
        )
        agent = NodeAgent(
            identity=identity,
            authorization_token=b"test-token",
            service_process_manager=manager,
        )
        await agent.start(controller_endpoint=endpoint)
        backend = NodeAgentServiceProcessBackend(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            controller_generation="controller_1",
            node_registry=registry,
            rpc_timeout_ms=5_000,
        )
        try:
            port = await backend.acquire(
                node_id="node_a",
                boot_id="boot_1",
                owner_instance_id="bad",
                generation=1,
            )
            invalid = _request("bad", 1, port, tmp_path)
            invalid = ServiceLaunchRequest(
                instance_id=invalid.instance_id,
                generation=invalid.generation,
                model_id=invalid.model_id,
                artifact_revision=invalid.artifact_revision,
                endpoint_id=invalid.endpoint_id,
                port_lease_id=invalid.port_lease_id,
                port=invalid.port,
                argv=("/bin/sh", "-c", "exit 0"),
                working_directory=invalid.working_directory,
                environment=invalid.environment,
            )
            with pytest.raises(RuntimeError, match="not allowed"):
                await backend.launch(invalid, _placement("bad"))
            assert await backend.release(port)

            timeout_port = await backend.acquire(
                node_id="node_a",
                boot_id="boot_1",
                owner_instance_id="timeout",
                generation=1,
            )
            request = _request("timeout", 1, timeout_port, tmp_path)
            request = ServiceLaunchRequest(
                instance_id=request.instance_id,
                generation=request.generation,
                model_id=request.model_id,
                artifact_revision=request.artifact_revision,
                endpoint_id=request.endpoint_id,
                port_lease_id=request.port_lease_id,
                port=request.port,
                argv=(
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
                ),
                working_directory=request.working_directory,
                environment=request.environment,
            )
            handle = await backend.launch(request, _placement("timeout"))
            await asyncio.sleep(0.1)
            stopped = await backend.stop(handle, timeout_ms=300)
            assert stopped.process_exited
            assert stopped.forced_termination
            assert stopped.port_released
            assert stopped.hbm_recovered
            assert await backend.release(timeout_port)
        finally:
            await backend.close()
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())
