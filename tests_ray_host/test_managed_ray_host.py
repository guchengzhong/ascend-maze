from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_managed_ray_host_owns_connection_lifecycle_in_isolated_process(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "runtime" / "control.sock"
    script = f"""
import asyncio
import ray
from pathlib import Path
from ascend_maze.control.ray_host import ManagedRayHost
from ascend_maze.control.local_rpc import UdsRuntimeClient
from ascend_maze.runtime.ray_cluster import RayClusterConfig

async def main():
    socket_path = Path({str(socket_path)!r})
    host = ManagedRayHost(
        ray_config=RayClusterConfig(
            namespace='managed-ray-host-test',
            local_num_cpus=2,
            local_object_store_memory=96 * 1024 * 1024,
        ),
        cluster_id='managed_cluster',
        authorization_token=b'test-token',
        config_fingerprint='c' * 64,
        environment_fingerprint='e' * 64,
        build_revision='test',
        node_capacities=(),
        control_socket_path=socket_path,
    )
    controller = await host.start()
    assert ray.is_initialized()
    assert controller.node_rpc_endpoint
    client = UdsRuntimeClient(socket_path)
    await client.get_controller_status()
    shutdown = asyncio.create_task(
        client.shutdown_controller(force=False, drain_timeout_ms=1_000)
    )
    await controller.wait_stopped()
    await host.close()
    result = await shutdown
    assert result['mode'] == 'graceful'
    assert result['cleanup_confirmed'] is True
    assert result['recording_complete'] is True
    assert result['exit_code'] == 0
    assert result['incomplete_resources'] == []
    assert not socket_path.exists()
    assert not ray.is_initialized()

asyncio.run(main())
"""
    environment = os.environ.copy()
    environment.pop("RAY_ADDRESS", None)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_managed_ray_host_reverses_partial_startup_failures(
    tmp_path: Path,
) -> None:
    script = f"""
import asyncio
import ray
from pathlib import Path
from ascend_maze.control.ray_host import ManagedRayHost
from ascend_maze.runtime.ray_cluster import RayClusterConfig

root = Path({str(tmp_path)!r})

class FailingAgent:
    def __init__(self):
        self.closed = False

    async def start(self, **kwargs):
        raise RuntimeError('injected head-agent startup failure')

    async def close(self, grace_seconds=0):
        self.closed = True

async def run_case(name, agent_factory=None, fail_local_rpc=False):
    runtime = root / name
    socket_path = runtime / 'control.sock'
    pid_path = runtime / 'controller.pid'
    agent = None if agent_factory is None else agent_factory()
    host = ManagedRayHost(
        ray_config=RayClusterConfig(
            namespace=f'startup-failure-{{name}}',
            local_num_cpus=2,
            local_object_store_memory=96 * 1024 * 1024,
        ),
        cluster_id=f'cluster_{{name}}',
        authorization_token=b'test-token',
        config_fingerprint='c' * 64,
        environment_fingerprint='e' * 64,
        build_revision='test',
        node_capacities=(),
        control_socket_path=socket_path,
        pid_lock_path=pid_path,
        head_node_agent_factory=(None if agent is None else lambda: agent),
    )
    if fail_local_rpc:
        original = host.controller
        from ascend_maze.control.local_rpc import LocalControlServer
        original_start = LocalControlServer.start
        async def fail_start(self):
            raise RuntimeError('injected UDS startup failure')
        LocalControlServer.start = fail_start
    try:
        try:
            await host.start()
        except RuntimeError as exc:
            assert 'injected' in str(exc)
        else:
            raise AssertionError('startup failure was not propagated')
    finally:
        if fail_local_rpc:
            LocalControlServer.start = original_start
        await host.close()
    if agent is not None:
        assert agent.closed
    assert not ray.is_initialized()
    assert not pid_path.exists()
    assert not socket_path.exists()

async def main():
    await run_case('head-agent', agent_factory=FailingAgent)
    await run_case('local-rpc', fail_local_rpc=True)

asyncio.run(main())
"""
    environment = os.environ.copy()
    environment.pop("RAY_ADDRESS", None)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
