from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_managed_ray_host_owns_connection_lifecycle_in_isolated_process() -> None:
    script = """
import asyncio
import ray
from ascend_maze.control.ray_host import ManagedRayHost
from ascend_maze.runtime.ray_cluster import RayClusterConfig

async def main():
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
    )
    controller = await host.start()
    assert ray.is_initialized()
    assert controller.node_rpc_endpoint
    await host.close()
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
