from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import os
from uuid import uuid4

import pytest
import ray

from ascend_maze.ascend import (
    AscendCorrectnessConfig,
    AscendDeviceSnapshot,
    AscendEnvironmentSnapshot,
    DcmiDeviceAdapter,
    create_ascend_correctness_config_snapshot,
    discover_ascend_environment,
)
from ascend_maze.contracts.config import ConfigSnapshot
from ascend_maze.runtime.ray_cluster import ManagedRayCluster, RayClusterConfig


@dataclass(frozen=True, slots=True)
class AscendAdmission:
    adapter: DcmiDeviceAdapter
    environment: AscendEnvironmentSnapshot
    config: AscendCorrectnessConfig
    config_snapshot: ConfigSnapshot
    device: AscendDeviceSnapshot


@pytest.fixture(scope="session")
def ascend_admission() -> AscendAdmission:
    if ray.__version__ != "2.55.1":
        raise RuntimeError(f"stage 4 requires ray==2.55.1, found {ray.__version__}")
    adapter = DcmiDeviceAdapter()
    devices = adapter.devices()
    eligible = [
        item
        for item in devices
        if item.health == "healthy"
        and not item.processes
        and item.total_hbm_mb >= 65_000
    ]
    if not eligible:
        raise RuntimeError("stage 4 requires one idle healthy 64 GiB Ascend NPU")
    environment = discover_ascend_environment(adapter, devices)
    expected = {
        "torch_npu": "2.7.1.post2",
        "ray": "2.55.1",
        "cloudpickle": "3.1.2",
        "cann": "9.0.0-beta.2",
        "driver": "25.3.rc1",
        "atb": "9.0.0",
        "atb_libmki_sha256": (
            "41d55d3994ab35b0460a0ce12aec1a35c6a9ed515d3d6424e654465a44d0f27f"
        ),
        "atb_libtbe_adapter_sha256": (
            "f9b332bd0fe8d8ba39f78fc1042b360c4517615d97a075fca5edd5393e69a108"
        ),
    }
    mismatches = {
        name: (environment.versions.get(name), version)
        for name, version in expected.items()
        if environment.versions.get(name) != version
    }
    if mismatches:
        raise RuntimeError(f"stage 4 environment version mismatch: {mismatches}")
    config = AscendCorrectnessConfig()
    config_snapshot = create_ascend_correctness_config_snapshot(
        config,
        environment,
        source_path="/etc/ascend-maze/correctness.toml",
        build_revision="stage4-test-build",
        created_at_ms=0,
    )
    return AscendAdmission(
        adapter=adapter,
        environment=environment,
        config=config,
        config_snapshot=config_snapshot,
        device=eligible[0],
    )


@pytest.fixture(scope="session")
def ascend_ray(ascend_admission: AscendAdmission) -> Iterator[str]:
    del ascend_admission
    if ray.is_initialized():
        raise RuntimeError("Ray was initialized before the stage 4 admission fixture")
    namespace = f"ascend-maze-stage4-{uuid4().hex}"
    variable = "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"
    previous = os.environ.get(variable)
    cluster = ManagedRayCluster(
        RayClusterConfig(
            namespace=namespace,
            include_dashboard=False,
            local_num_cpus=4,
            local_object_store_memory=256 * 1024 * 1024,
            disable_ray_npu_resource=True,
        )
    )
    cluster.start()
    try:
        assert os.environ[variable] == "0"
        assert ray.cluster_resources().get("NPU", 0) == 0
        yield namespace
    finally:
        cluster.close()
        if previous is None:
            assert variable not in os.environ
        else:
            assert os.environ[variable] == previous
