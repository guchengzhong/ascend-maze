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
        if item.health == "healthy" and not item.processes and item.total_hbm_mb >= 65_000
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
    os.environ[variable] = "0"
    ray.init(
        address="local",
        namespace=namespace,
        include_dashboard=False,
        num_cpus=4,
        object_store_memory=256 * 1024 * 1024,
        resources={"NPU": 0},
        log_to_driver=False,
    )
    try:
        yield namespace
    finally:
        ray.shutdown()
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
