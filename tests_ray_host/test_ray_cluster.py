from __future__ import annotations

import os
from typing import Any

import pytest

from ascend_maze.runtime import ray_cluster
from ascend_maze.runtime.ray_cluster import ManagedRayCluster, RayClusterConfig


OVERRIDE = "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"


def _cluster() -> ManagedRayCluster:
    return ManagedRayCluster(
        RayClusterConfig(
            namespace="stage4-ray-cluster-test",
            disable_ray_npu_resource=True,
        )
    )


def test_disable_ray_npu_resource_is_explicit_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    context = object()

    def initialize(**kwargs: object) -> object:
        calls.append(kwargs)
        assert os.environ[OVERRIDE] == "0"
        return context

    monkeypatch.setenv(OVERRIDE, "previous")
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_cluster.ray, "init", initialize)
    monkeypatch.setattr(ray_cluster.ray, "shutdown", lambda: None)
    cluster = _cluster()

    assert cluster.start() is context
    assert calls == [
        {
            "namespace": "stage4-ray-cluster-test",
            "include_dashboard": False,
            "address": "local",
            "resources": {"NPU": 0},
        }
    ]
    cluster.close()
    assert os.environ[OVERRIDE] == "previous"


def test_ray_start_failure_restores_absent_accelerator_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**kwargs: Any) -> None:
        del kwargs
        assert os.environ[OVERRIDE] == "0"
        raise RuntimeError("ray init failed")

    monkeypatch.delenv(OVERRIDE, raising=False)
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_cluster.ray, "init", fail)

    with pytest.raises(RuntimeError, match="ray init failed"):
        _cluster().start()
    assert OVERRIDE not in os.environ
