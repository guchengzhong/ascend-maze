from __future__ import annotations

from pathlib import Path
import signal
from typing import Any

import pytest

from ascend_maze.runtime import ray_cluster
from ascend_maze.runtime.ray_cluster import ManagedRayWorkerNode, SUPPORTED_RAY_VERSION


class _Process:
    def __init__(self, *, pid: int = 4321, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        del timeout
        self.returncode = 0
        return 0


def test_managed_ray_worker_uses_public_cli_and_owns_only_its_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False
    node_queries = 0
    process = _Process()
    popen_call: dict[str, Any] = {}
    signals: list[tuple[int, signal.Signals]] = []

    def is_initialized() -> bool:
        return initialized

    def initialize(**kwargs: object) -> None:
        nonlocal initialized
        assert kwargs == {"address": "10.0.0.1:6379", "namespace": "maze-test"}
        initialized = True

    def shutdown() -> None:
        nonlocal initialized
        initialized = False

    def nodes() -> list[dict[str, object]]:
        nonlocal node_queries
        node_queries += 1
        existing = {
            "NodeID": "head_node",
            "Alive": True,
            "NodeManagerAddress": "10.0.0.1",
        }
        if node_queries == 1:
            return [existing]
        return [
            existing,
            {
                "NodeID": "worker_node",
                "Alive": True,
                "NodeManagerAddress": "10.0.0.2",
            },
        ]

    def popen(command: tuple[str, ...], **kwargs: object) -> _Process:
        popen_call["command"] = command
        popen_call.update(kwargs)
        return process

    def killpg(process_id: int, signum: signal.Signals) -> None:
        signals.append((process_id, signum))

    monkeypatch.setattr(ray_cluster.ray, "__version__", SUPPORTED_RAY_VERSION)
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", is_initialized)
    monkeypatch.setattr(ray_cluster.ray, "init", initialize)
    monkeypatch.setattr(ray_cluster.ray, "shutdown", shutdown)
    monkeypatch.setattr(ray_cluster.ray, "nodes", nodes)
    monkeypatch.setattr(ray_cluster.subprocess, "Popen", popen)
    monkeypatch.setattr(ray_cluster.os, "killpg", killpg)
    monkeypatch.setenv("ASCEND_MAZE_TEST_SECRET", "must-not-enter-argv")

    worker = ManagedRayWorkerNode(
        address="10.0.0.1:6379",
        namespace="maze-test",
        node_ip="10.0.0.2",
        temp_directory=str(tmp_path / "ray"),
        num_cpus=4,
        log_path=tmp_path / "worker.log",
    )
    assert worker.start() == "worker_node"
    command = popen_call["command"]
    assert command[:2] == (
        str(Path(ray_cluster.sys.executable).with_name("ray")),
        "start",
    )
    assert "--address=10.0.0.1:6379" in command
    assert "--block" in command
    assert all("must-not-enter-argv" not in argument for argument in command)
    assert popen_call["start_new_session"] is True
    assert popen_call["stdin"] is ray_cluster.subprocess.DEVNULL
    environment = popen_call["env"]
    assert isinstance(environment, dict)
    assert environment["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"

    worker.close()
    assert signals == [(process.pid, signal.SIGTERM)]
    assert not initialized
    assert worker.process is None
    assert worker.node_id is None


def test_managed_ray_worker_start_failure_closes_driver_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False
    process = _Process(returncode=23)

    def initialize(**kwargs: object) -> None:
        nonlocal initialized
        del kwargs
        initialized = True

    def shutdown() -> None:
        nonlocal initialized
        initialized = False

    monkeypatch.setattr(ray_cluster.ray, "__version__", SUPPORTED_RAY_VERSION)
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: initialized)
    monkeypatch.setattr(ray_cluster.ray, "init", initialize)
    monkeypatch.setattr(ray_cluster.ray, "shutdown", shutdown)
    monkeypatch.setattr(ray_cluster.ray, "nodes", lambda: [])
    monkeypatch.setattr(ray_cluster.subprocess, "Popen", lambda *args, **kwargs: process)

    worker = ManagedRayWorkerNode(
        address="10.0.0.1:6379",
        namespace="maze-test",
        node_ip="10.0.0.2",
        temp_directory=str(tmp_path / "ray"),
        num_cpus=4,
        log_path=tmp_path / "worker.log",
    )
    with pytest.raises(RuntimeError, match="exited with code 23"):
        worker.start()
    assert not initialized
    assert worker.process is None
    assert worker.node_id is None
    assert worker._log_stream is None
