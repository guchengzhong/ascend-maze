from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ascend_maze.config import NodeBootstrapConfig, load_node_bootstrap
from ascend_maze.control import node_application
from ascend_maze.control.contracts import NodeRuntimePolicy
from ascend_maze.control.node_application import NodeApplication, NodeBootstrapResponse
from ascend_maze.control.process_lock import NodeProcessLock
from ascend_maze.contracts.runtime import RuntimeDeviceMapping
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.placement import NodeCapacity


ENVIRONMENT = "e" * 64


def _write_node_config(root: Path, extra: str = "") -> Path:
    token = root / "cluster.token"
    token.write_bytes(b"test-node-token")
    token.chmod(0o600)
    path = root / "node.toml"
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'cluster_id = "cluster_1"',
                'node_id = "node_b"',
                'node_ip = "10.0.0.2"',
                'controller_endpoint = "10.0.0.1:7123"',
                f'authorization_token_file = "{token.name}"',
                'runtime_directory = "runtime"',
                'ray_temp_directory = "ray-tmp"',
                'recording_root_directory = "records"',
                extra,
            )
        ),
        encoding="utf-8",
    )
    return path


def _config(root: Path) -> NodeBootstrapConfig:
    token = root / "cluster.token"
    token.write_bytes(b"test-node-token")
    token.chmod(0o600)
    runtime = (root / "runtime").resolve()
    return NodeBootstrapConfig(
        schema_version=1,
        source_path=str((root / "node.toml").resolve()),
        cluster_id="cluster_1",
        node_id="node_b",
        node_ip="10.0.0.2",
        controller_endpoint="10.0.0.1:7123",
        authorization_token_file=str(token.resolve()),
        runtime_directory=str(runtime),
        worker_rpc_bind_address="0.0.0.0:0",
        worker_advertised_host="10.0.0.2",
        ray_temp_directory=str((root / "ray").resolve()),
        ray_num_cpus=4,
        recording_root_directory=str((root / "records").resolve()),
    )


def test_node_bootstrap_config_is_strict_and_normalizes_paths(tmp_path: Path) -> None:
    path = _write_node_config(tmp_path)
    config = load_node_bootstrap(path)
    assert config.source_path == str(path.resolve())
    assert config.authorization_token_file == str((tmp_path / "cluster.token").resolve())
    assert config.runtime_directory == str((tmp_path / "runtime").resolve())
    assert config.ray_temp_directory == str((tmp_path / "ray-tmp").resolve())
    assert config.recording_root_directory == str((tmp_path / "records").resolve())
    assert config.device_mappings == ()

    path = _write_node_config(
        tmp_path,
        "device_mappings = ["
        '{ physical_device_id = "3", runtime_visible_device_id = "0", '
        "visible_device_index = 0 }]",
    )
    config = load_node_bootstrap(path)
    assert config.device_mappings == (RuntimeDeviceMapping("3", "0", 0),)

    path = _write_node_config(tmp_path, "unknown_option = true")
    with pytest.raises(ContractValidationError, match="unknown_option"):
        load_node_bootstrap(path)

    path = _write_node_config(tmp_path, "task_slots_total = 2")
    with pytest.raises(ContractValidationError, match="task_slots_total: unknown"):
        load_node_bootstrap(path)

    path = _write_node_config(tmp_path, "worker_rpc_bind_address = 1234")
    with pytest.raises(ContractValidationError, match="worker_rpc_bind_address"):
        load_node_bootstrap(path)

    path = _write_node_config(
        tmp_path,
        "device_mappings = ["
        '{ physical_device_id = "3", runtime_visible_device_id = "0" }, '
        '{ physical_device_id = "3", runtime_visible_device_id = "1" }]',
    )
    with pytest.raises(ContractValidationError, match="must be unique"):
        load_node_bootstrap(path)

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "schema_version = 1", "schema_version = true"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="schema_version"):
        load_node_bootstrap(path)


def test_node_process_lock_rejects_duplicate_generation_and_removes_own_file(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "runtime" / "node.pid").resolve()
    first = NodeProcessLock(path, node_generation="node_generation_1")
    first.acquire()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["node_generation"] == "node_generation_1"
    assert "controller_generation" not in payload

    second = NodeProcessLock(path, node_generation="node_generation_2")
    with pytest.raises(RuntimeError, match="another Node owns PID lock"):
        second.acquire()
    first.close()
    assert not path.exists()


def test_node_application_reverses_worker_start_when_agent_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    lifecycle: list[str] = []

    async def bootstrap(
        received: NodeBootstrapConfig,
        token: bytes,
    ) -> NodeBootstrapResponse:
        assert received is config
        assert token == b"test-node-token"
        return NodeBootstrapResponse(
            cluster_id=config.cluster_id,
            controller_generation="controller_1",
            config_fingerprint="c" * 64,
            environment_fingerprint=ENVIRONMENT,
            ray_address="10.0.0.1:6379",
            ray_namespace="maze-test",
            node_runtime_policy=NodeRuntimePolicy(),
        )

    class DeviceAdapter:
        def devices(self) -> tuple[object, ...]:
            return (SimpleNamespace(health="healthy"),)

        def verify_process_device(self, process_id: int, device_id: str) -> bool:
            del process_id, device_id
            return True

    class Worker:
        def __init__(self, **kwargs: object) -> None:
            assert "test-node-token" not in repr(kwargs)
            lifecycle.append("worker_created")

        def start(self) -> str:
            lifecycle.append("worker_started")
            return "ray_node_b"

        def close(self) -> None:
            lifecycle.append("worker_closed")

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            self.recorder = kwargs["recorder"]
            self.service_manager = kwargs["service_process_manager"]
            lifecycle.append("agent_created")

        async def start(self, **kwargs: object) -> str:
            del kwargs
            lifecycle.append("agent_start_failed")
            raise RuntimeError("injected registration failure")

        async def close(self, grace_seconds: float = 1.0) -> None:
            del grace_seconds
            lifecycle.append("agent_closed")
            await self.service_manager.close(1_000)
            await self.recorder.close(1_000)

    monkeypatch.setattr(node_application, "fetch_node_bootstrap", bootstrap)
    monkeypatch.setattr(
        node_application,
        "discover_ascend_environment",
        lambda adapter, devices: SimpleNamespace(environment_fingerprint=ENVIRONMENT),
    )
    monkeypatch.setattr(
        node_application,
        "build_ascend_node_capacity",
        lambda **kwargs: NodeCapacity(
            node_id=config.node_id,
            boot_id="boot_1",
            node_ip=config.node_ip,
            cpu_total=4,
            mem_total_mb=8_192,
            cpu_system_reserved=0,
            mem_system_reserved_mb=1_024,
            io_slots_total=8,
            observed_free_mem_mb=7_168,
        ),
    )
    monkeypatch.setattr(node_application, "_boot_id", lambda: "boot_1")
    monkeypatch.setattr(node_application, "ManagedRayWorkerNode", Worker)
    monkeypatch.setattr(node_application, "NodeAgent", Agent)

    application = NodeApplication(config, device_adapter=DeviceAdapter())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="injected registration failure"):
        asyncio.run(application.run())
    assert lifecycle == [
        "worker_created",
        "worker_started",
        "agent_created",
        "agent_start_failed",
        "agent_closed",
        "worker_closed",
    ]
    assert not (Path(config.runtime_directory) / "node.pid").exists()


def test_node_application_rejects_environment_before_starting_ray(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    async def bootstrap(
        received: NodeBootstrapConfig,
        token: bytes,
    ) -> NodeBootstrapResponse:
        del received, token
        return NodeBootstrapResponse(
            cluster_id=config.cluster_id,
            controller_generation="controller_1",
            config_fingerprint="c" * 64,
            environment_fingerprint="f" * 64,
            ray_address="10.0.0.1:6379",
            ray_namespace="maze-test",
            node_runtime_policy=NodeRuntimePolicy(),
        )

    class DeviceAdapter:
        def devices(self) -> tuple[object, ...]:
            return (SimpleNamespace(health="healthy"),)

    class UnexpectedWorker:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            raise AssertionError("Ray worker must not start after environment mismatch")

    monkeypatch.setattr(node_application, "fetch_node_bootstrap", bootstrap)
    monkeypatch.setattr(
        node_application,
        "discover_ascend_environment",
        lambda adapter, devices: SimpleNamespace(environment_fingerprint=ENVIRONMENT),
    )
    monkeypatch.setattr(node_application, "ManagedRayWorkerNode", UnexpectedWorker)
    application = NodeApplication(config, device_adapter=DeviceAdapter())  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError, match="environment fingerprint"):
        asyncio.run(application.run())
    assert not (Path(config.runtime_directory) / "node.pid").exists()
