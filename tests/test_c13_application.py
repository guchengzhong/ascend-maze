from __future__ import annotations

from pathlib import Path
from typing import cast

from ascend_maze.ascend import AscendDeviceSnapshot, DcmiDeviceAdapter
from ascend_maze.config import load_config
from ascend_maze.control.application import ControllerApplication
from ascend_maze.recording import ParquetRecorder


class _StaticDcmi:
    def __init__(self) -> None:
        self.inventory = (
            AscendDeviceSnapshot(
                physical_device_id="0",
                card_id=0,
                card_device_id=0,
                chip_type="910B3",
                chip_version="test",
                total_hbm_mb=65_536,
                used_hbm_mb=3_200,
                health="healthy",
                utilization=0.0,
            ),
        )

    def devices(self) -> tuple[AscendDeviceSnapshot, ...]:
        return self.inventory


def test_controller_application_builds_all_prestart_authorities(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    token = runtime / "cluster.token"
    token.write_bytes(b"test-cluster-token")
    token.chmod(0o600)
    config_path = tmp_path / "ascend-maze.toml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'profile = "correctness"',
                "[control]",
                f'runtime_directory = "{runtime}"',
                f'cluster_token_file = "{token}"',
                "[runtime.ray]",
                'namespace = "c13-build-test"',
                "local_num_cpus = 2",
                "object_store_memory_bytes = 134217728",
                "[recording]",
                'root_directory = "records"',
            )
        ),
        encoding="utf-8",
    )
    loaded = load_config(config_path, build_revision="test", created_at_ms=0)
    application = ControllerApplication(
        loaded,
        device_adapter=cast(DcmiDeviceAdapter, _StaticDcmi()),
    )
    host = application.build()

    assert host.config_fingerprint == loaded.snapshot.config_fingerprint
    assert host.controller_generation
    assert host.control_socket_path == Path(loaded.config.control.socket_path)
    assert host.pid_lock is not None
    assert not Path(loaded.config.control.pid_file).exists()
    assert isinstance(host.recorder, ParquetRecorder)
    assert host.head_node_agent_factory is not None
    assert host.node_registry is not None
    assert host.placement is not None
    assert host.node_runtime_policy.task_slots_total == (
        loaded.config.placement.task_slots_total
    )
    assert host.node_runtime_policy.recording_batch_size == (
        loaded.config.recording.batch_size
    )
    assert host.node_runtime_policy.hbm_recovery_tolerance_mb == (
        loaded.config.worker.hbm_recovery_tolerance_mb
    )
    assert host.recovery_path == Path(loaded.config.control.recovery_path)
    cursor_key = runtime / "recording.cursor.key"
    assert cursor_key.is_file()
    assert cursor_key.stat().st_mode & 0o077 == 0
