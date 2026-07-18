from __future__ import annotations

import asyncio
from pathlib import Path
import stat

from ascend_maze.control.local_rpc import (
    ControllerStatus,
    LocalControlServer,
    UdsRuntimeClient,
)


def test_local_control_status_uses_protected_unix_socket(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime_dir = tmp_path / "runtime"
        socket_path = runtime_dir / "control.sock"
        expected = ControllerStatus(
            controller_generation="controller_1",
            build_revision="test_build",
            environment_fingerprint="e" * 64,
            healthy_node_count=2,
        )
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: expected,
        )
        await server.start()
        assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        assert await UdsRuntimeClient(socket_path).get_controller_status() == expected
        await server.close(grace_seconds=0)
        assert not socket_path.exists()

    asyncio.run(scenario())
