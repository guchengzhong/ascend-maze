"""Minimal typed Head-local ControlService over a protected Unix socket."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any

import grpc

from ascend_maze.core.identifiers import new_id

from ascend_maze.control.proto import control_pb2 as _control_pb2
from ascend_maze.control.proto import control_pb2_grpc

control_pb2: Any = _control_pb2


@dataclass(frozen=True, slots=True)
class ControllerStatus:
    controller_generation: str
    build_revision: str
    environment_fingerprint: str
    healthy_node_count: int

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.controller_generation,
                self.build_revision,
                self.environment_fingerprint,
            )
        ):
            raise ValueError("controller status identity fields are required")
        if self.healthy_node_count < 0:
            raise ValueError("healthy_node_count must be non-negative")


class _LocalControlServicer:
    def __init__(self, owner: "LocalControlServer") -> None:
        self.owner = owner

    async def GetControllerStatus(
        self,
        request: Any,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> Any:
        if int(request.schema_version) != 1 or not request.request_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid request envelope")
        status = self.owner.status_provider()
        return control_pb2.GetControllerStatusResponse(
            request_id=request.request_id,
            controller_generation=status.controller_generation,
            status_code="ok",
            message="",
            build_revision=status.build_revision,
            environment_fingerprint=status.environment_fingerprint,
            healthy_node_count=status.healthy_node_count,
        )


class LocalControlServer:
    def __init__(
        self,
        *,
        socket_path: Path,
        status_provider: Callable[[], ControllerStatus],
    ) -> None:
        if not socket_path.is_absolute():
            raise ValueError("control socket path must be absolute")
        self.socket_path = socket_path
        self.status_provider = status_provider
        self._server: grpc.aio.Server | None = None
        self._socket_inode: int | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        parent = self.socket_path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(parent, 0o700)
        if self.socket_path.exists():
            raise RuntimeError(f"control socket already exists: {self.socket_path}")
        server = grpc.aio.server()
        control_pb2_grpc.add_LocalControlServicer_to_server(
            _LocalControlServicer(self), server
        )
        if server.add_insecure_port(_grpc_uds_target(self.socket_path)) == 0:
            raise RuntimeError(f"failed to bind control socket: {self.socket_path}")
        await server.start()
        self._server = server
        try:
            info = self.socket_path.stat()
        except FileNotFoundError:
            await server.stop(0)
            self._server = None
            raise RuntimeError("gRPC did not create the configured control socket") from None
        if not stat.S_ISSOCK(info.st_mode):
            await server.stop(0)
            self._server = None
            raise RuntimeError("configured control path is not a Unix socket")
        os.chmod(self.socket_path, 0o600)
        self._socket_inode = info.st_ino

    async def close(self, grace_seconds: float = 1.0) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        await server.stop(grace_seconds)
        try:
            info = self.socket_path.stat()
        except FileNotFoundError:
            return
        if info.st_ino == self._socket_inode and stat.S_ISSOCK(info.st_mode):
            self.socket_path.unlink()


class UdsRuntimeClient:
    def __init__(self, socket_path: Path, *, client_version: str = "0.1.0") -> None:
        if not socket_path.is_absolute():
            raise ValueError("control socket path must be absolute")
        self.socket_path = socket_path
        self.client_version = client_version

    async def get_controller_status(
        self, *, timeout_seconds: float = 5.0
    ) -> ControllerStatus:
        request_id = new_id("control_request")
        async with grpc.aio.insecure_channel(
            _grpc_uds_target(self.socket_path)
        ) as channel:
            stub = control_pb2_grpc.LocalControlStub(channel)
            response = await stub.GetControllerStatus(
                control_pb2.GetControllerStatusRequest(
                    schema_version=1,
                    request_id=request_id,
                    client_version=self.client_version,
                    deadline_ms=max(1, int(timeout_seconds * 1_000)),
                ),
                timeout=timeout_seconds,
            )
        if response.request_id != request_id or response.status_code != "ok":
            raise RuntimeError(response.message or "invalid ControlService response")
        return ControllerStatus(
            controller_generation=str(response.controller_generation),
            build_revision=str(response.build_revision),
            environment_fingerprint=str(response.environment_fingerprint),
            healthy_node_count=int(response.healthy_node_count),
        )


def _grpc_uds_target(path: Path) -> str:
    return f"unix:{path}"
