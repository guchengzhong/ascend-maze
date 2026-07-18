"""Managed Ray connection and Controller lifecycle for a Head process."""

from __future__ import annotations

from pathlib import Path

from ascend_maze.core.clock import Clock
from ascend_maze.placement import NodeCapacity
from ascend_maze.runtime.ray_cluster import ManagedRayCluster, RayClusterConfig

from ascend_maze.control.ray_controller import RayHostController


class ManagedRayHost:
    def __init__(
        self,
        *,
        ray_config: RayClusterConfig,
        cluster_id: str,
        authorization_token: bytes,
        config_fingerprint: str,
        environment_fingerprint: str,
        build_revision: str,
        node_capacities: tuple[NodeCapacity, ...],
        control_socket_path: Path | None = None,
        controller_generation: str | None = None,
        node_rpc_bind_address: str = "127.0.0.1:0",
        node_rpc_advertised_host: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.ray_cluster = ManagedRayCluster(ray_config)
        self.cluster_id = cluster_id
        self.authorization_token = authorization_token
        self.config_fingerprint = config_fingerprint
        self.environment_fingerprint = environment_fingerprint
        self.build_revision = build_revision
        self.node_capacities = node_capacities
        self.control_socket_path = control_socket_path
        self.controller_generation = controller_generation
        self.node_rpc_bind_address = node_rpc_bind_address
        self.node_rpc_advertised_host = node_rpc_advertised_host
        self.clock = clock
        self.controller: RayHostController | None = None

    async def start(self) -> RayHostController:
        if self.controller is not None:
            return self.controller
        self.ray_cluster.start()
        try:
            controller = RayHostController(
                cluster_id=self.cluster_id,
                authorization_token=self.authorization_token,
                ray_namespace=self.ray_cluster.config.namespace,
                config_fingerprint=self.config_fingerprint,
                environment_fingerprint=self.environment_fingerprint,
                build_revision=self.build_revision,
                node_capacities=self.node_capacities,
                control_socket_path=self.control_socket_path,
                controller_generation=self.controller_generation,
                node_rpc_bind_address=self.node_rpc_bind_address,
                node_rpc_advertised_host=self.node_rpc_advertised_host,
                clock=self.clock,
            )
            await controller.start()
        except Exception:
            self.ray_cluster.close()
            raise
        self.controller = controller
        return controller

    async def close(self) -> None:
        controller = self.controller
        self.controller = None
        try:
            if controller is not None:
                await controller.close()
        finally:
            self.ray_cluster.close()
