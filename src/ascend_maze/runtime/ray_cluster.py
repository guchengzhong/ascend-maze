"""Explicit lifecycle wrapper for a configured Ray Host connection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ray

SUPPORTED_RAY_VERSION = "2.55.1"


def validate_ray_version() -> None:
    if ray.__version__ != SUPPORTED_RAY_VERSION:
        raise RuntimeError(
            "Ray Host correctness requires "
            f"ray=={SUPPORTED_RAY_VERSION}; found {ray.__version__}"
        )


@dataclass(frozen=True, slots=True)
class RayClusterConfig:
    namespace: str
    address: str | None = None
    include_dashboard: bool = False
    local_num_cpus: int | None = None
    local_object_store_memory: int | None = None

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("Ray namespace is required")
        for name in ("local_num_cpus", "local_object_store_memory"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")


class ManagedRayCluster:
    def __init__(self, config: RayClusterConfig) -> None:
        self.config = config
        self._started = False

    def start(self) -> Any:
        if self._started:
            return ray.get_runtime_context()
        if ray.is_initialized():
            raise RuntimeError("Ray is already initialized outside ManagedRayCluster")
        validate_ray_version()
        kwargs: dict[str, object] = {
            "namespace": self.config.namespace,
            "include_dashboard": self.config.include_dashboard,
        }
        if self.config.address is not None:
            kwargs["address"] = self.config.address
        else:
            kwargs["address"] = "local"
            if self.config.local_num_cpus is not None:
                kwargs["num_cpus"] = self.config.local_num_cpus
            if self.config.local_object_store_memory is not None:
                kwargs["object_store_memory"] = self.config.local_object_store_memory
        ray_init: Any = ray.init
        context = ray_init(**kwargs)
        self._started = True
        return context

    def close(self) -> None:
        if not self._started:
            return
        ray.shutdown()
        self._started = False

    def live_node_ids(self) -> tuple[str, ...]:
        if not self._started:
            raise RuntimeError("Ray cluster is not started")
        return tuple(
            sorted(
                str(node["NodeID"])
                for node in ray.nodes()
                if bool(node.get("Alive"))
            )
        )
