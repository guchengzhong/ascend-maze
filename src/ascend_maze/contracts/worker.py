"""Host Worker identity and per-Attempt occupancy lease contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ascend_maze.core.errors import ContractValidationError


class WorkerProfile(str, Enum):
    CPU = "cpu"
    IO = "io"
    NPU_HOST = "npu_host"


@dataclass(frozen=True, slots=True)
class WorkerLease:
    worker_lease_id: str
    worker_endpoint_id: str
    worker_id: str
    worker_generation: int
    node_id: str
    boot_id: str
    profile: WorkerProfile
    source: str
    bound_device_id: str | None
    acquired_at_ms: int

    def __post_init__(self) -> None:
        for name in (
            "worker_lease_id",
            "worker_endpoint_id",
            "worker_id",
            "node_id",
            "boot_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractValidationError(f"{name} is required")
        if self.source not in {"standby", "cold_start"}:
            raise ContractValidationError("unsupported WorkerLease source")
        if not isinstance(self.profile, WorkerProfile):
            raise ContractValidationError("profile must be WorkerProfile")
        if (
            isinstance(self.worker_generation, bool)
            or not isinstance(self.worker_generation, int)
            or self.worker_generation < 1
            or isinstance(self.acquired_at_ms, bool)
            or not isinstance(self.acquired_at_ms, int)
            or self.acquired_at_ms < 0
        ):
            raise ContractValidationError("invalid WorkerLease generation or timestamp")
