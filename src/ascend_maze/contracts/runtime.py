"""Backend-neutral task dispatch contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol, runtime_checkable

from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.resources import ExecutionTarget, PlacementLease
from ascend_maze.core.canonical import CanonicalValue, freeze_canonical
from ascend_maze.core.errors import CanonicalizationError, ContractValidationError


@dataclass(frozen=True, slots=True)
class CodePackage:
    definition_id: str
    code_hash: str
    module: str
    qualname: str
    serialized_fallback: bytes | None
    serialized_payload_digest: str | None
    environment_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "definition_id",
            "code_hash",
            "module",
            "qualname",
            "environment_fingerprint",
        ):
            if not getattr(self, name):
                raise ContractValidationError(f"{name} is required")
        if self.serialized_fallback is None:
            if self.serialized_payload_digest is not None:
                raise ContractValidationError(
                    "serialized payload digest requires serialized fallback bytes"
                )
            return
        expected = hashlib.sha256(self.serialized_fallback).hexdigest()
        if self.serialized_payload_digest != expected:
            raise ContractValidationError("serialized payload digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        definition_id: str,
        code_hash: str,
        module: str,
        qualname: str,
        serialized_fallback: bytes | None,
        environment_fingerprint: str,
    ) -> "CodePackage":
        digest = (
            None
            if serialized_fallback is None
            else hashlib.sha256(serialized_fallback).hexdigest()
        )
        return cls(
            definition_id=definition_id,
            code_hash=code_hash,
            module=module,
            qualname=qualname,
            serialized_fallback=serialized_fallback,
            serialized_payload_digest=digest,
            environment_fingerprint=environment_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class CodeHandle:
    code_handle_id: str
    definition_id: str
    code_hash: str


@dataclass(frozen=True, slots=True)
class RuntimeArgument:
    name: str
    kind: str
    literal: CanonicalValue | None = None
    data_handle: DataHandle | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ContractValidationError("runtime argument name is required")
        if self.kind not in {"literal", "data_handle", "default_omitted"}:
            raise ContractValidationError(f"unsupported runtime argument kind: {self.kind}")
        if self.kind == "literal":
            if self.data_handle is not None:
                raise ContractValidationError("literal argument cannot carry DataHandle")
            try:
                object.__setattr__(self, "literal", freeze_canonical(self.literal))
            except CanonicalizationError as exc:
                raise ContractValidationError(
                    "runtime literal must be a canonical value"
                ) from exc
        if self.kind == "data_handle":
            if not isinstance(self.data_handle, DataHandle):
                raise ContractValidationError("data_handle argument requires DataHandle")
            if self.literal is not None:
                raise ContractValidationError(
                    "data_handle argument cannot carry a literal"
                )
        if self.kind == "default_omitted" and (
            self.literal is not None or self.data_handle is not None
        ):
            raise ContractValidationError(
                "default_omitted argument cannot carry a value"
            )


@dataclass(frozen=True, slots=True)
class ModelRouteLease:
    route_lease_id: str
    run_id: str
    task_id: str
    attempt: int
    model_id: str
    catalog_revision: str
    instance_id: str
    instance_generation: int
    adapter_name: str
    endpoint_id: str
    instance_node_id: str
    instance_boot_id: str
    affinity_key_hash: str
    created_at_ms: int
    dispatch_deadline_ms: int


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    dispatch_id: str
    run_id: str
    task_id: str
    attempt: int
    task_kind: str
    execution_target: ExecutionTarget
    model_route: ModelRouteLease | None
    code_handle: CodeHandle
    arguments: tuple[RuntimeArgument, ...]
    expected_outputs: tuple[str, ...]
    timeout_ms: int | None
    environment_fingerprint: str

    def __post_init__(self) -> None:
        if self.execution_target is ExecutionTarget.MODEL_SERVICE:
            if self.model_route is None:
                raise ContractValidationError(
                    "model service request requires ModelRouteLease"
                )
        elif self.model_route is not None:
            raise ContractValidationError(
                "local worker request cannot carry ModelRouteLease"
            )


@dataclass(frozen=True, slots=True)
class DispatchHandle:
    dispatch_id: str
    backend_name: str
    run_id: str
    task_id: str
    attempt: int
    lease_id: str
    route_lease_id: str | None
    worker_endpoint_id: str


@runtime_checkable
class RuntimeBackend(Protocol):
    async def start(self) -> None: ...

    async def prepare(
        self, definitions: tuple[CodePackage, ...]
    ) -> tuple[CodeHandle, ...]: ...

    async def dispatch(
        self,
        request: ExecutionRequest,
        lease: PlacementLease,
    ) -> DispatchHandle: ...

    async def cancel(self, handle: DispatchHandle, reason: str) -> None: ...

    async def release_code(self, handles: tuple[CodeHandle, ...]) -> None: ...

    async def close(self) -> None: ...
