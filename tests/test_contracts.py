from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from ascend_maze.contracts.config import ConfigSnapshot
from ascend_maze.contracts.data import DataHandle, DataStore, SharedFileRef
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.contracts.recording import ExecutionEvent, RecorderSink
from ascend_maze.contracts.resources import (
    ExecutionTarget,
    ReservationVector,
    ResourceDeclaration,
    ResourceSpec,
)
from ascend_maze.contracts.runtime import (
    CodeHandle,
    CodePackage,
    DispatchHandle,
    ExecutionRequest,
    RuntimeArgument,
    RuntimeBackend,
)
from ascend_maze.contracts.submission import (
    RunInputIdentity,
    SubmissionContract,
    SubmissionOptions,
    hash_session_key,
)
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.core.identifiers import GenerationRef, new_id, stable_id
from ascend_maze.core.time import monotonic_time_ms, wall_time_ms

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_resource_declaration_and_reservation_are_distinct() -> None:
    declaration = ResourceDeclaration.from_public(
        {"cpu_num": 2, "mem": 128, "npu_mem": 64, "io_num": 1}
    )
    spec = declaration.resolve(ResourceSpec(1, 32, 0, 0))
    assert spec == ResourceSpec(2, 128, 64, 1)
    vector = ReservationVector(2, 128, 1, 64, 1)
    assert vector.npu_slots == 1
    with pytest.raises(ContractValidationError):
        ReservationVector(1, 1, 0, 0, -1)


def test_gpu_mem_alias_warns_and_normalizes() -> None:
    with pytest.warns(DeprecationWarning):
        declaration = ResourceDeclaration.from_public({"gpu_mem": 12})
    assert declaration.npu_mem_mb == 12


def test_data_handle_identity_ignores_diagnostic_metadata() -> None:
    first = DataHandle("owner", "handle", metadata={"type": "str"})
    second = DataHandle("owner", "handle", metadata={"type": "bytes"})
    assert first.submission_identity() == second.submission_identity()
    digest = DataHandle("owner", "other", stable_digest=SHA_A)
    assert digest.submission_identity() == ("digest", SHA_A)


def test_shared_file_ref_is_canonical_and_content_identified(tmp_path: Path) -> None:
    ref = SharedFileRef(str(tmp_path / "folder" / ".." / "file"), SHA_A, 10)
    assert Path(ref.canonical_path).is_absolute()
    assert ref.content_sha256 == SHA_A
    with pytest.raises(ContractValidationError, match="absolute"):
        SharedFileRef("relative/file", SHA_A, 10)


def test_config_snapshot_is_deeply_frozen_and_fingerprinted(tmp_path: Path) -> None:
    source = {"scheduler": {"policy": "fcfs"}, "values": [1, 2]}
    snapshot = ConfigSnapshot.create(
        schema_version=1,
        project_version="0.1.0",
        source_path=str(tmp_path / "config.toml"),
        resolved=source,
        model_catalog_revision="none",
        build_revision="abc",
        runtime_versions={"python": "3.13"},
        created_at_ms=1,
    )
    source["values"].append(3)
    assert snapshot.resolved["values"] == (1, 2)
    assert len(snapshot.config_fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        snapshot.project_version = "changed"  # type: ignore[misc]


def test_submission_hash_sorts_inputs_and_uses_real_identity() -> None:
    literal = RunInputIdentity.from_small_value("a", {"x": 1})
    handle = RunInputIdentity.from_data_handle("b", DataHandle("owner", "h1"))
    options = SubmissionOptions(run_deadline_ms=1000, execution_options={"mode": "x"})
    kwargs = {
        "submission_id": "submission_1",
        "workflow_fingerprint": SHA_A,
        "session_key_hash": hash_session_key("session"),
        "options": options,
        "config_fingerprint": SHA_B,
    }
    left = SubmissionContract.create(
        input_identities=(literal, handle), **kwargs
    )
    right = SubmissionContract.create(
        input_identities=(handle, literal), **kwargs
    )
    assert left.submission_payload_hash == right.submission_payload_hash

    changed_handle = RunInputIdentity.from_data_handle(
        "b", DataHandle("owner", "h2")
    )
    changed = SubmissionContract.create(
        input_identities=(literal, changed_handle), **kwargs
    )
    assert changed.submission_payload_hash != left.submission_payload_hash
    with pytest.raises(ContractValidationError, match="does not match"):
        replace(left, submission_payload_hash=SHA_A)


def test_code_package_digest_checks_transport_bytes_only() -> None:
    package = CodePackage.create(
        definition_id="definition_1",
        code_hash=SHA_A,
        module="module",
        qualname="function",
        serialized_fallback=b"payload",
        environment_fingerprint=SHA_B,
    )
    assert package.serialized_payload_digest is not None
    with pytest.raises(ContractValidationError, match="digest mismatch"):
        CodePackage(
            definition_id="definition_1",
            code_hash=SHA_A,
            module="module",
            qualname="function",
            serialized_fallback=b"payload",
            serialized_payload_digest=SHA_A,
            environment_fingerprint=SHA_B,
        )


def test_error_and_event_payloads_are_deeply_frozen() -> None:
    details = {"nested": [1, 2]}
    error = ErrorInfo(
        schema_version=1,
        error_code="user_code_failed",
        category="user",
        origin="compiler",
        message="failed",
        retryable_hint=False,
        classification_confidence="exact",
        execution_phase="pre_attempt",
        run_id="run",
        task_id="task",
        attempt=0,
        occurred_at_ms=1,
        details=details,
    )
    event = ExecutionEvent(
        schema_version=1,
        event_id="event",
        experiment_id="experiment",
        run_id="run",
        task_id="task",
        attempt=0,
        lease_id=None,
        route_lease_id=None,
        model_instance_id=None,
        event_type="test",
        producer_id="controller",
        producer_sequence=1,
        node_id=None,
        device_id=None,
        monotonic_time_ms=1,
        wall_time_ms=1,
        duration_ms=0,
        payload=details,
    )
    details["nested"].append(3)
    assert error.details["nested"] == (1, 2)
    assert event.payload["nested"] == (1, 2)


def test_protocols_are_runtime_checkable() -> None:
    class Store:
        def put_staged(self, value, owner_generation):
            return DataHandle(owner_generation, "handle")

        def get(self, handle):
            return None

        def adopt(self, handles, owner):
            return None

        def release(self, handle):
            return None

        def release_many(self, handles):
            return None

        def state_of(self, handle):
            return "staged"

    class Sink:
        def emit(self, event):
            return True

    class Backend:
        async def start(self):
            return None

        async def prepare(self, definitions):
            return ()

        async def dispatch(self, request, lease):
            return DispatchHandle("d", "fake", "r", "t", 0, "l", None, "w")

        async def cancel(self, handle, reason):
            return None

        async def release_code(self, handles):
            return None

        async def close(self):
            return None

    assert isinstance(Store(), DataStore)
    assert isinstance(Sink(), RecorderSink)
    assert isinstance(Backend(), RuntimeBackend)


def test_runtime_request_enforces_execution_target_boundary() -> None:
    code = CodeHandle("code", "definition", SHA_A)
    with pytest.raises(ContractValidationError, match="requires ModelRouteLease"):
        ExecutionRequest(
            dispatch_id="dispatch",
            run_id="run",
            task_id="task",
            attempt=1,
            task_kind="npu",
            execution_target=ExecutionTarget.MODEL_SERVICE,
            model_route=None,
            code_handle=code,
            arguments=(RuntimeArgument("x", "literal", literal="x"),),
            expected_outputs=("x",),
            timeout_ms=None,
            environment_fingerprint=SHA_B,
        )


def test_runtime_literal_arguments_are_deeply_frozen() -> None:
    source = {"items": [1, 2]}
    argument = RuntimeArgument("value", "literal", literal=source)
    source["items"].append(3)
    assert argument.literal["items"] == (1, 2)
    with pytest.raises(ContractValidationError, match="cannot carry a literal"):
        RuntimeArgument(
            "value",
            "data_handle",
            literal="unexpected",
            data_handle=DataHandle("owner", "handle"),
        )


def test_identifier_generation_and_clocks() -> None:
    assert stable_id("task", "a", "b") == stable_id("task", "a", "b")
    assert stable_id("task", "a", "b") != stable_id("task", "b", "a")
    assert new_id("run").startswith("run_")
    assert GenerationRef("controller", 0).generation == 0
    assert isinstance(monotonic_time_ms(), int)
    assert isinstance(wall_time_ms(), int)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DataHandle("owner", "handle", stable_digest=1),
        lambda: SharedFileRef(1, SHA_A, 0),
        lambda: GenerationRef("controller", "bad"),
        lambda: ResourceDeclaration.from_public([("cpu_num", 1)]),
    ],
)
def test_invalid_contract_types_raise_structured_errors(factory) -> None:
    with pytest.raises(ContractValidationError):
        factory()
