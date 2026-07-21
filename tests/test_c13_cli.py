from __future__ import annotations

from collections.abc import AsyncIterator
import json
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest

from ascend_maze.cli import main as cli_main
from ascend_maze.cli import doctor
from ascend_maze.cli.main import (
    _decode_cli_input,
    _emit_result,
    _load_workflow_factory,
    main,
)
from ascend_maze.control.local_rpc import (
    ControlRpcError,
    UdsRuntimeClient,
    _raise_submit_transport,
)
from ascend_maze.contracts.data import SharedFileRef
from ascend_maze.core.errors import (
    ContractValidationError,
    EnvironmentValidationError,
    ModelValidationError,
)


def _write_workflow_module(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "from ascend_maze import Workflow, task",
                "",
                "@task",
                "def echo(value: str):",
                '    return {"value": value}',
                "",
                "def build():",
                '    workflow = Workflow("cli-factory")',
                '    value = workflow.input("value")',
                '    workflow.add_task(echo, inputs={"value": value})',
                "    return workflow",
                "",
                "def fail():",
                '    raise RuntimeError("factory failed")',
            )
        ),
        encoding="utf-8",
    )


def test_submit_transport_errors_map_to_retryable_builtin_types() -> None:
    deadline = grpc.aio.AioRpcError(
        grpc.StatusCode.DEADLINE_EXCEEDED,
        details="deadline",
    )
    with pytest.raises(TimeoutError, match="SubmitWorkflow RPC deadline exceeded"):
        _raise_submit_transport(deadline)

    unavailable = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE,
        details="unavailable",
    )
    with pytest.raises(ConnectionError, match="UNAVAILABLE"):
        _raise_submit_transport(unavailable)


def test_file_and_module_workflow_factories_compile_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "workflow_fixture.py"
    _write_workflow_module(source)
    first = _load_workflow_factory(f"{source}:build").compile()
    second = _load_workflow_factory(f"{source}:build").compile()
    assert first.canonical_ir_bytes == second.canonical_ir_bytes
    assert first.workflow_fingerprint == second.workflow_fingerprint
    assert tuple(first.tasks) == tuple(second.tasks)

    monkeypatch.syspath_prepend(str(tmp_path))
    imported = _load_workflow_factory("workflow_fixture:build").compile()
    assert imported.workflow_name == "cli-factory"

    with pytest.raises(ContractValidationError, match="workflow factory failed"):
        _load_workflow_factory(f"{source}:fail")
    with pytest.raises(ContractValidationError, match="cannot import"):
        _load_workflow_factory(f"{tmp_path / 'missing.py'}:build")


def test_workflow_module_import_failure_is_a_structured_local_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken_workflow.py"
    source.write_text(
        'raise RuntimeError("module initialization failed")\n',
        encoding="utf-8",
    )

    with pytest.raises(ContractValidationError, match="cannot import workflow factory"):
        _load_workflow_factory(f"{source}:build")


def test_json_result_has_schema_and_diagnostics_stay_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _emit_result({"status": "ready"}, True)
    result = json.loads(capsys.readouterr().out)
    assert result == {"schema_version": 1, "status": "ready"}

    async def rejected(self: UdsRuntimeClient, **kwargs: object) -> object:
        del self, kwargs
        raise ControlRpcError("state_rejected", "controller is draining")

    monkeypatch.setattr(UdsRuntimeClient, "get_controller_status", rejected)
    exit_code = main(
        ["--json", "cluster", "status", "--socket", "/tmp/missing-control.sock"]
    )
    captured = capsys.readouterr()
    assert exit_code == 5
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error == {
        "schema_version": 1,
        "status": "error",
        "error_code": "state_rejected",
        "message": "controller is draining",
    }
    assert "\x1b[" not in captured.err

    async def status(self: UdsRuntimeClient, **kwargs: object) -> object:
        del self, kwargs
        return object()

    async def incompatible(self: UdsRuntimeClient, **kwargs: object) -> None:
        del self, kwargs
        raise ControlRpcError("version_incompatible", "protocol version mismatch")

    monkeypatch.setattr(UdsRuntimeClient, "get_controller_status", status)
    monkeypatch.setattr(UdsRuntimeClient, "verify_compatibility", incompatible)
    exit_code = main(
        ["--json", "cluster", "status", "--socket", "/tmp/missing-control.sock"]
    )
    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error_code"] == "version_incompatible"

    async def unreachable(self: UdsRuntimeClient, **kwargs: object) -> object:
        del self, kwargs
        raise ConnectionError("socket unavailable")

    monkeypatch.setattr(UdsRuntimeClient, "get_controller_status", unreachable)
    exit_code = cli_main.main(
        ["--json", "cluster", "status", "--socket", "/tmp/missing-control.sock"]
    )
    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error_code"] == "controller_unreachable"


def test_uds_client_rejects_malformed_control_responses() -> None:
    client = UdsRuntimeClient(Path("/tmp/ascend-maze-test-control.sock"))
    mismatched = SimpleNamespace(
        schema_version=1,
        request_id="another_request",
        controller_generation="controller_1",
        status_code="ok",
        error_code="",
        message="",
        json_payload=b"{}",
    )
    with pytest.raises(ControlRpcError) as request_error:
        client._decode(mismatched, "expected_request")
    assert request_error.value.error_code == "control_protocol_invalid"

    malformed = SimpleNamespace(
        schema_version=1,
        request_id="expected_request",
        controller_generation="controller_1",
        status_code="ok",
        error_code="",
        message="",
        json_payload=b"{not-json",
    )
    with pytest.raises(ControlRpcError) as json_error:
        client._decode(malformed, "expected_request")
    assert json_error.value.error_code == "control_protocol_invalid"
    assert client.controller_generation is None

    incompatible = SimpleNamespace(
        schema_version=2,
        request_id="expected_request",
        controller_generation="controller_1",
        status_code="ok",
        error_code="",
        message="",
        json_payload=b"{}",
    )
    with pytest.raises(ControlRpcError) as schema_error:
        client._decode(incompatible, "expected_request")
    assert schema_error.value.error_code == "control_protocol_invalid"


def test_doctor_rejects_conflicting_maze_console_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = Path(doctor.sys.executable).resolve().parent
    monkeypatch.setattr(doctor.shutil, "which", lambda name: str(environment / name))
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "entry_points",
        lambda **kwargs: (
            importlib.metadata.EntryPoint(
                name="maze",
                value="ascend_maze.cli.main:main",
                group="console_scripts",
            ),
            importlib.metadata.EntryPoint(
                name="maze",
                value="maze.cli:main",
                group="console_scripts",
            ),
        ),
    )
    check = doctor._maze_executable_check()
    assert check.status == "fail"
    assert "maze.cli:main" in check.message


def test_cli_decodes_only_explicit_shared_file_values(tmp_path: Path) -> None:
    path = (tmp_path / "input.txt").resolve()
    path.write_text("payload", encoding="utf-8")
    tagged = {
        "$shared_file": {
            "canonical_path": str(path),
            "content_sha256": "2" * 64,
            "size_bytes": 7,
        }
    }
    decoded = _decode_cli_input(tagged)
    assert decoded == SharedFileRef(str(path), "2" * 64, 7)
    assert _decode_cli_input(str(path)) == str(path)
    assert _decode_cli_input({"path": str(path)}) == {"path": str(path)}
    with pytest.raises(ContractValidationError, match="SharedFileRef fields"):
        _decode_cli_input({"$shared_file": {"canonical_path": str(path)}})


def test_cli_distinguishes_config_environment_and_model_validation_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def model_failure(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ModelValidationError("model artifact is invalid")

    monkeypatch.setattr(cli_main, "load_config", model_failure)
    assert main(["--json", "models", "validate", "--config", "invalid.toml"]) == 4
    assert json.loads(capsys.readouterr().err)["error_code"] == (
        "model_validation_failed"
    )
    assert main(["--json", "config", "validate", "--config", "invalid.toml"]) == 2
    capsys.readouterr()

    def environment_failure(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise EnvironmentValidationError("NPU environment mismatch")

    monkeypatch.setattr(cli_main, "load_config", environment_failure)
    assert main(["--json", "controller", "start", "--config", "invalid.toml"]) == 4
    assert json.loads(capsys.readouterr().err)["error_code"] == (
        "environment_validation_failed"
    )


def test_node_cli_resolves_current_boot_and_maps_drain_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def status(self: UdsRuntimeClient, **kwargs: object) -> object:
        del self, kwargs
        return object()

    async def compatible(self: UdsRuntimeClient, **kwargs: object) -> None:
        del self, kwargs

    async def query(
        self: UdsRuntimeClient, operation: str, **kwargs: object
    ) -> dict[str, object]:
        del self, kwargs
        assert operation == "GetClusterSnapshot"
        return {
            "cluster": {
                "nodes": [
                    {
                        "capacity": {
                            "node_id": "node_a",
                            "boot_id": "boot_current",
                        }
                    }
                ]
            }
        }

    async def node_action(
        self: UdsRuntimeClient,
        operation: str,
        node_id: str,
        **kwargs: object,
    ) -> dict[str, object]:
        del self
        assert operation == "DrainNode"
        assert node_id == "node_a"
        assert kwargs["boot_id"] == "boot_current"
        return {
            "status": "draining",
            "timed_out": True,
            "cleanup_confirmed": False,
            "exit_code": 1,
        }

    monkeypatch.setattr(UdsRuntimeClient, "get_controller_status", status)
    monkeypatch.setattr(UdsRuntimeClient, "verify_compatibility", compatible)
    monkeypatch.setattr(UdsRuntimeClient, "query", query)
    monkeypatch.setattr(UdsRuntimeClient, "node_action", node_action)

    assert (
        main(
            [
                "--json",
                "node",
                "drain",
                "node_a",
                "--socket",
                "/tmp/control.sock",
            ]
        )
        == 5
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "draining"
    assert payload["schema_version"] == 1


def test_cluster_watch_refreshes_snapshot_and_resubscribes_after_version_jump(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def status(self: UdsRuntimeClient, **kwargs: object) -> object:
        del self, kwargs
        return object()

    async def compatible(self: UdsRuntimeClient, **kwargs: object) -> None:
        del self, kwargs

    query_versions = iter((1, 3))

    async def query(
        self: UdsRuntimeClient, operation: str, **kwargs: object
    ) -> dict[str, object]:
        del self, kwargs
        assert operation == "GetClusterSnapshot"
        return {"meta": {"snapshot_version": next(query_versions)}}

    watched_versions: list[int] = []

    async def watch_cluster(
        self: UdsRuntimeClient,
        *,
        after_snapshot_version: int,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, object]]:
        del self, kwargs
        watched_versions.append(after_snapshot_version)
        if len(watched_versions) == 1:
            yield {
                "events": [],
                "next_snapshot_version": 3,
                "snapshot_required": True,
            }
            return
        raise ControlRpcError("state_rejected", "test watch complete")
        yield  # pragma: no cover

    monkeypatch.setattr(UdsRuntimeClient, "get_controller_status", status)
    monkeypatch.setattr(UdsRuntimeClient, "verify_compatibility", compatible)
    monkeypatch.setattr(UdsRuntimeClient, "query", query)
    monkeypatch.setattr(UdsRuntimeClient, "watch_cluster", watch_cluster)

    exit_code = main(
        [
            "--json",
            "cluster",
            "resources",
            "--watch",
            "--socket",
            "/tmp/control.sock",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 5
    assert watched_versions == [1, 3]
    assert [json.loads(line)["meta"]["snapshot_version"] for line in captured.out.splitlines()] == [
        1,
        3,
    ]
    assert json.loads(captured.err)["error_code"] == "state_rejected"
