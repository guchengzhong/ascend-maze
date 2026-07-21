from __future__ import annotations

import argparse
import ast
import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ascend_maze.benchmark import cli as benchmark_cli
from ascend_maze.benchmark import c13_runtime
from ascend_maze.benchmark.c13_runtime import (
    C13BenchmarkRuntime,
    C13BenchmarkRuntimeFactory,
)
from ascend_maze.benchmark.loader import load_study_plan
from ascend_maze.benchmark.persistence import atomic_write_json
from ascend_maze.cli import main as cli_main
from ascend_maze.config import load_config_override_document
from ascend_maze.core.errors import ContractValidationError
from benchmark_fixtures import write_experiment_spec
from benchmark_workload_fixtures import build


def test_controller_override_document_is_strict_and_fingerprint_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_study_plan(write_experiment_spec(tmp_path))
    cell = plan.cells[0]
    override_path = tmp_path / "overrides.json"
    payload = {
        "schema_version": 1,
        "schema": "ascend-maze.controller-config-overrides.v1",
        "build_revision": plan.spec.build_revision,
        "expected_config_fingerprint": cell.config_snapshot.config_fingerprint,
        "overrides": [item.canonical_payload() for item in cell.overrides],
    }
    atomic_write_json(override_path, payload)
    document = load_config_override_document(override_path)
    assert document.build_revision == plan.spec.build_revision
    monkeypatch.setenv("ASCEND_MAZE_BUILD_REVISION", plan.spec.build_revision)
    loaded = cli_main._load_controller_start_config(
        argparse.Namespace(
            config=plan.spec.base_config_path,
            config_overrides=str(override_path),
        )
    )
    assert loaded.snapshot.config_fingerprint == cell.config_snapshot.config_fingerprint

    payload["unknown"] = True
    atomic_write_json(override_path, payload)
    with pytest.raises(ContractValidationError, match="unknown"):
        load_config_override_document(override_path)


def test_fresh_recovery_requires_a_stopped_controller(tmp_path: Path) -> None:
    plan = load_study_plan(write_experiment_spec(tmp_path))
    loaded = cli_main.load_config(
        plan.spec.base_config_path,
        build_revision=plan.spec.build_revision,
    )
    recovery = Path(loaded.config.control.recovery_path)
    recovery.parent.mkdir(parents=True, exist_ok=True)
    for path in (
        recovery,
        recovery.with_name(recovery.name + "-wal"),
        recovery.with_name(recovery.name + "-shm"),
    ):
        path.write_text("old", encoding="utf-8")
    pid_path = Path(loaded.config.control.pid_file)
    pid_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": 999_999_999,
                "controller_generation": "old",
                "process_start_ticks": 1,
            }
        ),
        encoding="utf-8",
    )
    cli_main._clear_stopped_controller_recovery(loaded)
    assert not recovery.exists()
    assert not pid_path.exists()

    socket_path = Path(loaded.config.control.socket_path)
    socket_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="socket exists"):
        cli_main._clear_stopped_controller_recovery(loaded)


def test_maze_bench_run_and_resume_emit_structured_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def execute(args: argparse.Namespace) -> dict[str, object]:
        calls.append(args.command)
        return {
            "schema_version": 1,
            "study_id": "study_" + "a" * 32,
            "study_directory": "/tmp/study",
            "state": "completed",
            "completed_trials": 1,
            "blocked_reason": None,
        }

    monkeypatch.setattr(benchmark_cli, "_run_or_resume", execute)
    assert benchmark_cli.main(["run", "spec.toml", "--output-root", "out"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "completed"
    assert benchmark_cli.main(["resume", "out/study"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "completed"
    assert calls == ["run", "resume"]


def test_formal_adapter_maps_only_public_c13_runtime_operations() -> None:
    class PublicClient:
        def __init__(self) -> None:
            self.actions: list[tuple[str, str, str | None, bool]] = []
            self.closed = False

        def close(self) -> None:
            self.closed = True

        async def get_controller_status(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(
                controller_generation="generation-1",
                build_revision="a" * 40,
                environment_fingerprint="e" * 64,
                healthy_node_count=1,
            )

        async def query(
            self,
            operation: str,
            *,
            resource_id: str = "",
            filter: str = "",
            **kwargs: object,
        ) -> dict[str, object]:
            del filter, kwargs
            if operation == "GetSystemSnapshot":
                return {
                    "meta": {"config_fingerprint": "f" * 64},
                    "nonterminal_run_count": 0,
                }
            if operation == "GetRun":
                return {"run": {"run_id": resource_id, "status": "succeeded"}}
            return {"operation": operation}

        async def submit(self, workflow: object, **kwargs: object) -> dict[str, object]:
            del workflow
            return {
                "state": "committed",
                "run_id": "run_" + "a" * 32,
                "replayed": False,
                "submission_id": kwargs["submission_id"],
            }

        async def watch_run(self, run_id: str, **kwargs: object):
            del run_id, kwargs
            yield {"run_terminal": True}

        async def run_action(
            self,
            operation: str,
            run_id: str,
            *,
            request_id: str | None = None,
            **kwargs: object,
        ) -> dict[str, object]:
            self.actions.append(
                (operation, run_id, request_id, bool(kwargs.get("force", False)))
            )
            if operation == "FlushRun":
                return {"recording_complete": True, "committed_files": []}
            return {}

        async def shutdown_controller(self, **kwargs: object) -> dict[str, object]:
            self.actions.append(
                ("ShutdownController", "", str(kwargs["request_id"]), False)
            )
            return {"cleanup_confirmed": True, "timed_out": False}

    async def scenario() -> None:
        public = PublicClient()
        adapter = C13BenchmarkRuntime(
            public,  # type: ignore[arg-type]
            process=None,
            stdout=None,
            stderr=None,
            shutdown_drain_timeout_ms=100,
        )
        before = await adapter.resource_snapshot()
        receipt = await adapter.submit(
            build(),
            inputs={"value": 1},
            submission_id="submission-a",
            run_deadline_ms=100,
        )
        assert receipt.run_id is not None
        terminal = await adapter.wait_terminal(
            receipt.run_id,
            deadline_monotonic_ms=10**15,
        )
        assert terminal.status == "succeeded"
        flushed = await adapter.flush_run(receipt.run_id, request_id="flush-id")
        assert flushed.recording_complete
        await adapter.cancel_run(receipt.run_id, request_id="cancel-id")
        await adapter.destroy_run(
            receipt.run_id,
            request_id="destroy-id",
            force=True,
        )
        _, recovery = await adapter.wait_for_recovery(
            before,
            run_ids=(receipt.run_id,),
            deadline_monotonic_ms=10**15,
        )
        assert recovery.recovered
        await adapter.shutdown(request_id="shutdown-id")
        assert public.closed
        assert [item[0] for item in public.actions] == [
            "FlushRun",
            "CancelRun",
            "DestroyRun",
            "ShutdownController",
        ]
        assert public.actions[2][3] is True

    asyncio.run(scenario())

    source_path = (
        Path(__file__).resolve().parents[1] / "src/ascend_maze/benchmark/c13_runtime.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "ascend_maze.control.application" not in imports
    assert "ascend_maze.control.controller" not in imports
    assert "create_subprocess_exec" in source
    assert "shell=" not in source


def test_shutdown_rpc_failure_still_reaps_managed_controller_process() -> None:
    class Client:
        calls = 0
        closed = 0

        def close(self) -> None:
            self.closed += 1

        async def shutdown_controller(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.calls += 1
            raise ConnectionError("injected shutdown transport failure")

    class Process:
        returncode: int | None = None
        terminated = False
        killed = False

        async def wait(self) -> int:
            if not self.terminated and not self.killed:
                raise TimeoutError("process still running")
            self.returncode = -15 if self.terminated else -9
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    async def scenario() -> None:
        client = Client()
        process = Process()
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        runtime = C13BenchmarkRuntime(
            client,  # type: ignore[arg-type]
            process=process,  # type: ignore[arg-type]
            stdout=stdout,
            stderr=stderr,
            shutdown_drain_timeout_ms=100,
        )

        result = await runtime.shutdown(request_id="shutdown-failure")
        repeated = await runtime.shutdown(request_id="shutdown-failure")

        assert repeated is result
        assert client.calls == 1
        assert client.closed == 1
        assert process.terminated and not process.killed
        assert result["cleanup_confirmed"] is False
        assert result["exit_code"] == -15
        assert "injected shutdown transport failure" in str(result["rpc_error"])
        assert stdout.closed and stderr.closed

    asyncio.run(scenario())


def test_formal_factory_starts_controller_with_argv_and_frozen_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Client:
        config_fingerprint: str | None = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    class Process:
        returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = 0

    async def scenario() -> None:
        plan = load_study_plan(write_experiment_spec(tmp_path))
        cell = plan.cells[0]
        matches = iter((False, True))

        async def controller_matches(*args: object, **kwargs: object) -> bool:
            del args, kwargs
            return next(matches)

        calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

        async def create_process(*argv: str, **kwargs: object) -> Process:
            calls.append((argv, kwargs))
            return Process()

        monkeypatch.setattr(c13_runtime, "RuntimeClient", Client)
        monkeypatch.setattr(c13_runtime, "_controller_matches", controller_matches)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        root = tmp_path / "trial"
        runtime = await C13BenchmarkRuntimeFactory(
            maze_command=("/opt/ascend-maze/bin/maze",)
        ).open(
            spec=plan.spec,
            cell=cell,
            trial_attempt_id="trial_attempt_" + "a" * 32,
            trial_directory=str(root),
            resume=False,
        )
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[:3] == (
            "/opt/ascend-maze/bin/maze",
            "controller",
            "start",
        )
        assert "--config-overrides" in argv
        assert argv[-1] == "--fresh-recovery"
        assert kwargs["start_new_session"] is True
        assert "shell" not in kwargs
        assert (root / "controller_started.json").is_file()
        assert isinstance(runtime, C13BenchmarkRuntime)
        if runtime.stdout is not None:
            runtime.stdout.close()
        if runtime.stderr is not None:
            runtime.stderr.close()

    asyncio.run(scenario())
