"""Formal C14 adapter implemented only with C13 public surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import sys
from typing import BinaryIO, cast

from ascend_maze.api.workflow import Workflow
from ascend_maze.benchmark.canonical import thaw
from ascend_maze.benchmark.contracts import CellSpec, ExperimentSpec
from ascend_maze.benchmark.persistence import atomic_write_json
from ascend_maze.benchmark.runtime import (
    BenchmarkRuntimeClient,
    BenchmarkRuntimeFactory,
    ResourceRecoveryResult,
    ResourceSnapshot,
    RunFlushResult,
    SubmissionReceipt,
    TerminalRunResult,
)
from ascend_maze.compiler.ir import CompiledWorkflow
from ascend_maze.config import load_config
from ascend_maze.control import RuntimeClient
from ascend_maze.core.errors import ExperimentValidationError
from ascend_maze.core.time import monotonic_time_ms, wall_time_ms


class C13BenchmarkRuntimeFactory(BenchmarkRuntimeFactory):
    def __init__(
        self,
        *,
        maze_command: Sequence[str] | None = None,
        startup_timeout_seconds: float = 60.0,
    ) -> None:
        command = tuple(maze_command or _default_maze_command())
        if not command or any(not item for item in command):
            raise ValueError("maze command must be a non-empty argv")
        if startup_timeout_seconds <= 0:
            raise ValueError("Controller startup timeout must be positive")
        self.maze_command = command
        self.startup_timeout_seconds = startup_timeout_seconds

    async def open(
        self,
        *,
        spec: ExperimentSpec,
        cell: CellSpec,
        trial_attempt_id: str,
        trial_directory: str,
        resume: bool,
    ) -> BenchmarkRuntimeClient:
        root = Path(trial_directory).resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        override_path = root / "controller_config_overrides.json"
        override_payload = {
            "schema_version": 1,
            "schema": "ascend-maze.controller-config-overrides.v1",
            "build_revision": spec.build_revision,
            "expected_config_fingerprint": cell.config_snapshot.config_fingerprint,
            "overrides": [item.canonical_payload() for item in cell.overrides],
        }
        if override_path.exists():
            from ascend_maze.benchmark.persistence import load_json_object

            if dict(
                load_json_object(override_path, description="config overrides")
            ) != (override_payload):
                raise ExperimentValidationError(
                    "Controller config overrides changed during resume"
                )
        else:
            atomic_write_json(override_path, override_payload)
        loaded = load_config(
            spec.base_config_path,
            build_revision=spec.build_revision,
            config_overrides=tuple(
                (item.path, thaw(item.value)) for item in cell.overrides
            ),
        )
        if (
            loaded.snapshot.config_fingerprint
            != cell.config_snapshot.config_fingerprint
        ):
            raise ExperimentValidationError(
                "managed Controller ConfigSnapshot does not match the frozen Cell"
            )
        client = RuntimeClient(
            Path(loaded.config.control.socket_path),
            max_inline_control_bytes=loaded.config.control.max_inline_control_bytes,
            shared_filesystem_roots=loaded.config.data.shared_filesystem_roots,
        )
        client.config_fingerprint = cell.config_snapshot.config_fingerprint
        process: asyncio.subprocess.Process | None = None
        stdout: BinaryIO | None = None
        stderr: BinaryIO | None = None
        started_marker = root / "controller_started.json"
        matched = await _controller_matches(
            client,
            expected_build_revision=spec.build_revision,
            expected_environment=spec.workload.required_environment_fingerprint,
            expected_config_fingerprint=cell.config_snapshot.config_fingerprint,
        )
        if not matched:
            stdout = (root / "controller.stdout.log").open("ab", buffering=0)
            stderr = (root / "controller.stderr.log").open("ab", buffering=0)
            env = os.environ.copy()
            env["ASCEND_MAZE_BUILD_REVISION"] = spec.build_revision
            argv = (
                *self.maze_command,
                "controller",
                "start",
                "--config",
                spec.base_config_path,
                "--config-overrides",
                str(override_path),
            )
            if not started_marker.exists():
                argv = (*argv, "--fresh-recovery")
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
            deadline = asyncio.get_running_loop().time() + self.startup_timeout_seconds
            while not await _controller_matches(
                client,
                expected_build_revision=spec.build_revision,
                expected_environment=spec.workload.required_environment_fingerprint,
                expected_config_fingerprint=cell.config_snapshot.config_fingerprint,
            ):
                if process.returncode is not None:
                    stdout.close()
                    stderr.close()
                    raise ExperimentValidationError(
                        f"managed Controller exited during startup: {process.returncode}"
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    process.terminate()
                    await process.wait()
                    stdout.close()
                    stderr.close()
                    raise TimeoutError("managed Controller startup deadline expired")
                await asyncio.sleep(0.05)
        if not started_marker.exists():
            atomic_write_json(
                started_marker,
                {
                    "schema_version": 1,
                    "trial_attempt_id": trial_attempt_id,
                    "resumed_open": resume,
                    "config_fingerprint": cell.config_snapshot.config_fingerprint,
                },
            )
        return C13BenchmarkRuntime(
            client,
            process=process,
            stdout=stdout,
            stderr=stderr,
            shutdown_drain_timeout_ms=loaded.config.control.shutdown_drain_timeout_ms,
        )


class C13BenchmarkRuntime(BenchmarkRuntimeClient):
    def __init__(
        self,
        client: RuntimeClient,
        *,
        process: asyncio.subprocess.Process | None,
        stdout: BinaryIO | None,
        stderr: BinaryIO | None,
        shutdown_drain_timeout_ms: int,
    ) -> None:
        self.client = client
        self.process = process
        self.stdout = stdout
        self.stderr = stderr
        self.shutdown_drain_timeout_ms = shutdown_drain_timeout_ms

    async def resource_snapshot(self) -> ResourceSnapshot:
        status = await self.client.get_controller_status()
        system, cluster, workers, models, recorder = await asyncio.gather(
            self.client.query("GetSystemSnapshot"),
            self.client.query("GetClusterSnapshot", filter="resources"),
            self.client.query("GetWorkerPools"),
            self.client.query("GetModelInstances"),
            self.client.query("GetRecorderStatus"),
        )
        meta = _mapping(system.get("meta"), "system snapshot meta")
        fingerprint = _string(meta.get("config_fingerprint"), "config fingerprint")
        return ResourceSnapshot.create(
            captured_at_wall_ms=wall_time_ms(),
            controller_generation=status.controller_generation,
            config_fingerprint=fingerprint,
            payload={
                "controller_status": {
                    "build_revision": status.build_revision,
                    "environment_fingerprint": status.environment_fingerprint,
                    "healthy_node_count": status.healthy_node_count,
                },
                "system": system,
                "cluster_resources": cluster,
                "worker_pools": workers,
                "model_instances": models,
                "recorder": recorder,
            },
        )

    async def submit(
        self,
        workflow: object,
        *,
        inputs: dict[str, object],
        submission_id: str,
        run_deadline_ms: int | None,
    ) -> SubmissionReceipt:
        if not isinstance(workflow, (Workflow, CompiledWorkflow)):
            raise ExperimentValidationError(
                "benchmark workload is not a Workflow or CompiledWorkflow"
            )
        outcome = await self.client.submit(
            workflow,
            inputs=inputs,
            submission_id=submission_id,
            run_deadline_ms=run_deadline_ms,
        )
        state = _string(outcome.get("state"), "submission state")
        raw_run_id = outcome.get("run_id")
        run_id = None if raw_run_id is None else _string(raw_run_id, "Run ID")
        raw_error = outcome.get("error")
        return SubmissionReceipt(
            submission_id=submission_id,
            state=state,
            run_id=run_id,
            replayed=bool(outcome.get("replayed", False)),
            error=None if raw_error is None else str(raw_error),
        )

    async def wait_terminal(
        self, run_id: str, *, deadline_monotonic_ms: int
    ) -> TerminalRunResult:
        remaining_ms = deadline_monotonic_ms - monotonic_time_ms()
        if remaining_ms <= 0:
            raise TimeoutError("Run terminal deadline expired")
        async for _ in self.client.watch_run(
            run_id,
            timeout_seconds=remaining_ms / 1_000,
        ):
            pass
        shown = await self.client.query("GetRun", resource_id=run_id)
        snapshot = _mapping(shown.get("run"), "Run snapshot")
        status = _string(snapshot.get("status"), "Run status")
        return TerminalRunResult.create(run_id, status, shown)

    async def flush_run(self, run_id: str, *, request_id: str) -> RunFlushResult:
        result = await self.client.run_action("FlushRun", run_id, request_id=request_id)
        raw_files = result.get("committed_files", [])
        if not isinstance(raw_files, list) or any(
            not isinstance(path, str) or not path for path in raw_files
        ):
            raise ExperimentValidationError("C13 FlushResult files are invalid")
        recording_complete = result.get("recording_complete")
        if not isinstance(recording_complete, bool):
            raise ExperimentValidationError(
                "C13 FlushResult recording_complete is invalid"
            )
        return RunFlushResult.create(
            run_id,
            recording_complete,
            tuple(raw_files),
            result,
        )

    async def cancel_run(self, run_id: str, *, request_id: str) -> None:
        await self.client.run_action("CancelRun", run_id, request_id=request_id)

    async def destroy_run(self, run_id: str, *, request_id: str) -> None:
        await self.client.run_action("DestroyRun", run_id, request_id=request_id)

    async def wait_for_recovery(
        self,
        before: ResourceSnapshot,
        *,
        run_ids: tuple[str, ...],
        deadline_monotonic_ms: int,
    ) -> tuple[ResourceSnapshot, ResourceRecoveryResult]:
        del before
        while True:
            snapshot = await self.resource_snapshot()
            payload = thaw(snapshot.payload)
            remaining_ids = tuple(
                run_id for run_id in run_ids if _contains_value(payload, run_id)
            )
            system = _mapping(
                _mapping(payload, "resource payload").get("system"),
                "system snapshot",
            )
            nonterminal = system.get("nonterminal_run_count")
            recovered = not remaining_ids and nonterminal == 0
            if recovered:
                return snapshot, ResourceRecoveryResult.create(
                    recovered=True,
                    checked_at_wall_ms=wall_time_ms(),
                    reason_code=None,
                    details={"remaining_run_ids": [], "nonterminal_run_count": 0},
                )
            if monotonic_time_ms() >= deadline_monotonic_ms:
                return snapshot, ResourceRecoveryResult.create(
                    recovered=False,
                    checked_at_wall_ms=wall_time_ms(),
                    reason_code="resource_recovery_failed",
                    details={
                        "remaining_run_ids": list(remaining_ids),
                        "nonterminal_run_count": nonterminal,
                    },
                )
            await asyncio.sleep(
                min(
                    0.1,
                    max(0.001, (deadline_monotonic_ms - monotonic_time_ms()) / 1_000),
                )
            )

    async def shutdown(self, *, request_id: str) -> Mapping[str, object]:
        try:
            result = await self.client.shutdown_controller(
                request_id=request_id,
                drain_timeout_ms=self.shutdown_drain_timeout_ms,
                timeout_seconds=max(5.0, self.shutdown_drain_timeout_ms / 1_000 + 5),
            )
            normalized = dict(result)
            if self.process is not None:
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=10.0)
                except TimeoutError:
                    self.process.terminate()
                    await self.process.wait()
                    normalized["cleanup_confirmed"] = False
                    normalized["timed_out"] = True
                if self.process.returncode not in {None, 0}:
                    normalized["cleanup_confirmed"] = False
                    normalized["exit_code"] = self.process.returncode
            return normalized
        finally:
            if self.stdout is not None:
                self.stdout.close()
                self.stdout = None
            if self.stderr is not None:
                self.stderr.close()
                self.stderr = None


async def _controller_matches(
    client: RuntimeClient,
    *,
    expected_build_revision: str,
    expected_environment: str,
    expected_config_fingerprint: str,
) -> bool:
    try:
        status = await client.get_controller_status(timeout_seconds=0.5)
        if (
            status.build_revision != expected_build_revision
            or status.environment_fingerprint != expected_environment
        ):
            raise ExperimentValidationError(
                "running Controller identity does not match the frozen Study"
            )
        await client.verify_compatibility(timeout_seconds=0.5)
        system = await client.query("GetSystemSnapshot", timeout_seconds=0.5)
        meta = _mapping(system.get("meta"), "system snapshot meta")
        if meta.get("config_fingerprint") != expected_config_fingerprint:
            raise ExperimentValidationError(
                "running Controller config does not match the frozen Cell"
            )
        return True
    except ExperimentValidationError:
        raise
    except Exception:
        return False


def _default_maze_command() -> tuple[str, ...]:
    adjacent = Path(sys.executable).with_name("maze")
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return (str(adjacent),)
    discovered = shutil.which("maze")
    if discovered is not None:
        return (discovered,)
    return (sys.executable, "-m", "ascend_maze.cli.main")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentValidationError(f"{name} is not an object")
    return cast(Mapping[str, object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentValidationError(f"{name} is invalid")
    return value


def _contains_value(value: object, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_value(item, expected) for item in value)
    return False
