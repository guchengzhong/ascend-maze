"""PREPARING/COMMITTED/ABORTED transaction around SchedulerCore commit."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib

from ascend_maze.compiler.ir import CompiledWorkflow
from ascend_maze.contracts.data import DataHandle, DataStore
from ascend_maze.contracts.recording import ExecutionRecorder, RunRecordingContext
from ascend_maze.contracts.runtime import CodePackage
from ascend_maze.contracts.runtime import CodeHandle
from ascend_maze.contracts.submission import (
    RunInputIdentity,
    SubmissionContract,
    SubmissionState,
)
from ascend_maze.core.clock import Clock, SystemClock
from ascend_maze.core.errors import ResponseLostError, SubmissionConflictError
from ascend_maze.core.identifiers import new_id
from ascend_maze.data import InMemoryDataStore, RunDataIndexRegistry
from ascend_maze.lifecycle import DeadlineManager, RunStateManager
from ascend_maze.lifecycle import RunSnapshot
from ascend_maze.placement import NodeCapacity, PlacementManager
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.resources import DeclaredOnlyAnchorProvider
from ascend_maze.runtime import FakeRuntimeBackend
from ascend_maze.scheduler import (
    DestroyResult,
    FcfsPolicy,
    HeterogeneousPartitioner,
    SchedulerCore,
)
from ascend_maze.scheduler.core import SchedulerRuntimeBackend


@dataclass(frozen=True, slots=True)
class SubmitRequest:
    compiled: CompiledWorkflow
    code_packages: tuple[CodePackage, ...]
    workflow_inputs: tuple[tuple[str, DataHandle], ...]
    contract: SubmissionContract

    def input_map(self) -> dict[str, DataHandle]:
        return dict(self.workflow_inputs)


@dataclass(frozen=True, slots=True)
class SubmissionOutcome:
    submission_id: str
    state: SubmissionState
    run_id: str | None
    submission_payload_hash: str
    replayed: bool
    error: str | None = None


@dataclass(slots=True)
class _SubmissionRecord:
    payload_hash: str
    state: SubmissionState
    run_id: str | None = None
    error: str | None = None


class InMemoryController:
    """Own the stage-two components and expose an in-process control surface."""

    def __init__(
        self,
        *,
        config_fingerprint: str,
        environment_fingerprint: str,
        build_revision: str,
        node_capacities: tuple[NodeCapacity, ...],
        controller_generation: str | None = None,
        clock: Clock | None = None,
        data_store: DataStore | None = None,
        recorder: ExecutionRecorder | None = None,
        runtime: SchedulerRuntimeBackend | None = None,
    ) -> None:
        self.config_fingerprint = config_fingerprint
        self.environment_fingerprint = environment_fingerprint
        self.build_revision = build_revision
        self.controller_generation = controller_generation or new_id("controller")
        self.clock = clock or SystemClock()
        self.data_store: DataStore = data_store or InMemoryDataStore()
        self.indexes = RunDataIndexRegistry(
            controller_generation=self.controller_generation,
            data_store=self.data_store,
        )
        self.state = RunStateManager()
        self.deadlines = DeadlineManager()
        self.anchors = DeclaredOnlyAnchorProvider(
            environment_fingerprint=environment_fingerprint
        )
        self.placement = PlacementManager()
        for capacity in node_capacities:
            self.placement.register_node(capacity)
        self.recorder: ExecutionRecorder = recorder or InMemoryRecorder()
        if runtime is None:
            if not isinstance(self.data_store, InMemoryDataStore):
                raise TypeError("default FakeRuntime requires InMemoryDataStore")
            runtime = FakeRuntimeBackend(
                data_store=self.data_store,
                owner_generation=self.controller_generation,
                environment_fingerprint=environment_fingerprint,
            )
        self.runtime: SchedulerRuntimeBackend = runtime
        self.core = SchedulerCore(
            state=self.state,
            deadlines=self.deadlines,
            indexes=self.indexes,
            anchors=self.anchors,
            placement=self.placement,
            runtime=self.runtime,
            recorder=self.recorder,
            policy=FcfsPolicy(),
            partitioner=HeterogeneousPartitioner(),
            clock=self.clock,
        )
        self._submissions: dict[str, _SubmissionRecord] = {}
        self._submit_lock = asyncio.Lock()
        self._failure_points: set[str] = set()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.core.start()
        self._started = True

    async def close(self) -> None:
        if not self._started:
            return
        await self.core.shutdown()
        await self.runtime.close()
        await self.recorder.close(1_000)
        self._started = False

    def inject_submit_failure(self, point: str) -> None:
        if point not in {"after_prepare", "after_open_run", "before_commit"}:
            raise ValueError("unknown submission failure point")
        self._failure_points.add(point)

    async def submit(
        self,
        request: SubmitRequest,
        *,
        lose_response_after_commit: bool = False,
    ) -> SubmissionOutcome:
        if not self._started:
            raise RuntimeError("controller is not started")
        contract = request.contract
        async with self._submit_lock:
            existing = self._submissions.get(contract.submission_id)
            if existing is not None:
                if existing.payload_hash != contract.submission_payload_hash:
                    raise SubmissionConflictError(
                        "submission_id already exists with a different payload"
                    )
                if existing.state is SubmissionState.COMMITTED:
                    return SubmissionOutcome(
                        submission_id=contract.submission_id,
                        state=existing.state,
                        run_id=existing.run_id,
                        submission_payload_hash=existing.payload_hash,
                        replayed=True,
                    )
                if existing.state is SubmissionState.ABORTED:
                    return SubmissionOutcome(
                        submission_id=contract.submission_id,
                        state=existing.state,
                        run_id=None,
                        submission_payload_hash=existing.payload_hash,
                        replayed=True,
                        error=existing.error,
                    )
                raise RuntimeError("submission is unexpectedly still PREPARING")

            record = _SubmissionRecord(
                payload_hash=contract.submission_payload_hash,
                state=SubmissionState.PREPARING,
            )
            self._submissions[contract.submission_id] = record
            code_handles: tuple[CodeHandle, ...] = ()
            provisional_run_id: str | None = None
            recording_open = False
            try:
                self._validate_request(request)
                code_handles = await self.runtime.prepare(request.code_packages)
                self._maybe_fail("after_prepare")
                provisional_run_id = new_id("run")
                self.recorder.open_run(
                    RunRecordingContext(
                        schema_version=1,
                        experiment_id=provisional_run_id,
                        run_id=provisional_run_id,
                        workflow_fingerprint=request.compiled.workflow_fingerprint,
                        config_fingerprint=self.config_fingerprint,
                        environment_fingerprint=self.environment_fingerprint,
                        build_revision=self.build_revision,
                        started_wall_time_ms=self.clock.wall_ms(),
                        initial_expected_producer_ids=("controller",),
                    )
                )
                recording_open = True
                self._maybe_fail("after_open_run")
                self._maybe_fail("before_commit")
                submitted_at = self.clock.monotonic_ms()
                deadline_at = (
                    None
                    if contract.options.run_deadline_ms is None
                    else submitted_at + contract.options.run_deadline_ms
                )
                await self.core.commit_run(
                    run_id=provisional_run_id,
                    compiled=request.compiled,
                    workflow_inputs=request.input_map(),
                    code_handles=code_handles,
                    session_key_hash=contract.session_key_hash,
                    submitted_at_ms=submitted_at,
                    deadline_at_ms=deadline_at,
                )
            except Exception as exc:
                if code_handles:
                    await self.runtime.release_code(code_handles)
                if recording_open and provisional_run_id is not None:
                    self.recorder.abort_run(provisional_run_id)
                record.state = SubmissionState.ABORTED
                record.error = f"{type(exc).__name__}: {exc}"
                return SubmissionOutcome(
                    submission_id=contract.submission_id,
                    state=SubmissionState.ABORTED,
                    run_id=None,
                    submission_payload_hash=contract.submission_payload_hash,
                    replayed=False,
                    error=record.error,
                )

            record.state = SubmissionState.COMMITTED
            record.run_id = provisional_run_id
            outcome = SubmissionOutcome(
                submission_id=contract.submission_id,
                state=SubmissionState.COMMITTED,
                run_id=provisional_run_id,
                submission_payload_hash=contract.submission_payload_hash,
                replayed=False,
            )
            if lose_response_after_commit:
                raise ResponseLostError(
                    "submission committed but its response was intentionally lost"
                )
            return outcome

    async def wait_run(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> RunSnapshot:
        return await self.core.wait_terminal(run_id, timeout_seconds=timeout_seconds)

    async def cancel_run(
        self,
        run_id: str,
        *,
        reason: str = "user_cancelled",
    ) -> RunSnapshot:
        return await self.core.cancel_run(run_id, reason=reason)

    async def destroy_run(
        self,
        run_id: str,
        *,
        force: bool = False,
    ) -> DestroyResult:
        return await self.core.destroy_run(run_id, force=force)

    def result(self, run_id: str, task_id: str) -> dict[str, object]:
        return self.core.result(run_id, task_id)

    def snapshot(self, run_id: str) -> RunSnapshot:
        return self.core.snapshot(run_id)

    def submission_outcome(self, submission_id: str) -> SubmissionOutcome:
        record = self._submissions[submission_id]
        return SubmissionOutcome(
            submission_id=submission_id,
            state=record.state,
            run_id=record.run_id,
            submission_payload_hash=record.payload_hash,
            replayed=True,
            error=record.error,
        )

    def _validate_request(self, request: SubmitRequest) -> None:
        compiled = request.compiled
        contract = request.contract
        if (
            hashlib.sha256(compiled.canonical_ir_bytes).hexdigest()
            != compiled.workflow_fingerprint
        ):
            raise ValueError("compiled workflow fingerprint does not match IR bytes")
        if contract.workflow_fingerprint != compiled.workflow_fingerprint:
            raise ValueError("submission workflow fingerprint mismatch")
        if contract.config_fingerprint != self.config_fingerprint:
            raise ValueError("submission config fingerprint mismatch")
        inputs = request.input_map()
        if len(inputs) != len(request.workflow_inputs):
            raise ValueError("workflow input names must be unique")
        if set(inputs) != set(compiled.workflow_inputs):
            raise ValueError("workflow input names do not match compiled workflow")
        package_by_definition = {
            package.definition_id: package for package in request.code_packages
        }
        if len(package_by_definition) != len(request.code_packages):
            raise ValueError("CodePackage definition IDs must be unique")
        if set(package_by_definition) != set(compiled.definitions):
            raise ValueError("CodePackage definitions do not match workflow")
        for definition_id, definition in compiled.definitions.items_tuple():
            package = package_by_definition[definition_id]
            if package.code_hash != definition.code_hash:
                raise ValueError("CodePackage code hash does not match definition")
        expected_identities = tuple(
            RunInputIdentity.from_data_handle(name, inputs[name])
            for name in sorted(inputs)
        )
        if expected_identities != contract.input_identities:
            raise ValueError("submission input identities do not match handles")
        for handle in inputs.values():
            if handle.owner_generation != self.controller_generation:
                raise ValueError("input handle belongs to another owner generation")
            if self.data_store.state_of(handle) != "staged":
                raise ValueError("new submission input handle must be staged")

    def _maybe_fail(self, point: str) -> None:
        if point in self._failure_points:
            self._failure_points.remove(point)
            raise RuntimeError(f"injected submission failure at {point}")
