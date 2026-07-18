"""Serial C3-C9 coordination for the stage-two FakeRuntime closure."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ascend_maze.compiler.ir import (
    CompiledWorkflow,
    DefaultBinding,
    LiteralBinding,
    OutputBinding,
    TaskDefinition,
    WorkflowInputBinding,
)
from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.contracts.recording import ExecutionEvent, FlushResult
from ascend_maze.contracts.runtime import (
    CodeHandle,
    DispatchHandle,
    ExecutionRequest,
    RuntimeArgument,
)
from ascend_maze.core.clock import Clock, SystemClock
from ascend_maze.core.canonical import FrozenMap, freeze_canonical
from ascend_maze.core.errors import (
    RunDataIndexError,
    RunNotTerminalError,
    StateTransitionError,
)
from ascend_maze.core.identifiers import new_id
from ascend_maze.data.index import (
    RunDataIndexRef,
    RunDataIndexRegistry,
    RunDataTombstone,
)
from ascend_maze.fault import CleanupBarrier, RecoveryAction, RecoveryPolicy
from ascend_maze.lifecycle.deadlines import DeadlineEvent, DeadlineKind, DeadlineManager
from ascend_maze.lifecycle.state import (
    AttemptSnapshot,
    AttemptStatus,
    RunSnapshot,
    RunStateManager,
    RunStatus,
    TaskStatus,
)
from ascend_maze.placement.manager import LeaseStatus, PlacementManager
from ascend_maze.recording.in_memory import InMemoryRecorder, NoopRecorder
from ascend_maze.resources.anchors import DeclaredOnlyAnchorProvider
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.runtime.fake import FakeRuntimeBackend
from ascend_maze.scheduler.contracts import (
    QueueToken,
    SchedulableTaskView,
    SchedulingPolicy,
    TaskKey,
)
from ascend_maze.scheduler.partitioners import HeterogeneousPartitioner


@dataclass(frozen=True, slots=True)
class DestroyResult:
    run_id: str
    tombstone: RunDataTombstone
    flush_result: FlushResult
    code_handles_released: int


@dataclass(slots=True)
class _RunExecution:
    compiled: CompiledWorkflow
    code_handles: tuple[CodeHandle, ...]
    code_by_definition: dict[str, CodeHandle]
    index_ref: RunDataIndexRef
    destroyed: DestroyResult | None = None


@dataclass(slots=True)
class _QueuedRecord:
    view: SchedulableTaskView
    partition: str


@dataclass(frozen=True, slots=True)
class _DispatchRecord:
    handle: DispatchHandle
    lease_id: str


@dataclass(slots=True)
class _CommitCommand:
    run_id: str
    compiled: CompiledWorkflow
    workflow_inputs: dict[str, DataHandle]
    code_handles: tuple[CodeHandle, ...]
    session_key_hash: str | None
    submitted_at_ms: int
    deadline_at_ms: int | None
    future: asyncio.Future[RunDataIndexRef]


@dataclass(slots=True)
class _CancelCommand:
    run_id: str
    target: RunStatus
    reason: str
    future: asyncio.Future[RunSnapshot]


@dataclass(slots=True)
class _DestroyCommand:
    run_id: str
    force: bool
    future: asyncio.Future[DestroyResult]


@dataclass(slots=True)
class _ShutdownCommand:
    future: asyncio.Future[None]


@dataclass(slots=True)
class _WakeCommand:
    future: asyncio.Future[None]


class SchedulerCore:
    """One event-loop authority for lifecycle, deadlines, queues and leases."""

    def __init__(
        self,
        *,
        state: RunStateManager,
        deadlines: DeadlineManager,
        indexes: RunDataIndexRegistry,
        anchors: DeclaredOnlyAnchorProvider,
        placement: PlacementManager,
        runtime: FakeRuntimeBackend,
        recorder: InMemoryRecorder | NoopRecorder,
        policy: SchedulingPolicy,
        partitioner: HeterogeneousPartitioner,
        clock: Clock | None = None,
        placement_lookahead: int = 8,
        dispatch_timeout_ms: int = 5_000,
        recorder_flush_timeout_ms: int = 1_000,
        controller_producer_id: str = "controller",
        recovery: RecoveryPolicy | None = None,
    ) -> None:
        self.state = state
        self.deadlines = deadlines
        self.indexes = indexes
        self.anchors = anchors
        self.placement = placement
        self.runtime = runtime
        self.recorder = recorder
        self.policy = policy
        self.partitioner = partitioner
        self.clock = clock or SystemClock()
        self.placement_lookahead = placement_lookahead
        self.dispatch_timeout_ms = dispatch_timeout_ms
        self.recorder_flush_timeout_ms = recorder_flush_timeout_ms
        self.controller_producer_id = controller_producer_id
        self.recovery = recovery or RecoveryPolicy()
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._runner: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._runs: dict[str, _RunExecution] = {}
        self._queued: dict[TaskKey, _QueuedRecord] = {}
        self._queue_generations: dict[TaskKey, int] = {}
        self._enqueue_sequence = 0
        self._partition_cursor = 0
        self._dispatches: dict[str, _DispatchRecord] = {}
        self._seen_runtime_events: dict[str, str] = {}
        self._producer_sequence = 0
        self._terminal_waiters: dict[str, list[asyncio.Future[RunSnapshot]]] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self.runtime.set_event_sink(self.post_runtime_event)
        await self.runtime.start()
        self._running = True
        self._runner = asyncio.create_task(self._run_loop())

    async def commit_run(
        self,
        *,
        run_id: str,
        compiled: CompiledWorkflow,
        workflow_inputs: dict[str, DataHandle],
        code_handles: tuple[CodeHandle, ...],
        session_key_hash: str | None,
        submitted_at_ms: int,
        deadline_at_ms: int | None,
    ) -> RunDataIndexRef:
        future: asyncio.Future[RunDataIndexRef] = self._new_future()
        await self._queue.put(
            _CommitCommand(
                run_id,
                compiled,
                workflow_inputs,
                code_handles,
                session_key_hash,
                submitted_at_ms,
                deadline_at_ms,
                future,
            )
        )
        return await future

    async def cancel_run(self, run_id: str, *, reason: str) -> RunSnapshot:
        future: asyncio.Future[RunSnapshot] = self._new_future()
        await self._queue.put(
            _CancelCommand(run_id, RunStatus.CANCELLED, reason, future)
        )
        return await future

    async def destroy_run(self, run_id: str, *, force: bool = False) -> DestroyResult:
        future: asyncio.Future[DestroyResult] = self._new_future()
        await self._queue.put(_DestroyCommand(run_id, force, future))
        return await future

    async def wake_deadlines(self) -> None:
        future: asyncio.Future[None] = self._new_future()
        await self._queue.put(_WakeCommand(future))
        await future

    async def wait_terminal(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> RunSnapshot:
        snapshot = self.state.snapshot(run_id)
        if snapshot.terminal:
            return snapshot
        future: asyncio.Future[RunSnapshot] = self._new_future()
        self._terminal_waiters.setdefault(run_id, []).append(future)
        if timeout_seconds is None:
            return await future
        return await asyncio.wait_for(future, timeout_seconds)

    async def shutdown(self) -> None:
        if not self._running:
            return
        future: asyncio.Future[None] = self._new_future()
        await self._queue.put(_ShutdownCommand(future))
        await future
        assert self._runner is not None
        await self._runner
        self._runner = None

    def snapshot(self, run_id: str) -> RunSnapshot:
        return self.state.snapshot(run_id)

    def run_index_ref(self, run_id: str) -> RunDataIndexRef:
        return self._runs[run_id].index_ref

    def result(self, run_id: str, task_id: str) -> dict[str, object]:
        execution = self._runs[run_id]
        definition_id = execution.compiled.tasks[task_id].definition_id
        output_names = execution.compiled.definitions[definition_id].output_names
        index = self.indexes.get(run_id)
        return index.read_task_result(
            task_id,
            output_names,
            controller_generation=execution.index_ref.controller_generation,
            index_generation=execution.index_ref.index_generation,
        )

    def post_runtime_event(self, event: RuntimeEvent) -> None:
        if self._loop is None or not self._running:
            for _, handle in event.output_handles:
                try:
                    self.indexes.data_store.release(handle)
                except Exception:
                    pass
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._loop:
            self._queue.put_nowait(event)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def _run_loop(self) -> None:
        while self._running:
            item: object | None = None
            timeout = self._next_wait_seconds()
            try:
                if timeout is None:
                    item = await self._queue.get()
                else:
                    item = await asyncio.wait_for(self._queue.get(), timeout)
            except TimeoutError:
                pass
            try:
                if item is not None:
                    await self._process_item(item)
                await self._process_due_deadlines()
                if self._running:
                    await self._dispatch_pass()
            except Exception as exc:
                await self._interrupt_after_scheduler_failure(exc)

    def _next_wait_seconds(self) -> float | None:
        if not self.clock.automatic_wait:
            return None
        next_due = self.deadlines.next_due_at_ms()
        if next_due is None:
            return None
        return max(0.0, (next_due - self.clock.monotonic_ms()) / 1000)

    async def _process_item(self, item: object) -> None:
        try:
            if isinstance(item, _CommitCommand):
                commit_result = self._commit(item)
                item.future.set_result(commit_result)
                self._activate_run(item.run_id)
            elif isinstance(item, _CancelCommand):
                cancel_result = await self._terminate_run(
                    item.run_id, item.target, item.reason
                )
                item.future.set_result(cancel_result)
            elif isinstance(item, _DestroyCommand):
                destroy_result = await self._destroy(item.run_id, force=item.force)
                item.future.set_result(destroy_result)
            elif isinstance(item, _WakeCommand):
                item.future.set_result(None)
            elif isinstance(item, _ShutdownCommand):
                await self._shutdown_active_runs()
                self._running = False
                item.future.set_result(None)
            elif isinstance(item, RuntimeEvent):
                await self._handle_runtime_event(item)
            else:
                raise TypeError(f"unsupported SchedulerCore item: {type(item).__name__}")
        except Exception as exc:
            future = getattr(item, "future", None)
            if future is not None and not future.done():
                future.set_exception(exc)
            elif isinstance(item, RuntimeEvent):
                await self._fail_internal_runtime_event(item, exc)
            else:
                raise

    def _commit(self, command: _CommitCommand) -> RunDataIndexRef:
        if command.run_id in self._runs:
            raise RuntimeError(f"run already committed: {command.run_id}")
        self.state.create_run(
            run_id=command.run_id,
            compiled=command.compiled,
            routing_session_key_hash=command.session_key_hash,
            submitted_at_ms=command.submitted_at_ms,
            deadline_at_ms=command.deadline_at_ms,
        )
        try:
            index = self.indexes.create_and_adopt(
                run_id=command.run_id,
                workflow_inputs=command.workflow_inputs,
            )
        except Exception:
            self.state.remove_submitted_run(command.run_id)
            raise
        code_by_definition = {item.definition_id: item for item in command.code_handles}
        execution = _RunExecution(
            compiled=command.compiled,
            code_handles=command.code_handles,
            code_by_definition=code_by_definition,
            index_ref=index.reference,
        )
        self._runs[command.run_id] = execution
        if command.deadline_at_ms is not None:
            self.deadlines.register(
                kind=DeadlineKind.RUN,
                run_id=command.run_id,
                due_at_ms=command.deadline_at_ms,
            )
        return index.reference

    def _activate_run(self, run_id: str) -> None:
        started = self.state.start_run(run_id, self.clock.monotonic_ms())
        self._record(run_id, "run_submitted")
        for task_id in started.ready_task_ids:
            self._enqueue_ready(run_id, task_id)

    def _enqueue_ready(self, run_id: str, task_id: str) -> None:
        execution = self._runs[run_id]
        task_snapshot = self.state.snapshot(run_id).task(task_id)
        if task_snapshot.status is not TaskStatus.READY:
            return
        anchor = self.anchors.resolve(
            run_id=run_id,
            compiled=execution.compiled,
            task_id=task_id,
        )
        now = self.clock.monotonic_ms()
        if not self.state.mark_queued(run_id, task_id, now):
            return
        key = TaskKey(run_id, task_id)
        generation = self._queue_generations.get(key, 0) + 1
        self._queue_generations[key] = generation
        self._enqueue_sequence += 1
        queued_snapshot = self.state.snapshot(run_id).task(task_id)
        token = QueueToken(key, generation)
        view = SchedulableTaskView(
            queue_token=token,
            task_kind=anchor.task_kind,
            ready_at_ms=queued_snapshot.ready_at_ms or now,
            queued_at_ms=queued_snapshot.queued_at_ms or now,
            enqueue_sequence=self._enqueue_sequence,
            depth_from_entry=execution.compiled.depth_from_entry[task_id],
            depth_to_exit=execution.compiled.depth_to_exit[task_id],
            resource_anchor=anchor,
        )
        partition = self.partitioner.partition(view)
        self._queued[key] = _QueuedRecord(view, partition)
        self.policy.enqueue(partition, view)
        self._record(run_id, "task_queued", task_id=task_id)

    async def _dispatch_pass(self) -> None:
        if not self._queued:
            return
        partitions = sorted({record.partition for record in self._queued.values()})
        if not partitions:
            return
        progress = True
        while progress and self._queued:
            progress = False
            ordered = partitions[
                self._partition_cursor % len(partitions) :
            ] + partitions[: self._partition_cursor % len(partitions)]
            self._partition_cursor = (self._partition_cursor + 1) % len(partitions)
            for partition in ordered:
                proposals = self.policy.propose(partition, self.placement_lookahead)
                for proposal in proposals:
                    key = proposal.task_key
                    queued = self._queued.get(key)
                    if (
                        queued is None
                        or queued.view.queue_token.queue_generation
                        != proposal.queue_generation
                    ):
                        continue
                    anchor = queued.view.resource_anchor
                    if anchor.execution_target.value == "model_service":
                        self.state.set_pending_reason(
                            key.run_id, key.task_id, "model_route_unavailable"
                        )
                        continue
                    task_snapshot = self.state.snapshot(key.run_id).task(key.task_id)
                    next_attempt = task_snapshot.attempt_count + 1
                    now = self.clock.monotonic_ms()
                    placement = self.placement.try_reserve(
                        run_id=key.run_id,
                        task_id=key.task_id,
                        attempt=next_attempt,
                        anchor=anchor,
                        now_ms=now,
                        dispatch_deadline_ms=now + self.dispatch_timeout_ms,
                    )
                    if not placement.selected:
                        self.state.set_pending_reason(
                            key.run_id, key.task_id, placement.rejection_reason
                        )
                        continue
                    assert placement.lease is not None
                    dispatch_id = new_id("dispatch")
                    attempt = self.state.create_attempt(
                        run_id=key.run_id,
                        task_id=key.task_id,
                        dispatch_id=dispatch_id,
                        lease_id=placement.lease.lease_id,
                        node_id=placement.lease.node_id,
                        device_ids=(
                            ()
                            if placement.lease.npu_device_id is None
                            else (placement.lease.npu_device_id,)
                        ),
                        anchor_revision=anchor.revision,
                        now_ms=now,
                    )
                    self.policy.depart(queued.view.queue_token)
                    del self._queued[key]
                    request = self._build_request(
                        key.run_id,
                        key.task_id,
                        attempt.attempt,
                        dispatch_id,
                    )
                    try:
                        handle = await self.runtime.dispatch(request, placement.lease)
                    except Exception as exc:
                        error = self._error(
                            run_id=key.run_id,
                            task_id=key.task_id,
                            attempt=attempt.attempt,
                            dispatch_id=dispatch_id,
                            lease_id=placement.lease.lease_id,
                            error_code="worker_start_failed",
                            category="worker",
                            origin="runtime",
                            phase="dispatched",
                            message=f"{type(exc).__name__}: {exc}",
                        )
                        await self._handle_attempt_failure(
                            run_id=key.run_id,
                            task_id=key.task_id,
                            attempt=attempt.attempt,
                            dispatch_id=dispatch_id,
                            lease_id=placement.lease.lease_id,
                            error=error,
                            attempt_status=AttemptStatus.FAILED,
                            dispatch_handle=None,
                        )
                    else:
                        self._dispatches[dispatch_id] = _DispatchRecord(
                            handle=handle,
                            lease_id=placement.lease.lease_id,
                        )
                        self.deadlines.register(
                            kind=DeadlineKind.LEASE,
                            run_id=key.run_id,
                            task_id=key.task_id,
                            attempt=attempt.attempt,
                            due_at_ms=placement.lease.dispatch_deadline_ms,
                        )
                        self._record(
                            key.run_id,
                            "task_dispatched",
                            task_id=key.task_id,
                            attempt=attempt.attempt,
                            lease_id=placement.lease.lease_id,
                        )
                    progress = True
                    break

    def _build_request(
        self,
        run_id: str,
        task_id: str,
        attempt: int,
        dispatch_id: str,
    ) -> ExecutionRequest:
        execution = self._runs[run_id]
        node = execution.compiled.tasks[task_id]
        definition = execution.compiled.definitions[node.definition_id]
        index = self.indexes.get(run_id)
        ref = execution.index_ref
        arguments: list[RuntimeArgument] = []
        for binding in node.inputs:
            if isinstance(binding, LiteralBinding):
                arguments.append(
                    RuntimeArgument(binding.input_name, "literal", literal=binding.value)
                )
            elif isinstance(binding, DefaultBinding):
                arguments.append(RuntimeArgument(binding.input_name, "default_omitted"))
            elif isinstance(binding, WorkflowInputBinding):
                handle = index.workflow_input_handle(
                    binding.workflow_input_name,
                    controller_generation=ref.controller_generation,
                    index_generation=ref.index_generation,
                )
                arguments.append(
                    RuntimeArgument(binding.input_name, "data_handle", data_handle=handle)
                )
            elif isinstance(binding, OutputBinding):
                handle = index.task_output_handle(
                    binding.source_task_id,
                    binding.source_output,
                    controller_generation=ref.controller_generation,
                    index_generation=ref.index_generation,
                )
                arguments.append(
                    RuntimeArgument(binding.input_name, "data_handle", data_handle=handle)
                )
            else:
                raise TypeError(f"unsupported input binding: {type(binding).__name__}")
        return ExecutionRequest(
            dispatch_id=dispatch_id,
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            task_kind=definition.task_kind,
            execution_target=self.anchors.resolve(
                run_id=run_id,
                compiled=execution.compiled,
                task_id=task_id,
            ).execution_target,
            model_route=None,
            code_handle=execution.code_by_definition[definition.definition_id],
            arguments=tuple(arguments),
            expected_outputs=definition.output_names,
            timeout_ms=definition.timeout_ms,
            environment_fingerprint=self.runtime.environment_fingerprint,
        )

    async def _handle_runtime_event(self, event: RuntimeEvent) -> None:
        if event.event_id in self._seen_runtime_events:
            return
        self._seen_runtime_events[event.event_id] = event.run_id
        if event.run_id not in self._runs:
            self._release_event_outputs(event)
            return
        if event.kind is RuntimeEventKind.WORKER_STARTED:
            await self._worker_started(event)
        elif event.kind is RuntimeEventKind.TASK_RESULT:
            await self._task_result(event)
        elif event.kind in {
            RuntimeEventKind.TASK_FAILED,
            RuntimeEventKind.DISPATCH_FAILED,
        }:
            await self._task_failed(event)
        elif event.kind is RuntimeEventKind.TASK_CANCELLED:
            self.placement.release_lease(
                event.lease_id,
                now_ms=self.clock.monotonic_ms(),
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                reason="runtime_cancelled",
            )

    async def _worker_started(self, event: RuntimeEvent) -> None:
        if not self.state.matches_active_attempt(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
        ):
            await self._cancel_dispatch(event.dispatch_id, "late_worker_started")
            self._release_event_lease(event, "late_worker_started")
            return
        if not self.placement.bind_lease(
            event.lease_id, now_ms=self.clock.monotonic_ms()
        ):
            await self._fail_worker_start(
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                dispatch_id=event.dispatch_id,
                lease_id=event.lease_id,
                reason="PlacementLease could not be bound before WorkerStarted",
            )
            return
        self.deadlines.cancel(
            kind=DeadlineKind.LEASE,
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
        )
        result = self.state.worker_started(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
            now_ms=self.clock.monotonic_ms(),
        )
        if result.accepted:
            definition = self._definition(event.run_id, event.task_id)
            if definition.timeout_ms is not None:
                self.deadlines.register(
                    kind=DeadlineKind.TASK,
                    run_id=event.run_id,
                    task_id=event.task_id,
                    attempt=event.attempt,
                    due_at_ms=self.clock.monotonic_ms() + definition.timeout_ms,
                )
            self._record(
                event.run_id,
                "worker_started",
                task_id=event.task_id,
                attempt=event.attempt,
                lease_id=event.lease_id,
            )

    async def _task_result(self, event: RuntimeEvent) -> None:
        if not self.state.matches_active_attempt(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
        ):
            if not self._matches_published_result(event):
                self._release_event_outputs(event)
            self._release_event_lease(event, "late_result")
            return
        execution = self._runs[event.run_id]
        definition = self._definition(event.run_id, event.task_id)
        output_handles = dict(event.output_handles)
        try:
            index = self.indexes.get(event.run_id)
            index.publish_outputs(
                task_id=event.task_id,
                output_handles=output_handles,
                expected_output_names=definition.output_names,
                controller_generation=execution.index_ref.controller_generation,
                index_generation=execution.index_ref.index_generation,
            )
        except Exception as exc:
            self._release_event_outputs(event)
            error = self._error(
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                dispatch_id=event.dispatch_id,
                lease_id=event.lease_id,
                error_code="result_publish_failed",
                category="data",
                origin="data",
                phase="publishing",
                message=f"{type(exc).__name__}: {exc}",
            )
            await self._handle_attempt_failure(
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                dispatch_id=event.dispatch_id,
                lease_id=event.lease_id,
                error=error,
                attempt_status=AttemptStatus.FAILED,
                dispatch_handle=self._dispatches.get(event.dispatch_id),
            )
            return
        self.deadlines.cancel(
            kind=DeadlineKind.LEASE,
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
        )
        self.deadlines.cancel(
            kind=DeadlineKind.TASK,
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
        )
        self.placement.release_lease(
            event.lease_id,
            now_ms=self.clock.monotonic_ms(),
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            reason="succeeded",
        )
        result = self.state.attempt_succeeded(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
            now_ms=self.clock.monotonic_ms(),
        )
        if not result.accepted:
            self._release_event_outputs(event)
            return
        self._record(
            event.run_id,
            "task_succeeded",
            task_id=event.task_id,
            attempt=event.attempt,
            lease_id=event.lease_id,
        )
        for task_id in result.ready_task_ids:
            self._enqueue_ready(event.run_id, task_id)
        if result.run_terminal:
            await self._on_run_terminal(event.run_id)

    async def _task_failed(self, event: RuntimeEvent) -> None:
        if event.error is None:
            error = self._error(
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                dispatch_id=event.dispatch_id,
                lease_id=event.lease_id,
                error_code="backend_internal_error",
                category="control",
                origin="runtime",
                phase="dispatched",
                message="runtime failure event omitted ErrorInfo",
            )
        else:
            error = event.error
        await self._handle_attempt_failure(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
            lease_id=event.lease_id,
            error=error,
            attempt_status=AttemptStatus.FAILED,
            dispatch_handle=self._dispatches.get(event.dispatch_id),
        )

    async def _handle_attempt_failure(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt: int,
        dispatch_id: str,
        lease_id: str,
        error: ErrorInfo,
        attempt_status: AttemptStatus,
        dispatch_handle: _DispatchRecord | None,
    ) -> None:
        if not self.state.matches_active_attempt(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            dispatch_id=dispatch_id,
        ):
            self._release_attempt_lease(
                lease_id=lease_id,
                run_id=run_id,
                task_id=task_id,
                attempt=attempt,
                reason="late_failure",
            )
            return
        self.deadlines.cancel(
            kind=DeadlineKind.LEASE,
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
        )
        self.deadlines.cancel(
            kind=DeadlineKind.TASK,
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
        )
        self.placement.release_lease(
            lease_id,
            now_ms=self.clock.monotonic_ms(),
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            reason=error.error_code,
        )
        definition = self._definition(run_id, task_id)
        task_snapshot = self.state.snapshot(run_id).task(task_id)
        run_snapshot = self.state.snapshot(run_id)
        cleanup = CleanupBarrier(
            dispatch_invalidated=self.runtime.dispatch_invalidated(dispatch_id),
            worker_released=self.runtime.worker_released(dispatch_id),
            unpublished_data_released=True,
            route_released=True,
            placement_released=(
                self.placement.lease_snapshot(lease_id).status
                not in {LeaseStatus.RESERVED, LeaseStatus.BOUND}
            ),
        )
        decision = self.recovery.decide(
            definition=definition,
            error=error,
            attempt_count=task_snapshot.attempt_count,
            cleanup=cleanup,
            now_ms=self.clock.monotonic_ms(),
            run_deadline_at_ms=run_snapshot.deadline_at_ms,
        )
        if decision.action is RecoveryAction.RETRY:
            result = self.state.attempt_retry_wait(
                run_id=run_id,
                task_id=task_id,
                attempt=attempt,
                dispatch_id=dispatch_id,
                attempt_status=attempt_status,
                error=error,
                now_ms=self.clock.monotonic_ms(),
            )
            if result.accepted:
                assert decision.eligible_at_ms is not None
                eligible_at = decision.eligible_at_ms
                if definition.retry_backoff_ms == 0:
                    eligible = self.state.retry_eligible(
                        run_id=run_id,
                        task_id=task_id,
                        attempt=attempt,
                        now_ms=eligible_at,
                    )
                    for ready_id in eligible.ready_task_ids:
                        self._enqueue_ready(run_id, ready_id)
                else:
                    self.deadlines.register(
                        kind=DeadlineKind.RETRY,
                        run_id=run_id,
                        task_id=task_id,
                        attempt=attempt,
                        due_at_ms=eligible_at,
                    )
                self._record(
                    run_id,
                    "task_retry_wait",
                    task_id=task_id,
                    attempt=attempt,
                    lease_id=lease_id,
                    payload={
                        "decision_id": decision.decision_id,
                        "action": decision.action.value,
                        "reason": decision.reason,
                        "error_code": error.error_code,
                    },
                )
            return
        task_status = (
            TaskStatus.TIMED_OUT
            if attempt_status is AttemptStatus.TIMED_OUT
            else TaskStatus.FAILED
        )
        result = self.state.attempt_final_failure(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            dispatch_id=dispatch_id,
            attempt_status=attempt_status,
            task_status=task_status,
            error=error,
            now_ms=self.clock.monotonic_ms(),
        )
        if result.accepted:
            self._record(
                run_id,
                "task_failed",
                task_id=task_id,
                attempt=attempt,
                lease_id=lease_id,
                payload={
                    "decision_id": decision.decision_id,
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "error_code": error.error_code,
                },
            )
            await self._cleanup_cancelled_attempts(run_id, result.cancelled_attempts)
            await self._on_run_terminal(run_id)

    async def _process_due_deadlines(self) -> None:
        for event in self.deadlines.pop_due(self.clock.monotonic_ms()):
            await self._handle_deadline(event)

    async def _handle_deadline(self, event: DeadlineEvent) -> None:
        if event.kind is DeadlineKind.RUN:
            await self._terminate_run(
                event.run_id,
                RunStatus.TIMED_OUT,
                "run_timed_out",
            )
            return
        assert event.task_id is not None and event.attempt is not None
        if event.kind is DeadlineKind.RETRY:
            eligible = self.state.retry_eligible(
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                now_ms=self.clock.monotonic_ms(),
            )
            for task_id in eligible.ready_task_ids:
                self._enqueue_ready(event.run_id, task_id)
            return
        task = self.state.snapshot(event.run_id).task(event.task_id)
        if not task.attempts:
            return
        attempt = task.attempts[-1]
        if attempt.attempt != event.attempt:
            return
        if event.kind is DeadlineKind.LEASE:
            if attempt.status is not AttemptStatus.DISPATCHED:
                return
            self.placement.expire_lease(
                attempt.lease_id,
                now_ms=self.clock.monotonic_ms(),
            )
            if self.placement.lease_snapshot(attempt.lease_id).status is LeaseStatus.BOUND:
                return
            await self._fail_worker_start(
                run_id=event.run_id,
                task_id=event.task_id,
                attempt=event.attempt,
                dispatch_id=attempt.dispatch_id,
                lease_id=attempt.lease_id,
                reason="WorkerStarted was not received before the dispatch deadline",
            )
            return
        dispatch = self._dispatches.get(attempt.dispatch_id)
        if dispatch is not None:
            await self.runtime.cancel(dispatch.handle, "task_timeout")
        error = self._error(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=attempt.dispatch_id,
            lease_id=attempt.lease_id,
            error_code="task_timeout",
            category="timeout",
            origin="control",
            phase="user_code",
            message="task execution timeout",
        )
        await self._handle_attempt_failure(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=attempt.dispatch_id,
            lease_id=attempt.lease_id,
            error=error,
            attempt_status=AttemptStatus.TIMED_OUT,
            dispatch_handle=dispatch,
        )

    async def _terminate_run(
        self,
        run_id: str,
        target: RunStatus,
        reason: str,
    ) -> RunSnapshot:
        result = self.state.terminate_run(
            run_id=run_id,
            target=target,
            reason=reason,
            now_ms=self.clock.monotonic_ms(),
        )
        if result.accepted:
            await self._cleanup_cancelled_attempts(run_id, result.cancelled_attempts)
            self._record(run_id, f"run_{target.value}", payload={"reason": reason})
            await self._on_run_terminal(run_id)
        return self.state.snapshot(run_id)

    async def _cleanup_cancelled_attempts(
        self,
        run_id: str,
        attempts: tuple[AttemptSnapshot, ...],
    ) -> None:
        for attempt in attempts:
            dispatch = self._dispatches.get(attempt.dispatch_id)
            if dispatch is not None:
                await self.runtime.cancel(dispatch.handle, "run_terminal")
            self.deadlines.cancel(
                kind=DeadlineKind.LEASE,
                run_id=run_id,
                task_id=attempt.task_id,
                attempt=attempt.attempt,
            )
            self.deadlines.cancel(
                kind=DeadlineKind.TASK,
                run_id=run_id,
                task_id=attempt.task_id,
                attempt=attempt.attempt,
            )
            self.placement.release_lease(
                attempt.lease_id,
                now_ms=self.clock.monotonic_ms(),
                reason="run_terminal",
            )

    async def _on_run_terminal(self, run_id: str) -> None:
        self.deadlines.clear_run(run_id)
        for key, queued in list(self._queued.items()):
            if key.run_id == run_id:
                self.policy.depart(queued.view.queue_token)
                del self._queued[key]
        snapshot = self.state.snapshot(run_id)
        for future in self._terminal_waiters.pop(run_id, []):
            if not future.done():
                future.set_result(snapshot)

    async def _destroy(self, run_id: str, *, force: bool) -> DestroyResult:
        execution = self._runs[run_id]
        if execution.destroyed is not None:
            return execution.destroyed
        snapshot = self.state.snapshot(run_id)
        if not snapshot.terminal:
            raise RunNotTerminalError("run must be terminal before destroy")
        flush = await self.recorder.flush_run(
            run_id, self.recorder_flush_timeout_ms
        )
        if not flush.recording_complete and not force:
            raise RuntimeError("recording is incomplete; force is required")
        if self.placement.active_lease_count(run_id) != 0:
            raise RuntimeError("run still owns placement leases")
        tombstone = self.indexes.destroy(
            run_id,
            completed_at_ms=self.clock.monotonic_ms(),
        )
        self.placement.destroy_run_context(run_id)
        self.anchors.destroy_run(run_id)
        released_code_count = len(execution.code_handles)
        await self.runtime.release_code(execution.code_handles)
        execution.code_handles = ()
        execution.code_by_definition.clear()
        self.runtime.release_run(run_id)
        dispatch_ids = [
            dispatch_id
            for dispatch_id, record in self._dispatches.items()
            if record.handle.run_id == run_id
        ]
        for dispatch_id in dispatch_ids:
            del self._dispatches[dispatch_id]
        for key in [key for key in self._queue_generations if key.run_id == run_id]:
            del self._queue_generations[key]
        for event_id in [
            event_id
            for event_id, event_run_id in self._seen_runtime_events.items()
            if event_run_id == run_id
        ]:
            del self._seen_runtime_events[event_id]
        result = DestroyResult(
            run_id=run_id,
            tombstone=tombstone,
            flush_result=flush,
            code_handles_released=released_code_count,
        )
        execution.destroyed = result
        return result

    async def _shutdown_active_runs(self) -> None:
        for run_id in tuple(self._runs):
            snapshot = self.state.snapshot(run_id)
            if not snapshot.terminal:
                await self._terminate_run(
                    run_id,
                    RunStatus.INTERRUPTED,
                    "scheduler_shutdown",
                )

    async def _interrupt_after_scheduler_failure(self, exc: Exception) -> None:
        for run_id in tuple(self._runs):
            snapshot = self.state.snapshot(run_id)
            if snapshot.terminal:
                continue
            self._record(
                run_id,
                "scheduler_interrupted",
                payload={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            await self._terminate_run(
                run_id,
                RunStatus.INTERRUPTED,
                "scheduler_internal_error",
            )

    async def _cancel_dispatch(self, dispatch_id: str, reason: str) -> None:
        dispatch = self._dispatches.get(dispatch_id)
        if dispatch is not None:
            await self.runtime.cancel(dispatch.handle, reason)

    async def _fail_worker_start(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt: int,
        dispatch_id: str,
        lease_id: str,
        reason: str,
    ) -> None:
        if not self.state.matches_active_attempt(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            dispatch_id=dispatch_id,
        ):
            return
        await self._cancel_dispatch(dispatch_id, "worker_start_failed")
        error = self._error(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            dispatch_id=dispatch_id,
            lease_id=lease_id,
            error_code="worker_start_failed",
            category="worker",
            origin="control",
            phase="dispatched",
            message=reason,
        )
        await self._handle_attempt_failure(
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            dispatch_id=dispatch_id,
            lease_id=lease_id,
            error=error,
            attempt_status=AttemptStatus.FAILED,
            dispatch_handle=self._dispatches.get(dispatch_id),
        )

    def _release_event_outputs(self, event: RuntimeEvent) -> None:
        for _, handle in event.output_handles:
            try:
                self.indexes.data_store.release(handle)
            except Exception:
                pass

    def _release_event_lease(self, event: RuntimeEvent, reason: str) -> bool:
        return self._release_attempt_lease(
            lease_id=event.lease_id,
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            reason=reason,
        )

    def _release_attempt_lease(
        self,
        *,
        lease_id: str,
        run_id: str,
        task_id: str,
        attempt: int,
        reason: str,
    ) -> bool:
        try:
            return self.placement.release_lease(
                lease_id,
                now_ms=self.clock.monotonic_ms(),
                run_id=run_id,
                task_id=task_id,
                attempt=attempt,
                reason=reason,
            )
        except StateTransitionError:
            return False

    def _matches_published_result(self, event: RuntimeEvent) -> bool:
        execution = self._runs.get(event.run_id)
        if execution is None:
            return False
        try:
            return self.indexes.get(event.run_id).matches_published_outputs(
                task_id=event.task_id,
                output_handles=dict(event.output_handles),
                controller_generation=execution.index_ref.controller_generation,
                index_generation=execution.index_ref.index_generation,
            )
        except RunDataIndexError:
            return False

    async def _fail_internal_runtime_event(
        self,
        event: RuntimeEvent,
        exc: Exception,
    ) -> None:
        self._release_event_outputs(event)
        if not self.state.matches_active_attempt(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
        ):
            return
        error = self._error(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
            lease_id=event.lease_id,
            error_code="backend_internal_error",
            category="control",
            origin="control",
            phase="cleanup",
            message=f"{type(exc).__name__}: {exc}",
        )
        await self._handle_attempt_failure(
            run_id=event.run_id,
            task_id=event.task_id,
            attempt=event.attempt,
            dispatch_id=event.dispatch_id,
            lease_id=event.lease_id,
            error=error,
            attempt_status=AttemptStatus.FAILED,
            dispatch_handle=self._dispatches.get(event.dispatch_id),
        )

    def _definition(self, run_id: str, task_id: str) -> TaskDefinition:
        execution = self._runs[run_id]
        definition_id = execution.compiled.tasks[task_id].definition_id
        return execution.compiled.definitions[definition_id]

    def _record(
        self,
        run_id: str,
        event_type: str,
        *,
        task_id: str | None = None,
        attempt: int | None = None,
        lease_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        self._producer_sequence += 1
        frozen_payload = freeze_canonical(payload or {})
        if not isinstance(frozen_payload, FrozenMap):
            raise TypeError("recording payload must freeze to a mapping")
        event = ExecutionEvent(
            schema_version=1,
            event_id=new_id("event"),
            experiment_id=run_id,
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            lease_id=lease_id,
            route_lease_id=None,
            model_instance_id=None,
            event_type=event_type,
            producer_id=self.controller_producer_id,
            producer_sequence=self._producer_sequence,
            node_id=None,
            device_id=None,
            monotonic_time_ms=self.clock.monotonic_ms(),
            wall_time_ms=self.clock.wall_ms(),
            duration_ms=None,
            payload=frozen_payload,
        )
        try:
            self.recorder.emit(event)
        except Exception as exc:
            try:
                self.recorder.record_writer_error(
                    run_id,
                    f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass

    def _error(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt: int,
        dispatch_id: str,
        lease_id: str,
        error_code: str,
        category: str,
        origin: str,
        phase: str,
        message: str,
    ) -> ErrorInfo:
        return ErrorInfo(
            schema_version=1,
            error_code=error_code,
            category=category,
            origin=origin,
            message=message,
            retryable_hint=False,
            classification_confidence="exact",
            execution_phase=phase,
            run_id=run_id,
            task_id=task_id,
            attempt=attempt,
            dispatch_id=dispatch_id,
            lease_id=lease_id,
            occurred_at_ms=self.clock.monotonic_ms(),
        )

    def _new_future(self) -> asyncio.Future[Any]:
        return asyncio.get_running_loop().create_future()
