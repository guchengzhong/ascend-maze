"""Serial C3-C9 coordination shared by Fake and distributed runtimes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Callable
from time import perf_counter_ns
from typing import Any, Protocol

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
from ascend_maze.contracts.recording import (
    ExecutionEvent,
    ExecutionRecorder,
    FlushResult,
)
from ascend_maze.contracts.resources import PlacementLease, ResourceObservation
from ascend_maze.contracts.runtime import (
    CodeHandle,
    DispatchHandle,
    ExecutionRequest,
    RuntimeArgument,
    RuntimeBackend,
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
from ascend_maze.resources.anchors import ResourceAnchorProvider
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.scheduler.contracts import (
    DispatchProposal,
    QueuePartitioner,
    QueueToken,
    RunLifecycleAwarePolicy,
    SchedulableTaskView,
    SchedulingPolicy,
    TaskKey,
)


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


@dataclass(slots=True)
class _BlockedRecord:
    blocked_since_ms: int
    bypass_count: int
    last_reason: str


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


@dataclass(frozen=True, slots=True)
class _ResourceChanged:
    reason: str


class SchedulerRuntimeBackend(RuntimeBackend, Protocol):
    environment_fingerprint: str

    def set_event_sink(self, sink: Callable[[RuntimeEvent], None]) -> None: ...

    def dispatch_invalidated(self, dispatch_id: str) -> bool: ...

    def worker_released(self, dispatch_id: str) -> bool: ...

    def producer_for_lease(self, lease: PlacementLease) -> str | None: ...

    async def release_run(self, run_id: str) -> int: ...


class SchedulerCore:
    """One event-loop authority for lifecycle, deadlines, queues and leases."""

    def __init__(
        self,
        *,
        state: RunStateManager,
        deadlines: DeadlineManager,
        indexes: RunDataIndexRegistry,
        anchors: ResourceAnchorProvider,
        placement: PlacementManager,
        runtime: SchedulerRuntimeBackend,
        recorder: ExecutionRecorder,
        policy: SchedulingPolicy,
        partitioner: QueuePartitioner,
        clock: Clock | None = None,
        placement_lookahead: int = 8,
        max_bypass_count: int = 8,
        dispatch_timeout_ms: int = 5_000,
        recorder_flush_timeout_ms: int = 1_000,
        controller_producer_id: str = "controller",
        recovery: RecoveryPolicy | None = None,
    ) -> None:
        if placement_lookahead < 1:
            raise ValueError("placement_lookahead must be positive")
        if max_bypass_count < 0:
            raise ValueError("max_bypass_count must be non-negative")
        if policy.capabilities.requires_prediction:
            raise ValueError("this SchedulerCore has no task-time prediction provider")
        if policy.capabilities.uses_cluster_snapshot:
            raise ValueError("this SchedulerCore does not expose cluster snapshots to policies")
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
        self.max_bypass_count = max_bypass_count
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
        self._blocked: dict[TaskKey, _BlockedRecord] = {}
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

    def post_resource_changed(self, reason: str) -> bool:
        """Wake queued placement after an authoritative cluster resource change."""

        if not reason:
            raise ValueError("resource change reason is required")
        if self._loop is None or not self._running:
            return False
        event = _ResourceChanged(reason)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._loop:
            self._queue.put_nowait(event)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        return True

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
            except asyncio.TimeoutError:
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
                commit_result = await self._commit(item)
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
            elif isinstance(item, _ResourceChanged):
                for run_id in sorted({key.run_id for key in self._queued}):
                    self._record(
                        run_id,
                        "resource_changed",
                        payload={"reason": item.reason},
                    )
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

    async def _commit(self, command: _CommitCommand) -> RunDataIndexRef:
        if command.run_id in self._runs:
            raise RuntimeError(f"run already committed: {command.run_id}")
        self.state.create_run(
            run_id=command.run_id,
            compiled=command.compiled,
            routing_session_key_hash=command.session_key_hash,
            submitted_at_ms=command.submitted_at_ms,
            deadline_at_ms=command.deadline_at_ms,
        )
        total_value_tasks = sum(
            command.compiled.definitions[node.definition_id].task_kind == "npu"
            for node in command.compiled.tasks.values()
        )
        try:
            if isinstance(self.policy, RunLifecycleAwarePolicy):
                self.policy.register_run(
                    run_id=command.run_id,
                    submitted_at_ms=command.submitted_at_ms,
                    total_value_tasks=total_value_tasks,
                )
        except Exception:
            self.state.remove_submitted_run(command.run_id)
            raise
        try:
            index = await asyncio.to_thread(
                self.indexes.create_and_adopt,
                run_id=command.run_id,
                workflow_inputs=command.workflow_inputs,
            )
        except Exception:
            if isinstance(self.policy, RunLifecycleAwarePolicy):
                self.policy.unregister_run(command.run_id)
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
                policy_started_ns = perf_counter_ns()
                proposals = self.policy.propose(partition, self.placement_lookahead)
                policy_select_ms = (perf_counter_ns() - policy_started_ns) / 1_000_000
                blocked_before: list[TaskKey] = []
                for proposal_rank, proposal in enumerate(proposals, start=1):
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
                        reason = "model_route_unavailable"
                        self.state.set_pending_reason(
                            key.run_id, key.task_id, reason
                        )
                        self._mark_blocked(key, reason)
                        self._record_scheduling_decision(
                            run_id=key.run_id,
                            task_id=key.task_id,
                            partition=partition,
                            proposal=proposal,
                            proposal_rank=proposal_rank,
                            placement_selected=False,
                            pending_reason=reason,
                            policy_select_ms=policy_select_ms,
                            placement_ms=0.0,
                        )
                        blocked_before.append(key)
                        if self._bypass_exhausted(key):
                            break
                        continue
                    task_snapshot = self.state.snapshot(key.run_id).task(key.task_id)
                    next_attempt = task_snapshot.attempt_count + 1
                    now = self.clock.monotonic_ms()
                    placement_started_ns = perf_counter_ns()
                    placement = self.placement.try_reserve(
                        run_id=key.run_id,
                        task_id=key.task_id,
                        attempt=next_attempt,
                        anchor=anchor,
                        now_ms=now,
                        dispatch_deadline_ms=now + self.dispatch_timeout_ms,
                    )
                    placement_ms = (perf_counter_ns() - placement_started_ns) / 1_000_000
                    if not placement.selected:
                        if (
                            placement.rejection_reason
                            == "resource_request_unsatisfiable"
                        ):
                            self.policy.depart(queued.view.queue_token)
                            del self._queued[key]
                            self._blocked.pop(key, None)
                            self._record_scheduling_decision(
                                run_id=key.run_id,
                                task_id=key.task_id,
                                partition=partition,
                                proposal=proposal,
                                proposal_rank=proposal_rank,
                                placement_selected=False,
                                pending_reason=placement.rejection_reason,
                                policy_select_ms=policy_select_ms,
                                placement_ms=placement_ms,
                            )
                            await self._fail_pre_attempt_unsatisfiable(
                                key.run_id,
                                key.task_id,
                                anchor.effective.npu_mem_mb,
                            )
                            progress = True
                            break
                        reason = placement.rejection_reason or "placement_unavailable"
                        self.state.set_pending_reason(
                            key.run_id, key.task_id, reason
                        )
                        self._mark_blocked(key, reason)
                        self._record_scheduling_decision(
                            run_id=key.run_id,
                            task_id=key.task_id,
                            partition=partition,
                            proposal=proposal,
                            proposal_rank=proposal_rank,
                            placement_selected=False,
                            pending_reason=reason,
                            policy_select_ms=policy_select_ms,
                            placement_ms=placement_ms,
                        )
                        blocked_before.append(key)
                        if self._bypass_exhausted(key):
                            break
                        continue
                    assert placement.lease is not None
                    self._record_scheduling_decision(
                        run_id=key.run_id,
                        task_id=key.task_id,
                        partition=partition,
                        proposal=proposal,
                        proposal_rank=proposal_rank,
                        placement_selected=True,
                        pending_reason=None,
                        policy_select_ms=policy_select_ms,
                        placement_ms=placement_ms,
                    )
                    self._expect_runtime_producer(key.run_id, placement.lease)
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
                    self._blocked.pop(key, None)
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
                        for blocked_key in blocked_before:
                            blocked = self._blocked.get(blocked_key)
                            if blocked is not None and blocked_key in self._queued:
                                blocked.bypass_count += 1
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
            await self._release_event_outputs(event)
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
                await self._release_event_outputs(event)
            self._release_event_lease(event, "late_result")
            return
        execution = self._runs[event.run_id]
        definition = self._definition(event.run_id, event.task_id)
        output_handles = dict(event.output_handles)
        try:
            index = self.indexes.get(event.run_id)
            await asyncio.to_thread(
                index.publish_outputs,
                task_id=event.task_id,
                output_handles=output_handles,
                expected_output_names=definition.output_names,
                controller_generation=execution.index_ref.controller_generation,
                index_generation=execution.index_ref.index_generation,
            )
        except Exception as exc:
            await self._release_event_outputs(event)
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
            await self._release_event_outputs(event)
            return
        if isinstance(self.policy, RunLifecycleAwarePolicy):
            self.policy.task_succeeded(
                run_id=event.run_id,
                task_id=event.task_id,
                task_kind=definition.task_kind,
            )
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
            resource_observation=event.resource_observation,
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
        resource_observation: ResourceObservation | None = None,
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
        precondition_reason = self.recovery.retry_precondition_reason(
            definition=definition,
            error=error,
            attempt_count=task_snapshot.attempt_count,
            cleanup=cleanup,
            now_ms=self.clock.monotonic_ms(),
            run_deadline_at_ms=run_snapshot.deadline_at_ms,
        )
        retry_block_reason: str | None = None
        if error.error_code == "npu_oom":
            execution = self._runs[run_id]
            observed_peak = None
            if resource_observation is not None:
                observed_peak = (
                    resource_observation.peak_npu_process_hbm_mb
                    or resource_observation.peak_npu_reserved_mb
                    or resource_observation.peak_npu_allocated_mb
                )
            if precondition_reason is None:
                reanchor = self.anchors.reanchor_after_oom(
                    run_id=run_id,
                    compiled=execution.compiled,
                    task_id=task_id,
                    observed_peak_npu_mem_mb=observed_peak,
                )
                if not reanchor.created:
                    retry_block_reason = reanchor.reason
                elif (
                    reanchor.anchor.effective.npu_mem_mb
                    > self.placement.max_single_npu_allocatable_hbm_mb()
                ):
                    retry_block_reason = "oom_reanchor_unsatisfiable"
                    error = self._error(
                        run_id=run_id,
                        task_id=task_id,
                        attempt=attempt,
                        dispatch_id=dispatch_id,
                        lease_id=lease_id,
                        error_code="resource_request_unsatisfiable",
                        category="configuration",
                        origin="placement",
                        phase="cleanup",
                        message=(
                            "OOM reanchor exceeds every single-NPU allocatable capacity"
                        ),
                    )
                anchor = reanchor.anchor
                anchor_created = reanchor.created
                anchor_reason = reanchor.reason
                previous_npu_mem_mb = reanchor.previous_npu_mem_mb
            else:
                anchor = self.anchors.resolve(
                    run_id=run_id,
                    compiled=execution.compiled,
                    task_id=task_id,
                )
                anchor_created = False
                anchor_reason = precondition_reason
                previous_npu_mem_mb = anchor.effective.npu_mem_mb
            self._record(
                run_id,
                "resource_anchor_oom",
                task_id=task_id,
                attempt=attempt,
                lease_id=lease_id,
                payload={
                    "created": anchor_created,
                    "reason": anchor_reason,
                    "previous_npu_mem_mb": previous_npu_mem_mb,
                    "new_npu_mem_mb": anchor.effective.npu_mem_mb,
                    "observed_peak_npu_mem_mb": observed_peak,
                    "revision": anchor.revision,
                },
            )
        decision = self.recovery.decide(
            definition=definition,
            error=error,
            attempt_count=task_snapshot.attempt_count,
            cleanup=cleanup,
            now_ms=self.clock.monotonic_ms(),
            run_deadline_at_ms=run_snapshot.deadline_at_ms,
            retry_block_reason=retry_block_reason,
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

    async def _fail_pre_attempt_unsatisfiable(
        self,
        run_id: str,
        task_id: str,
        requested_npu_hbm_mb: int,
    ) -> None:
        now = self.clock.monotonic_ms()
        error = self._error(
            run_id=run_id,
            task_id=task_id,
            attempt=0,
            dispatch_id=None,
            lease_id=None,
            error_code="resource_request_unsatisfiable",
            category="configuration",
            origin="placement",
            phase="pre_attempt",
            message=(
                f"requested NPU HBM {requested_npu_hbm_mb} MB exceeds every "
                "single-device allocatable capacity"
            ),
        )
        result = self.state.pre_attempt_final_failure(
            run_id=run_id,
            task_id=task_id,
            error=error,
            now_ms=now,
        )
        if result.accepted:
            self._record(
                run_id,
                "task_failed",
                task_id=task_id,
                payload={"error_code": error.error_code, "attempt": 0},
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
                self._blocked.pop(key, None)
        snapshot = self.state.snapshot(run_id)
        assert snapshot.finished_at_ms is not None
        if isinstance(self.policy, RunLifecycleAwarePolicy):
            try:
                self.policy.run_terminal(
                    run_id=run_id,
                    status=snapshot.status.value,
                    finished_at_ms=snapshot.finished_at_ms,
                )
            except Exception as exc:
                self._record(
                    run_id,
                    "policy_lifecycle_error",
                    payload={
                        "hook": "run_terminal",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
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
        tombstone = await asyncio.to_thread(
            self.indexes.destroy,
            run_id,
            completed_at_ms=self.clock.monotonic_ms(),
        )
        self.placement.destroy_run_context(run_id)
        self.anchors.destroy_run(run_id)
        released_code_count = len(execution.code_handles)
        await self.runtime.release_code(execution.code_handles)
        execution.code_handles = ()
        execution.code_by_definition.clear()
        await self.runtime.release_run(run_id)
        dispatch_ids = [
            dispatch_id
            for dispatch_id, record in self._dispatches.items()
            if record.handle.run_id == run_id
        ]
        for dispatch_id in dispatch_ids:
            del self._dispatches[dispatch_id]
        for key in [key for key in self._queue_generations if key.run_id == run_id]:
            del self._queue_generations[key]
            self._blocked.pop(key, None)
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

    async def _release_event_outputs(self, event: RuntimeEvent) -> None:
        def release() -> None:
            for _, handle in event.output_handles:
                try:
                    self.indexes.data_store.release(handle)
                except Exception:
                    pass

        await asyncio.to_thread(release)

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
        await self._release_event_outputs(event)
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

    def _mark_blocked(self, key: TaskKey, reason: str) -> None:
        blocked = self._blocked.get(key)
        if blocked is None:
            self._blocked[key] = _BlockedRecord(
                blocked_since_ms=self.clock.monotonic_ms(),
                bypass_count=0,
                last_reason=reason,
            )
        else:
            blocked.last_reason = reason

    def _bypass_exhausted(self, key: TaskKey) -> bool:
        blocked = self._blocked.get(key)
        return blocked is not None and blocked.bypass_count >= self.max_bypass_count

    def _record_scheduling_decision(
        self,
        *,
        run_id: str,
        task_id: str,
        partition: str,
        proposal: DispatchProposal,
        proposal_rank: int,
        placement_selected: bool,
        pending_reason: str | None,
        policy_select_ms: float,
        placement_ms: float,
    ) -> None:
        blocked = self._blocked.get(proposal.task_key)
        try:
            policy_metadata: object = dict(proposal.policy_metadata)
        except Exception as exc:
            policy_metadata = {
                "metadata_error": f"{type(exc).__name__}: {exc}",
            }
        self._record(
            run_id,
            "scheduling_decision",
            task_id=task_id,
            payload={
                "policy_name": self.policy.name,
                "policy_version": self.policy.version,
                "partition": partition,
                "proposal_rank": proposal_rank,
                "policy_metadata": policy_metadata,
                "placement_selected": placement_selected,
                "pending_reason": pending_reason,
                "score_compute_ms": proposal.score_compute_ms,
                "policy_select_ms": policy_select_ms,
                "placement_ms": placement_ms,
                "queue_length": sum(
                    queued.partition == partition for queued in self._queued.values()
                ),
                "blocked_since_ms": (
                    None if blocked is None else blocked.blocked_since_ms
                ),
                "bypass_count": 0 if blocked is None else blocked.bypass_count,
            },
        )

    def _expect_runtime_producer(self, run_id: str, lease: PlacementLease) -> None:
        producer_id = self.runtime.producer_for_lease(lease)
        if producer_id is None:
            return
        try:
            self.recorder.expect_producer(run_id, producer_id)
        except Exception as exc:
            try:
                self.recorder.record_writer_error(
                    run_id,
                    f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass

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
        try:
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
        dispatch_id: str | None,
        lease_id: str | None,
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
