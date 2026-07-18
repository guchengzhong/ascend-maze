"""In-memory C8 implementation for deterministic control-plane tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from ascend_maze.contracts.recording import (
    ExecutionEvent,
    FlushResult,
    RunRecordingContext,
)
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.core.time import monotonic_time_ms


@dataclass(slots=True)
class _RunRecording:
    context: RunRecordingContext
    expected_producers: set[str]
    seen_producers: set[str] = field(default_factory=set)
    last_sequence: dict[str, int] = field(default_factory=dict)
    event_ids: set[str] = field(default_factory=set)
    events: list[ExecutionEvent] = field(default_factory=list)
    dropped_control: int = 0
    sequence_gaps: int = 0
    writer_errors: list[str] = field(default_factory=list)
    flushed: FlushResult | None = None


class InMemoryRecorder:
    def __init__(self, *, control_capacity_per_run: int = 10_000) -> None:
        if control_capacity_per_run < 1:
            raise ValueError("control_capacity_per_run must be positive")
        self.control_capacity_per_run = control_capacity_per_run
        self._runs: dict[str, _RunRecording] = {}
        self._aborted_runs: set[str] = set()
        self._last_sequence: dict[str, int] = {}
        self._closed = False
        self._fail_next_open = False
        self._fail_next_emit = False
        self._lock = RLock()

    def inject_open_failure(self) -> None:
        with self._lock:
            self._fail_next_open = True

    def inject_emit_failure(self) -> None:
        with self._lock:
            self._fail_next_emit = True

    def open_run(self, context: RunRecordingContext) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder is closed")
            if self._fail_next_open:
                self._fail_next_open = False
                raise RuntimeError("injected recorder open failure")
            existing = self._runs.get(context.run_id)
            if existing is not None:
                if existing.context != context:
                    raise ContractValidationError(
                        "run recording context conflicts with existing context"
                    )
                return
            self._runs[context.run_id] = _RunRecording(
                context=context,
                expected_producers=set(context.initial_expected_producer_ids),
            )

    def abort_run(self, run_id: str) -> bool:
        with self._lock:
            existed = self._runs.pop(run_id, None) is not None
            self._aborted_runs.add(run_id)
            return existed

    def expect_producer(self, run_id: str, producer_id: str) -> None:
        if not producer_id:
            raise ContractValidationError("producer_id is required")
        with self._lock:
            self._require_run(run_id).expected_producers.add(producer_id)

    def emit(self, event: ExecutionEvent) -> bool:
        if event.run_id is None:
            return False
        with self._lock:
            recording = self._runs.get(event.run_id)
            if recording is None or recording.flushed is not None:
                return False
            if self._fail_next_emit:
                self._fail_next_emit = False
                raise RuntimeError("injected recorder emit failure")
            if event.event_id in recording.event_ids:
                return True
            if len(recording.events) >= self.control_capacity_per_run:
                recording.dropped_control += 1
                return False
            previous = self._last_sequence.get(event.producer_id)
            if previous is not None and event.producer_sequence != previous + 1:
                recording.sequence_gaps += 1
            self._last_sequence[event.producer_id] = event.producer_sequence
            recording.last_sequence[event.producer_id] = event.producer_sequence
            recording.seen_producers.add(event.producer_id)
            recording.event_ids.add(event.event_id)
            recording.events.append(event)
            return True

    def record_writer_error(self, run_id: str, message: str) -> None:
        with self._lock:
            recording = self._runs.get(run_id)
            if recording is not None and recording.flushed is None:
                recording.writer_errors.append(message)

    async def flush_run(self, run_id: str, timeout_ms: int) -> FlushResult:
        del timeout_ms
        started = monotonic_time_ms()
        with self._lock:
            recording = self._require_run(run_id)
            if recording.flushed is not None:
                return recording.flushed
            missing = recording.expected_producers - recording.seen_producers
            complete = (
                recording.dropped_control == 0
                and recording.sequence_gaps == 0
                and not missing
                and not recording.writer_errors
            )
            result = FlushResult(
                run_id=run_id,
                committed_files=(),
                dropped_control_event_count=recording.dropped_control,
                dropped_telemetry_count=0,
                sequence_gap_count=recording.sequence_gaps,
                missing_producer_count=len(missing),
                writer_errors=tuple(recording.writer_errors),
                recording_complete=complete,
                flush_duration_ms=max(0, monotonic_time_ms() - started),
            )
            recording.flushed = result
            return result

    async def close(self, timeout_ms: int) -> None:
        del timeout_ms
        with self._lock:
            self._closed = True

    def events(self, run_id: str) -> tuple[ExecutionEvent, ...]:
        with self._lock:
            return tuple(self._require_run(run_id).events)

    def context(self, run_id: str) -> RunRecordingContext:
        with self._lock:
            return self._require_run(run_id).context

    def is_aborted(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._aborted_runs

    @property
    def active_run_count(self) -> int:
        with self._lock:
            return len(self._runs)

    def _require_run(self, run_id: str) -> _RunRecording:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown recording run: {run_id}") from exc


class NoopRecorder:
    def open_run(self, context: RunRecordingContext) -> None:
        del context

    def abort_run(self, run_id: str) -> bool:
        del run_id
        return False

    def expect_producer(self, run_id: str, producer_id: str) -> None:
        del run_id, producer_id

    def emit(self, event: ExecutionEvent) -> bool:
        del event
        return True

    def record_writer_error(self, run_id: str, message: str) -> None:
        del run_id, message

    async def flush_run(self, run_id: str, timeout_ms: int) -> FlushResult:
        del timeout_ms
        return FlushResult(run_id, (), 0, 0, 0, 0, (), True, 0)

    async def close(self, timeout_ms: int) -> None:
        del timeout_ms
