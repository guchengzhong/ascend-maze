"""Bounded asynchronous Parquet execution recorder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
from queue import Empty, Full, Queue
import secrets
from threading import Condition, Event, RLock, Thread
import time
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from ascend_maze.contracts.recording import (
    ExecutionEvent,
    FlushResult,
    ParquetRecorderConfig,
    RunEventPage,
    RunRecordingContext,
)
from ascend_maze.core.canonical import (
    FrozenMap,
    canonical_bytes,
    canonical_digest,
    decode_canonical_bytes,
)
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.core.time import monotonic_time_ms
from ascend_maze.recording.cursor import CursorCodec, CursorPosition

_TELEMETRY_EVENT_TYPES = frozenset(
    {
        "device_resource_sample",
        "node_resource_sample",
        "recorder_health",
    }
)


@dataclass(frozen=True, slots=True)
class _QueuedEvent:
    event: ExecutionEvent
    channel: str


@dataclass(slots=True)
class _RunState:
    context: RunRecordingContext
    expected_producers: set[str]
    seen_producers: set[str] = field(default_factory=set)
    events_by_id: dict[str, ExecutionEvent] = field(default_factory=dict)
    dropped_control: int = 0
    dropped_telemetry: int = 0
    sequence_gaps: int = 0
    writer_errors: list[str] = field(default_factory=list)
    pending_events: int = 0
    accepting: bool = True
    aborted: bool = False
    context_file: str | None = None
    event_files: list[str] = field(default_factory=list)
    committed_files: list[str] = field(default_factory=list)
    producer_flushes: dict[str, FlushResult] = field(default_factory=dict)
    shard_sequences: dict[tuple[str, str, str], int] = field(default_factory=dict)
    flush_in_progress: bool = False
    flushed: FlushResult | None = None


class ParquetRecorder:
    """Keep emit non-blocking while a single background thread commits shards."""

    def __init__(
        self,
        config: ParquetRecorderConfig,
        *,
        cursor_signing_key: bytes | None = None,
    ) -> None:
        if not isinstance(config, ParquetRecorderConfig):
            raise TypeError("config must be ParquetRecorderConfig")
        self.config = config
        self.root = Path(config.root_directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self._control_queue: Queue[_QueuedEvent] = Queue(
            config.control_queue_capacity
        )
        self._telemetry_queue: Queue[_QueuedEvent] = Queue(
            config.telemetry_queue_capacity
        )
        self._runs: dict[str, _RunState] = {}
        self._aborted_runs: set[str] = set()
        self._event_owners: dict[str, str] = {}
        self._last_sequence: dict[str, int] = {}
        self._closed = False
        self._fail_next_write = False
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._stop = Event()
        self._cursor = CursorCodec(cursor_signing_key or secrets.token_bytes(32))
        self._writer = Thread(
            target=self._writer_loop,
            name="ascend-maze-parquet-recorder",
            daemon=True,
        )
        self._writer.start()

    def inject_write_failure(self) -> None:
        with self._lock:
            self._fail_next_write = True

    def open_run(self, context: RunRecordingContext) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder is closed")
            if context.run_id in self._aborted_runs:
                raise RuntimeError("run recording was aborted")
            existing = self._runs.get(context.run_id)
            if existing is not None:
                if existing.context != context:
                    raise ContractValidationError(
                        "run recording context conflicts with existing context"
                    )
                return
            self._runs[context.run_id] = _RunState(
                context=context,
                expected_producers=set(context.initial_expected_producer_ids),
            )

    def abort_run(self, run_id: str) -> bool:
        with self._condition:
            state = self._runs.get(run_id)
            if state is None:
                self._aborted_runs.add(run_id)
                return False
            if state.flushed is not None:
                raise RuntimeError("cannot abort a flushed recording")
            state.accepting = False
            state.aborted = True
            self._aborted_runs.add(run_id)
            files = tuple(state.committed_files)
            self._retire_aborted_run_if_idle(run_id, state)
            self._condition.notify_all()
        for path in files:
            self._discard_committed_file(Path(path))
        return True

    def expect_producer(self, run_id: str, producer_id: str) -> None:
        if not isinstance(producer_id, str) or not producer_id:
            raise ContractValidationError("producer_id is required")
        with self._lock:
            state = self._require_run(run_id)
            if state.flushed is not None or not state.accepting:
                raise RuntimeError("run recording no longer accepts producers")
            state.expected_producers.add(producer_id)

    def emit(self, event: ExecutionEvent) -> bool:
        if event.run_id is None:
            return False
        channel = (
            "telemetry"
            if event.event_type in _TELEMETRY_EVENT_TYPES
            else "control"
        )
        target = (
            self._telemetry_queue if channel == "telemetry" else self._control_queue
        )
        with self._condition:
            state = self._runs.get(event.run_id)
            if (
                self._closed
                or state is None
                or not state.accepting
                or state.flushed is not None
            ):
                return False
            if event.experiment_id != state.context.experiment_id:
                raise ContractValidationError(
                    "event experiment_id does not match RunRecordingContext"
                )
            if event.producer_id not in state.expected_producers:
                raise ContractValidationError("event producer is not expected for run")
            owner = self._event_owners.get(event.event_id)
            if owner is not None:
                existing_state = self._runs[owner]
                existing = existing_state.events_by_id[event.event_id]
                if existing != event:
                    raise ContractValidationError("event_id identifies conflicting events")
                return True
            previous = self._last_sequence.get(event.producer_id)
            if previous is not None and event.producer_sequence != previous + 1:
                state.sequence_gaps += 1
            try:
                target.put_nowait(_QueuedEvent(event, channel))
            except Full:
                if channel == "telemetry":
                    state.dropped_telemetry += 1
                else:
                    state.dropped_control += 1
                return False
            self._last_sequence[event.producer_id] = event.producer_sequence
            state.seen_producers.add(event.producer_id)
            state.events_by_id[event.event_id] = event
            self._event_owners[event.event_id] = event.run_id
            state.pending_events += 1
            self._condition.notify_all()
            return True

    def record_writer_error(self, run_id: str, message: str) -> None:
        if not isinstance(message, str) or not message:
            raise ContractValidationError("writer error message is required")
        with self._lock:
            state = self._runs.get(run_id)
            if state is not None and state.flushed is None:
                state.writer_errors.append(message)

    def merge_producer_flush(
        self,
        run_id: str,
        producer_id: str,
        result: FlushResult,
    ) -> None:
        if not isinstance(producer_id, str) or not producer_id:
            raise ContractValidationError("producer_id is required")
        if result.run_id != run_id:
            raise ContractValidationError("producer FlushResult run_id mismatch")
        with self._lock:
            state = self._require_run(run_id)
            if state.flushed is not None:
                raise RuntimeError("run recording is already flushed")
            if state.aborted or not state.accepting:
                raise RuntimeError("run recording no longer accepts producer flushes")
            if producer_id not in state.expected_producers:
                raise ContractValidationError("producer FlushResult is not expected for run")
            existing = state.producer_flushes.get(producer_id)
            if existing is not None and existing != result:
                raise ContractValidationError("producer FlushResult conflict")
        for path in result.committed_files:
            self._validate_remote_committed_file(path)
        with self._lock:
            state = self._require_run(run_id)
            if state.flushed is not None:
                raise RuntimeError("run recording is already flushed")
            if state.aborted or not state.accepting:
                raise RuntimeError("run recording no longer accepts producer flushes")
            existing = state.producer_flushes.get(producer_id)
            if existing is not None and existing != result:
                raise ContractValidationError("producer FlushResult conflict")
            state.producer_flushes[producer_id] = result

    async def flush_run(self, run_id: str, timeout_ms: int) -> FlushResult:
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        return await asyncio.to_thread(self._flush_blocking, run_id, timeout_ms)

    async def close(self, timeout_ms: int) -> None:
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        started = monotonic_time_ms()
        with self._lock:
            self._closed = True
            run_ids = tuple(
                run_id
                for run_id, state in self._runs.items()
                if not state.aborted and state.flushed is None
            )
        flush_errors: list[str] = []
        try:
            for run_id in run_ids:
                remaining = timeout_ms - (monotonic_time_ms() - started)
                if remaining <= 0:
                    flush_errors.append("recorder close deadline expired before flush")
                    break
                try:
                    await self.flush_run(run_id, remaining)
                except Exception as exc:
                    flush_errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            self._stop.set()
            remaining_seconds = max(
                0.001,
                (timeout_ms - (monotonic_time_ms() - started)) / 1_000,
            )
            await asyncio.to_thread(self._writer.join, remaining_seconds)
        if self._writer.is_alive():
            raise TimeoutError("Parquet recorder writer did not stop before deadline")
        if flush_errors:
            raise RuntimeError("; ".join(flush_errors))

    def get_run_events(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> RunEventPage:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ContractValidationError("page limit must be positive")
        if limit > self.config.max_page_size:
            raise ContractValidationError("page limit exceeds max_page_size")
        with self._lock:
            state = self._require_run(run_id)
            if state.flushed is None:
                raise RuntimeError("historical events are unavailable before flush")
            files = tuple(
                sorted(
                    path
                    for path in state.flushed.committed_files
                    if ".control." in Path(path).name
                    or ".telemetry." in Path(path).name
                )
            )
        manifest_digest = canonical_digest(files)
        position = CursorPosition(run_id, manifest_digest, 0, 0)
        if cursor is not None:
            position = self._cursor.decode(cursor)
            if position.run_id != run_id:
                raise ContractValidationError("cursor belongs to another run")
            if position.manifest_digest != manifest_digest:
                raise ContractValidationError("cursor manifest no longer matches")
        events: list[ExecutionEvent] = []
        file_index = position.file_index
        row_index = position.row_index
        while file_index < len(files) and len(events) < limit:
            rows = self._read_rows(Path(files[file_index]))
            while row_index < len(rows) and len(events) < limit:
                events.append(self._row_to_event(rows[row_index]))
                row_index += 1
            if row_index >= len(rows):
                file_index += 1
                row_index = 0
        exhausted = file_index >= len(files)
        next_cursor = None
        if not exhausted:
            next_cursor = self._cursor.encode(
                CursorPosition(run_id, manifest_digest, file_index, row_index)
            )
        return RunEventPage(tuple(events), next_cursor, exhausted)

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

    def _writer_loop(self) -> None:
        interval_seconds = self.config.flush_interval_ms / 1_000
        while (
            not self._stop.is_set()
            or not self._control_queue.empty()
            or not self._telemetry_queue.empty()
        ):
            batch = self._collect_batch(interval_seconds)
            if not batch:
                continue
            grouped: dict[tuple[str, str, str, str], list[ExecutionEvent]] = {}
            for item in batch:
                event = item.event
                assert event.run_id is not None
                key = (
                    event.run_id,
                    item.channel,
                    event.producer_id,
                    event.node_id or "_controller",
                )
                grouped.setdefault(key, []).append(event)
            for key, events in grouped.items():
                self._write_event_group(key, events)

    def _collect_batch(self, interval_seconds: float) -> list[_QueuedEvent]:
        first: _QueuedEvent | None = None
        try:
            first = self._control_queue.get(timeout=min(0.05, interval_seconds))
        except Empty:
            try:
                first = self._telemetry_queue.get(
                    timeout=min(0.05, interval_seconds)
                )
            except Empty:
                return []
        batch = [first]
        deadline = time.monotonic() + interval_seconds
        while len(batch) < self.config.batch_size and time.monotonic() < deadline:
            try:
                batch.append(self._control_queue.get_nowait())
                continue
            except Empty:
                pass
            try:
                batch.append(self._telemetry_queue.get_nowait())
                continue
            except Empty:
                time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))
        return batch

    def _write_event_group(
        self,
        key: tuple[str, str, str, str],
        events: list[ExecutionEvent],
    ) -> None:
        run_id, channel, producer_id, node_id = key
        final_path: Path | None = None
        keep_file = False
        try:
            with self._lock:
                state = self._runs.get(run_id)
                if state is None or state.aborted or state.flushed is not None:
                    return
                shard_key = (producer_id, node_id, channel)
                sequence = state.shard_sequences.get(shard_key, 0) + 1
                state.shard_sequences[shard_key] = sequence
                final_path = self._event_path(
                    state.context,
                    node_id=node_id,
                    producer_id=producer_id,
                    channel=channel,
                    sequence=sequence,
                )
                fail = self._fail_next_write
                self._fail_next_write = False
            if fail:
                raise RuntimeError("injected Parquet writer failure")
            rows = [self._event_to_row(event) for event in events]
            self._write_rows_atomic(final_path, rows, self._event_schema())
            with self._lock:
                state = self._runs.get(run_id)
                if (
                    state is not None
                    and not state.aborted
                    and state.flushed is None
                ):
                    path = str(final_path)
                    state.event_files.append(path)
                    state.committed_files.append(path)
                    keep_file = True
        except Exception as exc:
            with self._lock:
                state = self._runs.get(run_id)
                if state is not None:
                    state.writer_errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if final_path is not None and not keep_file and final_path.exists():
                self._discard_committed_file(final_path)
            with self._condition:
                state = self._runs.get(run_id)
                if state is not None:
                    state.pending_events = max(0, state.pending_events - len(events))
                    self._retire_aborted_run_if_idle(run_id, state)
                self._condition.notify_all()

    def _flush_blocking(self, run_id: str, timeout_ms: int) -> FlushResult:
        started = monotonic_time_ms()
        deadline = started + timeout_ms
        owns_flush = False
        try:
            # One caller owns finalization; concurrent callers reuse its immutable result.
            with self._condition:
                state = self._require_run(run_id)
                while state.flush_in_progress and state.flushed is None:
                    remaining = deadline - monotonic_time_ms()
                    if remaining <= 0:
                        raise TimeoutError("concurrent recorder flush did not finish")
                    self._condition.wait(remaining / 1_000)
                if state.flushed is not None:
                    return state.flushed
                if state.aborted:
                    raise RuntimeError("cannot flush an aborted recording")
                state.flush_in_progress = True
                owns_flush = True
                state.accepting = False
                while state.pending_events > 0:
                    remaining = deadline - monotonic_time_ms()
                    if remaining <= 0:
                        self._append_writer_error(
                            state, "recorder flush deadline expired"
                        )
                        break
                    self._condition.wait(remaining / 1_000)
                if state.aborted:
                    raise RuntimeError("cannot flush an aborted recording")
                needs_context = state.context_file is None
                context = state.context

            if needs_context:
                if monotonic_time_ms() >= deadline:
                    with self._lock:
                        self._append_writer_error(
                            self._require_run(run_id),
                            "recorder flush deadline expired before context commit",
                        )
                else:
                    path = self._context_path(context)
                    keep_context = False
                    try:
                        self._write_rows_atomic(
                            path,
                            [self._context_to_row(context)],
                            self._context_schema(),
                        )
                        with self._lock:
                            state = self._require_run(run_id)
                            # Abort may race after rename, so adoption is decided under the lock.
                            if not state.aborted and state.flushed is None:
                                state.context_file = str(path)
                                state.committed_files.append(str(path))
                                keep_context = True
                    except Exception as exc:
                        with self._lock:
                            self._append_writer_error(
                                self._require_run(run_id),
                                f"{type(exc).__name__}: {exc}",
                            )
                    finally:
                        if not keep_context and path.exists():
                            self._discard_committed_file(path)

            with self._condition:
                state = self._require_run(run_id)
                if state.aborted:
                    raise RuntimeError("cannot flush an aborted recording")
                result = self._build_flush_result(
                    run_id,
                    state,
                    max(0, monotonic_time_ms() - started),
                )
                state.flushed = result
                state.flush_in_progress = False
                owns_flush = False
                self._condition.notify_all()
                return result
        finally:
            if owns_flush:
                with self._condition:
                    remaining_state = self._runs.get(run_id)
                    if remaining_state is not None:
                        remaining_state.flush_in_progress = False
                        self._retire_aborted_run_if_idle(run_id, remaining_state)
                    self._condition.notify_all()

    def _build_flush_result(
        self,
        run_id: str,
        state: _RunState,
        duration_ms: int,
    ) -> FlushResult:
        remote_producers = set(state.producer_flushes)
        missing = state.expected_producers - state.seen_producers - remote_producers
        remote = tuple(state.producer_flushes.values())
        files = tuple(
            sorted(
                set(state.committed_files).union(
                    path for result in remote for path in result.committed_files
                )
            )
        )
        dropped_control = state.dropped_control + sum(
            result.dropped_control_event_count for result in remote
        )
        dropped_telemetry = state.dropped_telemetry + sum(
            result.dropped_telemetry_count for result in remote
        )
        sequence_gaps = state.sequence_gaps + sum(
            result.sequence_gap_count for result in remote
        )
        missing_count = len(missing) + sum(
            result.missing_producer_count for result in remote
        )
        writer_errors = tuple(state.writer_errors) + tuple(
            error for result in remote for error in result.writer_errors
        )
        complete = (
            dropped_control == 0
            and dropped_telemetry == 0
            and sequence_gaps == 0
            and missing_count == 0
            and not writer_errors
            and state.pending_events == 0
            and all(result.recording_complete for result in remote)
        )
        return FlushResult(
            run_id=run_id,
            committed_files=files,
            dropped_control_event_count=dropped_control,
            dropped_telemetry_count=dropped_telemetry,
            sequence_gap_count=sequence_gaps,
            missing_producer_count=missing_count,
            writer_errors=writer_errors,
            recording_complete=complete,
            flush_duration_ms=duration_ms,
        )

    @staticmethod
    def _append_writer_error(state: _RunState, message: str) -> None:
        if message not in state.writer_errors:
            state.writer_errors.append(message)

    def _retire_aborted_run_if_idle(self, run_id: str, state: _RunState) -> None:
        if not state.aborted or state.pending_events > 0 or state.flush_in_progress:
            return
        for event_id in state.events_by_id:
            if self._event_owners.get(event_id) == run_id:
                del self._event_owners[event_id]
        if self._runs.get(run_id) is state:
            del self._runs[run_id]

    def _validate_remote_committed_file(self, path: str) -> None:
        candidate = Path(path)
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ContractValidationError(
                f"producer committed file is unavailable: {path}"
            ) from exc
        if not resolved.is_file() or resolved.suffix != ".parquet":
            raise ContractValidationError(
                f"producer committed file is not a Parquet file: {path}"
            )
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ContractValidationError(
                "producer committed file is outside Recorder root_directory"
            ) from exc

    @staticmethod
    def _discard_committed_file(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except FileNotFoundError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _event_path(
        self,
        context: RunRecordingContext,
        *,
        node_id: str,
        producer_id: str,
        channel: str,
        sequence: int,
    ) -> Path:
        directory = (
            self.root
            / self._path_component(context.experiment_id)
            / self._path_component(node_id)
            / self._path_component(producer_id)
        )
        return directory / (
            f"{self._path_component(context.run_id)}.{channel}.{sequence:08d}.parquet"
        )

    def _context_path(self, context: RunRecordingContext) -> Path:
        producer_id = (
            context.initial_expected_producer_ids[0]
            if context.initial_expected_producer_ids
            else "_unassigned"
        )
        return (
            self.root
            / self._path_component(context.experiment_id)
            / "_context"
            / self._path_component(producer_id)
            / f"{self._path_component(context.run_id)}.context.parquet"
        )

    @staticmethod
    def _path_component(value: str) -> str:
        return quote(value, safe="")

    def _write_rows_atomic(self, path: Path, rows: list[dict[str, object]], schema: Any) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            table = pa.Table.from_pylist(rows, schema=schema)
            compression = None if self.config.compression == "none" else self.config.compression
            pq.write_table(table, temporary, compression=compression)
            descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _event_schema() -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                ("schema_version", pa.int32()),
                ("event_id", pa.string()),
                ("experiment_id", pa.string()),
                ("run_id", pa.string()),
                ("task_id", pa.string()),
                ("attempt", pa.int32()),
                ("lease_id", pa.string()),
                ("route_lease_id", pa.string()),
                ("model_instance_id", pa.string()),
                ("event_type", pa.string()),
                ("producer_id", pa.string()),
                ("producer_sequence", pa.int64()),
                ("node_id", pa.string()),
                ("device_id", pa.string()),
                ("monotonic_time_ms", pa.int64()),
                ("wall_time_ms", pa.int64()),
                ("duration_ms", pa.int64()),
                ("payload", pa.binary()),
            ],
            metadata={b"ascend_maze_schema": b"execution_event_v1"},
        )

    @staticmethod
    def _context_schema() -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                ("schema_version", pa.int32()),
                ("experiment_id", pa.string()),
                ("run_id", pa.string()),
                ("workflow_fingerprint", pa.string()),
                ("config_fingerprint", pa.string()),
                ("environment_fingerprint", pa.string()),
                ("build_revision", pa.string()),
                ("started_wall_time_ms", pa.int64()),
                ("initial_expected_producer_ids", pa.list_(pa.string())),
            ],
            metadata={b"ascend_maze_schema": b"run_recording_context_v1"},
        )

    @staticmethod
    def _event_to_row(event: ExecutionEvent) -> dict[str, object]:
        return {
            "schema_version": event.schema_version,
            "event_id": event.event_id,
            "experiment_id": event.experiment_id,
            "run_id": event.run_id,
            "task_id": event.task_id,
            "attempt": event.attempt,
            "lease_id": event.lease_id,
            "route_lease_id": event.route_lease_id,
            "model_instance_id": event.model_instance_id,
            "event_type": event.event_type,
            "producer_id": event.producer_id,
            "producer_sequence": event.producer_sequence,
            "node_id": event.node_id,
            "device_id": event.device_id,
            "monotonic_time_ms": event.monotonic_time_ms,
            "wall_time_ms": event.wall_time_ms,
            "duration_ms": event.duration_ms,
            "payload": canonical_bytes(event.payload),
        }

    @staticmethod
    def _context_to_row(context: RunRecordingContext) -> dict[str, object]:
        return {
            "schema_version": context.schema_version,
            "experiment_id": context.experiment_id,
            "run_id": context.run_id,
            "workflow_fingerprint": context.workflow_fingerprint,
            "config_fingerprint": context.config_fingerprint,
            "environment_fingerprint": context.environment_fingerprint,
            "build_revision": context.build_revision,
            "started_wall_time_ms": context.started_wall_time_ms,
            "initial_expected_producer_ids": list(
                context.initial_expected_producer_ids
            ),
        }

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, object]]:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()

    @staticmethod
    def _row_to_event(row: dict[str, object]) -> ExecutionEvent:
        payload_bytes = row["payload"]
        if not isinstance(payload_bytes, bytes):
            raise ContractValidationError("Parquet event payload must be bytes")
        payload = decode_canonical_bytes(payload_bytes)
        if not isinstance(payload, FrozenMap):
            raise ContractValidationError("Parquet event payload must be a mapping")
        return ExecutionEvent(
            schema_version=ParquetRecorder._required_int(row, "schema_version"),
            event_id=ParquetRecorder._required_str(row, "event_id"),
            experiment_id=ParquetRecorder._required_str(row, "experiment_id"),
            run_id=ParquetRecorder._required_str(row, "run_id"),
            task_id=ParquetRecorder._optional_str(row, "task_id"),
            attempt=ParquetRecorder._optional_int(row, "attempt"),
            lease_id=ParquetRecorder._optional_str(row, "lease_id"),
            route_lease_id=ParquetRecorder._optional_str(row, "route_lease_id"),
            model_instance_id=ParquetRecorder._optional_str(
                row, "model_instance_id"
            ),
            event_type=ParquetRecorder._required_str(row, "event_type"),
            producer_id=ParquetRecorder._required_str(row, "producer_id"),
            producer_sequence=ParquetRecorder._required_int(
                row, "producer_sequence"
            ),
            node_id=ParquetRecorder._optional_str(row, "node_id"),
            device_id=ParquetRecorder._optional_str(row, "device_id"),
            monotonic_time_ms=ParquetRecorder._required_int(
                row, "monotonic_time_ms"
            ),
            wall_time_ms=ParquetRecorder._required_int(row, "wall_time_ms"),
            duration_ms=ParquetRecorder._optional_int(row, "duration_ms"),
            payload=payload,
        )

    @staticmethod
    def _required_int(row: dict[str, object], name: str) -> int:
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractValidationError(f"Parquet {name} must be an integer")
        return value

    @staticmethod
    def _optional_int(row: dict[str, object], name: str) -> int | None:
        value = row[name]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractValidationError(f"Parquet {name} must be an integer")
        return value

    @staticmethod
    def _required_str(row: dict[str, object], name: str) -> str:
        value = row[name]
        if not isinstance(value, str) or not value:
            raise ContractValidationError(f"Parquet {name} must be a string")
        return value

    @staticmethod
    def _optional_str(row: dict[str, object], name: str) -> str | None:
        value = row[name]
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ContractValidationError(f"Parquet {name} must be a string")
        return value

    def _require_run(self, run_id: str) -> _RunState:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown recording run: {run_id}") from exc
