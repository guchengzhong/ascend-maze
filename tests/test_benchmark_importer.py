from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ascend_maze.benchmark.canonical import canonical_json_digest
from ascend_maze.benchmark.cli import main
from ascend_maze.benchmark.contracts import TrialManifest
from ascend_maze.benchmark.importer import validate_study
from ascend_maze.benchmark.loader import load_study_plan
from ascend_maze.benchmark.parquet_import import context_schema, event_schema
from ascend_maze.benchmark.persistence import atomic_write_bytes, atomic_write_json
from ascend_maze.contracts.recording import ExecutionEvent, RunRecordingContext
from ascend_maze.core.canonical import canonical_bytes
from ascend_maze.core.errors import ExperimentValidationError
from benchmark_fixtures import write_experiment_spec


class _Fixture:
    def __init__(
        self,
        study: Path,
        trial: Path,
        run_id: str,
        context_file: Path,
        event_file: Path,
        events: tuple[ExecutionEvent, ...],
    ) -> None:
        self.study = study
        self.trial = trial
        self.run_id = run_id
        self.context_file = context_file
        self.event_file = event_file
        self.events = events


def _event(
    sequence: int,
    event_type: str,
    *,
    run_id: str,
    task_id: str | None = None,
    attempt: int | None = None,
    lease_id: str | None = None,
    route_lease_id: str | None = None,
    model_instance_id: str | None = None,
    payload: dict[str, object] | None = None,
    monotonic_time_ms: int | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        schema_version=1,
        event_id=f"event_{sequence}",
        experiment_id=run_id,
        run_id=run_id,
        task_id=task_id,
        attempt=attempt,
        lease_id=lease_id,
        route_lease_id=route_lease_id,
        model_instance_id=model_instance_id,
        event_type=event_type,
        producer_id="controller",
        producer_sequence=sequence,
        node_id=None,
        device_id=None,
        monotonic_time_ms=(
            100 + sequence if monotonic_time_ms is None else monotonic_time_ms
        ),
        wall_time_ms=1_000 + sequence,
        duration_ms=None,
        payload=payload or {},
    )


def _valid_events(
    run_id: str, *, terminal_time: int = 108
) -> tuple[ExecutionEvent, ...]:
    return (
        _event(1, "run_submitted", run_id=run_id),
        _event(2, "task_queued", run_id=run_id, task_id="task_1"),
        _event(
            3,
            "task_dispatched",
            run_id=run_id,
            task_id="task_1",
            attempt=1,
            lease_id="placement_1",
            route_lease_id="route_1",
            model_instance_id="model_instance_1",
            payload={
                "dispatch_id": "dispatch_1",
                "model_id": "model_1",
                "instance_generation": 1,
            },
        ),
        _event(
            4,
            "worker_acquired",
            run_id=run_id,
            task_id="task_1",
            attempt=1,
            lease_id="placement_1",
            payload={"dispatch_id": "dispatch_1", "worker_lease_id": "worker_1"},
        ),
        _event(
            5,
            "model_route_released",
            run_id=run_id,
            task_id="task_1",
            attempt=1,
            route_lease_id="route_1",
            model_instance_id="model_instance_1",
            payload={"model_id": "model_1", "instance_generation": 1},
        ),
        _event(
            6,
            "worker_released",
            run_id=run_id,
            task_id="task_1",
            attempt=1,
            lease_id="placement_1",
            payload={"dispatch_id": "dispatch_1", "worker_lease_id": "worker_1"},
        ),
        _event(
            7,
            "task_succeeded",
            run_id=run_id,
            task_id="task_1",
            attempt=1,
            lease_id="placement_1",
            route_lease_id="route_1",
            model_instance_id="model_instance_1",
            payload={"dispatch_id": "dispatch_1"},
        ),
        _event(
            8,
            "run_terminal",
            run_id=run_id,
            payload={"status": "succeeded", "finished_at_ms": terminal_time},
            monotonic_time_ms=terminal_time,
        ),
    )


def _write_context(
    path: Path, context: RunRecordingContext, *, schema: pa.Schema | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": context.schema_version,
        "experiment_id": context.experiment_id,
        "run_id": context.run_id,
        "workflow_fingerprint": context.workflow_fingerprint,
        "config_fingerprint": context.config_fingerprint,
        "environment_fingerprint": context.environment_fingerprint,
        "build_revision": context.build_revision,
        "started_wall_time_ms": context.started_wall_time_ms,
        "initial_expected_producer_ids": list(context.initial_expected_producer_ids),
    }
    pq.write_table(pa.Table.from_pylist([row], schema=schema or context_schema()), path)


def _write_events(
    path: Path,
    events: tuple[ExecutionEvent, ...],
    *,
    schema: pa.Schema | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
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
        for event in events
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema or event_schema()), path)


def _write_analysis_inputs(
    trial: Path, *, trial_attempt_id: str, run_id: str, config_fingerprint: str
) -> None:
    run = {
        "phase": "measurement",
        "arrival_index": 0,
        "record_id": "record-a",
        "input_digest": "e" * 64,
        "submission_id": "submission-validation-1",
        "scheduled_offset_ms": 0,
        "scheduled_at_monotonic_ms": 1_000,
        "offered_at_monotonic_ms": 1_000,
        "issued_at_monotonic_ms": 1_000,
        "admitted_at_monotonic_ms": 1_001,
        "arrival_lateness_ms": 0,
        "run_id": run_id,
        "submission_replayed": False,
        "submission_error": None,
        "terminal_status": "succeeded",
        "terminal_at_monotonic_ms": 1_100,
        "flushed": True,
        "recording_complete": True,
        "destroyed": True,
    }
    empty = {
        "offered": 0,
        "issued": 0,
        "committed": 0,
        "terminal": 0,
        "succeeded": 0,
        "failed": 0,
        "timed_out": 0,
    }
    measurement = {
        "offered": 1,
        "issued": 1,
        "committed": 1,
        "terminal": 1,
        "succeeded": 1,
        "failed": 0,
        "timed_out": 0,
    }
    atomic_write_json(
        trial / "run_manifest.json",
        {
            "schema_version": 1,
            "schema": "ascend-maze.run-manifest.v1",
            "trial_attempt_id": trial_attempt_id,
            "runs": [run],
            "warmup_excluded_from_measurement": True,
            "warmup_counters": empty,
            "measurement_counters": measurement,
        },
    )
    schedule_schema = pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("trial_attempt_id", pa.string(), nullable=False),
            pa.field("mode", pa.string(), nullable=False),
            pa.field("phase", pa.string(), nullable=False),
            pa.field("arrival_index", pa.int64(), nullable=False),
            pa.field("scheduled_offset_ms", pa.int64(), nullable=True),
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("input_digest", pa.string(), nullable=False),
            pa.field("submission_id", pa.string(), nullable=False),
        ],
        metadata={
            b"ascend_maze.schema": b"ascend-maze.arrival-schedule.v1",
            b"ascend_maze.schema_version": b"1",
            b"ascend_maze.schedule_digest": b"f" * 64,
        },
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "schema_version": 1,
                    "trial_attempt_id": trial_attempt_id,
                    "mode": "poisson",
                    "phase": "measurement",
                    "arrival_index": 0,
                    "scheduled_offset_ms": 0,
                    "record_id": "record-a",
                    "input_digest": "e" * 64,
                    "submission_id": "submission-validation-1",
                }
            ],
            schema=schedule_schema,
        ),
        trial / "arrival_schedule.parquet",
    )
    snapshot = {
        "captured_at_wall_ms": 1_000,
        "controller_generation": "controller-generation-1",
        "config_fingerprint": config_fingerprint,
        "snapshot_digest": canonical_json_digest({}),
        "payload": {},
    }
    atomic_write_json(trial / "resource_before.json", snapshot)
    atomic_write_json(
        trial / "resource_after.json",
        {
            **snapshot,
            "captured_at_wall_ms": 1_200,
            "recovery": {
                "recovered": True,
                "checked_at_wall_ms": 1_200,
                "reason_code": None,
                "details": {},
            },
        },
    )


def _fixture(
    root: Path,
    *,
    events: tuple[ExecutionEvent, ...] | None = None,
    expected_producers: tuple[str, ...] = ("controller",),
    context_changes: dict[str, object] | None = None,
    flush_changes: dict[str, object] | None = None,
) -> _Fixture:
    root.mkdir(parents=True, exist_ok=True)
    spec_root = root / "spec"
    spec_root.mkdir()
    spec_path = write_experiment_spec(spec_root, study_kind="pilot", block_count=3)
    plan = load_study_plan(spec_path)
    trial_spec = plan.trials[0]
    cell = next(item for item in plan.cells if item.cell_id == trial_spec.cell_id)
    attempt = TrialManifest.planned(trial_spec)
    study = root / "study"
    trial = study / "trials" / attempt.trial_attempt_id
    trial.mkdir(parents=True)
    run_id = "run_validation_1"
    actual_events = events or _valid_events(run_id)
    context_values: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": run_id,
        "run_id": run_id,
        "workflow_fingerprint": plan.spec.workload.workflow_fingerprint,
        "config_fingerprint": cell.config_snapshot.config_fingerprint,
        "environment_fingerprint": plan.spec.workload.required_environment_fingerprint,
        "build_revision": plan.spec.build_revision,
        "started_wall_time_ms": 1_000,
        "initial_expected_producer_ids": expected_producers,
    }
    context_values.update(context_changes or {})
    context = RunRecordingContext(**context_values)  # type: ignore[arg-type]
    recording = root / "recording" / run_id
    context_file = recording / "_context" / "controller" / f"{run_id}.context.parquet"
    event_file = recording / "controller" / f"{run_id}.control.00000001.parquet"
    _write_context(context_file, context)
    _write_events(event_file, actual_events)
    committed = (str(context_file.resolve()), str(event_file.resolve()))
    manifest = TrialManifest(
        schema_version=1,
        trial_attempt_id=attempt.trial_attempt_id,
        trial_id=trial_spec.trial_id,
        attempt_index=0,
        state="valid",
        run_ids=(run_id,),
        experiment_ids=(run_id,),
        committed_files=committed,
    )
    atomic_write_json(trial / "trial_manifest.json", manifest.canonical_payload())
    _write_analysis_inputs(
        trial,
        trial_attempt_id=attempt.trial_attempt_id,
        run_id=run_id,
        config_fingerprint=cell.config_snapshot.config_fingerprint,
    )
    flush_payload: dict[str, object] = {
        "run_id": run_id,
        "committed_files": list(committed),
        "dropped_control_event_count": 0,
        "dropped_telemetry_count": 0,
        "sequence_gap_count": 0,
        "missing_producer_count": 0,
        "writer_errors": [],
        "recording_complete": True,
        "flush_duration_ms": 1,
    }
    flush_payload.update(flush_changes or {})
    atomic_write_json(
        trial / "flush_results.json",
        {
            "schema_version": 1,
            "trial_attempt_id": attempt.trial_attempt_id,
            "results": [
                {
                    "run_id": run_id,
                    "recording_complete": flush_payload["recording_complete"],
                    "committed_files": list(committed),
                    "payload": flush_payload,
                }
            ],
        },
    )
    study.mkdir(exist_ok=True)
    atomic_write_bytes(study / "study_plan.json", plan.canonical_bytes + b"\n")
    atomic_write_json(
        study / "study_manifest.json",
        {
            "schema_version": 1,
            "schema": "ascend-maze.study-manifest.v1",
            "study_id": plan.spec.study_id,
            "state": "running",
            "plan_sha256": canonical_json_digest(plan.canonical_payload()),
            "trials": [
                {
                    "trial_id": trial_spec.trial_id,
                    "trial_attempt_id": attempt.trial_attempt_id,
                    "state": "valid",
                    "relative_directory": str(trial.relative_to(study)),
                    "invalid_reasons": [],
                }
            ],
        },
    )
    return _Fixture(study, trial, run_id, context_file, event_file, actual_events)


def _trial_validation(fixture: _Fixture) -> dict[str, object]:
    return json.loads((fixture.trial / "validity.json").read_text(encoding="utf-8"))


def test_validate_imports_only_committed_files_and_is_deterministic(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    atomic_write_json(
        fixture.trial / "state.json",
        {"prompt": "must never be imported", "fabricated_metric": 0},
    )
    private_flush = (
        fixture.context_file.parents[1] / "_flush" / f"{fixture.run_id}.flush.json"
    )
    atomic_write_json(private_flush, {"prompt": "private C8 state"})
    first = validate_study(fixture.study)
    first_summary = (fixture.study / "validation_summary.json").read_bytes()
    first_trial = (fixture.trial / "validity.json").read_bytes()
    first_raw = (fixture.trial / "raw_files.json").read_bytes()

    assert first["study_valid"] is True
    validation = _trial_validation(fixture)
    assert validation["trial_valid"] is True
    assert validation["reason_codes"] == []
    assert validation["index_counts"] == {
        "attempt": 1,
        "dispatch": 1,
        "event": 8,
        "model_instance": 1,
        "placement_lease": 1,
        "producer": 1,
        "route_lease": 1,
        "run": 1,
        "task": 1,
        "worker_lease": 1,
    }
    assert all(item["valid"] for item in validation["metric_valid"])
    raw = json.loads((fixture.trial / "raw_files.json").read_text())
    assert len(raw["files"]) == 2
    for item in raw["files"]:
        source = Path(item["source_path"])
        assert item["size_bytes"] == source.stat().st_size
        assert item["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert validate_study(fixture.study) == first
    assert (fixture.study / "validation_summary.json").read_bytes() == first_summary
    assert (fixture.trial / "validity.json").read_bytes() == first_trial
    assert (fixture.trial / "raw_files.json").read_bytes() == first_raw


def test_validate_bytes_are_stable_across_process_hash_seeds(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    outputs: list[bytes] = []
    for seed in ("1", "999"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ascend_maze.benchmark.cli",
                "validate",
                str(fixture.study),
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        assert completed.stderr == b""
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


def test_validate_cli_reports_valid_invalid_and_contract_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path / "valid")
    assert main(["validate", str(fixture.study)]) == 0
    assert json.loads(capsys.readouterr().out)["study_valid"] is True

    fixture.event_file.write_bytes(b"not parquet")
    invalid = _fixture(tmp_path / "invalid")
    invalid.event_file.write_bytes(b"not parquet")
    assert main(["validate", str(invalid.study)]) == 1
    assert json.loads(capsys.readouterr().out)["study_valid"] is False

    assert main(["validate", str(tmp_path / "missing")]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_code"] == "experiment_validation_failed"


@pytest.mark.parametrize(
    ("corruption", "reason"),
    [
        ("footer", "parquet_footer_invalid"),
        ("metadata", "parquet_metadata_invalid"),
        ("schema", "parquet_schema_invalid"),
    ],
)
def test_parquet_footer_metadata_and_schema_are_strict(
    tmp_path: Path, corruption: str, reason: str
) -> None:
    fixture = _fixture(tmp_path)
    if corruption == "footer":
        fixture.event_file.write_bytes(b"PAR1corrupt")
    elif corruption == "metadata":
        _write_events(
            fixture.event_file,
            fixture.events,
            schema=event_schema().with_metadata(
                {b"ascend_maze_schema": b"unexpected_v1"}
            ),
        )
    else:
        fields = list(event_schema())
        fields[5] = pa.field("attempt", pa.int64())
        _write_events(
            fixture.event_file,
            fixture.events,
            schema=pa.schema(fields, metadata=event_schema().metadata),
        )
    result = validate_study(fixture.study)
    assert result["study_valid"] is False
    assert reason in _trial_validation(fixture)["reason_codes"]


def test_raw_snapshot_detects_valid_parquet_content_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    assert validate_study(fixture.study)["study_valid"] is True
    original_raw = (fixture.trial / "raw_files.json").read_bytes()
    changed = tuple(
        replace(event, payload={"status": "succeeded", "changed": True})
        if event.event_type == "run_terminal"
        else event
        for event in fixture.events
    )
    _write_events(fixture.event_file, changed)
    assert validate_study(fixture.study)["study_valid"] is False
    assert "committed_file_hash_changed" in _trial_validation(fixture)["reason_codes"]
    assert (fixture.trial / "raw_files.json").read_bytes() == original_raw


def test_corrupt_raw_snapshot_is_not_repaired_or_overwritten(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    assert validate_study(fixture.study)["study_valid"] is True
    raw_path = fixture.trial / "raw_files.json"
    payload = json.loads(raw_path.read_text())
    payload["content_digest"] = "0" * 64
    atomic_write_json(raw_path, payload)
    corrupted = raw_path.read_bytes()
    with pytest.raises(ExperimentValidationError, match="digest"):
        validate_study(fixture.study)
    assert raw_path.read_bytes() == corrupted


@pytest.mark.parametrize("suffix", ["unflushed.00000002.parquet", ".pending.tmp"])
def test_unlisted_and_temporary_same_run_files_are_never_imported(
    tmp_path: Path, suffix: str
) -> None:
    fixture = _fixture(tmp_path)
    extra = fixture.event_file.with_name(f"{fixture.run_id}.{suffix}")
    if suffix.endswith(".parquet"):
        sensitive = tuple(
            replace(event, payload={"prompt": "must not be read"})
            if event.event_type == "worker_acquired"
            else event
            for event in fixture.events
        )
        _write_events(extra, sensitive)
    else:
        shutil.copyfile(fixture.event_file, extra)
    assert validate_study(fixture.study)["study_valid"] is False
    validation = _trial_validation(fixture)
    assert "unlisted_trial_file" in validation["reason_codes"]
    assert "privacy_violation" not in validation["reason_codes"]
    assert validation["index_counts"]["event"] == len(fixture.events)
    raw = json.loads((fixture.trial / "raw_files.json").read_text())
    assert [item["source_path"] for item in raw["ignored_files"]] == [str(extra)]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"recording_complete": False}, "recording_incomplete"),
        ({"dropped_control_event_count": 1}, "dropped_control_events"),
        ({"dropped_telemetry_count": 1}, "dropped_telemetry_events"),
        ({"sequence_gap_count": 1}, "sequence_gap_reported"),
        ({"missing_producer_count": 1}, "missing_producer_reported"),
        ({"writer_errors": ["write failed"]}, "writer_error"),
    ],
)
def test_flush_result_health_is_not_equated_with_recording_complete(
    tmp_path: Path, changes: dict[str, object], reason: str
) -> None:
    fixture = _fixture(tmp_path, flush_changes=changes)
    assert validate_study(fixture.study)["study_valid"] is False
    assert reason in _trial_validation(fixture)["reason_codes"]


def test_missing_producer_and_sequence_gap_are_detected(tmp_path: Path) -> None:
    missing = _fixture(
        tmp_path / "missing", expected_producers=("controller", "node_agent")
    )
    validate_study(missing.study)
    assert "producer_missing" in _trial_validation(missing)["reason_codes"]

    run_id = "run_validation_1"
    events = list(_valid_events(run_id))
    events[-1] = replace(events[-1], producer_sequence=10)
    gap = _fixture(tmp_path / "gap", events=tuple(events))
    validate_study(gap.study)
    assert "producer_sequence_gap" in _trial_validation(gap)["reason_codes"]

    reversed_events = list(_valid_events(run_id))
    reversed_events[1] = replace(reversed_events[1], producer_sequence=3)
    reversed_events[2] = replace(reversed_events[2], producer_sequence=2)
    reversal = _fixture(tmp_path / "reversal", events=tuple(reversed_events))
    validate_study(reversal.study)
    assert "producer_sequence_reversal" in _trial_validation(reversal)["reason_codes"]


def test_context_identity_path_and_manifest_intersection_are_strict(
    tmp_path: Path,
) -> None:
    wrong_context = _fixture(
        tmp_path / "context", context_changes={"build_revision": "wrong-build"}
    )
    validate_study(wrong_context.study)
    assert (
        "context_identity_mismatch" in _trial_validation(wrong_context)["reason_codes"]
    )

    relative = _fixture(tmp_path / "relative")
    trial_manifest = json.loads((relative.trial / "trial_manifest.json").read_text())
    flush_results = json.loads((relative.trial / "flush_results.json").read_text())
    replacement = relative.event_file.name
    old = str(relative.event_file.resolve())
    trial_manifest["committed_files"] = [
        replacement if item == old else item
        for item in trial_manifest["committed_files"]
    ]
    result = flush_results["results"][0]
    result["committed_files"] = [
        replacement if item == old else item for item in result["committed_files"]
    ]
    result["payload"]["committed_files"] = list(result["committed_files"])
    atomic_write_json(relative.trial / "trial_manifest.json", trial_manifest)
    atomic_write_json(relative.trial / "flush_results.json", flush_results)
    validate_study(relative.study)
    assert "committed_file_path_invalid" in _trial_validation(relative)["reason_codes"]

    intersection = _fixture(tmp_path / "intersection")
    manifest_payload = json.loads(
        (intersection.trial / "trial_manifest.json").read_text()
    )
    manifest_payload["committed_files"].remove(str(intersection.event_file.resolve()))
    atomic_write_json(intersection.trial / "trial_manifest.json", manifest_payload)
    validate_study(intersection.study)
    validation = _trial_validation(intersection)
    assert "committed_file_manifest_mismatch" in validation["reason_codes"]
    assert validation["index_counts"]["event"] == 0


def test_committed_temporary_file_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    temporary = fixture.event_file.with_name(f".{fixture.run_id}.pending.tmp")
    fixture.event_file.rename(temporary)
    old = str(fixture.event_file.resolve())
    replacement = str(temporary.resolve())
    trial_manifest = json.loads((fixture.trial / "trial_manifest.json").read_text())
    flush_results = json.loads((fixture.trial / "flush_results.json").read_text())
    trial_manifest["committed_files"] = [
        replacement if item == old else item
        for item in trial_manifest["committed_files"]
    ]
    result = flush_results["results"][0]
    result["committed_files"] = [
        replacement if item == old else item for item in result["committed_files"]
    ]
    result["payload"]["committed_files"] = list(result["committed_files"])
    atomic_write_json(fixture.trial / "trial_manifest.json", trial_manifest)
    atomic_write_json(fixture.trial / "flush_results.json", flush_results)
    validate_study(fixture.study)
    assert "committed_file_temporary" in _trial_validation(fixture)["reason_codes"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("task", "task_reference_dangling"),
        ("attempt", "task_attempt_reference_dangling"),
        ("dispatch", "dispatch_reference_dangling"),
        ("placement", "placement_lease_reference_dangling"),
        ("worker", "worker_lease_reference_dangling"),
        ("route", "route_lease_reference_dangling"),
        ("model", "model_instance_reference_dangling"),
    ],
)
def test_dangling_associations_are_reported(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    run_id = "run_validation_1"
    events = list(_valid_events(run_id))
    if mutation == "task":
        events[6] = replace(events[6], task_id="missing_task")
    elif mutation == "attempt":
        events[6] = replace(events[6], attempt=2)
    elif mutation == "dispatch":
        events[5] = replace(
            events[5],
            payload={"dispatch_id": "missing_dispatch", "worker_lease_id": "worker_1"},
        )
    elif mutation == "placement":
        events[6] = replace(events[6], lease_id="missing_placement")
    elif mutation == "worker":
        events[5] = replace(
            events[5],
            payload={"dispatch_id": "dispatch_1", "worker_lease_id": "missing_worker"},
        )
    elif mutation == "route":
        events[6] = replace(events[6], route_lease_id="missing_route")
    else:
        events[4] = replace(events[4], model_instance_id="missing_model")
    fixture = _fixture(tmp_path, events=tuple(events))
    validate_study(fixture.study)
    assert reason in _trial_validation(fixture)["reason_codes"]


def test_conflicting_dispatch_and_lease_identities_are_rejected(tmp_path: Path) -> None:
    run_id = "run_validation_1"
    events = list(_valid_events(run_id))
    events.append(
        _event(
            9,
            "task_dispatched",
            run_id=run_id,
            task_id="task_1",
            attempt=2,
            lease_id="placement_1",
            payload={"dispatch_id": "dispatch_1"},
        )
    )
    fixture = _fixture(tmp_path, events=tuple(events))
    validate_study(fixture.study)
    reasons = _trial_validation(fixture)["reason_codes"]
    assert "dispatch_reference_dangling" in reasons
    assert "placement_lease_reference_dangling" in reasons


def test_duplicate_events_terminal_conflicts_and_privacy_are_invalid(
    tmp_path: Path,
) -> None:
    run_id = "run_validation_1"
    duplicate = list(_valid_events(run_id))
    duplicate[1] = replace(duplicate[1], event_id=duplicate[0].event_id)
    duplicate_fixture = _fixture(tmp_path / "duplicate", events=tuple(duplicate))
    validate_study(duplicate_fixture.study)
    assert "event_id_duplicate" in _trial_validation(duplicate_fixture)["reason_codes"]

    conflict_events = (
        *_valid_events(run_id),
        replace(_valid_events(run_id)[-1], event_id="event_9", producer_sequence=9),
    )
    conflict = _fixture(tmp_path / "terminal", events=conflict_events)
    validate_study(conflict.study)
    assert "terminal_event_conflict" in _trial_validation(conflict)["reason_codes"]

    sensitive = list(_valid_events(run_id))
    sensitive[3] = replace(sensitive[3], payload={"prompt": "private text"})
    privacy = _fixture(tmp_path / "privacy", events=tuple(sensitive))
    validate_study(privacy.study)
    validation = _trial_validation(privacy)
    assert "privacy_violation" in validation["reason_codes"]
    serialized = json.dumps(validation, sort_keys=True)
    assert "private text" not in serialized


def test_terminal_and_interval_facts_are_not_inferred(tmp_path: Path) -> None:
    run_id = "run_validation_1"
    no_terminal = _fixture(tmp_path / "terminal", events=_valid_events(run_id)[:-1])
    validate_study(no_terminal.study)
    assert "terminal_event_missing" in _trial_validation(no_terminal)["reason_codes"]

    inverted_events = list(_valid_events(run_id))
    inverted_events[6] = replace(inverted_events[6], monotonic_time_ms=50)
    inverted = _fixture(tmp_path / "interval", events=tuple(inverted_events))
    validate_study(inverted.study)
    reasons = _trial_validation(inverted)["reason_codes"]
    assert "task_attempt_interval_inverted" in reasons
    assert "placement_lease_interval_inverted" in reasons

    open_attempt_events = tuple(
        event for event in _valid_events(run_id) if event.event_type != "task_succeeded"
    )
    open_attempt = _fixture(tmp_path / "open-attempt", events=open_attempt_events)
    validate_study(open_attempt.study)
    reasons = _trial_validation(open_attempt)["reason_codes"]
    assert "task_attempt_interval_open" in reasons
    assert "placement_lease_interval_open" in reasons

    open_route_events = tuple(
        event
        for event in _valid_events(run_id)
        if event.event_type != "model_route_released"
    )
    open_route = _fixture(tmp_path / "open-route", events=open_route_events)
    validate_study(open_route.study)
    assert "route_lease_interval_open" in _trial_validation(open_route)["reason_codes"]

    open_worker_events = tuple(
        event
        for event in _valid_events(run_id)
        if event.event_type != "worker_released"
    )
    open_worker = _fixture(tmp_path / "open-worker", events=open_worker_events)
    validate_study(open_worker.study)
    assert (
        "worker_lease_interval_open" in _trial_validation(open_worker)["reason_codes"]
    )


def test_missing_file_stays_missing_and_legal_slow_trial_remains_valid(
    tmp_path: Path,
) -> None:
    missing = _fixture(tmp_path / "missing")
    missing.event_file.unlink()
    first = validate_study(missing.study)
    validation = _trial_validation(missing)
    assert "committed_file_missing" in validation["reason_codes"]
    assert validation["index_counts"]["event"] == 0
    assert all(item["valid"] is False for item in validation["metric_valid"])
    assert all("value" not in item for item in validation["metric_valid"])
    first_raw = (missing.trial / "raw_files.json").read_bytes()
    first_validation = (missing.trial / "validity.json").read_bytes()
    assert validate_study(missing.study) == first
    assert (missing.trial / "raw_files.json").read_bytes() == first_raw
    assert (missing.trial / "validity.json").read_bytes() == first_validation
    assert "committed_file_hash_changed" not in validation["reason_codes"]

    run_id = "run_validation_1"
    slow = _fixture(
        tmp_path / "slow",
        events=_valid_events(run_id, terminal_time=7 * 24 * 60 * 60 * 1_000),
    )
    assert validate_study(slow.study)["study_valid"] is True


def test_arrival_lateness_only_invalidates_dct_and_throughput(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run_manifest_path = fixture.trial / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["runs"][0]["arrival_lateness_ms"] = 11
    atomic_write_json(run_manifest_path, run_manifest)

    assert validate_study(fixture.study)["study_valid"] is True
    validity = _trial_validation(fixture)
    assert validity["trial_valid"] is True
    metrics = {item["metric_name"]: item for item in validity["metric_valid"]}
    assert metrics["dct_ms"] == {
        "metric_name": "dct_ms",
        "valid": False,
        "reason_codes": ["arrival_lateness_exceeded"],
    }
    assert metrics["throughput_success_per_s"]["valid"] is False
