from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from ascend_maze.benchmark import calibration, orchestrator
from ascend_maze.benchmark.admission import (
    HOST_AUDIT_SCHEMA,
    _verify_frozen_model_manifest,
    host_recovery_issues,
    model_artifact_manifest,
)
from ascend_maze.benchmark.aggregation import _apply_guard_decisions
from ascend_maze.benchmark.c13_runtime import C13BenchmarkRuntimeFactory
from ascend_maze.benchmark.loader import load_study_plan
from ascend_maze.benchmark.indexes import metric_validity
from ascend_maze.benchmark.metrics import (
    CORRECTNESS_GUARD_METRICS,
    RunFact,
    extract_metric,
)
from ascend_maze.benchmark.microbenchmarks import (
    MeasuredMicrobenchmarkRuntime,
    _base_config,
    _measure_c7,
    _microbenchmark_spec,
    prepare_microbenchmark_specs,
)
from ascend_maze.benchmark.persistence import atomic_write_json
from ascend_maze.benchmark.workloads.component import build as build_component
from ascend_maze.benchmark.workloads.qwen3_4b import build as build_qwen
from ascend_maze.config import load_config
from ascend_maze.compiler.analyzer import analyse_callable
from ascend_maze.contracts.recording import ExecutionEvent
from ascend_maze.control.application import _worker_pool_config
from ascend_maze.core.errors import ContractValidationError


ROOT = Path(__file__).resolve().parents[1]
BUILD_REVISION = "a" * 40


def _portable_candidate(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model"
    model.mkdir()
    catalog = (ROOT / "experiments/c14e/model_catalog.toml").read_text(
        encoding="utf-8"
    )
    catalog = catalog.replace(
        "../../../model_weight/model_from_hf/Qwen3-4B",
        "model",
    )
    (root / "model_catalog.toml").write_text(catalog, encoding="utf-8")
    config = (ROOT / "experiments/c14e/performance.candidate.toml").read_text(
        encoding="utf-8"
    )
    path = root / "performance.toml"
    path.write_text(config, encoding="utf-8")
    return path


def _component_spec(root: Path, suite: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "performance.toml"
    config.write_text(_base_config(root), encoding="ascii")
    fingerprint = build_component().compile().workflow_fingerprint
    dataset = root / "dataset.json"
    atomic_write_json(
        dataset,
        {
            "schema_version": 1,
            "workflow_fingerprint": fingerprint,
            "records": [
                {"record_id": f"record-{index:03d}", "inputs": {"value": index}}
                for index in range(100)
            ],
        },
    )
    spec = root / f"{suite}.toml"
    spec.write_text(
        _microbenchmark_spec(
            suite=suite,
            build_revision=BUILD_REVISION,
            config=config,
            dataset=dataset,
            workflow_fingerprint=fingerprint,
        ),
        encoding="ascii",
    )
    return spec


def test_qwen_workload_dataset_and_candidate_profile_are_consistent(
    tmp_path: Path,
) -> None:
    fingerprint = build_qwen().compile().workflow_fingerprint
    dataset = json.loads(
        (ROOT / "experiments/c14e/qwen3_4b_dataset.json").read_text(
            encoding="utf-8"
        )
    )
    assert fingerprint == "ad0db2b49b0d4833dfa5cf794425f23f4f0991a368152bd4a768aad7691185a9"
    assert dataset["workflow_fingerprint"] == fingerprint
    assert len(dataset["records"]) == 32
    assert len({item["record_id"] for item in dataset["records"]}) == 32

    loaded = load_config(
        _portable_candidate(tmp_path),
        build_revision=BUILD_REVISION,
        created_at_ms=0,
    )
    assert loaded.config.profile == "performance"
    assert loaded.snapshot.model_catalog_revision == "qwen3-4b-hf-6dc0981b8829"
    assert loaded.config.placement.allow_colocation is True
    assert loaded.config.placement.task_slots_total == 2
    assert loaded.config.worker.standby_min_idle == 2
    assert loaded.config.worker.standby_max_idle == 2


def test_task_code_hash_uses_version_neutral_ast_payload() -> None:
    from ascend_maze.benchmark.workloads.qwen3_4b import normalize_prompt

    analysis = analyse_callable(normalize_prompt)
    assert "kwonlyargs" in analysis.normalized_ast
    assert "type_params" not in analysis.normalized_ast


def test_worker_pool_translation_preserves_standby_ablation(tmp_path: Path) -> None:
    loaded = load_config(
        _portable_candidate(tmp_path),
        build_revision=BUILD_REVISION,
        created_at_ms=0,
    )
    standby = _worker_pool_config(loaded.config)
    assert standby.mode == "zero_hbm_standby"
    assert {profile.profile.value for profile in standby.profiles} == {
        "cpu",
        "io",
        "npu_host",
    }
    assert all(profile.min_idle == 2 for profile in standby.profiles)
    assert all(profile.max_idle == 2 for profile in standby.profiles)

    cold_config = replace(
        loaded.config,
        worker=replace(
            loaded.config.worker,
            standby_min_idle=0,
            standby_max_idle=0,
        ),
    )
    cold = _worker_pool_config(cold_config)
    assert cold.mode == "cold_start"
    assert all(profile.min_idle == 0 for profile in cold.profiles)
    assert all(profile.max_idle == 0 for profile in cold.profiles)


def test_model_manifest_hashes_complete_file_set(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    index = model / "model.safetensors.index.json"
    index.write_bytes(b'{"weight_map":{}}\n')
    (model / "config.json").write_bytes(b'{"model_type":"qwen3"}\n')
    (model / "weights.safetensors").write_bytes(b"weights-v1")

    first = model_artifact_manifest(model)
    second = model_artifact_manifest(model)
    assert first == second
    assert first["file_count"] == 3
    assert first["artifact_revision"] == hashlib.sha256(index.read_bytes()).hexdigest()

    (model / "weights.safetensors").write_bytes(b"weights-v2")
    with pytest.raises(ContractValidationError, match="artifact .* changed"):
        _verify_frozen_model_manifest(model, first)
    changed = model_artifact_manifest(model)
    assert changed["content_digest"] != first["content_digest"]

    (model / "linked").symlink_to(model / "config.json")
    with pytest.raises(ContractValidationError, match="symbolic link"):
        model_artifact_manifest(model)


def test_host_recovery_reports_every_resource_dimension() -> None:
    before = {
        "schema": HOST_AUDIT_SCHEMA,
        "devices": [
            {
                "physical_device_id": "0",
                "used_hbm_mb": 100,
                "health": "healthy",
                "processes": [],
            }
        ],
    }
    after = {
        "schema": HOST_AUDIT_SCHEMA,
        "relevant_processes": [{"pid": 10}],
        "relevant_listeners": [{"pid": 10, "local_address": "127.0.0.1:1"}],
        "devices": [
            {
                "physical_device_id": "0",
                "used_hbm_mb": 165,
                "health": "unhealthy",
                "processes": [{"pid": 11}],
            }
        ],
    }
    assert host_recovery_issues(before, after, hbm_tolerance_mb=64) == (
        "related_process_residual",
        "related_listener_residual",
        "npu_unhealthy:0",
        "npu_process_residual:0",
        "npu_hbm_not_recovered:0",
    )
    assert host_recovery_issues(before, {}, hbm_tolerance_mb=64) == (
        "host_audit_invalid",
    )


def test_prepare_14e_is_deterministic_and_pilot_is_nonformal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    config = _portable_candidate(repository)
    dataset = repository / "experiments/c14e/qwen3_4b_dataset.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(
        (ROOT / "experiments/c14e/qwen3_4b_dataset.json").read_bytes()
    )
    monkeypatch.setattr(calibration, "_repository_root", lambda _: repository)
    monkeypatch.setattr(
        calibration,
        "_clean_build_revision",
        lambda _: BUILD_REVISION,
    )

    first = calibration.prepare_c14e_specs(
        base_config=config,
        output_directory=tmp_path / "first",
        study_kind="pilot",
    )
    second = calibration.prepare_c14e_specs(
        base_config=config,
        output_directory=tmp_path / "second",
        study_kind="pilot",
    )
    assert len(first) == 3
    assert [path.read_bytes() for path in first] == [
        path.read_bytes() for path in second
    ]
    assert calibration.spec_bundle_digest(first) == calibration.spec_bundle_digest(
        second
    )
    plan = load_study_plan(first[0])
    assert plan.spec.study_kind == "pilot"
    assert plan.spec.block_count == 3
    assert plan.spec.windows.measurement_duration_ms == 20_000


def test_benchmark_only_override_changes_cell_identity_not_config(
    tmp_path: Path,
) -> None:
    plan = load_study_plan(_component_spec(tmp_path, "c12"))
    reference = next(cell for cell in plan.cells if cell.name == "no_fault_reference")
    bookkeeping = next(cell for cell in plan.cells if cell.name == "fault_bookkeeping")
    assert reference.cell_id != bookkeeping.cell_id
    assert (
        reference.config_snapshot.config_fingerprint
        == bookkeeping.config_snapshot.config_fingerprint
    )
    assert [difference.path for difference in bookkeeping.differences] == [
        "benchmark.c12_bookkeeping"
    ]

    factory = C13BenchmarkRuntimeFactory(maze_command=("maze",))

    async def rejected() -> None:
        with pytest.raises(
            ContractValidationError,
            match="benchmark-only overrides",
        ):
            await factory.open(
                spec=plan.spec,
                cell=bookkeeping,
                trial_attempt_id="trial_attempt_test",
                trial_directory=str(tmp_path / "trial"),
                resume=False,
            )

    asyncio.run(rejected())


def test_measured_runtime_submission_replay_uses_full_submission_id(
    tmp_path: Path,
) -> None:
    plan = load_study_plan(_component_spec(tmp_path / "spec", "c12"))
    cell = next(item for item in plan.cells if item.name == "no_fault_reference")
    trial = tmp_path / "trial"
    trial.mkdir()
    runtime = MeasuredMicrobenchmarkRuntime(
        suite="c12",
        spec=plan.spec,
        cell=cell,
        trial_attempt_id="trial_attempt_test",
        trial_directory=trial,
    )

    async def scenario() -> None:
        first = await runtime.submit(
            None,
            inputs={},
            submission_id="submission-a-same-tail",
            run_deadline_ms=None,
        )
        replay = await runtime.submit(
            None,
            inputs={},
            submission_id="submission-a-same-tail",
            run_deadline_ms=None,
        )
        distinct = await runtime.submit(
            None,
            inputs={},
            submission_id="submission-b-same-tail",
            run_deadline_ms=None,
        )
        assert replay.replayed is True
        assert replay.run_id == first.run_id
        assert distinct.run_id != first.run_id
        assert len(runtime.runs) == 2
        await runtime.shutdown(request_id="shutdown")

    asyncio.run(scenario())


def test_direct_microbenchmark_samples_and_order_guard() -> None:
    event = ExecutionEvent(
        schema_version=1,
        event_id="event-1",
        experiment_id="run-1",
        run_id="run-1",
        task_id="task-1",
        attempt=1,
        lease_id="lease-1",
        route_lease_id=None,
        model_instance_id=None,
        event_type="microbenchmark_sample",
        producer_id="controller",
        producer_sequence=1,
        node_id=None,
        device_id=None,
        monotonic_time_ms=1,
        wall_time_ms=1,
        duration_ms=None,
        payload={"metric_name": "dct_ms", "value": 3.5},
    )
    run = RunFact("run-1", "measurement", 0, 0, 0, 10, "succeeded", 0, 0, 0)
    extraction = extract_metric(
        "dct_ms",
        events=(event,),
        runs=(run,),
        measurement_duration_ms=1,
        recording_complete=True,
    )
    assert [sample.value for sample in extraction.samples] == [3.5]
    assert metric_validity(
        ("dct_ms",),
        run_ids=("run-1",),
        events=(event,),
        trial_integrity_valid=True,
    )[0].valid is True

    rows: list[dict[str, object]] = []
    for metric_name in sorted(CORRECTNESS_GUARD_METRICS):
        rows.append(_comparison_row(metric_name))
    order = _comparison_row("scheduling_order_match")
    rows.append(order)
    performance = _comparison_row("dct_ms")
    rows.append(performance)
    _apply_guard_decisions(rows)
    assert performance["guard_decision"] == "pass"
    assert performance["decision"] == "pass"

    order["relative_effect_pct"] = -100.0
    order["ci95_lower"] = -100.0
    _apply_guard_decisions(rows)
    assert performance["guard_decision"] == "fail"
    assert performance["decision"] == "fail"


def _comparison_row(metric_name: str) -> dict[str, object]:
    is_positive_guard = metric_name in {"scheduling_order_match", "success_rate"}
    return {
        "baseline_cell_id": "baseline",
        "candidate_cell_id": "candidate",
        "metric_name": metric_name,
        "paired_blocks": 10,
        "relative_effect_pct": 0.0 if is_positive_guard else None,
        "absolute_effect": 0.0,
        "ci95_lower": 0.0 if is_positive_guard else None,
        "guard_decision": "pending",
        "guard_reasons": [],
        "decision": "pass",
    }


def test_missing_trial_analysis_history_is_rebuilt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = "trial_attempt_123"
    completed = {"trial": {"trial_attempt_id": attempt}}
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator,
        "_run_analysis_pipeline",
        lambda _root, trial_attempt_id: calls.append(trial_attempt_id),
    )
    orchestrator._ensure_analysis_history(tmp_path, completed)
    assert calls == [attempt]

    atomic_write_json(
        tmp_path / "analysis_history" / f"{attempt}.json",
        {
            "schema_version": 1,
            "trial_attempt_id": attempt,
            "pipeline": ["validate", "aggregate", "report"],
            "validation_digest": "v",
            "aggregate_manifest_digest": "a",
            "report_digest": "r",
        },
    )
    calls.clear()
    orchestrator._ensure_analysis_history(tmp_path, completed)
    assert calls == []


def test_c7_microbenchmark_keeps_full_queue_and_cleans_placement_history() -> None:
    measured = _measure_c7()
    assert len(measured["scheduler_policy_select_ms"]) == 10_000
    assert len(measured["scheduler_total_ms"]) == 10_000
    assert all(value >= 0 for values in measured.values() for value in values)


def test_microbenchmark_preparation_accepts_module_file_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ascend_maze.benchmark.microbenchmarks as microbenchmarks

    monkeypatch.setattr(
        microbenchmarks,
        "_clean_build_revision",
        lambda _: BUILD_REVISION,
    )
    bundle = prepare_microbenchmark_specs(tmp_path)
    assert set(bundle.spec_paths) == {"c7", "c8", "c12", "c13"}
    assert all(path.is_file() for path in bundle.spec_paths.values())
    for path in bundle.spec_paths.values():
        load_study_plan(path)
