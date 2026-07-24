from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from ascend_maze.contracts.resources import ReservationVector
from ascend_maze.placement import NodeCapacity, NpuCapacity, PlacementManager


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "logical_cluster_performance.py"
PREPARE_PATH = ROOT / "deploy" / "logical_cluster" / "prepare_control_plane.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


performance = _load(TOOL_PATH, "logical_cluster_performance_test")
prepare = _load(PREPARE_PATH, "logical_cluster_prepare_test")


def test_arrival_ratio_builds_fixed_window_offsets() -> None:
    assert performance._arrival_offsets_ms(  # noqa: SLF001
        arrival_ratio=0.25,
        average_workflow_seconds=30.0,
        admission_window_seconds=130.0,
    ) == (0, 120_000)


def test_cases_and_paired_order_alternate_first_executor() -> None:
    args = argparse.Namespace(
        mode=["batch", "arrival"],
        batch_size=[1, 2],
        arrival_ratio=[0.25],
        average_workflow_seconds=30.0,
        arrival_window_seconds=130.0,
    )
    cases = performance._build_cases(args)  # noqa: SLF001
    assert [item.case_id for item in cases] == [
        "batch-1",
        "batch-2",
        "arrival-ratio-0p25",
    ]
    order = performance._execution_order(cases, "paired")  # noqa: SLF001
    assert [(item[0].case_id, item[1]) for item in order] == [
        ("batch-1", "maze"),
        ("batch-1", "ray"),
        ("batch-2", "ray"),
        ("batch-2", "maze"),
        ("arrival-ratio-0p25", "maze"),
        ("arrival-ratio-0p25", "ray"),
    ]


def test_fixed_mixed_batch20_manifest_is_deterministic_and_covers_all_workflows() -> None:
    samples = []
    query_index = 0
    for dataset, workflow in performance.qwen_smoke.WORKFLOW_MODULES:
        family = (
            "vision"
            if (dataset, workflow) in performance.qwen_smoke.VISION_WORKFLOWS
            else "text"
        )
        for offset in range(2):
            samples.append(
                performance.qwen_smoke.SampleSpec(
                    dataset=dataset,
                    workflow=workflow,
                    family=family,
                    dag_id=f"{dataset}-{workflow}-{offset}",
                    query_index=query_index,
                    inputs={},
                    source_files=(),
                    expected_answer="",
                    vision_mode="true_multimodal" if family == "vision" else None,
                )
            )
            query_index += 1

    manifest = performance._build_fixed_batch20_manifest(samples)  # noqa: SLF001
    entries = manifest["entries"]

    assert manifest["request_count"] == 20
    assert len(entries) == 20
    assert len({item["sample_id"] for item in entries}) == 20
    assert {(item["dataset"], item["workflow"]) for item in entries} == set(
        performance.qwen_smoke.WORKFLOW_MODULES
    )
    assert [item["selection_reason"] for item in entries].count(
        "first_sample_per_workflow"
    ) == 14
    assert [item["selection_reason"] for item in entries].count(
        "second_sample_visual_workflow"
    ) == 3
    assert [item["selection_reason"] for item in entries].count(
        "second_sample_dataset_text_representative"
    ) == 3
    assert sum(item["family"] == "vision" for item in entries) == 6
    assert sum(item["family"] == "text" for item in entries) == 14
    assert all(
        item["target_model_id"]
        == (
            performance.VISION_MODEL_ID
            if item["family"] == "vision"
            else performance.TEXT_MODEL_ID
        )
        for item in entries
    )
def test_request_aggregate_uses_client_e2e_and_makespan() -> None:
    aggregate = performance._aggregate_requests(  # noqa: SLF001
        [
            {
                "status": "succeeded",
                "client_e2e_ms": 100,
                "client_e2e_started_at_ms": 1_000,
                "client_e2e_finished_at_ms": 1_100,
            },
            {
                "status": "succeeded",
                "client_e2e_ms": 200,
                "client_e2e_started_at_ms": 1_000,
                "client_e2e_finished_at_ms": 1_200,
            },
        ],
        mode="batch",
        admission_window_seconds=None,
    )
    assert aggregate["p95_e2e_ms"] == 195.0
    assert aggregate["makespan_ms"] == 200
    assert aggregate["throughput_requests_per_second"] == 10.0


def test_resource_aggregate_keeps_cluster_and_per_device_metrics() -> None:
    samples = [
        {
            "timestamp_ms": 900,
            "cluster_cpu_utilization_pct": 1.0,
            "cluster_npu_utilization_pct": 0.0,
            "max_device_npu_utilization_pct": 0.0,
            "cluster_hbm_used_mb": 100,
            "containers": [],
            "npus": [],
            "errors": [],
        },
        {
            "timestamp_ms": 1_100,
            "cluster_cpu_utilization_pct": 20.0,
            "cluster_npu_utilization_pct": 25.0,
            "max_device_npu_utilization_pct": 80.0,
            "cluster_hbm_used_mb": 1_100,
            "containers": [
                {"node_id": "0", "cpu_utilization_pct": 20.0},
            ],
            "npus": [
                {
                    "physical_device_id": "0",
                    "utilization_pct": 80.0,
                    "used_hbm_mb": 1_000,
                    "processes": [{"pid": 1}, {"pid": 2}],
                }
            ],
            "errors": [],
        },
    ]
    aggregate = performance._aggregate_resources(  # noqa: SLF001
        samples,
        started_at_ms=1_000,
        finished_at_ms=1_200,
    )
    assert aggregate["sample_count"] == 1
    assert aggregate["peak_incremental_hbm_mb"] == 1_000
    assert aggregate["cluster_cpu_utilization_pct"]["mean"] == 20.0
    assert aggregate["incremental_cluster_cpu_utilization_pct"] == 19.0
    assert aggregate["per_device"]["0"]["utilization_pct"]["max"] == 80.0
    assert aggregate["per_device"]["0"]["npu_process_count"]["max"] == 2.0


def test_breakdowns_group_family_workflow_and_task_timings() -> None:
    records = [
        {
            "status": "succeeded",
            "dataset": "gaia",
            "workflow": "vision",
            "family": "vision",
            "client_e2e_ms": 200,
            "client_e2e_started_at_ms": 1_000,
            "client_e2e_finished_at_ms": 1_200,
            "transformers_local_records": [
                {"model_load_ms": 50, "generate_ms": 75}
            ],
            "task_timings": [{"worker_startup_ms": 10, "output_put_ms": 5}],
            "dispatch_lifecycle": [{"queue_to_dispatch_ms": 20}],
        },
        {
            "status": "succeeded",
            "dataset": "tbench",
            "workflow": "retail_cancel",
            "family": "text",
            "client_e2e_ms": 100,
            "client_e2e_started_at_ms": 1_000,
            "client_e2e_finished_at_ms": 1_100,
            "transformers_local_records": [
                {"model_load_ms": 40, "generate_ms": 60}
            ],
        },
    ]

    aggregate = performance._aggregate_breakdowns(  # noqa: SLF001
        records,
        mode="batch",
        admission_window_seconds=None,
    )

    assert aggregate["overall"]["requests"]["p95_e2e_ms"] == 195.0
    assert aggregate["families"]["vision"]["requests"]["request_count"] == 1
    assert (
        aggregate["workflows"]["tbench.retail_cancel"]["requests"]["succeeded"]
        == 1
    )
    assert aggregate["overall"]["timings"]["model_load_ms"]["mean"] == 45.0
    assert aggregate["families"]["vision"]["timings"]["queue_to_dispatch_ms"][
        "mean"
    ] == 20.0


def test_performance_profile_enables_scheduler_pool_and_replicas() -> None:
    controller = prepare._controller_config("performance")  # noqa: SLF001
    catalog = prepare._model_catalog("performance")  # noqa: SLF001
    assert 'profile = "performance"' in controller
    assert "controller-transformers-performance-v3.sqlite3" in controller
    assert 'policy = "hacs_no_tp"' in controller
    assert 'anchor_strategy = "static"' in controller
    assert "standby_min_idle = 1" in controller
    assert "max_tasks_per_worker = 1" in controller
    assert catalog.count("max_replicas = 8") == 2
    assert catalog.count("max_parallel_starts = 8") == 2
    assert catalog.count("scale_cooldown_ms = 0") == 2
    assert catalog.count("scale_down_idle_ms = 0") == 2
    assert 'catalog_revision = "logical-performance-v3-' in catalog


def _model_reservation(model: dict[str, object]) -> ReservationVector:
    return ReservationVector(
        cpu_num=int(model["instance_cpu_num"]),
        host_mem_mb=int(model["instance_host_mem_mb"]),
        io_slots=0,
        npu_hbm_mb=int(model["instance_hbm_mb"]),
        npu_slots=int(model["npu_slots"]),
    )


def _one_npu_node(*, total_hbm_mb: int) -> NodeCapacity:
    return NodeCapacity(
        node_id="node-0",
        boot_id="boot-0",
        node_ip="127.0.0.1",
        cpu_total=20,
        mem_total_mb=131_072,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=8,
        npus=(
            NpuCapacity(
                device_id="0",
                chip_type="910B3",
                total_hbm_mb=total_hbm_mb,
                system_reserved_hbm_mb=4_096,
                task_slots_total=2,
                observed_free_hbm_mb=total_hbm_mb - 3_210,
            ),
        ),
        observed_free_mem_mb=131_072,
    )


def _reserve_model(
    placement: PlacementManager,
    model: dict[str, object],
    instance_id: str,
):
    return placement.reserve_model_instance(
        instance_id=instance_id,
        generation=1,
        resources=_model_reservation(model),
        allow_colocation=bool(model["allow_colocation"]),
        now_ms=1,
        startup_deadline_ms=10_000,
    )


def test_calibrated_text_and_vision_instances_share_only_when_hbm_fits() -> None:
    document = tomllib.loads(prepare._model_catalog("performance"))  # noqa: SLF001
    models = {item["model_id"]: item for item in document["models"]}
    text = models["qwen3-4b-e2e"]
    vision = models["qwen2_5-vl-3b-e2e"]

    assert (
        text["weight_hbm_mb"],
        text["runtime_hbm_mb"],
        text["kv_cache_hbm_mb"],
        text["instance_hbm_mb"],
        text["allow_colocation"],
    ) == (8_192, 4_096, 1_536, 13_824, True)
    assert (
        vision["weight_hbm_mb"],
        vision["runtime_hbm_mb"],
        vision["kv_cache_hbm_mb"],
        vision["instance_hbm_mb"],
        vision["allow_colocation"],
    ) == (8_192, 3_072, 512, 11_776, True)

    fitting = PlacementManager(npu_hbm_headroom_mb=1_024)
    fitting.register_node(_one_npu_node(total_hbm_mb=65_536))
    first = _reserve_model(fitting, text, "text-1")
    second = _reserve_model(fitting, vision, "vision-1")
    assert first.selected and second.selected
    assert first.lease is not None and second.lease is not None
    assert first.lease.npu_device_id == second.lease.npu_device_id == "0"
    assert fitting.snapshot().nodes[0].per_npu_reserved == (
        ("0", 13_824 + 11_776, 2),
    )

    insufficient = PlacementManager(npu_hbm_headroom_mb=1_024)
    insufficient.register_node(_one_npu_node(total_hbm_mb=29_500))
    assert _reserve_model(insufficient, text, "text-2").selected
    blocked = _reserve_model(insufficient, vision, "vision-2")
    assert not blocked.selected
    assert blocked.rejection_reason == "insufficient_npu_hbm"


def test_dispatch_lifecycle_records_starting_to_running_timeline() -> None:
    lifecycle = performance._dispatch_lifecycle(  # noqa: SLF001
        [
            {
                "events": [
                    {
                        "event_type": "task_queued",
                        "task_id": "task_1",
                        "attempt": None,
                        "sequence": 9,
                        "monotonic_time_ms": 990,
                        "payload": {},
                    },
                    {
                        "event_type": "task_dispatched",
                        "task_id": "task_1",
                        "attempt": 1,
                        "sequence": 10,
                        "monotonic_time_ms": 1_000,
                        "payload": {
                            "dispatch_id": "dispatch_1",
                            "node_id": "node-1",
                        },
                    },
                    {
                        "event_type": "dispatch_prepared",
                        "task_id": "task_1",
                        "attempt": 1,
                        "sequence": 14,
                        "monotonic_time_ms": 1_650,
                        "payload": {"dispatch_prepare_ms": 649.5},
                    },
                    {
                        "event_type": "worker_started",
                        "task_id": "task_1",
                        "attempt": 1,
                        "sequence": 15,
                        "monotonic_time_ms": 1_680,
                        "payload": {"node_id": "node-1", "worker_pid": 42},
                    },
                ]
            }
        ],
        {"task_1": "load_model"},
    )
    assert lifecycle == [
        {
            "task_id": "task_1",
            "task_name": "load_model",
            "attempt": 1,
            "dispatch_id": "dispatch_1",
            "node_id": "node-1",
            "worker_pid": 42,
            "task_queued_sequence": 9,
            "task_dispatched_sequence": 10,
            "dispatch_prepared_sequence": 14,
            "running_sequence": 15,
            "task_queued_at_ms": 990,
            "task_dispatched_at_ms": 1_000,
            "dispatch_prepared_at_ms": 1_650,
            "running_at_ms": 1_680,
            "dispatch_prepare_ms": 649.5,
            "queue_to_dispatch_ms": 10,
            "dispatch_to_prepared_ms": 650,
            "prepared_to_running_ms": 30,
            "dispatch_to_running_ms": 680,
        }
    ]


def test_wait_maze_terminal_returns_runtime_task_timings() -> None:
    class Client:
        async def watch_run(self, run_id: str, **kwargs: object):
            del run_id, kwargs
            yield {"events": [{"event_type": "run_succeeded"}]}

        async def query(self, operation: str, **kwargs: object):
            del operation, kwargs
            return {
                "run": {"run_id": "run_1", "status": "succeeded"},
                "runtime_task_timings": [
                    {
                        "task_id": "task_1",
                        "worker_startup_ms": 12,
                        "inference_metrics": [
                            {"model_load_ms": 4, "generate_ms": 7}
                        ],
                    }
                ],
            }

    run, batches, timings = asyncio.run(
        performance._wait_maze_terminal(Client(), "run_1", 10.0)  # noqa: SLF001
    )
    assert run["status"] == "succeeded"
    assert len(batches) == 1
    assert timings[0]["worker_startup_ms"] == 12
    assert timings[0]["inference_metrics"][0]["model_load_ms"] == 4


def test_maze_request_hard_timeout_returns_a_failed_record() -> None:
    original = performance._run_maze_request  # noqa: SLF001

    async def never_finishes(**kwargs: object) -> dict[str, object]:
        del kwargs
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    sample = argparse.Namespace(
        sample_id="gaia-text-0",
        dataset="gaia",
        workflow="text",
        family="text",
    )
    performance._run_maze_request = never_finishes  # type: ignore[assignment]  # noqa: SLF001
    try:
        record = asyncio.run(
            performance._run_maze_request_bounded(  # noqa: SLF001
                client=object(),
                workflow=object(),
                compiled=object(),
                task_names={},
                sample=sample,
                target_model_id=performance.TEXT_MODEL_ID,
                case_id="batch-20",
                request_index=1,
                timeout_seconds=10.0,
                hard_timeout_seconds=0.01,
            )
        )
    finally:
        performance._run_maze_request = original  # type: ignore[assignment]  # noqa: SLF001

    assert record["status"] == "failed"
    assert record["sample_id"] == "gaia-text-0"
    assert record["hard_timeout_seconds"] == 0.01
    assert "hard deadline" in str(record["error"])


def test_report_states_pilot_boundaries() -> None:
    report = performance._render_report(  # noqa: SLF001
        {
            "results": [
                {
                    "executor": "maze",
                    "case": {"case_id": "batch-1"},
                    "aggregate": {
                        "succeeded": 1,
                        "request_count": 1,
                        "p95_e2e_ms": 123.0,
                        "throughput_requests_per_second": 1.0,
                    },
                    "resources": {},
                }
            ]
        }
    )
    assert "batch-1" in report
    assert "P95 和吞吐量用于 Pilot 对比" in report
    assert "不代表真实跨机网络性能" in report


def _resume_fixture(tmp_path: Path) -> tuple[Path, argparse.Namespace]:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    text_model = tmp_path / "text-model"
    vision_model = tmp_path / "vision-model"
    text_model.mkdir()
    vision_model.mkdir()
    case = performance.WorkloadCase(
        case_id="batch-20",
        mode="batch",
        request_count=20,
        launch_offsets_ms=(0,) * 20,
        batch_size=20,
    )
    manifest = {
        "schema_version": 1,
        "request_count": 20,
        "entries": [],
    }
    control = {
        "profile": "performance",
        "controller_config_sha256": "controller-sha",
        "model_catalog_sha256": "catalog-sha",
    }
    plan = {
        "schema_version": performance.SCHEMA_VERSION,
        "objective": performance.OBJECTIVE,
        "executor": "paired",
        "cases": [case.payload()],
        "execution_order": [
            {
                "ordinal": 1,
                "case_id": "batch-20",
                "executor": "maze",
                "pair_position": 1,
            },
            {
                "ordinal": 2,
                "case_id": "batch-20",
                "executor": "ray",
                "pair_position": 2,
            },
        ],
        "contract": {
            "workload_manifest": manifest,
            "text_model_path": str(text_model),
            "vision_model_path": str(vision_model),
            "request_timeout_seconds": 3600.0,
            "case_timeout_seconds": 4500.0,
            "ray_task_num_cpus": 20.0,
        },
        "control_environment": control,
    }
    (output_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (output_dir / "workload_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    args = argparse.Namespace(
        state_root=tmp_path,
        executor="paired",
        workload_manifest=None,
        text_model_path=Path("unused-text"),
        vision_model_path=Path("unused-vision"),
        request_timeout_seconds=1.0,
        case_timeout_seconds=1.0,
        ray_task_num_cpus=1.0,
    )
    return output_dir, args


def test_resume_loads_frozen_plan_and_rejects_manifest_drift(tmp_path: Path) -> None:
    output_dir, args = _resume_fixture(tmp_path)
    original = performance._control_environment  # noqa: SLF001
    performance._control_environment = lambda _state_root: {  # type: ignore[assignment]  # noqa: SLF001,E501
        "profile": "performance",
        "controller_config_sha256": "controller-sha",
        "model_catalog_sha256": "catalog-sha",
    }
    try:
        plan, cases, order = performance._load_resume_state(  # noqa: SLF001
            args, output_dir
        )
        assert plan["executor"] == "paired"
        assert [item.case_id for item in cases] == ["batch-20"]
        assert [item[1] for item in order] == ["maze", "ray"]
        assert args.request_timeout_seconds == 3600.0
        assert args.case_timeout_seconds == 4500.0
        assert args.workload_manifest == output_dir / "workload_manifest.json"

        (output_dir / "workload_manifest.json").write_text(
            json.dumps({"request_count": 19}), encoding="utf-8"
        )
        with pytest.raises(
            performance.PerformancePilotError,
            match="manifest differs",
        ):
            performance._load_resume_state(args, output_dir)  # noqa: SLF001
    finally:
        performance._control_environment = original  # type: ignore[assignment]  # noqa: SLF001


def test_resume_reuses_only_complete_resource_evidence_and_archives_partial(
    tmp_path: Path,
) -> None:
    output_dir, _args = _resume_fixture(tmp_path)
    case = performance.WorkloadCase(
        case_id="batch-20",
        mode="batch",
        request_count=20,
        launch_offsets_ms=(0,) * 20,
        batch_size=20,
    )
    maze_dir = output_dir / "cases" / "batch-20" / "maze"
    maze_dir.mkdir(parents=True)
    resources = maze_dir / "resource_samples.jsonl"
    resources.write_text("{}\n", encoding="utf-8")
    result = {
        "executor": "maze",
        "case": case.payload(),
        "process": {"exit_code": 0},
        "aggregate": {"failed": 0},
        "resources": {"sample_count": 1},
        "resource_samples_path": str(resources),
        "physical_hbm_recovery": {"recovered": True},
        "control_recovery": {"recovered": True},
    }
    (maze_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    reused = performance._completed_case_result(  # noqa: SLF001
        output_dir, case, "maze"
    )
    assert reused is not None

    result["resources"] = {"sample_count": 0}
    (maze_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert (
        performance._completed_case_result(output_dir, case, "maze")  # noqa: SLF001
        is None
    )

    ray_dir = output_dir / "cases" / "batch-20" / "ray"
    ray_dir.mkdir(parents=True)
    (ray_dir / "runner.json").write_text('{"succeeded": 20}', encoding="utf-8")
    (ray_dir / "stdout.log").write_text("20/20\n", encoding="utf-8")
    archive = performance._archive_incomplete_case(  # noqa: SLF001
        output_dir, case, "ray"
    )
    assert archive is not None
    assert (archive / "runner.json").is_file()
    assert (archive / "stdout.log").is_file()
    assert not (ray_dir / "runner.json").exists()


def test_resume_cli_requires_an_existing_output_directory_argument() -> None:
    with pytest.raises(SystemExit):
        performance.parse_args(["--resume"])
