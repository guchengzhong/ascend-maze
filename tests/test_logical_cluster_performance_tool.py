from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


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
    assert aggregate["per_device"]["0"]["utilization_pct"]["max"] == 80.0


def test_performance_profile_enables_scheduler_pool_and_replicas() -> None:
    controller = prepare._controller_config("performance")  # noqa: SLF001
    catalog = prepare._model_catalog("performance")  # noqa: SLF001
    assert 'profile = "performance"' in controller
    assert 'policy = "hacs_no_tp"' in controller
    assert 'anchor_strategy = "static"' in controller
    assert "standby_min_idle = 1" in controller
    assert "max_tasks_per_worker = 1" in controller
    assert catalog.count("max_replicas = 8") == 2
    assert catalog.count("max_parallel_starts = 8") == 2


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
    assert "P95 在单请求或双请求 Pilot" in report
    assert "不代表真实跨机网络性能" in report
