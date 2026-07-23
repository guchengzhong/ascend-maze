from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "ray_baseline_performance.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("ray_baseline_performance", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ray_performance_builds_fixed_workload_plan() -> None:
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--dataset",
            "tbench",
            "--workflow",
            "retail_cancel",
            "--family",
            "text",
            "--samples-per-workflow",
            "1",
            "--warmup-iterations",
            "1",
            "--measurement-iterations",
            "2",
            "--repeats",
            "2",
            "--concurrency",
            "3",
            "--model-actor-concurrency",
            "3",
            "--target-qps",
            "0.5",
        ]
    )
    args.data_root = REPO_ROOT / "data"
    args.text_model_path = Path("/models/text")
    args.vision_model_path = Path("/models/vision")
    samples, failures = tool._discover(args)  # noqa: SLF001

    plan = tool._build_plan(  # noqa: SLF001
        args=args,
        output_dir=REPO_ROOT / "experiments" / "ray_baseline_performance" / "unit",
        samples=samples,
        discovery_failures=failures,
    )

    assert not failures
    assert plan["objective"] == "ray_performance_baseline"
    assert plan["executor"]["worker_max_calls"] == 1
    assert plan["executor"]["workflow_concurrency"] == 3
    assert plan["executor"]["model_actor_concurrency"] == 3
    assert plan["executor"]["target_qps"] == 0.5
    family = plan["workload"]["families"]["text"]
    assert len(family["samples"]) == 1
    assert len(family["warmup"]) == 1
    assert len(family["measurement"]) == 4
    assert {item["stage"] for item in family["measurement"]} == {"measurement"}
    assert [item["repeat"] for item in family["measurement"]] == [1, 1, 2, 2]


def test_ray_performance_family_defaults_to_text_but_can_select_vision() -> None:
    tool = _load_tool()

    default_args = tool.parse_args(["--plan-only"])
    vision_args = tool.parse_args(["--plan-only", "--family", "vision"])

    assert default_args.family == ["text"]
    assert vision_args.family == ["vision"]


def test_ray_performance_plan_records_vision_launch_workarounds() -> None:
    tool = _load_tool()
    args = tool.parse_args(["--plan-only", "--family", "vision"])
    args.data_root = REPO_ROOT / "data"
    args.text_model_path = Path("/models/text")
    args.vision_model_path = Path("/models/vision")

    plan = tool._build_plan(  # noqa: SLF001
        args=args,
        output_dir=REPO_ROOT / "experiments" / "ray_baseline_performance" / "unit",
        samples=[],
        discovery_failures=[],
    )

    assert plan["models"]["vision"]["launch_options"] == {
        "generation_config": "vllm",
        "qwen2_5_vl_cpu_unique_consecutive_workaround": True,
    }


def test_ray_performance_batch_mode_uses_batch_size_not_iterations() -> None:
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--dataset",
            "tbench",
            "--workflow",
            "retail_cancel",
            "--family",
            "text",
            "--samples-per-workflow",
            "1",
            "--arrival-mode",
            "batch",
            "--batch-size",
            "4",
            "--measurement-iterations",
            "99",
            "--repeats",
            "1",
        ]
    )
    args.data_root = REPO_ROOT / "data"
    args.text_model_path = Path("/models/text")
    args.vision_model_path = Path("/models/vision")
    samples, failures = tool._discover(args)  # noqa: SLF001

    plan = tool._build_plan(  # noqa: SLF001
        args=args,
        output_dir=REPO_ROOT / "experiments" / "ray_baseline_performance" / "unit",
        samples=samples,
        discovery_failures=failures,
    )

    family = plan["workload"]["families"]["text"]
    assert plan["workload"]["arrival_mode"] == "batch"
    assert plan["workload"]["batch_size"] == 4
    assert len(family["measurement"]) == 4
    assert {item["planned_launch_offset_ms"] for item in family["measurement"]} == {0}
    assert [item["sequence"] for item in family["measurement"]] == [1, 2, 3, 4]


def test_ray_performance_poisson_arrival_ratio_builds_maze_style_rate() -> None:
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--dataset",
            "tbench",
            "--workflow",
            "retail_cancel",
            "--family",
            "text",
            "--samples-per-workflow",
            "1",
            "--arrival-mode",
            "poisson",
            "--arrival-ratio",
            "0.5",
            "--avg-workflow-time-seconds",
            "45",
            "--measurement-window-seconds",
            "180",
            "--seed",
            "1",
        ]
    )
    args.data_root = REPO_ROOT / "data"
    args.text_model_path = Path("/models/text")
    args.vision_model_path = Path("/models/vision")
    samples, failures = tool._discover(args)  # noqa: SLF001

    plan = tool._build_plan(  # noqa: SLF001
        args=args,
        output_dir=REPO_ROOT / "experiments" / "ray_baseline_performance" / "unit",
        samples=samples,
        discovery_failures=failures,
    )

    workload = plan["workload"]
    family = workload["families"]["text"]
    assert workload["arrival_mode"] == "poisson"
    assert workload["arrival_ratio"] == 0.5
    assert workload["avg_workflow_time_seconds"] == 45.0
    assert workload["effective_arrival_rate"] == 0.5 / 45
    assert workload["arrival_rate_source"] == "arrival_ratio"
    assert family["measurement"]
    assert family["measurement"][0]["planned_launch_offset_ms"] == 0
    assert all(
        0 <= item["planned_launch_offset_ms"] < 180_000
        for item in family["measurement"]
    )


def test_ray_performance_aggregates_latency_and_tokens() -> None:
    tool = _load_tool()
    records = [
        {
            "status": "succeeded",
            "duration_ms": 1000,
            "performance": {"stage": "measurement"},
            "tasks": [
                {"task_name": "a", "duration_ms": 100},
                {"task_name": "b", "duration_ms": 900},
            ],
            "inference_records": [
                {"duration_ms": 800, "input_tokens": 10, "output_tokens": 20}
            ],
        },
        {
            "status": "succeeded",
            "duration_ms": 3000,
            "performance": {"stage": "measurement"},
            "tasks": [
                {"task_name": "a", "duration_ms": 300},
                {"task_name": "b", "duration_ms": 2700},
            ],
            "inference_records": [
                {"duration_ms": 1200, "input_tokens": 30, "output_tokens": 60}
            ],
        },
        {
            "status": "failed:task",
            "duration_ms": 500,
            "performance": {"stage": "measurement"},
            "failure": {"error_code": "unit_failed"},
        },
        {
            "status": "succeeded",
            "duration_ms": 10,
            "performance": {"stage": "warmup"},
            "tasks": [],
            "inference_records": [],
        },
    ]

    aggregate = tool._aggregate_records(  # noqa: SLF001
        records,
        measurement_started_at_ms=1_000,
        measurement_finished_at_ms=5_000,
    )

    assert aggregate["total"] == 3
    assert aggregate["succeeded"] == 2
    assert aggregate["failed"] == 1
    assert aggregate["success_rate"] == 2 / 3
    assert aggregate["workflow_latency_ms"]["mean"] == 2000
    assert aggregate["chat_latency_ms"]["count"] == 2
    assert aggregate["input_tokens"] == 40
    assert aggregate["output_tokens"] == 80
    assert aggregate["output_tokens_per_wall_second"] == 20
    assert aggregate["failure_reasons"] == {"unit_failed": 1}


def test_ray_performance_plan_only_cli_writes_plan(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--plan-only",
            "--dataset",
            "tbench",
            "--workflow",
            "retail_cancel",
            "--family",
            "text",
            "--samples-per-workflow",
            "1",
            "--warmup-iterations",
            "1",
            "--measurement-iterations",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}:{REPO_ROOT / 'tools'}",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "RAY_PERF_RESULT plan_only_succeeded" in completed.stdout
    plan = json.loads((tmp_path / "performance_plan.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert plan["objective"] == "ray_performance_baseline"
    assert plan["workload"]["warmup_iterations"] == 1
    assert plan["workload"]["measurement_iterations"] == 2
    assert summary["result"] == "plan_only_succeeded"
