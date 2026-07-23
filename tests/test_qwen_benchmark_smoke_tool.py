from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ascend_maze.contracts.recording import ExecutionEvent


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "qwen_benchmark_smoke.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("qwen_benchmark_smoke", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen_benchmark_smoke_discovers_one_sample_per_workflow() -> None:
    tool = _load_tool()

    samples, failures = tool.discover_samples(
        data_root=REPO_ROOT / "data",
        datasets=set(),
        workflows=set(),
        families=set(),
        samples_per_workflow=1,
        sample_offset=0,
        max_inline_file_bytes=64 * 1024 * 1024,
    )

    assert not failures
    assert len(samples) == 14
    assert {sample.family for sample in samples} == {"text", "vision"}
    assert {
        (sample.dataset, sample.workflow)
        for sample in samples
    } == set(tool.WORKFLOW_MODULES)


def test_qwen_benchmark_smoke_parses_tbench_questions_without_tau_bench() -> None:
    tool = _load_tool()

    retail_cancel = tool._parse_tbench_task_file(  # noqa: SLF001
        REPO_ROOT / "data" / "tbench" / "question" / "cancel_ins.py"
    )

    first = retail_cancel["55f2e5e7-6a59-4802-9fa8-179f1a6a4e85"]
    assert "Cancel order #W7602708" in first["instruction"]
    assert first["user_id"] == "juan_rossi_6696"
    assert first["actions"] == [
        {
            "name": "cancel_pending_order",
            "kwargs": {
                "order_id": "#W7602708",
                "reason": "no longer needed",
            },
        }
    ]


def test_qwen_benchmark_smoke_adds_explicit_tbench_smoke_overrides() -> None:
    tool = _load_tool()

    samples, failures = tool.discover_samples(
        data_root=REPO_ROOT / "data",
        datasets={"tbench"},
        workflows={"retail_cancel"},
        families={"text"},
        samples_per_workflow=1,
        sample_offset=0,
        max_inline_file_bytes=64 * 1024 * 1024,
    )

    assert not failures
    assert len(samples) == 1
    metadata = samples[0].inputs["metadata"]
    assert metadata["smoke_override_mode"] == "tbench_ground_truth_actions_or_regex"
    assert json.loads(metadata["llm_output_override"]) == [
        {
            "order_id": "#W7602708",
            "reason": "no longer needed",
        }
    ]


def test_qwen_benchmark_smoke_can_disable_tbench_smoke_overrides() -> None:
    tool = _load_tool()

    samples, failures = tool.discover_samples(
        data_root=REPO_ROOT / "data",
        datasets={"tbench"},
        workflows={"retail_cancel"},
        families={"text"},
        samples_per_workflow=1,
        sample_offset=0,
        max_inline_file_bytes=64 * 1024 * 1024,
        tbench_smoke_overrides=False,
    )

    assert not failures
    assert "llm_output_override" not in samples[0].inputs["metadata"]
    assert "smoke_override_mode" not in samples[0].inputs["metadata"]


def test_qwen_benchmark_smoke_summarizes_gaia_file_inputs() -> None:
    tool = _load_tool()

    samples, failures = tool.discover_samples(
        data_root=REPO_ROOT / "data",
        datasets={"gaia"},
        workflows={"file"},
        families={"text"},
        samples_per_workflow=1,
        sample_offset=0,
        max_inline_file_bytes=64 * 1024 * 1024,
    )

    assert not failures
    assert len(samples) == 1
    metadata = samples[0].inputs["metadata"]
    supplementary_files = samples[0].inputs["supplementary_files"]
    payload = next(iter(supplementary_files.values()))
    assert metadata["gaia_file_smoke_mode"] == "file_summary_not_full_inline"
    assert isinstance(payload, str)
    assert "GAIA file smoke summary" in payload
    assert "not GAIA file-answer accuracy" in payload


def test_qwen_benchmark_smoke_rewrites_model_anchors_before_compile() -> None:
    tool = _load_tool()
    samples, _ = tool.discover_samples(
        data_root=REPO_ROOT / "data",
        datasets={"gaia"},
        workflows={"reason"},
        families={"text"},
        samples_per_workflow=1,
        sample_offset=0,
        max_inline_file_bytes=64 * 1024 * 1024,
    )
    assert len(samples) == 1

    workflow, aliases = tool._build_workflow(  # noqa: SLF001
        samples[0].dataset,
        samples[0].workflow,
        "local-qwen-smoke",
    )
    compiled = workflow.compile()
    compiled_models = {
        node.model_anchor.model
        for _, node in compiled.tasks.items_tuple()
        if node.model_anchor is not None
    }

    assert aliases == {
        "deepseek-r1-32b": "local-qwen-smoke",
        "qwen3-32b": "local-qwen-smoke",
    }
    assert compiled_models == {"local-qwen-smoke"}


def test_qwen_benchmark_smoke_collects_task_timing_records() -> None:
    tool = _load_tool()

    class Runtime:
        def task_timing_records(self, run_id: str):
            assert run_id == "run_1"
            return (
                {
                    "run_id": "run_1",
                    "task_id": "task_a",
                    "attempt": 1,
                    "task_total_ms": 15,
                    "input_fetch_ms": 2,
                    "callable_execute_ms": 10,
                    "chat_request_ms": 7,
                    "output_put_ms": 1,
                    "task_runtime_overhead_ms": 2,
                    "callable_minus_chat_ms": 3,
                },
            )

    class Controller:
        runtime = Runtime()

    records = tool._task_timing_records(  # noqa: SLF001
        Controller(),
        "run_1",
        {"task_alpha": "task_a"},
    )
    assert records == [
        {
            "run_id": "run_1",
            "task_id": "task_a",
            "attempt": 1,
            "task_total_ms": 15,
            "input_fetch_ms": 2,
            "callable_execute_ms": 10,
            "chat_request_ms": 7,
            "output_put_ms": 1,
            "task_runtime_overhead_ms": 2,
            "callable_minus_chat_ms": 3,
            "task_name": "task_alpha",
        }
    ]
    assert tool._task_timing_summary(records) == {  # noqa: SLF001
        "task_count": 1,
        "task_total_ms": 15,
        "dispatch_prepare_ms": 0,
        "worker_startup_ms": 0,
        "dispatch_wait_ms": 0,
        "input_fetch_ms": 2,
        "callable_execute_ms": 10,
        "chat_request_ms": 7,
        "output_put_ms": 1,
        "task_runtime_overhead_ms": 2,
        "callable_minus_chat_ms": 3,
    }


def test_qwen_benchmark_smoke_records_ray_binding_evidence() -> None:
    tool = _load_tool()
    event = ExecutionEvent(
        schema_version=1,
        event_id="event_1",
        experiment_id="run_1",
        run_id="run_1",
        task_id="task_a",
        attempt=1,
        lease_id="lease_1",
        route_lease_id=None,
        model_instance_id=None,
        event_type="task_dispatched",
        producer_id="controller:test",
        producer_sequence=1,
        node_id=None,
        device_id=None,
        monotonic_time_ms=1,
        wall_time_ms=1,
        duration_ms=None,
        payload={
            "node_id": "node_a",
            "affinity_hit": True,
            "input_object_refs": (
                {
                    "input_name": "backend_data",
                    "data_handle_id": "data_1",
                    "object_ref_id": "object_1",
                },
            ),
        },
    )

    class Recorder:
        def events(self, run_id: str):
            assert run_id == "run_1"
            return (event,)

    class Controller:
        recorder = Recorder()

    records = tool._run_event_records(  # noqa: SLF001
        Controller(),
        "run_1",
        {"task_alpha": "task_a"},
    )

    assert len(records) == 1
    assert records[0]["task_name"] == "task_alpha"
    assert records[0]["event_type"] == "task_dispatched"
    assert records[0]["payload"] == {
        "node_id": "node_a",
        "affinity_hit": True,
        "input_object_refs": [
            {
                "input_name": "backend_data",
                "data_handle_id": "data_1",
                "object_ref_id": "object_1",
            }
        ],
    }


def test_residual_vllm_processes_only_reports_owned_process_groups(
    monkeypatch,
) -> None:
    tool = _load_tool()
    process_table = "\n".join(
        (
            "100 1 100 S python -m vllm.entrypoints.openai.api_server "
            "--model /models/qwen --port 32060",
            "200 1 200 S python -m vllm.entrypoints.openai.api_server "
            "--model /models/qwen --port 32271",
            "201 200 200 S python -m vllm.entrypoints.openai.api_server "
            "--model /models/qwen --port 32271",
        )
    )
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=process_table),
    )

    residual = tool._residual_vllm_processes(  # noqa: SLF001
        (Path("/models/qwen"),),
        (32060, 32271),
        owned_process_group_ids=(100,),
    )

    assert residual == [process_table.splitlines()[0]]


def test_qwen_benchmark_smoke_rejects_incomplete_model_artifact(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    model_dir = tmp_path / "empty-model"
    model_dir.mkdir()

    try:
        tool.validate_model_artifact(model_dir)
    except tool.SmokePreflightError as exc:
        assert "model config is missing" in str(exc)
    else:  # pragma: no cover - makes the failure message clearer than pytest.raises.
        raise AssertionError("empty model directory should fail validation")

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    try:
        tool.validate_model_artifact(model_dir)
    except tool.SmokePreflightError as exc:
        assert "model weights are missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("config-only model directory should fail validation")

    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"fake")
    manifest = tool.validate_model_artifact(model_dir)
    assert manifest["weight_file_count"] == 1


def test_qwen_benchmark_smoke_plan_only_cli_writes_plan(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--plan-only",
            "--dataset",
            "tbench",
            "--workflow",
            "retail_cancel",
            "--samples-per-workflow",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SMOKE_RESULT plan_only_succeeded" in completed.stdout
    assert (tmp_path / "plan.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["inference_backend"] == "vllm"
    assert plan["text_model"]["max_model_len"] == 10240
    assert plan["vision_model"]["model_id"] == "qwen2_5-vl-3b-smoke"
    assert plan["vision_model"]["path"].endswith("Qwen2.5-VL-3B-Instruct")
    assert plan["vision_model"]["max_model_len"] == 12288
    assert plan["vision_model"]["max_num_batched_tokens"] == 4096


def test_qwen_benchmark_smoke_plan_records_transformers_backend(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--plan-only",
            "--inference-backend",
            "transformers",
            "--dataset",
            "tbench",
            "--workflow",
            "retail_cancel",
            "--family",
            "text",
            "--samples-per-workflow",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["inference_backend"] == "transformers"
    assert [sample["family"] for sample in plan["samples"]] == ["text"]


def test_qwen_benchmark_smoke_configures_transformers_vision_path() -> None:
    tool = _load_tool()

    options = tool._transformers_local_launch_options(  # noqa: SLF001
        device_id="6",
        request_timeout_ms=900_000,
        runtime_paths=("/opt/ascend/host_aicpu",),
        trust_remote_code=False,
        is_vision=True,
    )

    assert options == {
        "device_id": "6",
        "enable_thinking": False,
        "generation_method": "manual_greedy",
        "model_kind": "vision_language",
        "qwen2_5_vl_cpu_unique_consecutive_workaround": True,
        "request_timeout_ms": 900_000,
        "runtime_library_paths": ("/opt/ascend/host_aicpu",),
        "trust_remote_code": False,
    }
