from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "ray_baseline_smoke.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("ray_baseline_smoke", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ray_baseline_plan_uses_plain_ray_executor() -> None:
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
            "--plan-only",
        ]
    )
    args.data_root = REPO_ROOT / "data"
    args.text_model_path = Path("/models/text")
    args.vision_model_path = Path("/models/vision")
    samples, failures = tool._discover(args)  # noqa: SLF001

    plan = tool._build_plan(  # noqa: SLF001
        args=args,
        output_dir=REPO_ROOT / "experiments" / "ray_baseline_smoke" / "unit",
        samples=samples,
        discovery_failures=failures,
    )

    assert not failures
    assert len(samples) == 1
    assert plan["objective"] == "ray_correctness_baseline"
    assert plan["inference_backend"] == "vllm"
    assert plan["executor"] == {
        "kind": "plain_ray_task_actor",
        "dag_policy": "sequential_topological_order",
        "worker_max_calls": 1,
        "uses_ascend_maze_controller": False,
        "uses_ascend_maze_scheduler": False,
        "uses_ascend_maze_runtime_client": False,
    }
    assert plan["text_model"]["model_id"] == "qwen3-4b-smoke"


def test_ray_baseline_transformers_plan_and_config() -> None:
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
            "--inference-backend",
            "transformers",
            "--plan-only",
        ]
    )
    args.data_root = REPO_ROOT / "data"
    args.text_model_path = Path("/models/text")
    args.vision_model_path = Path("/models/vision")
    samples, failures = tool._discover(args)  # noqa: SLF001

    plan = tool._build_plan(  # noqa: SLF001
        args=args,
        output_dir=REPO_ROOT / "experiments" / "ray_baseline_smoke" / "unit",
        samples=samples,
        discovery_failures=failures,
    )
    config = tool._family_transformers_config(  # noqa: SLF001
        args=args,
        family="text",
        preflight={"runtime_library_paths": ("/aicpu",)},
    )

    assert not failures
    assert plan["inference_backend"] == "transformers"
    assert config == {
        "family": "text",
        "model_id": "qwen3-4b-smoke",
        "model_path": "/models/text",
        "tokenizer_path": "/models/text",
        "device_id": "0",
        "dtype": "bfloat16",
        "generation_method": "manual_greedy",
        "model_kind": "text",
        "max_model_len": 10240,
        "trust_remote_code": False,
        "enable_thinking": False,
        "request_timeout_ms": 180_000,
        "runtime_library_paths": ("/aicpu",),
    }


def test_ray_baseline_transformers_vision_config() -> None:
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--inference-backend",
            "transformers",
            "--family",
            "vision",
        ]
    )
    args.vision_model_path = Path("/models/qwen2.5-vl")

    config = tool._family_transformers_config(  # noqa: SLF001
        args=args,
        family="vision",
        preflight={"runtime_library_paths": ("/aicpu",)},
    )

    assert config == {
        "family": "vision",
        "model_id": "qwen2_5-vl-3b-smoke",
        "model_path": "/models/qwen2.5-vl",
        "tokenizer_path": "/models/qwen2.5-vl",
        "device_id": "0",
        "dtype": "bfloat16",
        "generation_method": "manual_greedy",
        "model_kind": "vision_language",
        "max_model_len": 12288,
        "trust_remote_code": True,
        "enable_thinking": False,
        "request_timeout_ms": 180_000,
        "runtime_library_paths": ("/aicpu",),
        "qwen2_5_vl_cpu_unique_consecutive_workaround": True,
    }


def test_ray_baseline_vllm_argv_uses_model_alias_and_vision_options() -> None:
    tool = _load_tool()

    argv = tool._build_vllm_argv(  # noqa: SLF001
        python_executable=Path("/opt/python"),
        host="127.0.0.1",
        port=31441,
        model_path=Path("/weights/Qwen2.5-VL-3B-Instruct"),
        served_model_name="qwen2_5-vl-3b-smoke",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        trust_remote_code=True,
        generation_config="vllm",
    )

    assert argv[:3] == ["/opt/python", "-m", "vllm.entrypoints.openai.api_server"]
    assert argv[argv.index("--served-model-name") + 1] == "qwen2_5-vl-3b-smoke"
    assert argv[argv.index("--model") + 1] == "/weights/Qwen2.5-VL-3B-Instruct"
    assert argv[argv.index("--max-model-len") + 1] == "4096"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "4096"
    assert "--trust-remote-code" in argv
    assert "--no-enable-prefix-caching" in argv
    assert argv[argv.index("--generation-config") + 1] == "vllm"


def test_ray_baseline_vision_service_uses_runtime_workarounds() -> None:
    tool = _load_tool()
    env = tool._service_environment(  # noqa: SLF001
        base_env={"PYTHONPATH": "/existing/path"},
        device_id="3",
        log_level="INFO",
        runtime_preloads={},
        runtime_library_paths=(),
        qwen2_5_vl_cpu_unique_consecutive_workaround=True,
    )

    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert env["ASCEND_MAZE_QWEN25VL_CPU_UNIQUE_CONSECUTIVE"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[0].endswith(
        "ascend_maze/inference/adapters/vllm_runtime_patches"
    )


def test_ray_baseline_plan_records_vision_launch_workarounds() -> None:
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--family",
            "vision",
            "--dataset",
            "gaia",
            "--workflow",
            "vision",
            "--samples-per-workflow",
            "1",
            "--plan-only",
        ]
    )
    args.data_root = REPO_ROOT / "data"
    args.text_model_path = Path("/models/text")
    args.vision_model_path = Path("/models/vision")
    samples, failures = tool._discover(args)  # noqa: SLF001

    plan = tool._build_plan(  # noqa: SLF001
        args=args,
        output_dir=REPO_ROOT / "experiments" / "ray_baseline_smoke" / "unit",
        samples=samples,
        discovery_failures=failures,
    )

    assert not failures
    assert plan["vision_model"]["launch_options"] == {
        "generation_config": "vllm",
        "qwen2_5_vl_cpu_unique_consecutive_workaround": True,
    }


def test_ray_baseline_resolves_compiled_workflow_bindings() -> None:
    tool = _load_tool()
    samples, failures = tool.qwen_smoke.discover_samples(
        data_root=REPO_ROOT / "data",
        datasets={"tbench"},
        workflows={"retail_cancel"},
        families={"text"},
        samples_per_workflow=1,
        sample_offset=0,
        max_inline_file_bytes=64 * 1024 * 1024,
    )
    assert not failures
    sample = samples[0]
    workflow, _aliases = tool.qwen_smoke._build_workflow(  # noqa: SLF001
        sample.dataset,
        sample.workflow,
        "qwen3-4b-smoke",
    )
    compiled = workflow.compile()
    first_task_id, second_task_id = compiled.topological_order[:2]
    first_node = compiled.tasks[first_task_id]
    second_node = compiled.tasks[second_task_id]

    first_kwargs = tool._resolve_task_kwargs(  # noqa: SLF001
        compiled=compiled,
        node=first_node,
        workflow_inputs=sample.inputs,
        task_outputs={},
    )

    assert first_kwargs["dag_id"] == sample.dag_id
    assert "Cancel order" in first_kwargs["question"]

    first_outputs = {
        "answer": "",
        "backend_data": {"orders": []},
        "dag_id": sample.dag_id,
        "instruction": "instruction",
        "metadata": {"k": "v"},
        "prompt": "prompt",
        "succ_task_feat": {},
    }
    second_kwargs = tool._resolve_task_kwargs(  # noqa: SLF001
        compiled=compiled,
        node=second_node,
        workflow_inputs=sample.inputs,
        task_outputs={first_task_id: first_outputs},
    )

    assert second_kwargs == {
        "dag_id": sample.dag_id,
        "instruction": "instruction",
        "prompt": "prompt",
        "metadata": {"k": "v"},
        "backend_data": {"orders": []},
    }


def test_ray_baseline_persists_failure_records(tmp_path: Path) -> None:
    tool = _load_tool()
    record = {
        "schema_version": 1,
        "sample": {"sample_id": "unit.failed"},
        "status": "failed:task",
        "failure": {"error_code": "unit_failure"},
    }

    succeeded = tool._persist_sample_record(  # noqa: SLF001
        records_path=tmp_path / "text_records.jsonl",
        failures_path=tmp_path / "text_failures.jsonl",
        record=record,
    )

    assert succeeded is False
    records = (tmp_path / "text_records.jsonl").read_text().splitlines()
    failures = (tmp_path / "text_failures.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert len(failures) == 1
    assert json.loads(records[0]) == record
    assert json.loads(failures[0]) == record


def test_ray_baseline_plan_only_cli_writes_plan(tmp_path: Path) -> None:
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
    assert "RAY_BASELINE_RESULT plan_only_succeeded" in completed.stdout
    plan = json.loads((tmp_path / "plan.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert plan["objective"] == "ray_correctness_baseline"
    assert plan["executor"]["uses_ascend_maze_scheduler"] is False
    assert len(plan["samples"]) == 1
    assert summary["result"] == "plan_only_succeeded"
