from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


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
    assert plan["vision_model"]["model_id"] == "qwen2_5-vl-3b-smoke"
    assert plan["vision_model"]["path"].endswith("Qwen2.5-VL-3B-Instruct")
    assert plan["vision_model"]["max_model_len"] == 4096
    assert plan["vision_model"]["max_num_batched_tokens"] == 4096
