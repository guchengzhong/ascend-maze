from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ascend_maze.benchmark.cli import build_parser, main
from benchmark_fixtures import write_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MODULES = {"ray", "torch_npu", "vllm"}


def test_maze_bench_plan_emits_one_stable_json_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = write_experiment_spec(tmp_path)
    assert main(["plan", str(spec)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    payload = json.loads(captured.out)
    assert payload["schema"] == "ascend-maze.study-plan.v1"
    assert len(payload["cells"]) == 5
    assert len(payload["trials"]) == 50

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("Ascend-Maze benchmark 0.1.0")


def test_maze_bench_reports_structured_local_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["plan", str(tmp_path / "missing.toml")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["schema_version"] == 1
    assert error["status"] == "error"
    assert error["error_code"] == "experiment_validation_failed"

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "experiment.toml"])


def test_plan_bytes_are_identical_across_process_hash_seeds(tmp_path: Path) -> None:
    spec = write_experiment_spec(tmp_path)
    outputs: list[bytes] = []
    for seed in ("1", "777"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-m", "ascend_maze.benchmark.cli", "plan", str(spec)],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
        )
        assert completed.stderr == b""
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
    first = json.loads(outputs[0])
    second = json.loads(outputs[1])
    assert first["study_id"] == second["study_id"]
    assert [item["cell_id"] for item in first["cells"]] == [
        item["cell_id"] for item in second["cells"]
    ]
    assert [item["trial_id"] for item in first["trials"]] == [
        item["trial_id"] for item in second["trials"]
    ]


def test_planner_import_and_execution_do_not_load_heavy_runtimes(
    tmp_path: Path,
) -> None:
    spec = write_experiment_spec(tmp_path)
    script = (
        "import json,sys; "
        "from ascend_maze.benchmark.loader import load_study_plan; "
        f"load_study_plan({str(spec)!r}); "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.split('.')[0] in {'ray','torch_npu','vllm'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []

    violations: list[tuple[str, str]] = []
    for path in (ROOT / "src" / "ascend_maze" / "benchmark").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module.split(".")[0]]
            else:
                continue
            for name in imported:
                if name in FORBIDDEN_MODULES:
                    violations.append((str(path.relative_to(ROOT)), name))
                if name == "ascend_maze" and isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith("ascend_maze.control"):
                        violations.append((str(path.relative_to(ROOT)), module))
    assert violations == []
