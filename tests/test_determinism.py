from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _probe(seed: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "hashseed_probe.py")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_compiled_workflow_is_identical_across_hash_seeds() -> None:
    assert _probe("1") == _probe("777")


def test_public_import_does_not_load_runtime_heavy_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import ascend_maze; "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name.split('.')[0] in {'ray','torch_npu','vllm'})))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_common_modules_do_not_import_runtime_heavy_dependencies() -> None:
    forbidden = {"ray", "torch_npu", "vllm"}
    allowed_runtime_modules = {
        "ray": {
            "src/ascend_maze/data/ray_store.py",
            "src/ascend_maze/runtime/ray_backend.py",
            "src/ascend_maze/runtime/ray_cluster.py",
            "src/ascend_maze/runtime/ray_node_registry.py",
            "src/ascend_maze/runtime/ray_worker.py",
            "src/ascend_maze/runtime/ray_worker_pool.py",
        },
        "torch_npu": {
            "src/ascend_maze/inference/adapters/transformers_local.py",
        },
    }
    violations: list[tuple[str, str]] = []
    for path in (ROOT / "src" / "ascend_maze").rglob("*.py"):
        relative_path = str(path.relative_to(ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in forbidden and relative_path not in allowed_runtime_modules.get(
                    name, set()
                ):
                    violations.append((relative_path, name))
    assert violations == []
