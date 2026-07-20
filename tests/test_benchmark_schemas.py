from __future__ import annotations

from pathlib import Path

import pytest

from ascend_maze.benchmark.schema_registry import (
    SCHEMA_FILES,
    load_schema,
    schema_digests,
)
from ascend_maze.core.errors import ExperimentValidationError


def test_all_versioned_benchmark_schemas_are_packaged_and_fingerprinted() -> None:
    assert len(SCHEMA_FILES) == 16
    digests = dict(schema_digests())
    assert tuple(sorted(digests)) == SCHEMA_FILES
    assert all(len(value) == 64 for value in digests.values())
    for name in SCHEMA_FILES:
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert str(schema["$id"]).endswith(name)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
    with pytest.raises(ExperimentValidationError, match="unknown benchmark schema"):
        load_schema("future.v2.schema.json")


def test_pyproject_packages_schemas_and_exposes_dedicated_entry_point() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'maze-bench = "ascend_maze.benchmark.cli:main"' in content
    assert '"benchmark/schemas/*.json"' in content
