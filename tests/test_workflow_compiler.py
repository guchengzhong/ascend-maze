from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ascend_maze import Workflow, task
from ascend_maze.compiler import CompileOptions
from ascend_maze.compiler.ir import (
    DefaultBinding,
    LiteralBinding,
    OutputBinding,
    WorkflowInputBinding,
)
from ascend_maze.core.errors import (
    LiteralSizeError,
    WorkflowFrozenError,
    WorkflowValidationError,
)
from task_fixtures import (
    barrier,
    finish,
    load_text,
    local_npu_task,
    service_task,
    summarize,
    unanchored_npu_task,
)


@task
def task_with_mutable_default(value: str, options: dict = {"items": [1, 2]}):
    return {"result": value}


def build_workflow(name: str = "document-summary"):
    workflow = Workflow(name)
    path = workflow.input("path")
    loaded = workflow.add_task(load_text, inputs={"path": path}, task_name="load")
    options = {"upper": False, "buckets": {2, 1}}
    summary = workflow.add_task(
        summarize,
        inputs={"text": loaded.outputs["text"], "options": options},
        task_name="summary",
    )
    gate = workflow.add_task(barrier, task_name="gate")
    result = workflow.add_task(
        finish,
        inputs={"summary": summary.outputs["summary"]},
        task_name="finish",
    )
    workflow.add_edge(gate, result)
    return workflow, options, loaded, summary, gate, result


def test_compile_builds_structured_immutable_ir() -> None:
    workflow, _, loaded, summary, gate, result = build_workflow()
    compiled = workflow.compile()
    assert workflow.frozen
    assert compiled.workflow_inputs == ("path",)
    assert len(compiled.definitions) == 4
    assert isinstance(compiled.tasks[loaded.task_id].inputs[0], WorkflowInputBinding)
    summary_bindings = compiled.tasks[summary.task_id].inputs
    assert isinstance(summary_bindings[0], OutputBinding)
    assert isinstance(summary_bindings[1], LiteralBinding)
    assert isinstance(summary_bindings[2], DefaultBinding)
    assert compiled.predecessors[result.task_id] == tuple(
        sorted((summary.task_id, gate.task_id))
    )
    assert compiled.depth_from_entry[loaded.task_id] == 0
    assert compiled.depth_to_exit[result.task_id] == 0
    assert compiled.workflow_fingerprint
    assert compiled.canonical_ir_bytes
    with pytest.raises(FrozenInstanceError):
        compiled.tasks[summary.task_id].task_name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        compiled.tasks["new"] = compiled.tasks[summary.task_id]  # type: ignore[index]


def test_compile_snapshots_literals_and_is_idempotent() -> None:
    workflow, options, _, summary, _, _ = build_workflow()
    compiled = workflow.compile()
    before = compiled.canonical_ir_bytes
    options["upper"] = True
    options["buckets"].add(9)
    assert compiled.canonical_ir_bytes == before
    literal = compiled.tasks[summary.task_id].inputs[1]
    assert isinstance(literal, LiteralBinding)
    assert literal.value["upper"] is False
    assert workflow.compile() is compiled
    with pytest.raises(WorkflowFrozenError):
        workflow.add_task(barrier)


def test_compile_snapshots_default_value_digests() -> None:
    workflow = Workflow("mutable-default")
    node = workflow.add_task(task_with_mutable_default, inputs={"value": "x"})
    compiled = workflow.compile()
    before = compiled.canonical_ir_bytes
    defaults = task_with_mutable_default.__defaults__
    assert defaults is not None
    defaults[0]["items"].append(3)
    assert compiled.canonical_ir_bytes == before
    definition = compiled.definitions[compiled.tasks[node.task_id].definition_id]
    assert definition.default_value_digests


def test_same_function_added_multiple_times_has_stable_names_and_one_definition() -> None:
    workflow = Workflow("repeated")
    first = workflow.add_task(barrier)
    second = workflow.add_task(barrier)
    compiled = workflow.compile()
    assert first.task_name == "barrier"
    assert second.task_name == "barrier_2"
    assert first.task_id != second.task_id
    assert len(compiled.definitions) == 1


def test_unknown_missing_duplicate_and_cross_workflow_inputs_fail() -> None:
    unknown = Workflow("unknown")
    unknown.add_task(finish, inputs={"bad": "x"})
    with pytest.raises(WorkflowValidationError, match="unknown inputs"):
        unknown.compile()

    missing = Workflow("missing")
    missing.add_task(finish)
    with pytest.raises(WorkflowValidationError, match="missing required"):
        missing.compile()

    duplicate = Workflow("duplicate")
    duplicate.add_task(barrier, task_name="same")
    with pytest.raises(WorkflowValidationError, match="already used"):
        duplicate.add_task(barrier, task_name="same")

    left = Workflow("left")
    source = left.add_task(finish, inputs={"summary": "x"})
    right = Workflow("right")
    right.add_task(finish, inputs={"summary": source.outputs["result"]})
    with pytest.raises(WorkflowValidationError, match="another workflow"):
        right.compile()


def test_nested_output_references_are_rejected() -> None:
    workflow = Workflow("nested")
    source = workflow.add_task(finish, inputs={"summary": "x"})
    with pytest.raises(WorkflowValidationError, match="top-level"):
        workflow.add_task(
            summarize,
            inputs={
                "text": "x",
                "options": {"source": source.outputs["result"]},
            },
        )


def test_cycle_is_rejected() -> None:
    workflow = Workflow("cycle")
    left = workflow.add_task(barrier, task_name="left")
    right = workflow.add_task(barrier, task_name="right")
    workflow.add_edge(left, right)
    workflow.add_edge(right, left)
    with pytest.raises(WorkflowValidationError, match="cycle"):
        workflow.compile()


def test_model_resource_rules_are_checked_at_compile_time() -> None:
    service = Workflow("service")
    service.add_task(
        service_task,
        inputs={"prompt": "hello"},
        model_anchor={"model": "qwen-small", "mode": "service"},
    )
    service.compile()

    invalid_service = Workflow("invalid-service")
    invalid_service.add_task(
        local_npu_task,
        inputs={"value": "x"},
        model_anchor={"model": "qwen-small", "mode": "service"},
    )
    with pytest.raises(WorkflowValidationError, match="cannot explicitly reserve"):
        invalid_service.compile()

    local = Workflow("local")
    local.add_task(local_npu_task, inputs={"value": "x"})
    local.compile()

    unanchored = Workflow("unanchored")
    unanchored.add_task(unanchored_npu_task, inputs={"value": "x"})
    with pytest.raises(WorkflowValidationError, match="positive npu_mem"):
        unanchored.compile()

    catalog_local = Workflow("catalog-local")
    catalog_local.add_task(
        unanchored_npu_task,
        inputs={"value": "x"},
        model_anchor={"model": "qwen-small", "mode": "local_worker"},
    )
    catalog_local.compile()


def test_literal_limits_are_enforced_per_value_and_in_total() -> None:
    per_value = Workflow("per-value")
    per_value.add_task(finish, inputs={"summary": "x" * 100})
    with pytest.raises(LiteralSizeError, match="max_literal_value_bytes"):
        per_value.compile(
            CompileOptions(
                max_literal_value_bytes=32,
                max_compiled_literal_bytes=256,
            )
        )

    total = Workflow("total")
    total.add_task(
        summarize,
        inputs={"text": "x" * 20, "options": {"value": "y" * 20}},
    )
    with pytest.raises(LiteralSizeError, match="max_compiled_literal_bytes"):
        total.compile(
            CompileOptions(
                max_literal_value_bytes=80,
                max_compiled_literal_bytes=80,
            )
        )

    with pytest.raises(ValueError, match="cannot exceed"):
        CompileOptions(
            max_literal_value_bytes=81,
            max_compiled_literal_bytes=80,
        )


def test_fingerprint_covers_literals_models_and_edges() -> None:
    def fingerprint(*, literal: str, edge: bool, model: str) -> str:
        workflow = Workflow("fingerprint")
        gate = workflow.add_task(barrier, task_name="gate")
        node = workflow.add_task(
            service_task,
            inputs={"prompt": literal},
            task_name="service",
            model_anchor={"model": model, "mode": "service"},
        )
        if edge:
            workflow.add_edge(gate, node)
        return workflow.compile().workflow_fingerprint

    baseline = fingerprint(literal="a", edge=False, model="m1")
    assert fingerprint(literal="b", edge=False, model="m1") != baseline
    assert fingerprint(literal="a", edge=True, model="m1") != baseline
    assert fingerprint(literal="a", edge=False, model="m2") != baseline
