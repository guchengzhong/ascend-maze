from __future__ import annotations

from importlib import import_module

import pytest

from ascend_maze.compiler.ir import OutputBinding, WorkflowInputBinding


MIGRATED_WORKFLOW_MODULES = (
    "workflows.gaia.file",
    "workflows.gaia.reason",
    "workflows.gaia.speech",
    "workflows.gaia.vision",
    "workflows.openagi.document_qa",
    "workflows.openagi.image_captioning_complex",
    "workflows.openagi.multimodal_vqa_complex",
    "workflows.openagi.text_processing_multilingual",
    "workflows.tbench.airline_book",
    "workflows.tbench.airline_cancel",
    "workflows.tbench.retail_cancel",
    "workflows.tbench.retail_cancel_modify",
    "workflows.tbench.retail_modify",
    "workflows.tbench.retail_return",
)

EXPECTED_INPUTS = (
    "answer",
    "dag_id",
    "metadata",
    "question",
    "supplementary_files",
)


@pytest.mark.parametrize("module_name", MIGRATED_WORKFLOW_MODULES)
def test_migrated_workflow_builds_compileable_ascend_maze_ir(
    module_name: str,
) -> None:
    module = import_module(module_name)

    workflow = module.build()
    compiled = workflow.compile()
    task_by_name = {task.task_name: task for _, task in compiled.tasks.items_tuple()}
    task_id_by_name = {
        task.task_name: task.task_id for _, task in compiled.tasks.items_tuple()
    }

    assert compiled.workflow_name == module.SPEC.name
    assert compiled.workflow_inputs == EXPECTED_INPUTS
    assert set(task_by_name) == {node.name for node in module.SPEC.nodes}
    assert len(compiled.tasks) == len(module.SPEC.nodes)
    assert compiled.workflow_fingerprint
    assert compiled.canonical_ir_bytes

    for node in module.SPEC.nodes:
        task_node = task_by_name[node.name]
        definition = compiled.definitions[task_node.definition_id]
        assert definition.task_kind == node.kind
        if node.kind == "npu":
            assert task_node.model_anchor is not None
            assert task_node.model_anchor.mode == "service"
            assert task_node.model_anchor.model
        else:
            assert task_node.model_anchor is None

    explicit_edges = {
        (task_id_by_name[source], task_id_by_name[target])
        for source, target in module.SPEC.edges
    }
    data_edges = {
        (binding.source_task_id, task.task_id)
        for _, task in compiled.tasks.items_tuple()
        for binding in task.inputs
        if isinstance(binding, OutputBinding)
    }
    actual_edges = {
        (source_id, target_id)
        for source_id, task in compiled.tasks.items_tuple()
        for target_id in compiled.successors[task.task_id]
    }
    assert actual_edges == explicit_edges | data_edges


@pytest.mark.parametrize("module_name", MIGRATED_WORKFLOW_MODULES)
def test_migrated_workflow_data_bindings_are_internal_and_resolvable(
    module_name: str,
) -> None:
    module = import_module(module_name)

    compiled = module.build().compile()
    for _, task_node in compiled.tasks.items_tuple():
        definition = compiled.definitions[task_node.definition_id]
        assert not {
            "context",
            "dag_context",
            "data_handle",
            "object_ref",
        }.intersection(definition.input_names)
        for binding in task_node.inputs:
            assert binding.input_name in definition.input_names
            if isinstance(binding, WorkflowInputBinding):
                assert binding.workflow_input_name in compiled.workflow_inputs
            elif isinstance(binding, OutputBinding):
                source = compiled.tasks[binding.source_task_id]
                source_definition = compiled.definitions[source.definition_id]
                assert binding.source_output in source_definition.output_names
                assert task_node.task_id in compiled.successors[source.task_id]
