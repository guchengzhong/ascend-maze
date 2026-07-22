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

    expected_edges = {
        (task_id_by_name[source], task_id_by_name[target])
        for source, target in module.SPEC.edges
    }
    actual_edges = {
        (source_id, target_id)
        for source_id, task in compiled.tasks.items_tuple()
        for target_id in compiled.successors[task.task_id]
    }
    assert actual_edges == expected_edges


@pytest.mark.parametrize("module_name", MIGRATED_WORKFLOW_MODULES)
def test_migrated_workflow_data_bindings_follow_original_dag_edges(
    module_name: str,
) -> None:
    module = import_module(module_name)

    compiled = module.build().compile()
    task_by_name = {task.task_name: task for _, task in compiled.tasks.items_tuple()}
    predecessor_names = {node.name: [] for node in module.SPEC.nodes}
    for source, target in module.SPEC.edges:
        predecessor_names[target].append(source)

    for node in module.SPEC.nodes:
        task_node = task_by_name[node.name]
        state_inputs = [
            item
            for item in task_node.inputs
            if isinstance(item, (OutputBinding, WorkflowInputBinding))
        ]
        if not predecessor_names[node.name]:
            assert {
                item.workflow_input_name
                for item in state_inputs
                if isinstance(item, WorkflowInputBinding)
            } == {
                "answer",
                "dag_id",
                "metadata",
                "question",
                "supplementary_files",
            }
            continue

        expected_sources = {
            task_by_name[parent_name].task_id
            for parent_name in predecessor_names[node.name]
        }
        actual_sources = {
            item.source_task_id
            for item in state_inputs
            if isinstance(item, OutputBinding)
        }
        assert actual_sources == expected_sources
