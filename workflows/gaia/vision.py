"""Ascend-Maze-native GAIA vision workflow."""

from __future__ import annotations

from ascend_maze import Workflow, task

from workflows._common import WorkflowSpec, edges, nodes, spec_inputs
from workflows.gaia._common import (
    inference_features,
    metadata_dict,
    response_or_override,
    summarize_image_file,
    text_features,
    vision_content_parts,
    vision_prompt,
)

SPEC = WorkflowSpec(
    name="maze-gaia-vision",
    source="gaia",
    kind="vision",
    nodes=nodes(
        (
            ("task1_obtain_content", "cpu", None),
            ("task2_vlm_process", "npu", "qwen2.5-vl-32b"),
            ("task3_output_final_answer", "io", None),
        )
    ),
    edges=edges(
        (
            ("task1_obtain_content", "task2_vlm_process"),
            ("task2_vlm_process", "task3_output_final_answer"),
        )
    ),
)

INPUTS = spec_inputs()


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 1024})
def task1_obtain_content(
    dag_id: str,
    question: str,
    answer: str = "",
    supplementary_files: object = None,
    metadata: object = None,
) -> dict[str, object]:
    if not question:
        raise ValueError(f"task {dag_id} missing Question")
    normalized_metadata = metadata_dict(metadata)
    image_summary = summarize_image_file(supplementary_files)
    prompt = vision_prompt(question, image_summary["image_features"])
    features = text_features(prompt)
    return {
        "dag_id": dag_id,
        "question": question,
        "answer": answer,
        "metadata": normalized_metadata,
        "file_name": image_summary["file_name"],
        "image_bytes": image_summary["image_bytes"],
        "image_features": image_summary["image_features"],
        "succ_task_feat": {"task2_vlm_process": features},
    }


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 512}, max_retries=0)
def task2_vlm_process(
    dag_id: str,
    question: str,
    image_bytes: bytes,
    image_features: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    from ascend_maze.inference import chat

    prompt = vision_prompt(question, image_features)
    response = chat(
        [
            {
                "role": "user",
                "content": vision_content_parts(
                    question,
                    image_bytes,
                    image_features,
                ),
            }
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    vlm_answer = response_or_override(metadata, "vlm_output_override", response)
    return {
        "dag_id": dag_id,
        "vlm_answer": vlm_answer,
        "raw_model_output": response.text,
        "curr_task_feat": {
            **inference_features(prompt, response),
            "vision_input_mode": "true_multimodal" if image_bytes else "text_only",
        },
    }


@task(task_kind="io", resources={"cpu_num": 1, "mem": 1024, "io_num": 1})
def task3_output_final_answer(
    dag_id: str,
    vlm_answer: str,
) -> dict[str, object]:
    return {
        "dag_id": dag_id,
        "final_answer": vlm_answer,
    }


def build() -> Workflow:
    workflow = Workflow(SPEC.name)
    dag_id = workflow.input("dag_id")
    question = workflow.input("question")
    answer = workflow.input("answer")
    supplementary_files = workflow.input("supplementary_files")
    metadata = workflow.input("metadata")

    prepared = workflow.add_task(
        task1_obtain_content,
        task_name="task1_obtain_content",
        inputs={
            "dag_id": dag_id,
            "question": question,
            "answer": answer,
            "supplementary_files": supplementary_files,
            "metadata": metadata,
        },
    )
    answered = workflow.add_task(
        task2_vlm_process,
        task_name="task2_vlm_process",
        model_anchor={"model": "qwen2.5-vl-32b", "mode": "service"},
        inputs={
            "dag_id": prepared.outputs["dag_id"],
            "question": prepared.outputs["question"],
            "image_bytes": prepared.outputs["image_bytes"],
            "image_features": prepared.outputs["image_features"],
            "metadata": prepared.outputs["metadata"],
        },
    )
    workflow.add_task(
        task3_output_final_answer,
        task_name="task3_output_final_answer",
        inputs={
            "dag_id": answered.outputs["dag_id"],
            "vlm_answer": answered.outputs["vlm_answer"],
        },
    )
    return workflow
