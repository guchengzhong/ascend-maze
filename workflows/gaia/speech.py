"""Ascend-Maze-native GAIA speech workflow."""

from __future__ import annotations

from ascend_maze import Workflow, task

from workflows._common import WorkflowSpec, edges, nodes, spec_inputs
from workflows.gaia._common import (
    gaia_deepseek_prompt,
    gaia_fusion_prompt,
    gaia_question_prompt,
    inference_features,
    metadata_dict,
    response_or_override,
    speech_transcription_prompt,
    summarize_audio_file,
    text_feature_for_answer,
    text_features,
)

SPEC = WorkflowSpec(
    name="maze-gaia-speech",
    source="gaia",
    kind="speech",
    nodes=nodes(
        (
            ("task1_speech_process", "cpu", None),
            ("task2_speech_process", "npu", "whisper-large-v3"),
            ("task3_llm_process_qwen", "npu", "qwen3-32b"),
            ("task4_llm_process_deepseek", "npu", "deepseek-r1-32b"),
            ("task5_llm_fuse_answer", "npu", "qwen3-32b"),
        )
    ),
    edges=edges(
        (
            ("task1_speech_process", "task2_speech_process"),
            ("task2_speech_process", "task3_llm_process_qwen"),
            ("task2_speech_process", "task4_llm_process_deepseek"),
            ("task3_llm_process_qwen", "task5_llm_fuse_answer"),
            ("task4_llm_process_deepseek", "task5_llm_fuse_answer"),
        )
    ),
)

INPUTS = spec_inputs()


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 1024})
def task1_speech_process(
    dag_id: str,
    question: str,
    answer: str = "",
    supplementary_files: object = None,
    metadata: object = None,
) -> dict[str, object]:
    if not question:
        raise ValueError(f"task {dag_id} missing Question")
    normalized_metadata = metadata_dict(metadata)
    audio_summary = summarize_audio_file(supplementary_files)
    return {
        "dag_id": dag_id,
        "question": question,
        "answer": answer,
        "metadata": normalized_metadata,
        "file_name": audio_summary["file_name"],
        "audio_bytes": audio_summary["audio_bytes"],
        "audio_features": audio_summary["audio_features"],
        "succ_task_feat": audio_summary["audio_features"],
    }


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 512}, max_retries=0)
def task2_speech_process(
    dag_id: str,
    question: str,
    audio_bytes: bytes,
    audio_features: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    from ascend_maze.inference import chat

    del audio_bytes
    prompt = speech_transcription_prompt(question, audio_features)
    response = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0,
    )
    processed_content = response_or_override(metadata, "transcript_override", response)
    next_prompt = gaia_question_prompt(
        question,
        "Extracted text from speech",
        processed_content,
    )
    features = text_features(next_prompt)
    return {
        "dag_id": dag_id,
        "question": question,
        "metadata": metadata,
        "processed_content": processed_content,
        "audio_features": audio_features,
        "curr_task_feat": audio_features,
        "raw_model_output": response.text,
        "succ_task_feat": {
            "task3_llm_process_qwen": {
                "text_length": features["text_length"],
                "token_count": features["token_count"],
                "reason": 1,
            },
            "task4_llm_process_deepseek": {
                "text_length": features["text_length"],
                "token_count": features["token_count"],
                "reason": 0,
            },
        },
    }


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 512}, max_retries=0)
def task3_llm_process_qwen(
    dag_id: str,
    question: str,
    processed_content: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    from ascend_maze.inference import chat

    prompt = gaia_question_prompt(
        question,
        "Extracted text from speech",
        processed_content,
    )
    response = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0,
    )
    qwen_answer = response_or_override(metadata, "qwen_output_override", response)
    text1_feature = text_feature_for_answer("text1", qwen_answer)
    next_prompt = gaia_fusion_prompt(question, qwen_answer, "")
    return {
        "dag_id": dag_id,
        "question": question,
        "metadata": metadata,
        "qwen_answer": qwen_answer,
        "raw_model_output": response.text,
        "text1_feature": text1_feature,
        "curr_task_feat": inference_features(prompt, response, reason=0),
        "succ_task_feat": {
            "task5_llm_fuse_answer": {
                "prompt_length": len(next_prompt),
                "prompt_token_count": text_features(next_prompt)["token_count"],
                "text1_length": text1_feature["text1_length"],
                "text1_token_count": text1_feature["text1_token_count"],
                "reason": 0,
            }
        },
    }


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 512}, max_retries=0)
def task4_llm_process_deepseek(
    dag_id: str,
    question: str,
    processed_content: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    from ascend_maze.inference import chat

    prompt = gaia_deepseek_prompt(
        question,
        "Extracted text from speech",
        processed_content,
    )
    response = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0,
    )
    deepseek_answer = response_or_override(
        metadata,
        "deepseek_output_override",
        response,
    )
    text2_feature = text_feature_for_answer("text2", deepseek_answer)
    next_prompt = gaia_fusion_prompt(question, "", deepseek_answer)
    return {
        "dag_id": dag_id,
        "question": question,
        "metadata": metadata,
        "deepseek_answer": deepseek_answer,
        "raw_model_output": response.text,
        "text2_feature": text2_feature,
        "curr_task_feat": inference_features(prompt, response, reason=1),
        "succ_task_feat": {
            "task5_llm_fuse_answer": {
                "prompt_length": len(next_prompt),
                "prompt_token_count": text_features(next_prompt)["token_count"],
                "text2_length": text2_feature["text2_length"],
                "text2_token_count": text2_feature["text2_token_count"],
                "reason": 0,
            }
        },
    }


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 512}, max_retries=0)
def task5_llm_fuse_answer(
    dag_id: str,
    question: str,
    qwen_answer: str,
    deepseek_answer: str,
    text1_feature: dict[str, object],
    text2_feature: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    from ascend_maze.inference import chat

    prompt = gaia_fusion_prompt(question, qwen_answer, deepseek_answer)
    response = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0,
    )
    final_answer = response_or_override(metadata, "fuse_output_override", response)
    merged_features = inference_features(prompt, response, reason=0)
    return {
        "dag_id": dag_id,
        "final_answer": final_answer,
        "raw_model_output": response.text,
        "curr_task_feat": {
            "prompt_length": merged_features["text_length"],
            "prompt_token_count": merged_features["token_count"],
            "text1_length": text1_feature["text1_length"],
            "text1_token_count": text1_feature["text1_token_count"],
            "text2_length": text2_feature["text2_length"],
            "text2_token_count": text2_feature["text2_token_count"],
            "reason": 0,
        },
    }


def build() -> Workflow:
    workflow = Workflow(SPEC.name)
    dag_id = workflow.input("dag_id")
    question = workflow.input("question")
    answer = workflow.input("answer")
    supplementary_files = workflow.input("supplementary_files")
    metadata = workflow.input("metadata")

    prepared = workflow.add_task(
        task1_speech_process,
        task_name="task1_speech_process",
        inputs={
            "dag_id": dag_id,
            "question": question,
            "answer": answer,
            "supplementary_files": supplementary_files,
            "metadata": metadata,
        },
    )
    transcribed = workflow.add_task(
        task2_speech_process,
        task_name="task2_speech_process",
        model_anchor={"model": "whisper-large-v3", "mode": "service"},
        inputs={
            "dag_id": prepared.outputs["dag_id"],
            "question": prepared.outputs["question"],
            "audio_bytes": prepared.outputs["audio_bytes"],
            "audio_features": prepared.outputs["audio_features"],
            "metadata": prepared.outputs["metadata"],
        },
    )
    qwen = workflow.add_task(
        task3_llm_process_qwen,
        task_name="task3_llm_process_qwen",
        model_anchor={"model": "qwen3-32b", "mode": "service"},
        inputs={
            "dag_id": transcribed.outputs["dag_id"],
            "question": transcribed.outputs["question"],
            "processed_content": transcribed.outputs["processed_content"],
            "metadata": transcribed.outputs["metadata"],
        },
    )
    deepseek = workflow.add_task(
        task4_llm_process_deepseek,
        task_name="task4_llm_process_deepseek",
        model_anchor={"model": "deepseek-r1-32b", "mode": "service"},
        inputs={
            "dag_id": transcribed.outputs["dag_id"],
            "question": transcribed.outputs["question"],
            "processed_content": transcribed.outputs["processed_content"],
            "metadata": transcribed.outputs["metadata"],
        },
    )
    workflow.add_task(
        task5_llm_fuse_answer,
        task_name="task5_llm_fuse_answer",
        model_anchor={"model": "qwen3-32b", "mode": "service"},
        inputs={
            "dag_id": qwen.outputs["dag_id"],
            "question": qwen.outputs["question"],
            "qwen_answer": qwen.outputs["qwen_answer"],
            "deepseek_answer": deepseek.outputs["deepseek_answer"],
            "text1_feature": qwen.outputs["text1_feature"],
            "text2_feature": deepseek.outputs["text2_feature"],
            "metadata": qwen.outputs["metadata"],
        },
    )
    return workflow
