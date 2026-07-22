from __future__ import annotations

from importlib import import_module
from pathlib import Path

from workflows.gaia._common import process_document_file


GAIA_MODULES = (
    "workflows.gaia.file",
    "workflows.gaia.reason",
    "workflows.gaia.speech",
    "workflows.gaia.vision",
)


EXPECTED_OUTPUTS_BY_TASK = {
    "workflows.gaia.file": {
        "task1_file_process": (
            "answer",
            "dag_id",
            "file_info",
            "metadata",
            "processed_content",
            "question",
            "succ_task_feat",
        ),
        "task2_llm_process_qwen": (
            "curr_task_feat",
            "dag_id",
            "metadata",
            "question",
            "qwen_answer",
            "raw_model_output",
            "succ_task_feat",
            "text1_feature",
        ),
        "task3_llm_process_deepseek": (
            "curr_task_feat",
            "dag_id",
            "deepseek_answer",
            "metadata",
            "question",
            "raw_model_output",
            "succ_task_feat",
            "text2_feature",
        ),
        "task4_llm_fuse_answer": (
            "curr_task_feat",
            "dag_id",
            "final_answer",
            "raw_model_output",
        ),
    },
    "workflows.gaia.reason": {
        "task1_obtain_content": (
            "answer",
            "dag_id",
            "metadata",
            "prompt_context",
            "question",
            "succ_task_feat",
        ),
        "task2_llm_process_qwen": (
            "curr_task_feat",
            "dag_id",
            "metadata",
            "question",
            "qwen_answer",
            "raw_model_output",
            "succ_task_feat",
            "text1_feature",
        ),
        "task3_llm_process_deepseek": (
            "curr_task_feat",
            "dag_id",
            "deepseek_answer",
            "metadata",
            "question",
            "raw_model_output",
            "succ_task_feat",
            "text2_feature",
        ),
        "task4_llm_fuse_answer": (
            "curr_task_feat",
            "dag_id",
            "final_answer",
            "raw_model_output",
        ),
    },
    "workflows.gaia.speech": {
        "task1_speech_process": (
            "answer",
            "audio_bytes",
            "audio_features",
            "dag_id",
            "file_name",
            "metadata",
            "question",
            "succ_task_feat",
        ),
        "task2_speech_process": (
            "audio_features",
            "curr_task_feat",
            "dag_id",
            "metadata",
            "processed_content",
            "question",
            "raw_model_output",
            "succ_task_feat",
        ),
        "task3_llm_process_qwen": (
            "curr_task_feat",
            "dag_id",
            "metadata",
            "question",
            "qwen_answer",
            "raw_model_output",
            "succ_task_feat",
            "text1_feature",
        ),
        "task4_llm_process_deepseek": (
            "curr_task_feat",
            "dag_id",
            "deepseek_answer",
            "metadata",
            "question",
            "raw_model_output",
            "succ_task_feat",
            "text2_feature",
        ),
        "task5_llm_fuse_answer": (
            "curr_task_feat",
            "dag_id",
            "final_answer",
            "raw_model_output",
        ),
    },
    "workflows.gaia.vision": {
        "task1_obtain_content": (
            "answer",
            "dag_id",
            "file_name",
            "image_bytes",
            "image_features",
            "metadata",
            "question",
            "succ_task_feat",
        ),
        "task2_vlm_process": (
            "curr_task_feat",
            "dag_id",
            "raw_model_output",
            "vlm_answer",
        ),
        "task3_output_final_answer": (
            "dag_id",
            "final_answer",
        ),
    },
}


def test_gaia_workflow_static_output_keys_are_explicit() -> None:
    for module_name in GAIA_MODULES:
        module = import_module(module_name)
        compiled = module.build().compile()
        definitions_by_task_name = {
            task.task_name: compiled.definitions[task.definition_id]
            for _, task in compiled.tasks.items_tuple()
        }

        expected = EXPECTED_OUTPUTS_BY_TASK[module_name]
        assert set(definitions_by_task_name) == set(expected)
        for task_name, output_names in expected.items():
            assert definitions_by_task_name[task_name].output_names == output_names


def test_gaia_workflows_do_not_use_legacy_maze_runtime_patterns() -> None:
    forbidden = (
        "context.get",
        "context.put",
        "import ray",
        "agentos",
        "@gpu",
        "@cpu",
        "@io",
        "RemoteLlmRoute",
    )
    gaia_root = Path(__file__).resolve().parents[1] / "workflows" / "gaia"
    for source_file in sorted(gaia_root.glob("*.py")):
        text = source_file.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text, f"{source_file} contains {pattern!r}"


def test_gaia_workflows_do_not_guess_plain_string_paths() -> None:
    common_text = (
        Path(__file__).resolve().parents[1]
        / "workflows"
        / "gaia"
        / "_common.py"
    ).read_text(encoding="utf-8")

    assert "Path(payload)" not in common_text
    assert "open(payload" not in common_text
    assert "read_bytes()" in common_text
    assert "SharedFileRef" in common_text


def test_gaia_plain_string_file_payload_is_literal_content(tmp_path) -> None:
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("this should not be read as file content", encoding="utf-8")

    processed = process_document_file({"note.txt": str(secret_file)})

    assert processed["processed_content"] == str(secret_file)
    assert "this should not be read" not in processed["processed_content"]
