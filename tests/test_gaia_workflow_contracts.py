from __future__ import annotations

from importlib import import_module
import inspect
from pathlib import Path

from workflows.gaia._common import model_runtime_inputs, process_document_file


GAIA_MODULES = (
    "workflows.gaia.file",
    "workflows.gaia.reason",
    "workflows.gaia.speech",
    "workflows.gaia.vision",
)


EXPECTED_OUTPUTS_BY_TASK = {
    "workflows.gaia.file": {
        "task1_file_process": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "processed_content",
            "start_time",
            "succ_task_feat",
            "time_record",
        ),
        "task2_llm_process_qwen": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "qwen_answer",
            "start_time",
            "succ_task_feat",
            "text1_feature",
            "time_record",
        ),
        "task3_llm_process_deepseek": (
            "curr_task_feat",
            "dag_id",
            "deepseek_answer",
            "end_time",
            "start_time",
            "succ_task_feat",
            "text2_feature",
            "time_record",
        ),
        "task4_llm_fuse_answer": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "final_answer",
            "start_time",
            "time_record",
        ),
    },
    "workflows.gaia.reason": {
        "task1_obtain_content": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "start_time",
            "succ_task_feat",
            "time_record",
        ),
        "task2_llm_process_qwen": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "qwen_answer",
            "start_time",
            "succ_task_feat",
            "text1_feature",
            "time_record",
        ),
        "task3_llm_process_deepseek": (
            "curr_task_feat",
            "dag_id",
            "deepseek_answer",
            "end_time",
            "start_time",
            "succ_task_feat",
            "text2_feature",
            "time_record",
        ),
        "task4_llm_fuse_answer": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "final_answer",
            "start_time",
            "time_record",
        ),
    },
    "workflows.gaia.speech": {
        "task1_speech_process": (
            "audio_features",
            "end_time",
            "file_content",
            "start_time",
            "succ_task_feat",
            "time_record",
        ),
        "task2_speech_process": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "processed_content",
            "start_time",
            "succ_task_feat",
            "time_record",
        ),
        "task3_llm_process_qwen": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "qwen_answer",
            "start_time",
            "succ_task_feat",
            "text1_feature",
            "time_record",
        ),
        "task4_llm_process_deepseek": (
            "curr_task_feat",
            "dag_id",
            "deepseek_answer",
            "end_time",
            "start_time",
            "succ_task_feat",
            "text2_feature",
            "time_record",
        ),
        "task5_llm_fuse_answer": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "final_answer",
            "start_time",
            "time_record",
        ),
    },
    "workflows.gaia.vision": {
        "task1_obtain_content": (
            "curr_task_feat",
            "dag_id",
            "end_time",
            "file_content",
            "start_time",
            "succ_task_feat",
            "task2_vlm_process_feature",
            "time_record",
        ),
        "task2_vlm_process": (
            "curr_task_feat",
            "end_time",
            "start_time",
            "task_id",
            "time_record",
            "vlm_answer",
        ),
        "task3_output_final_answer": (
            "dag_id",
            "end_time",
            "final_answer",
            "start_time",
            "time_record",
        ),
    },
}


EXPECTED_INPUTS_BY_TASK = {
    "workflows.gaia.file": {
        "task1_file_process": ("dag_id", "question", "supplementary_files"),
        "task2_llm_process_qwen": (
            "dag_id",
            "processed_content",
            "question",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task2_llm_process_qwen_request_api_url",
        ),
        "task3_llm_process_deepseek": (
            "dag_id",
            "processed_content",
            "question",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task3_llm_process_deepseek_request_api_url",
        ),
        "task4_llm_fuse_answer": (
            "qwen_answer",
            "deepseek_answer",
            "dag_id",
            "question",
            "text1_feature",
            "text2_feature",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task4_llm_fuse_answer_request_api_url",
        ),
    },
    "workflows.gaia.reason": {
        "task1_obtain_content": ("dag_id", "question"),
        "task2_llm_process_qwen": (
            "dag_id",
            "question",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task2_llm_process_qwen_request_api_url",
        ),
        "task3_llm_process_deepseek": (
            "dag_id",
            "question",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task3_llm_process_deepseek_request_api_url",
        ),
        "task4_llm_fuse_answer": (
            "qwen_answer",
            "deepseek_answer",
            "dag_id",
            "question",
            "text1_feature",
            "text2_feature",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task4_llm_fuse_answer_request_api_url",
        ),
    },
    "workflows.gaia.speech": {
        "task1_speech_process": ("supplementary_files",),
        "task2_speech_process": (
            "dag_id",
            "question",
            "file_content",
            "model_folder",
            "audio_features",
        ),
        "task3_llm_process_qwen": (
            "dag_id",
            "processed_content",
            "question",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task3_llm_process_qwen_request_api_url",
        ),
        "task4_llm_process_deepseek": (
            "dag_id",
            "processed_content",
            "question",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task4_llm_process_deepseek_request_api_url",
        ),
        "task5_llm_fuse_answer": (
            "qwen_answer",
            "deepseek_answer",
            "dag_id",
            "question",
            "text1_feature",
            "text2_feature",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task5_llm_fuse_answer_request_api_url",
        ),
    },
    "workflows.gaia.vision": {
        "task1_obtain_content": ("dag_id", "question", "supplementary_files"),
        "task2_vlm_process": (
            "dag_id",
            "question",
            "file_content",
            "task2_vlm_process_feature",
            "use_online_model",
            "model_folder",
            "temperature",
            "max_tokens",
            "top_p",
            "repetition_penalty",
            "task2_vlm_process_request_api_url",
        ),
        "task3_output_final_answer": ("dag_id", "vlm_answer"),
    },
}


def test_gaia_model_runtime_defaults_to_4096_output_tokens() -> None:
    assert model_runtime_inputs("request_api_url")["max_tokens"] == 4096


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


def test_gaia_task_parameters_match_original_context_get_fields() -> None:
    for module_name, expected_by_task in EXPECTED_INPUTS_BY_TASK.items():
        module = import_module(module_name)
        for task_name, expected in expected_by_task.items():
            parameters = inspect.signature(getattr(module, task_name)).parameters
            assert tuple(parameters) == expected


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
