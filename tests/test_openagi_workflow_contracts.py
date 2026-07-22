from __future__ import annotations

from importlib import import_module
from pathlib import Path

from workflows.openagi._common import list_inline_images, read_named_text_file


OPENAGI_MODULES = (
    "workflows.openagi.document_qa",
    "workflows.openagi.image_captioning_complex",
    "workflows.openagi.multimodal_vqa_complex",
    "workflows.openagi.text_processing_multilingual",
)


def test_openagi_workflow_static_output_keys_are_explicit() -> None:
    for module_name in OPENAGI_MODULES:
        module = import_module(module_name)
        compiled = module.build().compile()
        task_by_name = {
            task.task_name: task
            for _, task in compiled.tasks.items_tuple()
        }
        for task_name, task_node in task_by_name.items():
            definition = compiled.definitions[task_node.definition_id]
            assert definition.output_names
            assert definition.output_names != ("state",)
            if task_name.endswith("output_final_answer"):
                assert "final_answer" in definition.output_names


def test_openagi_workflows_do_not_use_legacy_maze_runtime_patterns() -> None:
    forbidden = (
        "context.get",
        "context.put",
        "import ray",
        "agentos",
        "@gpu",
        "@cpu",
        "@io",
        "RemoteLlmRoute",
        "build_workflow(",
    )
    openagi_root = Path(__file__).resolve().parents[1] / "workflows" / "openagi"
    for source_file in sorted(openagi_root.glob("*.py")):
        text = source_file.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text, f"{source_file} contains {pattern!r}"


def test_openagi_plain_string_file_payload_is_literal_content(tmp_path) -> None:
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("this should not be read as file content", encoding="utf-8")

    processed = read_named_text_file({"context.txt": str(secret_file)}, "context.txt")

    assert processed["content"] == str(secret_file)
    assert "this should not be read" not in processed["content"]


def test_openagi_plain_string_image_payload_is_inline_bytes(tmp_path) -> None:
    secret_image = tmp_path / "secret.png"
    secret_image.write_bytes(b"real image bytes should not be read")

    images = list_inline_images({"image.png": str(secret_image)})

    assert len(images) == 1
    assert images[0]["content"] == str(secret_image).encode("utf-8")
    assert images[0]["content"] != secret_image.read_bytes()
