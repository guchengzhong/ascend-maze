from __future__ import annotations

from openagi_workflow_helpers import run_openagi_workflow
from workflows.openagi import document_qa


def test_openagi_document_qa_workflow_runs_with_fake_inference(tmp_path) -> None:
    questions = "\n".join(
        [
            "What is the first fact?",
            "What is the second fact?",
            "What is the third fact?",
            "What is the fourth fact?",
            "What is the fifth fact?",
        ]
    )
    results, invoke_count, request_count = run_openagi_workflow(
        document_qa,
        tmp_path=tmp_path,
        submission_id="openagi_document_qa_smoke",
        model_ids=("qwen3-32b",),
        inputs={
            "dag_id": "openagi_document_qa_smoke",
            "question": questions,
            "answer": "",
            "supplementary_files": {
                "context.txt": (
                    b"Fact one is alpha.\nFact two is beta.\nFact three is gamma.\n"
                    b"Fact four is delta.\nFact five is epsilon."
                )
            },
            "metadata": {
                "structure_output_override": "five short fact lines",
                "batch1_output_overrides": ["alpha"],
                "batch2_output_overrides": ["beta"],
                "batch3_output_overrides": ["gamma", "delta", "epsilon"],
            },
        },
    )

    read_file = results["task2_read_file"]
    assert "Fact one" in read_file["document_content"]
    prepared = results["task4b_prepare_qa_context"]
    assert len(prepared["qa_context"]["question_batches"]) == 3  # type: ignore[index]
    final = results["task8_output_final_answer"]
    assert final["final_answer"] == "alpha\nbeta\ngamma\ndelta\nepsilon"
    assert invoke_count == 6
    assert request_count == 6
