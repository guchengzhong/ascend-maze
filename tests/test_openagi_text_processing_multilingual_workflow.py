from __future__ import annotations

from openagi_workflow_helpers import run_openagi_workflow
from workflows.openagi import text_processing_multilingual


def test_openagi_text_processing_multilingual_workflow_runs_with_fake_inference(
    tmp_path,
) -> None:
    questions = "\n".join(
        [
            "Please answer in German: item one?",
            "Please answer in German: item two?",
            "Please answer in German: item three?",
            "Please answer in German: item four?",
            "Please answer in German: item five?",
        ]
    )
    results, invoke_count, request_count = run_openagi_workflow(
        text_processing_multilingual,
        tmp_path=tmp_path,
        submission_id="openagi_text_processing_smoke",
        model_ids=("qwen3-32b",),
        inputs={
            "dag_id": "openagi_text_processing_smoke",
            "question": questions,
            "answer": "",
            "supplementary_files": {
                "text.txt": b"Alpha is first. Beta is second. Gamma is third."
            },
            "metadata": {
                "translation_output_override": "Alpha ist zuerst. Beta ist zweite.",
                "summary_output_override": "short summary",
                "sentiment_output_override": "positive",
                "text_batch1_output_overrides": ["eins", "zwei"],
                "text_batch2_output_overrides": ["drei", "vier"],
                "text_batch3_output_overrides": ["funf"],
            },
        },
    )

    detected = results["task3_language_detect"]
    assert detected["source_language"] == "en"
    translated = results["task4_translate_text"]
    assert translated["target_language"] == "de"
    final = results["task9_output_final_answer"]
    assert final["final_answer"] == "eins\nzwei\ndrei\nvier\nfunf"
    assert invoke_count == 8
    assert request_count == 8
