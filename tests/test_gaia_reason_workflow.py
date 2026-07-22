from __future__ import annotations

from gaia_workflow_helpers import run_gaia_workflow
from workflows.gaia import reason as gaia_reason


def test_gaia_reason_workflow_runs_with_fake_inference(tmp_path) -> None:
    results, invoke_count, request_count = run_gaia_workflow(
        gaia_reason,
        tmp_path=tmp_path,
        submission_id="gaia_reason_smoke",
        model_ids=("qwen3-32b", "deepseek-r1-32b"),
        inputs={
            "dag_id": "gaia_reason_smoke",
            "question": "If Alice has two apples and buys three more, how many?",
            "answer": "5",
            "supplementary_files": {},
            "metadata": {
                "qwen_output_override": "FINAL ANSWER: 5",
                "deepseek_output_override": "FINAL ANSWER: 5",
                "fuse_output_override": "FINAL ANSWER: 5",
            },
        },
    )

    assert results["task1_obtain_content"]["prompt_context"].startswith("If Alice")
    final = results["task4_llm_fuse_answer"]
    assert final["final_answer"] == "FINAL ANSWER: 5"
    assert invoke_count == 3
    assert request_count == 3
