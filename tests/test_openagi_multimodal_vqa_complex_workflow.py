from __future__ import annotations

import base64

from openagi_workflow_helpers import run_openagi_workflow
from workflows.openagi import multimodal_vqa_complex


def _images() -> dict[str, bytes]:
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
        "/6X+XioAAAAASUVORK5CYII="
    )
    return {f"vqa_{index}.png": one_pixel_png for index in range(1, 6)}


def test_openagi_multimodal_vqa_complex_workflow_runs_with_fake_inference(
    tmp_path,
) -> None:
    results, invoke_count, request_count = run_openagi_workflow(
        multimodal_vqa_complex,
        tmp_path=tmp_path,
        submission_id="openagi_multimodal_vqa_smoke",
        model_ids=("qwen2.5-vl-32b",),
        inputs={
            "dag_id": "openagi_multimodal_vqa_smoke",
            "question": "What color is the tiny image?",
            "answer": "",
            "supplementary_files": _images(),
            "metadata": {
                "vqa_a_overrides": ["answer one"],
                "vqa_b_overrides": ["answer two"],
                "vqa_c_overrides": ["answer three"],
                "vqa_d_overrides": ["answer four", "answer five"],
            },
        },
    )

    merged = results["task4_merge_results"]
    assert len(merged["final_answers"]) == 5
    assert results["task4a_vlm_process"]["curr_task_feat"]["vision_input_mode"] == "true_multimodal"  # type: ignore[index]
    assert results["task4a_vlm_process"]["curr_task_feat"]["true_multimodal_count"] == 1  # type: ignore[index]
    assert results["task4d_vlm_process"]["curr_task_feat"]["vision_input_mode"] == "true_multimodal"  # type: ignore[index]
    assert results["task4d_vlm_process"]["curr_task_feat"]["true_multimodal_count"] == 2  # type: ignore[index]
    final = results["task5_output_final_answer"]
    assert "Answer for vqa_1.png:\nanswer one" in final["final_answer"]
    assert "Answer for vqa_5.png:\nanswer five" in final["final_answer"]
    assert invoke_count == 5
    assert request_count == 5
