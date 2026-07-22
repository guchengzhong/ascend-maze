from __future__ import annotations

import base64

from gaia_workflow_helpers import run_gaia_workflow
from workflows.gaia import vision as gaia_vision


def test_gaia_vision_workflow_runs_with_fake_inference(tmp_path) -> None:
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
        "/6X+XioAAAAASUVORK5CYII="
    )
    results, invoke_count, request_count = run_gaia_workflow(
        gaia_vision,
        tmp_path=tmp_path,
        submission_id="gaia_vision_smoke",
        model_ids=("qwen2.5-vl-32b",),
        inputs={
            "dag_id": "gaia_vision_smoke",
            "question": "What is shown in the tiny image?",
            "answer": "pixel",
            "supplementary_files": {"pixel.png": one_pixel_png},
            "metadata": {
                "vlm_output_override": "FINAL ANSWER: pixel",
            },
        },
    )

    prepared = results["task1_obtain_content"]
    assert prepared["image_features"]["size_bytes"] == len(one_pixel_png)  # type: ignore[index]
    vlm = results["task2_vlm_process"]
    assert "[1 image(s)]" in str(vlm["raw_model_output"])
    assert vlm["curr_task_feat"]["vision_input_mode"] == "true_multimodal"  # type: ignore[index]
    final = results["task3_output_final_answer"]
    assert final["final_answer"] == "FINAL ANSWER: pixel"
    assert invoke_count == 1
    assert request_count == 1
