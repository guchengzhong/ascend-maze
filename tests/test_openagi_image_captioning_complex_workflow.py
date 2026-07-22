from __future__ import annotations

import base64

from openagi_workflow_helpers import run_openagi_workflow
from workflows.openagi import image_captioning_complex


def _images() -> dict[str, bytes]:
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
        "/6X+XioAAAAASUVORK5CYII="
    )
    return {f"image_{index}.png": one_pixel_png for index in range(1, 6)}


def test_openagi_image_captioning_complex_workflow_runs_with_fake_inference(
    tmp_path,
) -> None:
    results, invoke_count, request_count = run_openagi_workflow(
        image_captioning_complex,
        tmp_path=tmp_path,
        submission_id="openagi_image_caption_smoke",
        model_ids=("blip-image-captioning", "easyocr", "qwen2.5-vl-32b"),
        inputs={
            "dag_id": "openagi_image_caption_smoke",
            "question": "Describe these images in English.",
            "answer": "",
            "supplementary_files": _images(),
            "metadata": {
                "blip_caption_overrides": [
                    "caption one",
                    "caption two",
                    "caption three",
                    "caption four",
                    "caption five",
                ],
                "ocr_text_overrides": ["", "", "", "", ""],
                "description_a_overrides": ["description one"],
                "description_b_overrides": ["description two"],
                "description_c_overrides": ["description three"],
                "description_d_overrides": ["description four", "description five"],
            },
        },
    )

    merged = results["task5_merge_results"]
    assert len(merged["final_descriptions"]) == 5
    assert results["task3a_extract_blip_captions"]["curr_task_feat"]["vision_input_mode"] == "true_multimodal"  # type: ignore[index]
    assert results["task3a_extract_blip_captions"]["curr_task_feat"]["true_multimodal_count"] == 5  # type: ignore[index]
    assert results["task3b_extract_ocr_text"]["curr_task_feat"]["vision_input_mode"] == "true_multimodal"  # type: ignore[index]
    assert results["task5a_vlm_process"]["curr_task_feat"]["true_multimodal_count"] == 1  # type: ignore[index]
    assert results["task5d_vlm_process"]["curr_task_feat"]["true_multimodal_count"] == 2  # type: ignore[index]
    final = results["task6_output_final_answer"]
    assert "Image image_1.png: description one" in final["final_answer"]
    assert "Image image_5.png: description five" in final["final_answer"]
    assert invoke_count == 15
    assert request_count == 15
