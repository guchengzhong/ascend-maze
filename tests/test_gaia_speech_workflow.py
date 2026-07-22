from __future__ import annotations

from io import BytesIO
import wave

from gaia_workflow_helpers import run_gaia_workflow
from workflows.gaia import speech as gaia_speech


def _tiny_wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


def test_gaia_speech_workflow_runs_with_fake_inference(tmp_path) -> None:
    results, invoke_count, request_count = run_gaia_workflow(
        gaia_speech,
        tmp_path=tmp_path,
        submission_id="gaia_speech_smoke",
        model_ids=("whisper-large-v3", "qwen3-32b", "deepseek-r1-32b"),
        inputs={
            "dag_id": "gaia_speech_smoke",
            "question": "What color does the speaker mention?",
            "answer": "blue",
            "supplementary_files": {"clip.wav": _tiny_wav_bytes()},
            "metadata": {
                "transcript_override": "The speaker says blue.",
                "qwen_output_override": "FINAL ANSWER: blue",
                "deepseek_output_override": "FINAL ANSWER: blue",
                "fuse_output_override": "FINAL ANSWER: blue",
            },
        },
    )

    prepared = results["task1_speech_process"]
    assert prepared["audio_features"]["sample_rate"] == 16_000  # type: ignore[index]
    transcribed = results["task2_speech_process"]
    assert transcribed["processed_content"] == "The speaker says blue."
    final = results["task5_llm_fuse_answer"]
    assert final["final_answer"] == "FINAL ANSWER: blue"
    assert invoke_count == 4
    assert request_count == 4
