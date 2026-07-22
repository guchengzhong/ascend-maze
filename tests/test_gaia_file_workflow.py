from __future__ import annotations

from hashlib import sha256

from ascend_maze.contracts.data import SharedFileRef
from gaia_workflow_helpers import run_gaia_workflow
from workflows.gaia import file as gaia_file


def test_gaia_file_workflow_runs_with_shared_file_and_fake_inference(tmp_path) -> None:
    document = tmp_path / "gaia-note.txt"
    content = b"The spreadsheet note says the final city is Kyoto."
    document.write_bytes(content)
    file_ref = SharedFileRef(str(document), sha256(content).hexdigest(), len(content))

    results, invoke_count, request_count = run_gaia_workflow(
        gaia_file,
        tmp_path=tmp_path,
        submission_id="gaia_file_smoke",
        model_ids=("qwen3-32b", "deepseek-r1-32b"),
        shared_filesystem_roots=(str(tmp_path),),
        inputs={
            "dag_id": "gaia_file_smoke",
            "question": "Which city is mentioned in the note?",
            "answer": "Kyoto",
            "supplementary_files": file_ref,
            "metadata": {
                "qwen_output_override": "Qwen says FINAL ANSWER: Kyoto",
                "deepseek_output_override": "DeepSeek says FINAL ANSWER: Kyoto",
                "fuse_output_override": "FINAL ANSWER: Kyoto",
            },
        },
    )

    prepared = results["task1_file_process"]
    assert prepared["file_info"]["source_kind"] == "shared_file"  # type: ignore[index]
    assert "Kyoto" in prepared["processed_content"]
    final = results["task4_llm_fuse_answer"]
    assert final["final_answer"] == "FINAL ANSWER: Kyoto"
    assert invoke_count == 3
    assert request_count == 3
