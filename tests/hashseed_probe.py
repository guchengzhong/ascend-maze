"""Emit deterministic compiled identity for PYTHONHASHSEED tests."""

from __future__ import annotations

from base64 import b64encode
import json

from ascend_maze import Workflow
from ascend_maze.contracts.submission import (
    RunInputIdentity,
    SubmissionContract,
    SubmissionOptions,
    hash_session_key,
)
from task_fixtures import barrier, finish, load_text, summarize


def main() -> None:
    workflow = Workflow("determinism-probe")
    path = workflow.input("path")
    loaded = workflow.add_task(load_text, inputs={"path": path})
    processed = workflow.add_task(
        summarize,
        inputs={
            "text": loaded.outputs["text"],
            "options": {"upper": True, "buckets": {3, 1, 2}},
        },
    )
    gate = workflow.add_task(barrier)
    result = workflow.add_task(
        finish,
        inputs={"summary": processed.outputs["summary"]},
    )
    workflow.add_edge(gate, result)
    compiled = workflow.compile()
    submission = SubmissionContract.create(
        submission_id="submission_determinism_probe",
        workflow_fingerprint=compiled.workflow_fingerprint,
        input_identities=(
            RunInputIdentity.from_small_value(
                "path",
                {"parts": {"b", "a"}, "path": "/shared/input.txt"},
            ),
        ),
        session_key_hash=hash_session_key("determinism-session"),
        options=SubmissionOptions(
            run_deadline_ms=60_000,
            execution_options={"labels": {"beta", "alpha"}},
        ),
        config_fingerprint="a" * 64,
    )
    print(
        json.dumps(
            {
                "canonical_ir": b64encode(compiled.canonical_ir_bytes).decode("ascii"),
                "task_ids": list(compiled.tasks),
                "topological_order": compiled.topological_order,
                "predecessors": list(compiled.predecessors.items()),
                "successors": list(compiled.successors.items()),
                "workflow_fingerprint": compiled.workflow_fingerprint,
                "submission_payload_hash": submission.submission_payload_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
