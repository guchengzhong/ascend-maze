from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from ascend_maze import Workflow
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.contracts.data import SharedFileRef, shared_file_from_handle
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.lifecycle import RunStatus
from ascend_maze.placement import NodeCapacity
from task_fixtures import finish, read_shared_file


def _controller() -> InMemoryController:
    return InMemoryController(
        config_fingerprint="c" * 64,
        environment_fingerprint="e" * 64,
        build_revision="test",
        node_capacities=(
            NodeCapacity(
                node_id="node_a",
                boot_id="boot_a",
                node_ip="127.0.0.1",
                cpu_total=2,
                mem_total_mb=512,
                cpu_system_reserved=0,
                mem_system_reserved_mb=0,
                io_slots_total=1,
                observed_free_mem_mb=512,
            ),
        ),
    )


def _file_ref(path: Path) -> SharedFileRef:
    content = path.read_bytes()
    return SharedFileRef(
        str(path.resolve()),
        hashlib.sha256(content).hexdigest(),
        len(content),
    )


def test_shared_file_uses_explicit_identity_and_head_validation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        shared_root = tmp_path / "shared"
        shared_root.mkdir()
        path = shared_root / "input.txt"
        path.write_text("shared payload", encoding="utf-8")
        file_ref = _file_ref(path)
        controller = _controller()
        await controller.start()
        client = InMemoryRuntimeClient(
            controller,
            shared_filesystem_roots=(str(shared_root),),
        )
        workflow = Workflow("shared-file")
        source = workflow.input("source")
        task = workflow.add_task(read_shared_file, inputs={"file_ref": source})
        prepared = client.prepare_submission(
            workflow,
            inputs={"source": file_ref},
            submission_id="shared_file_submission",
        )
        identity = prepared.request.contract.input_identities[0]
        assert identity.identity_kind == "shared_file"
        assert identity.identity == (
            file_ref.canonical_path,
            file_ref.content_sha256,
            str(file_ref.size_bytes),
        )
        handle = prepared.request.workflow_inputs[0][1]
        assert shared_file_from_handle(handle) == file_ref

        outcome = await client.submit_prepared(prepared)
        assert outcome.run_id is not None
        terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED
        assert controller.result(outcome.run_id, task.task_id) == {
            "content": "shared payload",
            "size": len(b"shared payload"),
        }
        await controller.destroy_run(outcome.run_id)

        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        with pytest.raises(ContractValidationError, match="outside"):
            client.prepare_submission(
                workflow,
                inputs={"source": _file_ref(outside)},
                submission_id="outside_shared_file",
            )
        bad_digest = SharedFileRef(
            file_ref.canonical_path,
            "0" * 64,
            file_ref.size_bytes,
        )
        with pytest.raises(ContractValidationError, match="content_sha256"):
            client.prepare_submission(
                workflow,
                inputs={"source": bad_digest},
                submission_id="bad_shared_file_digest",
            )

        plain = Workflow("plain-string-path")
        value = plain.input("value")
        plain.add_task(finish, inputs={"summary": value})
        plain_prepared = client.prepare_submission(
            plain,
            inputs={"value": str(tmp_path / "does-not-exist")},
            submission_id="plain_string_path",
        )
        assert plain_prepared.request.contract.input_identities[0].identity_kind == (
            "data_handle"
        )
        await client.submit_prepared(plain_prepared)
        await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())
