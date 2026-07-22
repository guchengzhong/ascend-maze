from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_node, make_spec


def run_gaia_workflow(
    module: ModuleType,
    *,
    inputs: Mapping[str, object],
    model_ids: tuple[str, ...],
    submission_id: str,
    tmp_path: Path,
    shared_filesystem_roots: tuple[str, ...] = (),
) -> tuple[dict[str, dict[str, object]], int, int]:
    async def scenario() -> tuple[dict[str, dict[str, object]], int, int]:
        specs = tuple(
            make_spec(tmp_path / model_id, model_id=model_id)
            for model_id in model_ids
        )
        controller, inference, adapter = make_controller(
            specs,
            nodes=(make_node(npu_count=max(1, len(set(model_ids)))),),
        )
        await controller.start()
        try:
            workflow = module.build()
            compiled = workflow.compile()
            task_id_by_name = {
                task.task_name: task.task_id
                for _, task in compiled.tasks.items_tuple()
            }
            client = InMemoryRuntimeClient(
                controller,
                shared_filesystem_roots=shared_filesystem_roots,
            )
            outcome = await client.submit(
                workflow,
                inputs=dict(inputs),
                submission_id=submission_id,
                session_key=f"{submission_id}_session",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id,
                timeout_seconds=5,
            )
            assert terminal.status is RunStatus.SUCCEEDED
            results = {
                task_name: controller.result(outcome.run_id, task_id)
                for task_name, task_id in task_id_by_name.items()
            }
            request_count = len(inference.request_records())
            invoke_count = adapter.invoke_count
            await controller.destroy_run(outcome.run_id)
            return results, invoke_count, request_count
        finally:
            await controller.close()

    return asyncio.run(scenario())
