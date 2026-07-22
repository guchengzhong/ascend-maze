from __future__ import annotations

import asyncio

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_spec
from workflows.tbench import retail_cancel


def _retail_cancel_inputs() -> dict[str, object]:
    return {
        "dag_id": "retail_cancel_smoke",
        "question": "Please cancel order #ORDER-1 because I no longer need it.",
        "answer": "",
        "metadata": {
            "llm_output_override": (
                '{"order_id": "#ORDER-1", "reason": "no longer needed"}'
            )
        },
        "supplementary_files": {
            "products.json": {},
            "users.json": {
                "user_1": {
                    "payment_methods": {
                        "gift_card_1": {
                            "source": "gift_card",
                            "id": "gift_card_1",
                            "balance": 5.0,
                        }
                    }
                }
            },
            "orders.json": {
                "#ORDER-1": {
                    "order_id": "#ORDER-1",
                    "user_id": "user_1",
                    "address": {},
                    "items": [],
                    "fulfillments": [],
                    "status": "pending",
                    "payment_history": [
                        {
                            "transaction_type": "payment",
                            "amount": 12.5,
                            "payment_method_id": "gift_card_1",
                        }
                    ],
                }
            },
        },
    }


def test_retail_cancel_workflow_runs_with_fake_inference(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "qwen3-32b", model_id="qwen3-32b")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        try:
            workflow = retail_cancel.build()
            compiled = workflow.compile()
            task_id_by_name = {
                task.task_name: task.task_id
                for _, task in compiled.tasks.items_tuple()
            }

            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs=_retail_cancel_inputs(),
                submission_id="retail_cancel_smoke",
                session_key="retail_cancel_test",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id,
                timeout_seconds=2,
            )

            assert terminal.status is RunStatus.SUCCEEDED
            assert adapter.invoke_count == 1
            assert len(inference.request_records()) == 1

            executed = controller.result(
                outcome.run_id,
                task_id_by_name["task2_execute_cancel"],
            )
            order = executed["backend_data"]["orders"]["#ORDER-1"]  # type: ignore[index]
            assert order["status"] == "cancelled"
            assert order["cancel_reason"] == "no longer needed"
            assert order["payment_history"][1] == {
                "transaction_type": "refund",
                "amount": 12.5,
                "payment_method_id": "gift_card_1",
            }
            user = executed["backend_data"]["users"]["user_1"]  # type: ignore[index]
            assert user["payment_methods"]["gift_card_1"]["balance"] == 17.5

            final = controller.result(
                outcome.run_id,
                task_id_by_name["task3_output_result"],
            )
            assert final["status"] == "done"
            assert final["cancel_results"][0]["status"] == "success"  # type: ignore[index]
            assert "#ORDER-1" in final["result"]

            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())
