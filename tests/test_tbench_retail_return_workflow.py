from __future__ import annotations

import asyncio

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_spec
from workflows.tbench import retail_return


def _retail_return_inputs() -> dict[str, object]:
    return {
        "dag_id": "retail_return_smoke",
        "question": (
            "I am Ava Moore at ava@example.com. Please return the water bottle "
            "from order #ORDER-2 to my credit card."
        ),
        "answer": "",
        "metadata": {
            "llm_output_override": (
                "{"
                '"order_id": "#ORDER-2", '
                '"items": ["Water Bottle"], '
                '"reason": "changed my mind", '
                '"email": "ava@example.com", '
                '"payment_method_id": "credit_card_1"'
                "}"
            )
        },
        "supplementary_files": {
            "products.json": {},
            "users.json": {
                "user_2": {
                    "name": {
                        "first_name": "Ava",
                        "last_name": "Moore",
                    },
                    "address": {"zip": "78234"},
                    "email": "ava@example.com",
                    "payment_methods": {
                        "credit_card_1": {
                            "source": "credit_card",
                            "id": "credit_card_1",
                        },
                        "gift_card_1": {
                            "source": "gift_card",
                            "id": "gift_card_1",
                            "balance": 0.0,
                        },
                    },
                    "orders": ["#ORDER-2"],
                }
            },
            "orders.json": {
                "#ORDER-2": {
                    "order_id": "#ORDER-2",
                    "user_id": "user_2",
                    "address": {},
                    "items": [
                        {
                            "name": "Water Bottle",
                            "product_id": "product_water",
                            "item_id": "item_water",
                            "price": 47.76,
                            "options": {"color": "red"},
                        },
                        {
                            "name": "Bookshelf",
                            "product_id": "product_bookshelf",
                            "item_id": "item_bookshelf",
                            "price": 463.04,
                            "options": {"color": "black"},
                        },
                    ],
                    "fulfillments": [],
                    "status": "delivered",
                    "payment_history": [
                        {
                            "transaction_type": "payment",
                            "amount": 510.8,
                            "payment_method_id": "credit_card_1",
                        }
                    ],
                }
            },
        },
    }


def test_retail_return_workflow_runs_with_fake_inference(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "qwen3-32b", model_id="qwen3-32b")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        try:
            workflow = retail_return.build()
            compiled = workflow.compile()
            task_id_by_name = {
                task.task_name: task.task_id
                for _, task in compiled.tasks.items_tuple()
            }

            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs=_retail_return_inputs(),
                submission_id="retail_return_smoke",
                session_key="retail_return_test",
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
                task_id_by_name["task4_execute_return"],
            )
            order = executed["backend_data"]["orders"]["#ORDER-2"]  # type: ignore[index]
            assert order["status"] == "return requested"
            assert order["return_items"] == ["item_water"]
            assert order["return_payment_method_id"] == "credit_card_1"
            assert executed["final_result"]["status"] == "success"  # type: ignore[index]

            final = controller.result(
                outcome.run_id,
                task_id_by_name["task5_output_result"],
            )
            assert final["status"] == "success"
            assert final["final_result"]["result"]["return_items"] == [  # type: ignore[index]
                "item_water"
            ]
            assert "#ORDER-2" in final["result"]

            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())
