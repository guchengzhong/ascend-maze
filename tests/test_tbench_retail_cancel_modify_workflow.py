from __future__ import annotations

import asyncio

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_spec
from workflows.tbench import retail_cancel_modify


def _retail_cancel_modify_inputs() -> dict[str, object]:
    return {
        "dag_id": "retail_cancel_modify_smoke",
        "question": (
            "Cancel #ORDER-CANCEL because it was ordered by mistake, and change "
            "the red water bottle in #ORDER-MOD to blue."
        ),
        "answer": "",
        "metadata": {
            "llm_output_override": (
                "{"
                '"user_info": {"email": "zoe@example.com"}, '
                '"cancellation": {'
                '"order_id": "#ORDER-CANCEL", '
                '"reason": "ordered by mistake"'
                "}, "
                '"modification": {'
                '"order_id": "#ORDER-MOD", '
                '"item_to_modify": {'
                '"name": "Water Bottle", '
                '"attributes": {"color": "red"}'
                "}, "
                '"new_item_spec": {"attributes": {"color": "blue"}}, '
                '"payment_method_id": "credit_card_4"'
                "}"
                "}"
            )
        },
        "supplementary_files": {
            "products.json": {
                "product_water": {
                    "name": "Water Bottle",
                    "product_id": "product_water",
                    "variants": {
                        "item_red": {
                            "item_id": "item_red",
                            "options": {"color": "red", "size": "1L"},
                            "available": True,
                            "price": 10.0,
                        },
                        "item_blue": {
                            "item_id": "item_blue",
                            "options": {"color": "blue", "size": "1L"},
                            "available": True,
                            "price": 12.0,
                        },
                    },
                }
            },
            "users.json": {
                "user_4": {
                    "name": {
                        "first_name": "Zoe",
                        "last_name": "Lane",
                    },
                    "address": {"zip": "94101"},
                    "email": "zoe@example.com",
                    "payment_methods": {
                        "gift_card_4": {
                            "source": "gift_card",
                            "id": "gift_card_4",
                            "balance": 3.0,
                        },
                        "credit_card_4": {
                            "source": "credit_card",
                            "id": "credit_card_4",
                        },
                    },
                    "orders": ["#ORDER-CANCEL", "#ORDER-MOD"],
                }
            },
            "orders.json": {
                "#ORDER-CANCEL": {
                    "order_id": "#ORDER-CANCEL",
                    "user_id": "user_4",
                    "address": {},
                    "items": [],
                    "fulfillments": [],
                    "status": "pending",
                    "payment_history": [
                        {
                            "transaction_type": "payment",
                            "amount": 15.0,
                            "payment_method_id": "gift_card_4",
                        }
                    ],
                },
                "#ORDER-MOD": {
                    "order_id": "#ORDER-MOD",
                    "user_id": "user_4",
                    "address": {},
                    "items": [
                        {
                            "name": "Water Bottle",
                            "product_id": "product_water",
                            "item_id": "item_red",
                            "price": 10.0,
                            "options": {"color": "red", "size": "1L"},
                        }
                    ],
                    "fulfillments": [],
                    "status": "pending",
                    "payment_history": [
                        {
                            "transaction_type": "payment",
                            "amount": 10.0,
                            "payment_method_id": "credit_card_4",
                        }
                    ],
                },
            },
        },
    }


def test_retail_cancel_modify_workflow_runs_with_fake_inference(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "qwen3-32b", model_id="qwen3-32b")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        try:
            workflow = retail_cancel_modify.build()
            compiled = workflow.compile()
            task_id_by_name = {
                task.task_name: task.task_id
                for _, task in compiled.tasks.items_tuple()
            }

            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs=_retail_cancel_modify_inputs(),
                submission_id="retail_cancel_modify_smoke",
                session_key="retail_cancel_modify_test",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id,
                timeout_seconds=2,
            )

            assert terminal.status is RunStatus.SUCCEEDED
            assert adapter.invoke_count == 1
            assert len(inference.request_records()) == 1

            user_lookup = controller.result(
                outcome.run_id,
                task_id_by_name["task2a_find_user"],
            )
            assert user_lookup["user_lookup"]["status"] == "success"  # type: ignore[index]

            executed = controller.result(
                outcome.run_id,
                task_id_by_name["task3_execute_operations"],
            )
            orders = executed["backend_data"]["orders"]  # type: ignore[index]
            cancelled = orders["#ORDER-CANCEL"]
            modified = orders["#ORDER-MOD"]
            assert cancelled["status"] == "cancelled"
            assert cancelled["cancel_reason"] == "ordered by mistake"
            assert cancelled["payment_history"][1] == {
                "transaction_type": "refund",
                "amount": 15.0,
                "payment_method_id": "gift_card_4",
            }
            users = executed["backend_data"]["users"]  # type: ignore[index]
            assert users["user_4"]["payment_methods"]["gift_card_4"]["balance"] == 18.0

            assert modified["status"] == "pending (item modified)"
            assert modified["items"][0]["item_id"] == "item_blue"
            assert modified["items"][0]["price"] == 12.0
            assert modified["items"][0]["options"] == {"color": "blue", "size": "1L"}
            assert modified["payment_history"][1] == {
                "transaction_type": "payment",
                "amount": 2.0,
                "payment_method_id": "credit_card_4",
            }
            assert executed["final_result"]["status"] == "success"  # type: ignore[index]

            final = controller.result(
                outcome.run_id,
                task_id_by_name["task4_output_result"],
            )
            assert final["status"] == "success"
            assert "cancellation_result" in final["result"]
            assert "modification_result" in final["result"]

            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())
