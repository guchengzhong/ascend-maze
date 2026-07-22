from __future__ import annotations

import asyncio

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_spec
from workflows.tbench import retail_modify


def _retail_modify_inputs() -> dict[str, object]:
    return {
        "dag_id": "retail_modify_smoke",
        "question": (
            "Please change the payment for order #ORDER-3 to credit_card_2 "
            "and ship it to 99 New Street, Austin, TX 78701."
        ),
        "answer": "",
        "metadata": {
            "llm_output_override": (
                "{"
                '"payment_modification": {'
                '"order_id": "#ORDER-3", '
                '"payment_method_id": "credit_card_2"'
                "}, "
                '"order_address_modification": {'
                '"order_id": "#ORDER-3", '
                '"address1": "99 New Street", '
                '"address2": "Unit 5", '
                '"city": "Austin", '
                '"state": "TX", '
                '"country": "USA", '
                '"zip": "78701"'
                "}"
                "}"
            )
        },
        "supplementary_files": {
            "products.json": {},
            "users.json": {
                "user_3": {
                    "name": {
                        "first_name": "Mia",
                        "last_name": "Stone",
                    },
                    "address": {"zip": "78759"},
                    "email": "mia@example.com",
                    "payment_methods": {
                        "gift_card_2": {
                            "source": "gift_card",
                            "id": "gift_card_2",
                            "balance": 2.0,
                        },
                        "credit_card_2": {
                            "source": "credit_card",
                            "id": "credit_card_2",
                        },
                    },
                    "orders": ["#ORDER-3"],
                }
            },
            "orders.json": {
                "#ORDER-3": {
                    "order_id": "#ORDER-3",
                    "user_id": "user_3",
                    "address": {
                        "address1": "1 Old Road",
                        "address2": "",
                        "city": "Dallas",
                        "state": "TX",
                        "country": "USA",
                        "zip": "75201",
                    },
                    "items": [],
                    "fulfillments": [],
                    "status": "pending",
                    "payment_history": [
                        {
                            "transaction_type": "payment",
                            "amount": 20.0,
                            "payment_method_id": "gift_card_2",
                        }
                    ],
                }
            },
        },
    }


def test_retail_modify_workflow_runs_with_fake_inference(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "qwen3-32b", model_id="qwen3-32b")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        try:
            workflow = retail_modify.build()
            compiled = workflow.compile()
            task_id_by_name = {
                task.task_name: task.task_id
                for _, task in compiled.tasks.items_tuple()
            }

            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs=_retail_modify_inputs(),
                submission_id="retail_modify_smoke",
                session_key="retail_modify_test",
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
            assert user_lookup["user_lookup"]["status"] == "skipped"  # type: ignore[index]

            executed = controller.result(
                outcome.run_id,
                task_id_by_name["task3_execute_modifications"],
            )
            order = executed["backend_data"]["orders"]["#ORDER-3"]  # type: ignore[index]
            assert order["payment_history"][1] == {
                "transaction_type": "payment",
                "amount": 20.0,
                "payment_method_id": "credit_card_2",
            }
            assert order["payment_history"][2] == {
                "transaction_type": "refund",
                "amount": 20.0,
                "payment_method_id": "gift_card_2",
            }
            assert order["address"] == {
                "address1": "99 New Street",
                "address2": "Unit 5",
                "city": "Austin",
                "state": "TX",
                "country": "USA",
                "zip": "78701",
            }
            user = executed["backend_data"]["users"]["user_3"]  # type: ignore[index]
            assert user["payment_methods"]["gift_card_2"]["balance"] == 22.0
            assert executed["final_result"]["status"] == "success"  # type: ignore[index]

            final = controller.result(
                outcome.run_id,
                task_id_by_name["task4_output_result"],
            )
            assert final["status"] == "success"
            assert "payment_modification_result" in final["result"]
            assert "#ORDER-3" in final["result"]

            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())
