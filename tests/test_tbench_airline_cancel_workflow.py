from __future__ import annotations

import asyncio

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_spec
from workflows.tbench import airline_cancel


def _airline_cancel_inputs() -> dict[str, object]:
    return {
        "dag_id": "airline_cancel_smoke",
        "question": (
            "Cancel reservation OLDRES for airline_user and rebook an economy "
            "SFO to LAX flight on 2024-05-22."
        ),
        "answer": "",
        "metadata": {
            "cancel_extract_output_override": (
                "{"
                '"user_id": "airline_user", '
                '"cancel_reservation_id": "OLDRES", '
                '"origin": "SFO", '
                '"destination": "LAX", '
                '"departure_date": "2024-05-22", '
                '"return_date": "", '
                '"cabin": "economy", '
                '"baggages": 1, '
                '"insurance": "no", '
                '"payment_preference": "", '
                '"constraints": ["direct flight"], '
                '"num_passengers": 1'
                "}"
            ),
            "flight_selection_output_override": (
                "{"
                '"outbound_flight_number": "HAT200", '
                '"return_flight_number": ""'
                "}"
            ),
        },
        "supplementary_files": {
            "flights.json": {
                "HAT200": {
                    "flight_number": "HAT200",
                    "origin": "SFO",
                    "destination": "LAX",
                    "scheduled_departure_time_est": "09:00",
                    "scheduled_arrival_time_est": "11:00",
                    "dates": {
                        "2024-05-22": {
                            "status": "available",
                            "available_seats": {
                                "basic_economy": 5,
                                "economy": 5,
                                "business": 2,
                            },
                            "prices": {
                                "basic_economy": 90,
                                "economy": 120,
                                "business": 280,
                            },
                        }
                    },
                }
            },
            "users.json": {
                "airline_user": {
                    "name": {
                        "first_name": "Ava",
                        "last_name": "Pilot",
                    },
                    "dob": "1990-01-01",
                    "payment_methods": {
                        "credit_card_air": {
                            "source": "credit_card",
                            "id": "credit_card_air",
                        }
                    },
                    "reservations": ["OLDRES"],
                }
            },
            "reservations.json": {
                "OLDRES": {
                    "reservation_id": "OLDRES",
                    "user_id": "airline_user",
                    "origin": "SEA",
                    "destination": "JFK",
                    "flight_type": "one_way",
                    "cabin": "economy",
                    "flights": [],
                    "passengers": [],
                    "payment_history": [
                        {
                            "payment_id": "credit_card_air",
                            "amount": 50,
                        }
                    ],
                    "created_at": "2024-05-01T00:00:00",
                    "total_baggages": 0,
                    "nonfree_baggages": 0,
                    "insurance": "no",
                }
            },
        },
    }


def test_airline_cancel_workflow_runs_with_fake_inference(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "qwen3-32b", model_id="qwen3-32b")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        try:
            workflow = airline_cancel.build()
            compiled = workflow.compile()
            task_id_by_name = {
                task.task_name: task.task_id
                for _, task in compiled.tasks.items_tuple()
            }

            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs=_airline_cancel_inputs(),
                submission_id="airline_cancel_smoke",
                session_key="airline_cancel_test",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id,
                timeout_seconds=2,
            )

            assert terminal.status is RunStatus.SUCCEEDED
            assert adapter.invoke_count == 2
            assert len(inference.request_records()) == 2

            final = controller.result(
                outcome.run_id,
                task_id_by_name["task6_book_new_reservation"],
            )
            assert final["status"] == "success"
            reservations = final["backend_data"]["reservations"]  # type: ignore[index]
            assert reservations["OLDRES"]["status"] == "cancelled"
            assert reservations["OLDRES"]["payment_history"][1] == {
                "payment_id": "credit_card_air",
                "amount": -50,
            }
            assert set(reservations) == {"OLDRES", "HATHAT"}
            new_reservation = reservations["HATHAT"]
            assert new_reservation["user_id"] == "airline_user"
            assert new_reservation["origin"] == "SFO"
            assert new_reservation["destination"] == "LAX"
            assert new_reservation["flights"] == [
                {
                    "flight_number": "HAT200",
                    "date": "2024-05-22",
                    "price": 120,
                    "origin": "SFO",
                    "destination": "LAX",
                }
            ]
            assert new_reservation["payment_history"] == [
                {
                    "payment_id": "credit_card_air",
                    "amount": 120.0,
                }
            ]
            users = final["backend_data"]["users"]  # type: ignore[index]
            assert users["airline_user"]["reservations"] == ["OLDRES", "HATHAT"]

            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())
