from __future__ import annotations

import asyncio

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_spec
from workflows.tbench import airline_book


def _selected_flight() -> dict[str, object]:
    return {
        "flight_number": "HAT100",
        "origin": "SFO",
        "destination": "LAX",
        "scheduled_departure_time_est": "08:00",
        "scheduled_arrival_time_est": "10:00",
        "date": "2024-05-20",
        "available_seats": {
            "basic_economy": 5,
            "economy": 5,
            "business": 2,
        },
        "prices": {
            "basic_economy": 80,
            "economy": 100,
            "business": 250,
        },
        "status": "available",
    }


def _airline_book_inputs() -> dict[str, object]:
    selected = _selected_flight()
    return {
        "dag_id": "airline_book_smoke",
        "question": (
            "Book one economy flight for airline_user from SFO to LAX on "
            "2024-05-20 with one bag and no insurance."
        ),
        "answer": "",
        "metadata": {
            "booking_extract_output_override": (
                "{"
                '"user_id": "airline_user", '
                '"origin": "SFO", '
                '"destination": "LAX", '
                '"date": "2024-05-20", '
                '"cabin": "economy", '
                '"baggages": 1, '
                '"insurance": "no", '
                '"constraints": ["direct flight"], '
                '"num_passengers": 1'
                "}"
            ),
            "itinerary_output_override": (
                "["
                f"{selected!r}"
                "]"
            ).replace("'", '"'),
        },
        "supplementary_files": {
            "flights.json": {
                "HAT100": {
                    "flight_number": "HAT100",
                    "origin": "SFO",
                    "destination": "LAX",
                    "scheduled_departure_time_est": "08:00",
                    "scheduled_arrival_time_est": "10:00",
                    "dates": {
                        "2024-05-20": {
                            "status": "available",
                            "available_seats": {
                                "basic_economy": 5,
                                "economy": 5,
                                "business": 2,
                            },
                            "prices": {
                                "basic_economy": 80,
                                "economy": 100,
                                "business": 250,
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
                    "reservations": [],
                }
            },
            "reservations.json": {},
        },
    }


def test_airline_book_workflow_runs_with_fake_inference(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "qwen3-32b", model_id="qwen3-32b")
        controller, inference, adapter = make_controller(spec)
        await controller.start()
        try:
            workflow = airline_book.build()
            compiled = workflow.compile()
            task_id_by_name = {
                task.task_name: task.task_id
                for _, task in compiled.tasks.items_tuple()
            }

            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs=_airline_book_inputs(),
                submission_id="airline_book_smoke",
                session_key="airline_book_test",
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
                task_id_by_name["task4_book_reservation"],
            )
            assert final["status"] == "success"
            reservations = final["backend_data"]["reservations"]  # type: ignore[index]
            assert set(reservations) == {"HATHAT"}
            reservation = reservations["HATHAT"]
            assert reservation["user_id"] == "airline_user"
            assert reservation["origin"] == "SFO"
            assert reservation["destination"] == "LAX"
            assert reservation["cabin"] == "economy"
            assert reservation["flights"] == [
                {
                    "flight_number": "HAT100",
                    "date": "2024-05-20",
                    "price": 100,
                    "origin": "SFO",
                    "destination": "LAX",
                }
            ]
            assert reservation["payment_history"] == [
                {
                    "payment_id": "credit_card_air",
                    "amount": 100.0,
                }
            ]
            users = final["backend_data"]["users"]  # type: ignore[index]
            assert users["airline_user"]["reservations"] == ["HATHAT"]

            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())
