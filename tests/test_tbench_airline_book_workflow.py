from __future__ import annotations

import asyncio
from dataclasses import replace
import inspect
import json

from ascend_maze.control import InMemoryRuntimeClient
from ascend_maze.compiler.ir import OutputBinding
from ascend_maze.inference.adapters.fake import FakeInferenceEngineAdapter
from ascend_maze.lifecycle import RunStatus
from inference_helpers import make_controller, make_spec
from workflows.tbench import airline_book
from workflows.tbench.airline_tools import (
    BookReservation,
    GetUserDetails,
    SearchDirectFlight,
    SearchOnestopFlight,
)


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
    return {
        "dag_id": "airline_book_smoke",
        "question": (
            "Book one economy flight for airline_user from SFO to LAX on "
            "2024-05-20 with one bag and no insurance."
        ),
        "answer": "",
        "metadata": {},
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


class _AirlineBookAdapter(FakeInferenceEngineAdapter):
    async def invoke_chat(self, context, request):  # type: ignore[no-untyped-def]
        response = await super().invoke_chat(context, request)
        prompt = str(request.messages[-1]["content"])
        if "# List of candidate itineraries" in prompt:
            text = json.dumps([_selected_flight()])
        else:
            text = json.dumps(
                {
                    "user_id": "airline_user",
                    "origin": "SFO",
                    "destination": "LAX",
                    "date": "2024-05-20",
                    "cabin": "economy",
                    "baggages": 1,
                    "insurance": "no",
                    "constraints": ["direct flight"],
                    "num_passengers": 1,
                }
            )
        return replace(response, text=text)


EXPECTED_INPUTS_BY_TASK = {
    "task0_init": ("dag_id", "question", "supplementary_files"),
    "task1_llm_process": (
        "dag_id",
        "instruction",
        "use_online_model",
        "model_folder",
        "temperature",
        "max_tokens",
        "top_p",
        "repetition_penalty",
        "task1_llm_process_request_api_url",
    ),
    "task2a_search_direct_flight": (
        "dag_id",
        "extracted_info",
        "backend_data",
        "instruction",
    ),
    "task2b_search_onestop_flight": (
        "dag_id",
        "extracted_info",
        "backend_data",
        "instruction",
    ),
    "task2c_get_user_details": ("dag_id", "user_id", "backend_data"),
    "task3_llm_fuse_process_filter_and_decide": (
        "dag_id",
        "instruction",
        "text1_feature",
        "text2_feature",
        "use_online_model",
        "model_folder",
        "temperature",
        "max_tokens",
        "top_p",
        "repetition_penalty",
        "direct_flights",
        "onestop_flights",
        "task3_llm_fuse_process_filter_and_decide_request_api_url",
    ),
    "task4_book_reservation": (
        "dag_id",
        "user_id",
        "backend_data",
        "extracted_info",
        "selected_journey",
        "user_details",
    ),
}


def test_airline_book_task_parameters_match_original_context_get_fields() -> None:
    for task_name, expected in EXPECTED_INPUTS_BY_TASK.items():
        parameters = inspect.signature(getattr(airline_book, task_name)).parameters
        assert tuple(parameters) == expected


def test_airline_book_preserves_original_tool_invoke_interfaces() -> None:
    assert tuple(inspect.signature(SearchDirectFlight.invoke).parameters) == (
        "data",
        "origin",
        "destination",
        "date",
    )
    assert tuple(inspect.signature(SearchOnestopFlight.invoke).parameters) == (
        "data",
        "origin",
        "destination",
        "date",
    )
    assert tuple(inspect.signature(GetUserDetails.invoke).parameters) == (
        "data",
        "user_id",
    )
    assert tuple(inspect.signature(BookReservation.invoke).parameters) == (
        "data",
        "user_id",
        "origin",
        "destination",
        "flight_type",
        "cabin",
        "flights",
        "passengers",
        "payment_methods",
        "total_baggages",
        "nonfree_baggages",
        "insurance",
    )


def test_airline_book_backend_data_is_one_shared_output_binding() -> None:
    compiled = airline_book.build().compile()
    tasks_by_name = {
        task.task_name: task for _, task in compiled.tasks.items_tuple()
    }
    initialized = tasks_by_name["task0_init"]
    consumers = (
        "task2a_search_direct_flight",
        "task2b_search_onestop_flight",
        "task2c_get_user_details",
        "task4_book_reservation",
    )

    for task_name in consumers:
        binding = next(
            item
            for item in tasks_by_name[task_name].inputs
            if item.input_name == "backend_data"
        )
        assert isinstance(binding, OutputBinding)
        assert binding.source_task_id == initialized.task_id
        assert binding.source_output == "backend_data"

    producers = [
        task.task_name
        for _, task in compiled.tasks.items_tuple()
        if "backend_data"
        in compiled.definitions[task.definition_id].output_names
    ]
    assert producers == ["task0_init"]


def test_airline_book_workflow_runs_with_fake_inference(tmp_path) -> None:
    async def scenario() -> None:
        spec = make_spec(tmp_path / "qwen3-32b", model_id="qwen3-32b")
        adapter = _AirlineBookAdapter()
        controller, inference, adapter = make_controller(spec, adapter=adapter)
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
            assert final["status"] == "done"
            assert final["result"] == final["booking_result"]
            reservation = json.loads(final["booking_result"])
            assert reservation["reservation_id"] == "HATHAT"
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
            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())
