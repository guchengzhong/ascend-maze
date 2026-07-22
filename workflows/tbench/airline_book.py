"""Ascend-Maze-native tau-bench airline booking workflow."""

from __future__ import annotations

from ascend_maze import Workflow, task

from workflows._common import WorkflowSpec, edges, nodes, spec_inputs
from workflows.tbench._common import (
    airline_booking_decision_prompt,
    airline_booking_extract_prompt,
    airline_candidate_journeys,
    book_airline_reservation,
    get_airline_user_details,
    inference_features,
    load_airline_backend_data,
    metadata_dict,
    parse_airline_booking_request,
    parse_selected_airline_journey,
    search_direct_flights,
    search_onestop_flights,
)

SPEC = WorkflowSpec(
    name="maze-tbench-airline-book",
    source="tbench",
    kind="airline_book",
    nodes=nodes(
        (
            ("task0_init", "cpu", None),
            ("task1_llm_process", "npu", "qwen3-32b"),
            ("task2a_search_direct_flight", "cpu", None),
            ("task2b_search_onestop_flight", "cpu", None),
            ("task2c_get_user_details", "cpu", None),
            ("task3_llm_fuse_process_filter_and_decide", "npu", "qwen3-32b"),
            ("task4_book_reservation", "cpu", None),
        )
    ),
    edges=edges(
        (
            ("task0_init", "task1_llm_process"),
            ("task1_llm_process", "task2a_search_direct_flight"),
            ("task1_llm_process", "task2b_search_onestop_flight"),
            ("task1_llm_process", "task2c_get_user_details"),
            (
                "task2a_search_direct_flight",
                "task3_llm_fuse_process_filter_and_decide",
            ),
            (
                "task2b_search_onestop_flight",
                "task3_llm_fuse_process_filter_and_decide",
            ),
            (
                "task2c_get_user_details",
                "task3_llm_fuse_process_filter_and_decide",
            ),
            (
                "task3_llm_fuse_process_filter_and_decide",
                "task4_book_reservation",
            ),
        )
    ),
)

INPUTS = spec_inputs()


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 1024})
def task0_init(
    dag_id: str,
    question: str,
    answer: str = "",
    supplementary_files: object = None,
    metadata: object = None,
) -> dict[str, object]:
    if not question:
        raise ValueError(f"task {dag_id} question field is empty")
    backend_data = load_airline_backend_data(supplementary_files)
    prompt = airline_booking_extract_prompt(question)
    normalized_metadata = metadata_dict(metadata)
    features = inference_features(prompt)
    return {
        "dag_id": dag_id,
        "instruction": question,
        "answer": answer,
        "backend_data": backend_data,
        "metadata": normalized_metadata,
        "prompt": prompt,
        "succ_task_feat": {"task1_llm_process": features},
    }


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 512}, max_retries=0)
def task1_llm_process(
    dag_id: str,
    instruction: str,
    prompt: str,
    metadata: dict[str, object],
    backend_data: dict[str, object],
) -> dict[str, object]:
    from ascend_maze.inference import chat

    response = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=768,
        temperature=0.0,
    )
    override = metadata.get("booking_extract_output_override")
    if not isinstance(override, str) or not override.strip():
        override = metadata.get("llm_output_override")
    if isinstance(override, str) and override.strip():
        llm_output = override
    else:
        llm_output = response.text
    booking_request = parse_airline_booking_request(llm_output)
    features = {
        "text_length": len(prompt),
        "token_count": len(prompt.split()),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
    return {
        "dag_id": dag_id,
        "instruction": instruction,
        "llm_output": llm_output,
        "raw_model_output": response.text,
        "booking_request": booking_request,
        "backend_data": backend_data,
        "metadata": metadata,
        "curr_task_feat": features,
    }


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 1024})
def task2a_search_direct_flight(
    dag_id: str,
    instruction: str,
    backend_data: dict[str, object],
    booking_request: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    direct_flights = search_direct_flights(
        backend_data,
        origin=str(booking_request.get("origin", "")),
        destination=str(booking_request.get("destination", "")),
        date=str(booking_request.get("date", "")),
    )
    text = str(direct_flights)
    return {
        "dag_id": dag_id,
        "instruction": instruction,
        "backend_data": backend_data,
        "booking_request": booking_request,
        "metadata": metadata,
        "direct_flights": direct_flights,
        "text1_feature": {
            "text1_length": len(text),
            "text1_token_count": len(text.split()),
        },
    }


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 1024})
def task2b_search_onestop_flight(
    dag_id: str,
    backend_data: dict[str, object],
    booking_request: dict[str, object],
) -> dict[str, object]:
    onestop_flights = search_onestop_flights(
        backend_data,
        origin=str(booking_request.get("origin", "")),
        destination=str(booking_request.get("destination", "")),
        date=str(booking_request.get("date", "")),
    )
    text = str(onestop_flights)
    return {
        "dag_id": dag_id,
        "onestop_flights": onestop_flights,
        "text2_feature": {
            "text2_length": len(text),
            "text2_token_count": len(text.split()),
        },
    }


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 1024})
def task2c_get_user_details(
    dag_id: str,
    backend_data: dict[str, object],
    booking_request: dict[str, object],
) -> dict[str, object]:
    user_lookup = get_airline_user_details(
        backend_data,
        str(booking_request.get("user_id", "")),
    )
    return {
        "dag_id": dag_id,
        "user_lookup": user_lookup,
    }


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 512}, max_retries=0)
def task3_llm_fuse_process_filter_and_decide(
    dag_id: str,
    instruction: str,
    backend_data: dict[str, object],
    booking_request: dict[str, object],
    metadata: dict[str, object],
    direct_flights: list[dict[str, object]],
    onestop_flights: list[list[dict[str, object]]],
    user_lookup: dict[str, object],
    text1_feature: dict[str, object],
    text2_feature: dict[str, object],
) -> dict[str, object]:
    from ascend_maze.inference import chat

    candidates = airline_candidate_journeys(direct_flights, onestop_flights)
    prompt = airline_booking_decision_prompt(instruction, candidates)
    response = chat(
        [{"role": "user", "content": prompt}],
        max_tokens=768,
        temperature=0.0,
    )
    override = metadata.get("itinerary_output_override")
    if isinstance(override, str) and override.strip():
        llm_output = override
    else:
        llm_output = response.text
    selected_journey = parse_selected_airline_journey(llm_output)
    user_details = user_lookup.get("user_details", {})
    if not isinstance(user_details, dict):
        user_details = {}
    features = {
        "prompt_length": len(prompt),
        "prompt_token_count": len(prompt.split()),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "text1_length": text1_feature.get("text1_length", 0),
        "text1_token_count": text1_feature.get("text1_token_count", 0),
        "text2_length": text2_feature.get("text2_length", 0),
        "text2_token_count": text2_feature.get("text2_token_count", 0),
    }
    return {
        "dag_id": dag_id,
        "llm_output": llm_output,
        "raw_model_output": response.text,
        "backend_data": backend_data,
        "booking_request": booking_request,
        "selected_journey": selected_journey,
        "user_details": user_details,
        "curr_task_feat": features,
    }


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 1024})
def task4_book_reservation(
    dag_id: str,
    backend_data: dict[str, object],
    booking_request: dict[str, object],
    selected_journey: list[dict[str, object]],
    user_details: dict[str, object],
) -> dict[str, object]:
    booking_result = book_airline_reservation(
        backend_data,
        booking_request,
        selected_journey,
        user_details,
    )
    return {
        "dag_id": dag_id,
        "status": booking_result.get("status", "error"),
        "backend_data": backend_data,
        "booking_result": booking_result,
        "result": booking_result,
    }


def build() -> Workflow:
    workflow = Workflow(SPEC.name)
    dag_id = workflow.input("dag_id")
    question = workflow.input("question")
    answer = workflow.input("answer")
    supplementary_files = workflow.input("supplementary_files")
    metadata = workflow.input("metadata")

    initialized = workflow.add_task(
        task0_init,
        task_name="task0_init",
        inputs={
            "dag_id": dag_id,
            "question": question,
            "answer": answer,
            "supplementary_files": supplementary_files,
            "metadata": metadata,
        },
    )
    extracted = workflow.add_task(
        task1_llm_process,
        task_name="task1_llm_process",
        model_anchor={"model": "qwen3-32b", "mode": "service"},
        inputs={
            "dag_id": initialized.outputs["dag_id"],
            "instruction": initialized.outputs["instruction"],
            "prompt": initialized.outputs["prompt"],
            "metadata": initialized.outputs["metadata"],
            "backend_data": initialized.outputs["backend_data"],
        },
    )
    direct = workflow.add_task(
        task2a_search_direct_flight,
        task_name="task2a_search_direct_flight",
        inputs={
            "dag_id": extracted.outputs["dag_id"],
            "instruction": extracted.outputs["instruction"],
            "backend_data": extracted.outputs["backend_data"],
            "booking_request": extracted.outputs["booking_request"],
            "metadata": extracted.outputs["metadata"],
        },
    )
    onestop = workflow.add_task(
        task2b_search_onestop_flight,
        task_name="task2b_search_onestop_flight",
        inputs={
            "dag_id": extracted.outputs["dag_id"],
            "backend_data": extracted.outputs["backend_data"],
            "booking_request": extracted.outputs["booking_request"],
        },
    )
    user = workflow.add_task(
        task2c_get_user_details,
        task_name="task2c_get_user_details",
        inputs={
            "dag_id": extracted.outputs["dag_id"],
            "backend_data": extracted.outputs["backend_data"],
            "booking_request": extracted.outputs["booking_request"],
        },
    )
    decided = workflow.add_task(
        task3_llm_fuse_process_filter_and_decide,
        task_name="task3_llm_fuse_process_filter_and_decide",
        model_anchor={"model": "qwen3-32b", "mode": "service"},
        inputs={
            "dag_id": direct.outputs["dag_id"],
            "instruction": direct.outputs["instruction"],
            "backend_data": direct.outputs["backend_data"],
            "booking_request": direct.outputs["booking_request"],
            "metadata": direct.outputs["metadata"],
            "direct_flights": direct.outputs["direct_flights"],
            "onestop_flights": onestop.outputs["onestop_flights"],
            "user_lookup": user.outputs["user_lookup"],
            "text1_feature": direct.outputs["text1_feature"],
            "text2_feature": onestop.outputs["text2_feature"],
        },
    )
    workflow.add_task(
        task4_book_reservation,
        task_name="task4_book_reservation",
        inputs={
            "dag_id": decided.outputs["dag_id"],
            "backend_data": decided.outputs["backend_data"],
            "booking_request": decided.outputs["booking_request"],
            "selected_journey": decided.outputs["selected_journey"],
            "user_details": decided.outputs["user_details"],
        },
    )
    return workflow
