from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "logical_cluster_e2e.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("logical_cluster_e2e", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_logical_cluster_e2e_selects_real_text_and_vision_samples() -> None:
    tool = _load_tool()

    text = tool._discover_sample(REPO_ROOT / "data", "text")  # noqa: SLF001
    vision = tool._discover_sample(REPO_ROOT / "data", "vision")  # noqa: SLF001

    assert (text.dataset, text.workflow) == ("tbench", "retail_cancel")
    assert (vision.dataset, vision.workflow) == ("gaia", "vision")
    assert vision.source_files


def test_logical_cluster_e2e_keeps_runtime_internals_out_of_task_contracts() -> None:
    tool = _load_tool()
    workflow, aliases = tool.smoke._build_workflow(  # noqa: SLF001
        "tbench",
        "retail_cancel",
        "qwen3-4b-e2e",
    )

    contracts, clean = tool._task_contracts(workflow)  # noqa: SLF001

    assert clean
    assert aliases == {"qwen3-32b": "qwen3-4b-e2e"}
    assert all(not item["forbidden_parameters"] for item in contracts)


def test_logical_cluster_e2e_derives_attempt_timing_and_device_evidence() -> None:
    tool = _load_tool()
    terminal = {
        "task_states": [
            {
                "task_id": "task_1",
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "succeeded",
                        "node_id": "node-3",
                        "device_ids": ["3"],
                        "dispatched_at_ms": 100,
                        "worker_started_at_ms": 125,
                        "finished_at_ms": 225,
                    }
                ],
            }
        ]
    }

    assert tool._task_timings(terminal, {"task_1": "model"}) == [  # noqa: SLF001
        {
            "task_id": "task_1",
            "task_name": "model",
            "attempt": 1,
            "status": "succeeded",
            "node_id": "node-3",
            "device_ids": ["3"],
            "dispatch_to_worker_ms": 25,
            "worker_execution_ms": 100,
            "attempt_total_ms": 125,
        }
    ]


def test_logical_cluster_e2e_can_require_text_tasks_to_cross_nodes() -> None:
    tool = _load_tool()
    timings = [
        {"task_id": "task_1", "node_id": "node-2"},
        {"task_id": "task_2", "node_id": "node-0"},
        {"task_id": "task_3", "node_id": "node-2"},
    ]

    node_ids = tool._task_node_ids(timings)  # noqa: SLF001
    tool._require_cross_node_text("text", node_ids, required=True)  # noqa: SLF001

    assert node_ids == ["node-0", "node-2"]


def test_logical_cluster_e2e_rejects_single_node_text_when_required() -> None:
    tool = _load_tool()

    with pytest.raises(RuntimeError, match="did not cross"):
        tool._require_cross_node_text(  # noqa: SLF001
            "text",
            ["node-0"],
            required=True,
        )


def test_logical_cluster_e2e_uses_service_instance_as_npu_evidence() -> None:
    tool = _load_tool()
    timings = [
        {
            "task_id": "task_model",
            "task_name": "service_model_task",
            "node_id": "node-2",
            "device_ids": [],
        }
    ]
    snapshots = [
        {
            "instances": [
                {
                    "instance_id": "model_instance_1",
                    "generation": 3,
                    "model_id": "qwen3-4b-e2e",
                    "node_id": "node-5",
                    "npu_device_id": "5",
                    "route_occupancy": 1,
                    "actual_request_inflight": 0,
                }
            ]
        },
        {
            "instances": [
                {
                    "instance_id": "model_instance_1",
                    "generation": 3,
                    "model_id": "qwen3-4b-e2e",
                    "node_id": "node-5",
                    "npu_device_id": "5",
                    "route_occupancy": 1,
                    "actual_request_inflight": 1,
                }
            ]
        },
    ]

    assert tool._model_device_evidence(  # noqa: SLF001
        timings,
        {"task_model"},
        snapshots,
        "qwen3-4b-e2e",
    ) == [
        {
            "source": "service_model_instance",
            "instance_id": "model_instance_1",
            "instance_generation": 3,
            "model_id": "qwen3-4b-e2e",
            "node_id": "node-5",
            "physical_device_id": "5",
            "route_occupancy_observed": True,
            "request_inflight_observed": True,
        }
    ]


def test_logical_cluster_e2e_ignores_released_worker_lease_history() -> None:
    tool = _load_tool()
    pool = {
        "worker_leases": [
            {"lease": {"worker_lease_id": "old"}, "released": True},
            {"lease": {"worker_lease_id": "active"}, "released": False},
        ]
    }

    assert tool._active_worker_leases(pool) == [  # noqa: SLF001
        {"lease": {"worker_lease_id": "active"}, "released": False}
    ]
