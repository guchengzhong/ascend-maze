from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


TOOL_PATH = Path(__file__).parents[1] / "tools" / "async_dispatch_acceptance.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "async_dispatch_acceptance_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


acceptance = _load_tool()


def test_attempt_validation_rejects_internal_retry() -> None:
    valid = {
        "run_id": "run_1",
        "task_states": [
            {
                "task_id": "task_1",
                "status": "succeeded",
                "attempt_count": 1,
                "attempts": [{"attempt": 1, "status": "succeeded"}],
            }
        ],
    }
    retried = {
        "run_id": "run_2",
        "task_states": [
            {
                "task_id": "task_2",
                "status": "succeeded",
                "attempt_count": 2,
                "attempts": [
                    {"attempt": 1, "status": "failed"},
                    {"attempt": 2, "status": "succeeded"},
                ],
            }
        ],
    }

    assert acceptance._attempt_violations([valid]) == []  # noqa: SLF001
    violations = acceptance._attempt_violations([retried])  # noqa: SLF001
    assert len(violations) == 1
    assert violations[0]["attempt_count"] == 2


def test_resource_signatures_ignore_replenished_worker_and_lease_ids() -> None:
    def snapshot(worker_id: str, lease_id: str) -> tuple[dict[str, object], dict[str, object]]:
        workers = {
            "worker_pool": {
                "workers": [
                    {
                        "worker_id": worker_id,
                        "node_id": "node-0",
                        "boot_id": "boot-1",
                        "profile": "cpu",
                        "state": "idle",
                    }
                ]
            }
        }
        cluster = {
            "cluster": {
                "active_leases": [
                    {
                        "status": "bound",
                        "lease": {
                            "lease_id": lease_id,
                            "reservation_kind": "standby_worker",
                            "run_id": None,
                            "node_id": "node-0",
                            "boot_id": "boot-1",
                            "npu_device_id": None,
                            "resources": {
                                "cpu_num": 1,
                                "host_mem_mb": 64,
                                "io_slots": 0,
                                "npu_hbm_mb": 0,
                                "npu_slots": 0,
                            },
                        },
                    }
                ]
            }
        }
        return workers, cluster

    before_workers, before_cluster = snapshot("worker_old", "lease_old")
    after_workers, after_cluster = snapshot("worker_new", "lease_new")

    assert acceptance._live_worker_signature(  # noqa: SLF001
        before_workers
    ) == acceptance._live_worker_signature(after_workers)  # noqa: SLF001
    assert acceptance._global_lease_signature(  # noqa: SLF001
        before_cluster
    ) == acceptance._global_lease_signature(after_cluster)  # noqa: SLF001


def test_cli_accepts_batch_larger_than_logical_node_count() -> None:
    args = acceptance.parse_args(["--batch-size", "20"])

    assert args.batch_size == [20]
