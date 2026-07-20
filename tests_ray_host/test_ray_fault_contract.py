from __future__ import annotations

from ascend_maze.runtime.ray_worker import (
    RAY_ONE_SHOT_FAULT_OPTIONS,
    RAY_STANDBY_FAULT_OPTIONS,
)


def test_ray_cannot_retry_or_restart_below_maze_attempt_authority() -> None:
    assert dict(RAY_ONE_SHOT_FAULT_OPTIONS.items_tuple()) == {
        "max_calls": 1,
        "max_retries": 0,
        "num_cpus": 0,
    }
    assert dict(RAY_STANDBY_FAULT_OPTIONS.items_tuple()) == {
        "max_restarts": 0,
        "max_task_retries": 0,
        "num_cpus": 0,
    }
