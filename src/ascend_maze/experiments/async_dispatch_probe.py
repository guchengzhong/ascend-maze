"""Source-backed Task used by the real-Ray asynchronous dispatch acceptance."""

from __future__ import annotations

from ascend_maze import task


@task(
    task_kind="cpu",
    resources={"cpu_num": 17, "mem": 64},
    max_retries=0,
)
def cold_dispatch_probe(probe_id: str, hold_seconds: float):
    import os
    import socket
    import time

    started_ns = time.time_ns()
    time.sleep(hold_seconds)
    return {
        "probe_id": probe_id,
        "hostname": socket.gethostname(),
        "worker_pid": os.getpid(),
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
    }
