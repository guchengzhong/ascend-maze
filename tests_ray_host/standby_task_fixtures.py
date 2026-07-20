from __future__ import annotations

from typing import IO
import subprocess
import sys
import time

from ascend_maze import task


_OPEN_FILES: list[IO[str]] = []
_CHILDREN: list[subprocess.Popen[bytes]] = []


@task(resources={"cpu_num": 1, "mem": 64})
def leak_file_descriptor(path: str):
    handle = open(path, "w", encoding="utf-8")
    handle.write("kept-open")
    handle.flush()
    _OPEN_FILES.append(handle)
    return {"result": "published"}


@task(
    resources={"cpu_num": 1, "mem": 64},
    timeout_seconds=0.05,
    max_retries=0,
)
def slow_cpu_task(value: str):
    time.sleep(10)
    return {"result": value}


@task(resources={"cpu_num": 1, "mem": 64})
def drain_slow_cpu_task(value: str):
    time.sleep(0.2)
    return {"result": value}


@task(resources={"cpu_num": 1, "mem": 64})
def leak_child_process():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _CHILDREN.append(child)
    return {"child_pid": child.pid}
