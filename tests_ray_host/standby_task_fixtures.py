from __future__ import annotations

from typing import IO
import time

from ascend_maze import task


_OPEN_FILES: list[IO[str]] = []


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
