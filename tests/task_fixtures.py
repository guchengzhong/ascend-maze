"""Stable source-backed tasks shared by compiler and subprocess tests."""

from ascend_maze import task


@task(task_kind="io", resources={"cpu_num": 1, "mem": 64, "io_num": 1})
def load_text(path: str):
    return {"text": path}


@task(
    task_kind="cpu",
    resources={"cpu_num": 2, "mem": 128},
    timeout_seconds=3.5,
    max_retries=2,
    retry_backoff_seconds=0.25,
    retry_on=["worker_lost", "task_timeout"],
)
def summarize(text: str, options: dict, max_length: int = 16):
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if options.get("upper"):
        return {"summary": text[:max_length].upper(), "size": len(text)}
    return {"size": len(text), "summary": text[:max_length]}


@task
def barrier():
    return {}


@task
def finish(summary: str):
    return {"result": summary}


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 128})
def service_task(prompt: str):
    return {"answer": prompt}


@task(task_kind="npu", resources={"npu_mem": 1024})
def local_npu_task(value: str):
    return {"value": value}


@task(task_kind="npu")
def unanchored_npu_task(value: str):
    return {"value": value}
