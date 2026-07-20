"""Stable source-backed tasks shared by compiler and subprocess tests."""

from ascend_maze import task
from ascend_maze.contracts.data import SharedFileRef


@task(task_kind="io", resources={"cpu_num": 1, "mem": 64, "io_num": 1})
def load_text(path: str):
    return {"text": path}


@task(task_kind="io", resources={"cpu_num": 1, "mem": 64, "io_num": 1})
def read_shared_file(file_ref: SharedFileRef):
    from pathlib import Path

    content = Path(file_ref.canonical_path).read_text(encoding="utf-8")
    return {"content": content, "size": file_ref.size_bytes}


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


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 64})
def inference_twice_task(prompt: str):
    from ascend_maze.inference import chat

    first = chat([{"role": "user", "content": prompt}], max_tokens=8)
    second = chat([{"role": "user", "content": first.text}], max_tokens=8)
    return {"answer": f"{first.text}|{second.text}"}


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 64})
def inference_zero_call_task(prompt: str):
    return {"answer": prompt}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 64},
    max_retries=1,
    retry_on=["worker_start_failed"],
)
def inference_retry_task(prompt: str):
    from ascend_maze.inference import chat

    response = chat([{"role": "user", "content": prompt}], max_tokens=8)
    return {"answer": response.text}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 64},
    timeout_seconds=0.02,
)
def inference_timeout_task(prompt: str):
    from ascend_maze.inference import chat

    response = chat([{"role": "user", "content": prompt}], max_tokens=8)
    return {"answer": response.text}


@task(task_kind="npu", resources={"npu_mem": 1024})
def local_npu_task(value: str):
    return {"value": value}


@task(task_kind="npu")
def unanchored_npu_task(value: str):
    return {"value": value}


@task(timeout_seconds=0.02, max_retries=0)
def timeout_task(value: str):
    return {"result": value}


@task(max_retries=0)
def user_failure_task(should_fail: bool):
    if should_fail:
        raise RuntimeError("requested failure")
    return {"result": "ok"}
