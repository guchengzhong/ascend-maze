from __future__ import annotations

from functools import partial
import inspect

import pytest

from ascend_maze import task
from ascend_maze.api.task import get_task_template
from ascend_maze.core.errors import TaskDefinitionError
from task_fixtures import barrier, summarize


def valid_branch(value: bool):
    if value:
        return {"x": 1, "y": 2}
    raise RuntimeError("terminal raise is allowed")


def inconsistent(value: bool):
    if value:
        return {"x": 1}
    return {"y": 2}


def dynamic_result():
    result = {"x": 1}
    return result


def bare_return():
    return


def fallthrough(value: bool):
    if value:
        return {"x": 1}


def unpack_result():
    other = {"x": 1}
    return {**other}


def dynamic_key(key: str):
    return {key: 1}


def generator_task():
    yield 1
    return {"x": 1}


async def async_task():
    return {"x": 1}


def positional_only(value, /):
    return {"x": value}


def variadic(*values):
    return {"x": values}


def only_raises():
    raise RuntimeError("no output contract")


class CallableObject:
    def __call__(self):
        return {"x": 1}


class MethodOwner:
    def method(self):
        return {"x": 1}


def test_decorated_task_remains_a_plain_callable() -> None:
    assert inspect.isfunction(summarize)
    assert summarize("abcdef", {}, 3) == {"size": 6, "summary": "abc"}
    template = get_task_template(summarize)
    assert template.analysis.output_names == ("size", "summary")
    assert template.timeout_ms == 3500
    assert template.retry_backoff_ms == 250
    assert template.retry_on == ("task_timeout", "worker_lost")


def test_raise_path_and_empty_control_task_are_supported() -> None:
    decorated = task(valid_branch)
    assert get_task_template(decorated).analysis.output_names == ("x", "y")
    assert get_task_template(barrier).analysis.output_names == ()


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (inconsistent, "inconsistent"),
        (dynamic_result, "dict literal"),
        (bare_return, "bare return"),
        (fallthrough, "fall through"),
        (unpack_result, "dict unpacking"),
        (dynamic_key, "static strings"),
        (generator_task, "generator"),
        (async_task, "async"),
        (positional_only, "explicitly named"),
        (variadic, "explicitly named"),
        (only_raises, "no direct dict return"),
    ],
)
def test_invalid_task_functions_fail_locally(candidate, message: str) -> None:
    with pytest.raises(TaskDefinitionError, match=message):
        task(candidate)


def test_unsupported_callable_shapes_are_rejected() -> None:
    with pytest.raises(TaskDefinitionError, match="lambda"):
        task(lambda: {"x": 1})
    with pytest.raises(TaskDefinitionError, match="inspect.isfunction"):
        task(partial(valid_branch, True))
    with pytest.raises(TaskDefinitionError, match="inspect.isfunction"):
        task(CallableObject())
    with pytest.raises(TaskDefinitionError, match="inspect.isfunction"):
        task(MethodOwner().method)


def test_non_empty_closure_is_rejected() -> None:
    captured = "value"

    def closure():
        return {"x": captured}

    with pytest.raises(TaskDefinitionError, match="closures"):
        task(closure)


def test_invalid_task_options_are_rejected() -> None:
    with pytest.raises(TaskDefinitionError, match="task_kind"):
        task(task_kind="gpu")(valid_branch)
    with pytest.raises(TaskDefinitionError, match="max_retries"):
        task(max_retries=-1)(valid_branch)
    with pytest.raises(TaskDefinitionError, match="positive"):
        task(timeout_seconds=0)(valid_branch)
    with pytest.raises(TaskDefinitionError, match="unknown stable error"):
        task(retry_on=["runtime_timeout"])(valid_branch)


def test_retry_defaults_follow_the_stable_error_contract() -> None:
    template = get_task_template(barrier)
    assert template.max_retries == 1
    assert template.retry_on == (
        "npu_oom",
        "runtime_node_unavailable",
        "worker_acquire_failed",
        "worker_start_failed",
    )
