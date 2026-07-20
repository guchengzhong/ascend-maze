"""Source-backed Workflow used by C14 orchestrator tests."""

from ascend_maze import Workflow, task


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 32})
def echo_value(value: int):
    return {"result": value}


def build() -> Workflow:
    workflow = Workflow("c14-benchmark-fixture")
    value = workflow.input("value")
    workflow.add_task(echo_value, inputs={"value": value})
    return workflow
