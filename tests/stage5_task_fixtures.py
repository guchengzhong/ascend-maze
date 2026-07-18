from ascend_maze import task


@task(task_kind="cpu", resources={"cpu_num": 1, "mem": 32}, max_retries=0)
def one_cpu_task(value: str):
    return {"value": value}


@task(task_kind="cpu", resources={"cpu_num": 2, "mem": 32}, max_retries=0)
def two_cpu_task(value: str):
    return {"value": value}
