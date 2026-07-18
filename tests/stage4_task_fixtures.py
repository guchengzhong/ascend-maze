from ascend_maze import task


@task(resources={"cpu_num": 1})
def statically_parallel(value: int):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(abs, [value]))[0]
    return {"result": result}


@task(task_kind="npu", resources={"npu_mem": 1024})
def statically_npu(value: int):
    import torch
    import torch_npu

    assert torch_npu is not None
    tensor = torch.tensor([value], device="npu:0")
    return {"result": int(tensor.cpu()[0])}


@task(task_kind="npu", resources={"npu_mem": 70_000}, max_retries=0)
def impossible_npu(value: int):
    return {"result": value}


@task(task_kind="npu", resources={"npu_mem": 1024}, max_retries=0)
def no_retry_npu(value: int):
    return {"result": value}
