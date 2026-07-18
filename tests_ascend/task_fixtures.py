from ascend_maze import task


@task(task_kind="npu", resources={"cpu_num": 1, "mem": 256, "npu_mem": 1024})
def npu_add(megabytes: int):
    import torch
    import torch_npu

    assert torch_npu is not None
    element_count = megabytes * 1024 * 1024 // 4
    value = torch.ones(element_count, dtype=torch.float32, device="npu:0")
    return {"result": int(value[:1024].sum().cpu())}


@task
def cpu_visible_device():
    import os

    return {"visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 256, "npu_mem": 1024},
    timeout_seconds=0.2,
    max_retries=0,
)
def npu_timeout():
    import time

    import torch
    import torch_npu

    assert torch_npu is not None
    value = torch.ones(16 * 1024 * 1024, dtype=torch.float32, device="npu:0")
    torch.npu.synchronize()
    time.sleep(30)
    return {"result": int(value[0].cpu())}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 256, "npu_mem": 1024},
    max_retries=0,
)
def npu_long_running():
    import time

    import torch
    import torch_npu

    assert torch_npu is not None
    value = torch.ones(16 * 1024 * 1024, dtype=torch.float32, device="npu:0")
    torch.npu.synchronize()
    time.sleep(30)
    return {"result": int(value[0].cpu())}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 256, "npu_mem": 1024},
    max_retries=3,
)
def npu_oom():
    import torch
    import torch_npu

    assert torch_npu is not None
    value = torch.empty(70 * 1024 * 1024 * 1024, dtype=torch.uint8, device="npu:0")
    return {"result": int(value[0].cpu())}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 256, "npu_mem": 1024},
    max_retries=0,
)
def npu_tensor_output():
    import torch
    import torch_npu

    assert torch_npu is not None
    return {"result": torch.ones(1, device="npu:0")}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 256, "npu_mem": 1024},
    max_retries=0,
)
def npu_sync_device_error():
    import torch
    import torch_npu

    assert torch_npu is not None
    torch.ones(1, device="npu:0")
    real_synchronize = torch.npu.synchronize

    def fail_at_mandatory_synchronize(device=0):
        del device
        return real_synchronize(1)

    torch.npu.synchronize = fail_at_mandatory_synchronize
    return {"result": 1}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 256, "npu_mem": 1024},
    max_retries=0,
)
def npu_user_error(should_fail: bool = True):
    import torch
    import torch_npu

    assert torch_npu is not None
    torch.ones(1, device="npu:0")
    if should_fail:
        raise RuntimeError("stage4 user failure")
    return {"result": 1}


@task(
    task_kind="npu",
    resources={"cpu_num": 1, "mem": 256, "npu_mem": 70_000},
    max_retries=0,
)
def impossible_multi_card_npu():
    return {"result": 1}
