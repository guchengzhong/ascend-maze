#!/usr/bin/env python3
"""Paired Ascend-Maze/plain-Ray performance pilot on the logical 8-node cluster."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for _path in (str(TOOLS_ROOT), str(SRC_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import logical_cluster_e2e as logical_e2e  # noqa: E402
import qwen_benchmark_smoke as qwen_smoke  # noqa: E402
import ray_baseline_smoke as ray_smoke  # noqa: E402


SCHEMA_VERSION = 1
OBJECTIVE = "logical_cluster_maze_ray_performance_pilot"
CONTAINER_NAME = "ascend-maze-logical-node-0"
CONTAINER_ENV = REPO_ROOT / "deploy" / "logical_cluster" / "container_env.sh"
DEFAULT_STATE_ROOT = (
    Path.home() / ".local" / "state" / "ascend-maze" / "logical-cluster"
)
DEFAULT_CONTROL_SOCKET = Path("/workspace/state/control-plane/control.sock")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_STATE_ROOT / "node-0" / "output" / "logical-cluster-performance"
)
TEXT_MODEL_ID = "qwen3-4b-e2e"
TEXT_MODEL_PATH = Path("/home/user2/workplace/model_weight/model_from_hf/Qwen3-4B")
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}


class PerformancePilotError(RuntimeError):
    """Expected pilot setup or execution failure."""


@dataclass(frozen=True, slots=True)
class WorkloadCase:
    case_id: str
    mode: str
    request_count: int
    launch_offsets_ms: tuple[int, ...]
    batch_size: int | None = None
    arrival_ratio: float | None = None
    arrival_rate_per_second: float | None = None
    average_workflow_seconds: float | None = None
    admission_window_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"batch", "arrival"}:
            raise ValueError("workload mode must be batch or arrival")
        if self.request_count < 1 or len(self.launch_offsets_ms) != self.request_count:
            raise ValueError("request_count must match launch offsets")
        if tuple(sorted(self.launch_offsets_ms)) != self.launch_offsets_ms:
            raise ValueError("launch offsets must be sorted")
        if any(item < 0 for item in self.launch_offsets_ms):
            raise ValueError("launch offsets must be non-negative")

    def payload(self) -> dict[str, object]:
        return _jsonable(asdict(self))


def _jsonable(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PerformancePilotError(f"JSON document is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * fraction
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        weight = rank - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def _arrival_offsets_ms(
    *,
    arrival_ratio: float,
    average_workflow_seconds: float,
    admission_window_seconds: float,
) -> tuple[int, ...]:
    if arrival_ratio <= 0:
        raise ValueError("arrival_ratio must be positive")
    if average_workflow_seconds <= 0 or admission_window_seconds <= 0:
        raise ValueError("arrival timing values must be positive")
    rate = arrival_ratio / average_workflow_seconds
    interval = 1.0 / rate
    offsets: list[int] = []
    offset = 0.0
    while offset < admission_window_seconds:
        offsets.append(round(offset * 1_000))
        offset += interval
    return tuple(offsets)


def _build_cases(args: argparse.Namespace) -> tuple[WorkloadCase, ...]:
    cases: list[WorkloadCase] = []
    modes = {str(item) for item in args.mode}
    if "batch" in modes:
        for size in args.batch_size:
            cases.append(
                WorkloadCase(
                    case_id=f"batch-{size}",
                    mode="batch",
                    request_count=int(size),
                    launch_offsets_ms=(0,) * int(size),
                    batch_size=int(size),
                )
            )
    if "arrival" in modes:
        for ratio in args.arrival_ratio:
            offsets = _arrival_offsets_ms(
                arrival_ratio=float(ratio),
                average_workflow_seconds=float(args.average_workflow_seconds),
                admission_window_seconds=float(args.arrival_window_seconds),
            )
            ratio_id = str(float(ratio)).replace(".", "p")
            cases.append(
                WorkloadCase(
                    case_id=f"arrival-ratio-{ratio_id}",
                    mode="arrival",
                    request_count=len(offsets),
                    launch_offsets_ms=offsets,
                    arrival_ratio=float(ratio),
                    arrival_rate_per_second=(
                        float(ratio) / float(args.average_workflow_seconds)
                    ),
                    average_workflow_seconds=float(args.average_workflow_seconds),
                    admission_window_seconds=float(args.arrival_window_seconds),
                )
            )
    if not cases:
        raise ValueError("at least one workload case is required")
    return tuple(cases)


def _execution_order(
    cases: Sequence[WorkloadCase], executor: str
) -> tuple[tuple[WorkloadCase, str, int], ...]:
    if executor in {"maze", "ray"}:
        return tuple((case, executor, 1) for case in cases)
    ordered: list[tuple[WorkloadCase, str, int]] = []
    for index, case in enumerate(cases):
        pair = ("maze", "ray") if index % 2 == 0 else ("ray", "maze")
        ordered.extend((case, name, position + 1) for position, name in enumerate(pair))
    return tuple(ordered)


def _aggregate_requests(
    records: Sequence[Mapping[str, object]],
    *,
    mode: str,
    admission_window_seconds: float | None,
) -> dict[str, object]:
    succeeded = [item for item in records if item.get("status") == "succeeded"]
    latencies = [
        float(item["client_e2e_ms"])
        for item in succeeded
        if isinstance(item.get("client_e2e_ms"), (int, float))
        and not isinstance(item.get("client_e2e_ms"), bool)
    ]
    starts = [
        int(item["client_e2e_started_at_ms"])
        for item in records
        if isinstance(item.get("client_e2e_started_at_ms"), int)
    ]
    finishes = [
        int(item["client_e2e_finished_at_ms"])
        for item in records
        if isinstance(item.get("client_e2e_finished_at_ms"), int)
    ]
    makespan_ms = max(finishes) - min(starts) if starts and finishes else 0
    completed_in_window = None
    window_throughput = None
    if mode == "arrival" and starts and admission_window_seconds is not None:
        deadline = min(starts) + round(admission_window_seconds * 1_000)
        completed_in_window = sum(
            item.get("status") == "succeeded"
            and isinstance(item.get("client_e2e_finished_at_ms"), int)
            and int(item["client_e2e_finished_at_ms"]) <= deadline
            for item in records
        )
        window_throughput = completed_in_window / admission_window_seconds
    failure_reasons: dict[str, int] = {}
    for item in records:
        if item.get("status") == "succeeded":
            continue
        reason = str(item.get("error") or item.get("status") or "unknown")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    return {
        "request_count": len(records),
        "succeeded": len(succeeded),
        "failed": len(records) - len(succeeded),
        "success_rate": len(succeeded) / len(records) if records else 0.0,
        "e2e_latency_ms": _stats(latencies),
        "p95_e2e_ms": _stats(latencies)["p95"],
        "makespan_ms": makespan_ms,
        "throughput_requests_per_second": (
            len(succeeded) / (makespan_ms / 1_000) if makespan_ms > 0 else 0.0
        ),
        "completed_in_admission_window": completed_in_window,
        "admission_window_throughput_requests_per_second": window_throughput,
        "failure_reasons": failure_reasons,
    }


class HostResourceMonitor:
    """Sample logical-node cgroups and all physical NPUs outside the containers."""

    def __init__(
        self,
        *,
        output_path: Path,
        interval_seconds: float,
        container_prefix: str = "ascend-maze-logical-node-",
    ) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.container_prefix = container_prefix
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._containers = self._discover_containers()
        self._previous_cpu: dict[str, tuple[int, int]] = {}
        from ascend_maze.ascend.dcmi import DcmiDeviceAdapter

        self._dcmi = DcmiDeviceAdapter()

    @staticmethod
    def _cpu_count(cpuset: str) -> int:
        total = 0
        for part in cpuset.split(","):
            if "-" in part:
                start, finish = part.split("-", 1)
                total += int(finish) - int(start) + 1
            elif part:
                total += 1
        return total

    def _discover_containers(self) -> tuple[dict[str, object], ...]:
        command = [
            "docker",
            "ps",
            "--filter",
            "label=com.ascend-maze.logical-cluster=true",
            "--format",
            "{{.Names}}",
        ]
        names = sorted(
            item.strip()
            for item in subprocess.check_output(command, text=True).splitlines()
            if item.strip().startswith(self.container_prefix)
        )
        if len(names) != 8:
            raise PerformancePilotError(
                f"expected 8 running logical containers, found {len(names)}"
            )
        containers: list[dict[str, object]] = []
        for name in names:
            payload = json.loads(
                subprocess.check_output(["docker", "inspect", name], text=True)
            )[0]
            container_id = str(payload["Id"])
            cpuset = str(payload["HostConfig"]["CpusetCpus"])
            cpu_path = (
                Path("/sys/fs/cgroup/cpu,cpuacct/docker")
                / container_id
                / "cpuacct.usage"
            )
            memory_path = (
                Path("/sys/fs/cgroup/memory/docker")
                / container_id
                / "memory.usage_in_bytes"
            )
            if not cpu_path.is_file() or not memory_path.is_file():
                raise PerformancePilotError(
                    f"logical container cgroup files are missing: {name}"
                )
            containers.append(
                {
                    "name": name,
                    "node_id": name.removeprefix(self.container_prefix),
                    "container_id": container_id,
                    "cpuset": cpuset,
                    "cpu_count": self._cpu_count(cpuset),
                    "cpu_path": cpu_path,
                    "memory_path": memory_path,
                }
            )
        return tuple(containers)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource monitor is already started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="logical-cluster-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> tuple[dict[str, object], ...]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(5.0, self.interval_seconds * 2))
        self._sample()
        with self.output_path.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(_jsonable(sample), sort_keys=True) + "\n")
        return tuple(self.samples)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        timestamp_ms = int(time.time() * 1_000)
        monotonic_ns = time.monotonic_ns()
        container_samples: list[dict[str, object]] = []
        errors: list[str] = []
        for container in self._containers:
            name = str(container["name"])
            try:
                cpu_usage_ns = int(Path(container["cpu_path"]).read_text().strip())
                memory_bytes = int(Path(container["memory_path"]).read_text().strip())
                previous = self._previous_cpu.get(name)
                cpu_percent = None
                if previous is not None:
                    previous_time_ns, previous_usage_ns = previous
                    elapsed = monotonic_ns - previous_time_ns
                    if elapsed > 0:
                        cpu_percent = max(
                            0.0,
                            (cpu_usage_ns - previous_usage_ns)
                            / elapsed
                            / int(container["cpu_count"])
                            * 100,
                        )
                self._previous_cpu[name] = (monotonic_ns, cpu_usage_ns)
                container_samples.append(
                    {
                        "name": name,
                        "node_id": container["node_id"],
                        "cpu_count": container["cpu_count"],
                        "cpu_usage_ns": cpu_usage_ns,
                        "cpu_utilization_pct": cpu_percent,
                        "memory_used_bytes": memory_bytes,
                    }
                )
            except Exception as exc:
                errors.append(f"cgroup:{name}:{type(exc).__name__}:{exc}")
        npu_samples: list[dict[str, object]] = []
        try:
            for device in self._dcmi.devices():
                npu_samples.append(
                    {
                        "physical_device_id": device.physical_device_id,
                        "utilization_pct": device.utilization,
                        "used_hbm_mb": device.used_hbm_mb,
                        "total_hbm_mb": device.total_hbm_mb,
                        "processes": [asdict(item) for item in device.processes],
                    }
                )
        except Exception as exc:
            errors.append(f"dcmi:{type(exc).__name__}:{exc}")
        cpu_values = [
            float(item["cpu_utilization_pct"])
            for item in container_samples
            if isinstance(item.get("cpu_utilization_pct"), (int, float))
        ]
        npu_values = [
            float(item["utilization_pct"])
            for item in npu_samples
            if isinstance(item.get("utilization_pct"), (int, float))
        ]
        self.samples.append(
            {
                "timestamp_ms": timestamp_ms,
                "monotonic_ns": monotonic_ns,
                "containers": container_samples,
                "npus": npu_samples,
                "cluster_cpu_utilization_pct": (
                    statistics.fmean(cpu_values) if cpu_values else None
                ),
                "cluster_npu_utilization_pct": (
                    statistics.fmean(npu_values) if npu_values else None
                ),
                "max_device_npu_utilization_pct": max(npu_values, default=None),
                "cluster_hbm_used_mb": sum(
                    int(item["used_hbm_mb"]) for item in npu_samples
                ),
                "errors": errors,
            }
        )


def _aggregate_resources(
    samples: Sequence[Mapping[str, object]],
    *,
    started_at_ms: int,
    finished_at_ms: int,
) -> dict[str, object]:
    selected = [
        item
        for item in samples
        if isinstance(item.get("timestamp_ms"), int)
        and started_at_ms <= int(item["timestamp_ms"]) <= finished_at_ms
    ]
    if not selected and samples:
        selected = list(samples)
    cpu_values = [
        float(item["cluster_cpu_utilization_pct"])
        for item in selected
        if isinstance(item.get("cluster_cpu_utilization_pct"), (int, float))
    ]
    npu_values = [
        float(item["cluster_npu_utilization_pct"])
        for item in selected
        if isinstance(item.get("cluster_npu_utilization_pct"), (int, float))
    ]
    max_npu_values = [
        float(item["max_device_npu_utilization_pct"])
        for item in selected
        if isinstance(item.get("max_device_npu_utilization_pct"), (int, float))
    ]
    hbm_values = [
        int(item["cluster_hbm_used_mb"])
        for item in selected
        if isinstance(item.get("cluster_hbm_used_mb"), int)
    ]
    baseline_hbm = None
    baseline_samples = [
        item
        for item in samples
        if isinstance(item.get("timestamp_ms"), int)
        and int(item["timestamp_ms"]) < started_at_ms
    ]
    baseline_cpu_values = [
        float(item["cluster_cpu_utilization_pct"])
        for item in baseline_samples
        if isinstance(item.get("cluster_cpu_utilization_pct"), (int, float))
    ]
    baseline_npu_values = [
        float(item["cluster_npu_utilization_pct"])
        for item in baseline_samples
        if isinstance(item.get("cluster_npu_utilization_pct"), (int, float))
    ]
    for item in reversed(samples):
        if (
            isinstance(item.get("timestamp_ms"), int)
            and int(item["timestamp_ms"]) <= started_at_ms
            and isinstance(item.get("cluster_hbm_used_mb"), int)
        ):
            baseline_hbm = int(item["cluster_hbm_used_mb"])
            break
    per_device: dict[str, dict[str, list[float]]] = {}
    per_node_cpu: dict[str, list[float]] = {}
    monitor_errors: list[str] = []
    for sample in selected:
        for error in sample.get("errors", []):  # type: ignore[union-attr]
            monitor_errors.append(str(error))
        for item in sample.get("containers", []):  # type: ignore[union-attr]
            if not isinstance(item, Mapping):
                continue
            value = item.get("cpu_utilization_pct")
            if isinstance(value, (int, float)):
                per_node_cpu.setdefault(str(item.get("node_id")), []).append(
                    float(value)
                )
        for item in sample.get("npus", []):  # type: ignore[union-attr]
            if not isinstance(item, Mapping):
                continue
            device_id = str(item.get("physical_device_id"))
            target = per_device.setdefault(device_id, {"utilization": [], "hbm": []})
            utilization = item.get("utilization_pct")
            hbm = item.get("used_hbm_mb")
            if isinstance(utilization, (int, float)):
                target["utilization"].append(float(utilization))
            if isinstance(hbm, int):
                target["hbm"].append(float(hbm))
    return {
        "sample_count": len(selected),
        "window_started_at_ms": started_at_ms,
        "window_finished_at_ms": finished_at_ms,
        "cluster_cpu_utilization_pct": _stats(cpu_values),
        "baseline_cluster_cpu_utilization_pct": _stats(baseline_cpu_values),
        "incremental_cluster_cpu_utilization_pct": (
            None
            if not cpu_values or not baseline_cpu_values
            else statistics.fmean(cpu_values) - statistics.fmean(baseline_cpu_values)
        ),
        "cluster_npu_utilization_pct": _stats(npu_values),
        "baseline_cluster_npu_utilization_pct": _stats(baseline_npu_values),
        "incremental_cluster_npu_utilization_pct": (
            None
            if not npu_values or not baseline_npu_values
            else statistics.fmean(npu_values) - statistics.fmean(baseline_npu_values)
        ),
        "max_device_npu_utilization_pct": _stats(max_npu_values),
        "cluster_hbm_used_mb": _stats(hbm_values),
        "baseline_cluster_hbm_used_mb": baseline_hbm,
        "peak_incremental_hbm_mb": (
            None
            if baseline_hbm is None or not hbm_values
            else max(hbm_values) - baseline_hbm
        ),
        "per_node_cpu_utilization_pct": {
            key: _stats(values) for key, values in sorted(per_node_cpu.items())
        },
        "per_device": {
            key: {
                "utilization_pct": _stats(value["utilization"]),
                "hbm_used_mb": _stats(value["hbm"]),
            }
            for key, value in sorted(per_device.items(), key=lambda item: int(item[0]))
        },
        "monitor_errors": monitor_errors,
    }


def _discover_text_sample(data_root: Path) -> Any:
    samples, failures = qwen_smoke.discover_samples(
        data_root=data_root,
        datasets={"tbench"},
        workflows={"retail_cancel"},
        families={"text"},
        samples_per_workflow=1,
        sample_offset=0,
        max_inline_file_bytes=64 * 1024 * 1024,
        tbench_smoke_overrides=True,
        gaia_file_smoke_summary=True,
    )
    if failures:
        raise PerformancePilotError(f"sample discovery failed: {failures}")
    if len(samples) != 1:
        raise PerformancePilotError(
            f"expected one retail_cancel sample, found {len(samples)}"
        )
    return samples[0]


def _task_timings(
    run: Mapping[str, object], task_names: Mapping[str, str]
) -> list[dict[str, object]]:
    return logical_e2e._task_timings(dict(run), dict(task_names))  # noqa: SLF001


def _dispatch_lifecycle(
    watch_batches: Sequence[Mapping[str, object]],
    task_names: Mapping[str, str],
) -> list[dict[str, object]]:
    lifecycle_types = {"task_dispatched", "dispatch_prepared", "worker_started"}
    attempts: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for batch in watch_batches:
        events = batch.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("event_type")
            task_id = event.get("task_id")
            attempt = event.get("attempt")
            if (
                event_type not in lifecycle_types
                or not isinstance(task_id, str)
                or not isinstance(attempt, int)
                or isinstance(attempt, bool)
            ):
                continue
            attempts.setdefault((task_id, attempt), {})[str(event_type)] = event

    def timestamp(event: Mapping[str, object] | None) -> int | None:
        if event is None:
            return None
        value = event.get("monotonic_time_ms")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def elapsed(start: int | None, finish: int | None) -> int | None:
        if start is None or finish is None:
            return None
        return max(0, finish - start)

    records: list[dict[str, object]] = []
    for (task_id, attempt), events in sorted(attempts.items()):
        dispatched = events.get("task_dispatched")
        prepared = events.get("dispatch_prepared")
        running = events.get("worker_started")
        dispatched_at = timestamp(dispatched)
        prepared_at = timestamp(prepared)
        running_at = timestamp(running)
        dispatch_payload = (
            dispatched.get("payload")
            if isinstance(dispatched, Mapping)
            and isinstance(dispatched.get("payload"), Mapping)
            else {}
        )
        prepared_payload = (
            prepared.get("payload")
            if isinstance(prepared, Mapping)
            and isinstance(prepared.get("payload"), Mapping)
            else {}
        )
        running_payload = (
            running.get("payload")
            if isinstance(running, Mapping)
            and isinstance(running.get("payload"), Mapping)
            else {}
        )
        records.append(
            {
                "task_id": task_id,
                "task_name": task_names.get(task_id, task_id),
                "attempt": attempt,
                "dispatch_id": dispatch_payload.get("dispatch_id"),
                "node_id": dispatch_payload.get("node_id"),
                "worker_pid": running_payload.get("worker_pid"),
                "task_dispatched_sequence": (
                    None if dispatched is None else dispatched.get("sequence")
                ),
                "dispatch_prepared_sequence": (
                    None if prepared is None else prepared.get("sequence")
                ),
                "running_sequence": (
                    None if running is None else running.get("sequence")
                ),
                "task_dispatched_at_ms": dispatched_at,
                "dispatch_prepared_at_ms": prepared_at,
                "running_at_ms": running_at,
                "dispatch_prepare_ms": prepared_payload.get("dispatch_prepare_ms"),
                "dispatch_to_prepared_ms": elapsed(dispatched_at, prepared_at),
                "prepared_to_running_ms": elapsed(prepared_at, running_at),
                "dispatch_to_running_ms": elapsed(dispatched_at, running_at),
            }
        )
    return records


async def _wait_maze_terminal(
    client: Any, run_id: str, timeout_seconds: float
) -> tuple[dict[str, object], list[dict[str, object]]]:
    watch_batches: list[dict[str, object]] = []
    async for batch in client.watch_run(run_id, timeout_seconds=timeout_seconds):
        watch_batches.append(batch)
    shown = await client.query(
        "GetRun", resource_id=run_id, timeout_seconds=min(30.0, timeout_seconds)
    )
    run = shown.get("run")
    if not isinstance(run, dict):
        raise PerformancePilotError("GetRun returned no terminal Run")
    if str(run.get("status")) not in TERMINAL_STATES:
        raise PerformancePilotError("WatchRun ended before a terminal state")
    return run, watch_batches


async def _run_maze_request(
    *,
    client: Any,
    workflow: Any,
    compiled: Any,
    task_names: Mapping[str, str],
    sample: Any,
    case_id: str,
    request_index: int,
    timeout_seconds: float,
) -> dict[str, object]:
    unique = f"{case_id}:{request_index}:{time.time_ns()}"
    submission_id = "perf-" + hashlib.sha256(unique.encode()).hexdigest()[:28]
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "executor": "maze",
        "case_id": case_id,
        "request_index": request_index,
        "sample_id": sample.sample_id,
        "submission_id": submission_id,
        "status": "not_started",
    }
    run_id: str | None = None
    destroyed = False
    e2e_started_perf = time.perf_counter()
    record["client_e2e_started_at_ms"] = int(time.time() * 1_000)
    try:
        stage = time.perf_counter()
        prepared = await client.prepare_submission(
            workflow,
            inputs=sample.inputs,
            submission_id=submission_id,
            session_key=f"{submission_id}-session",
            run_deadline_ms=round(timeout_seconds * 1_000),
        )
        record["prepare_submission_ms"] = round((time.perf_counter() - stage) * 1_000)
        stage = time.perf_counter()
        outcome = await client.submit_prepared(prepared, timeout_seconds=60.0)
        record["submit_roundtrip_ms"] = round((time.perf_counter() - stage) * 1_000)
        value = outcome.get("run_id")
        if not isinstance(value, str) or not value:
            raise PerformancePilotError(f"submission did not commit: {outcome}")
        run_id = value
        record["run_id"] = run_id
        terminal, watch_batches = await _wait_maze_terminal(
            client, run_id, timeout_seconds
        )
        record["terminal_status"] = terminal.get("status")
        record["watch_batch_count"] = len(watch_batches)
        record["dispatch_lifecycle"] = _dispatch_lifecycle(
            watch_batches, task_names
        )
        if terminal.get("status") != "succeeded":
            raise PerformancePilotError(f"Run terminated as {terminal.get('status')}")
        results = {}
        for task_id in compiled.exit_tasks:
            results[task_names[task_id]] = await client.materialize_task_result(
                run_id, task_id
            )
        record["exit_task_results"] = results
        record["status"] = "succeeded"
        record["client_e2e_finished_at_ms"] = int(time.time() * 1_000)
        record["client_e2e_ms"] = round(
            (time.perf_counter() - e2e_started_perf) * 1_000
        )
        record["task_timings"] = _task_timings(terminal, task_names)
        stage = time.perf_counter()
        record["destroy_result"] = await client.run_action(
            "DestroyRun", run_id, force=True, timeout_seconds=120.0
        )
        record["destroy_ms"] = round((time.perf_counter() - stage) * 1_000)
        destroyed = True
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        record.setdefault("client_e2e_finished_at_ms", int(time.time() * 1_000))
        record.setdefault(
            "client_e2e_ms", round((time.perf_counter() - e2e_started_perf) * 1_000)
        )
        if run_id is not None and not destroyed:
            try:
                await client.run_action(
                    "CancelRun",
                    run_id,
                    reason="performance_pilot_cleanup",
                    force=True,
                    timeout_seconds=30.0,
                )
            except Exception as exc:
                record.setdefault("cleanup_errors", []).append(
                    f"cancel:{type(exc).__name__}:{exc}"
                )
            try:
                await client.run_action(
                    "DestroyRun", run_id, force=True, timeout_seconds=120.0
                )
            except Exception as exc:
                record.setdefault("cleanup_errors", []).append(
                    f"destroy:{type(exc).__name__}:{exc}"
                )
    return record


async def _run_scheduled(
    launch_offsets_ms: Sequence[int], run_one: Any
) -> tuple[list[dict[str, object]], int, int]:
    loop = asyncio.get_running_loop()
    workload_started_at_ms = int(time.time() * 1_000)
    started = loop.time()
    tasks: list[asyncio.Task[dict[str, object]]] = []
    for request_index, offset_ms in enumerate(launch_offsets_ms, start=1):
        delay = started + offset_ms / 1_000 - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        actual_offset_ms = round((loop.time() - started) * 1_000)

        async def invoke(index: int, planned: int, actual: int) -> dict[str, object]:
            result = await run_one(index)
            result["planned_launch_offset_ms"] = planned
            result["actual_launch_offset_ms"] = actual
            return result

        tasks.append(
            asyncio.create_task(invoke(request_index, int(offset_ms), actual_offset_ms))
        )
    records = list(await asyncio.gather(*tasks))
    workload_finished_at_ms = int(time.time() * 1_000)
    return records, workload_started_at_ms, workload_finished_at_ms


async def _run_maze_worker(
    args: argparse.Namespace, case: WorkloadCase
) -> dict[str, object]:
    from ascend_maze.control.local_rpc import UdsRuntimeClient

    sample = _discover_text_sample(args.data_root)
    workflow, aliases = qwen_smoke._build_workflow(  # noqa: SLF001
        sample.dataset, sample.workflow, TEXT_MODEL_ID
    )
    compiled = workflow.compile()
    task_names = {
        task_id: task.task_name for task_id, task in compiled.tasks.items_tuple()
    }
    client = UdsRuntimeClient(args.control_socket)
    try:
        controller_status = await client.get_controller_status(timeout_seconds=10.0)
        if controller_status.healthy_node_count != 8:
            raise PerformancePilotError(
                f"expected 8 healthy nodes, found {controller_status.healthy_node_count}"
            )
        await client._ensure_data_store()  # noqa: SLF001

        async def run_one(index: int) -> dict[str, object]:
            return await _run_maze_request(
                client=client,
                workflow=workflow,
                compiled=compiled,
                task_names=task_names,
                sample=sample,
                case_id=case.case_id,
                request_index=index,
                timeout_seconds=float(args.request_timeout_seconds),
            )

        records, started_at_ms, finished_at_ms = await _run_scheduled(
            case.launch_offsets_ms, run_one
        )
        system_after = await client.query("GetSystemSnapshot", timeout_seconds=10.0)
        return {
            "schema_version": SCHEMA_VERSION,
            "objective": OBJECTIVE,
            "executor": "maze",
            "case": case.payload(),
            "sample": sample.manifest(),
            "model_aliases": aliases,
            "workflow_fingerprint": compiled.workflow_fingerprint,
            "controller_status": controller_status,
            "system_after": system_after,
            "workload_started_at_ms": started_at_ms,
            "workload_finished_at_ms": finished_at_ms,
            "records": records,
            "aggregate": _aggregate_requests(
                records,
                mode=case.mode,
                admission_window_seconds=case.admission_window_seconds,
            ),
        }
    finally:
        client.close()


def _transformers_config(args: argparse.Namespace) -> dict[str, object]:
    from ascend_maze.ascend.discovery import discover_aicpu_runtime_library_paths

    return {
        "family": "text",
        "model_id": TEXT_MODEL_ID,
        "model_path": str(args.text_model_path),
        "tokenizer_path": str(args.text_model_path),
        "device_id": "0",
        "dtype": "bfloat16",
        "generation_method": "manual_greedy",
        "model_kind": "text",
        "max_model_len": 10240,
        "trust_remote_code": True,
        "enable_thinking": False,
        "request_timeout_ms": round(float(args.request_timeout_seconds) * 1_000),
        "runtime_library_paths": tuple(discover_aicpu_runtime_library_paths()),
    }


async def _run_ray_worker(
    args: argparse.Namespace, case: WorkloadCase
) -> dict[str, object]:
    import ray

    sample = _discover_text_sample(args.data_root)
    ray.init(
        address="auto",
        namespace=f"ascend-maze-performance-{case.case_id}",
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}},
    )
    ray_task = ray.remote(
        num_cpus=float(args.ray_task_num_cpus),
        max_calls=ray_smoke.RAY_TASK_MAX_CALLS,
    )(ray_smoke._execute_workflow_task_remote)  # noqa: SLF001
    transformers_config = _transformers_config(args)
    try:
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive_nodes) != 8:
            raise PerformancePilotError(
                f"expected 8 healthy Ray nodes, found {len(alive_nodes)}"
            )

        async def run_one(index: int) -> dict[str, object]:
            record = await asyncio.to_thread(
                ray_smoke._run_one_sample_ray,  # noqa: SLF001
                ray_task=ray_task,
                service_actor=None,
                inference_backend="transformers",
                transformers_config=transformers_config,
                sample=sample,
                target_model_id=TEXT_MODEL_ID,
                run_timeout_seconds=float(args.request_timeout_seconds),
                run_salt=f"{case.case_id}-{index}",
            )
            latency = record.get("latency_metrics")
            if not isinstance(latency, Mapping):
                latency = {}
            return {
                "schema_version": SCHEMA_VERSION,
                "executor": "ray",
                "case_id": case.case_id,
                "request_index": index,
                "sample_id": sample.sample_id,
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "client_e2e_started_at_ms": record.get("client_e2e_started_at_ms"),
                "client_e2e_finished_at_ms": record.get("client_e2e_finished_at_ms"),
                "client_e2e_ms": latency.get("client_e2e_ms"),
                "error": record.get("error"),
                "task_timings": record.get("task_timing_records", []),
                "tasks": record.get("tasks", []),
                "transformers_local_records": record.get(
                    "transformers_local_records", []
                ),
                "raw_record": record,
            }

        records, started_at_ms, finished_at_ms = await _run_scheduled(
            case.launch_offsets_ms, run_one
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "objective": OBJECTIVE,
            "executor": "ray",
            "case": case.payload(),
            "sample": sample.manifest(),
            "worker_max_calls": ray_smoke.RAY_TASK_MAX_CALLS,
            "ray_task_num_cpus": float(args.ray_task_num_cpus),
            "ray_nodes": [
                {
                    "node_id": node.get("NodeID"),
                    "node_ip": node.get("NodeManagerAddress"),
                    "cpu": node.get("Resources", {}).get("CPU"),
                }
                for node in alive_nodes
            ],
            "transformers_config": transformers_config,
            "workload_started_at_ms": started_at_ms,
            "workload_finished_at_ms": finished_at_ms,
            "records": records,
            "aggregate": _aggregate_requests(
                records,
                mode=case.mode,
                admission_window_seconds=case.admission_window_seconds,
            ),
        }
    finally:
        ray.shutdown()


def _case_from_file(path: Path) -> WorkloadCase:
    payload = _read_json(path)
    return WorkloadCase(
        case_id=str(payload["case_id"]),
        mode=str(payload["mode"]),
        request_count=int(payload["request_count"]),
        launch_offsets_ms=tuple(int(item) for item in payload["launch_offsets_ms"]),  # type: ignore[index]
        batch_size=(
            None if payload.get("batch_size") is None else int(payload["batch_size"])
        ),
        arrival_ratio=(
            None
            if payload.get("arrival_ratio") is None
            else float(payload["arrival_ratio"])
        ),
        arrival_rate_per_second=(
            None
            if payload.get("arrival_rate_per_second") is None
            else float(payload["arrival_rate_per_second"])
        ),
        average_workflow_seconds=(
            None
            if payload.get("average_workflow_seconds") is None
            else float(payload["average_workflow_seconds"])
        ),
        admission_window_seconds=(
            None
            if payload.get("admission_window_seconds") is None
            else float(payload["admission_window_seconds"])
        ),
    )


def _run_internal_worker(args: argparse.Namespace) -> int:
    if args.case_file is None or args.result_file is None:
        raise SystemExit(
            "--case-file and --result-file are required for internal worker"
        )
    case = _case_from_file(args.case_file)
    started_at_ms = int(time.time() * 1_000)
    try:
        result = asyncio.run(
            _run_maze_worker(args, case)
            if args.internal_worker == "maze"
            else _run_ray_worker(args, case)
        )
        result["worker_process_started_at_ms"] = started_at_ms
        result["worker_process_finished_at_ms"] = int(time.time() * 1_000)
        result["worker_environment"] = {
            "python": sys.version,
            "executable": sys.executable,
            "pid": os.getpid(),
        }
        _write_json(args.result_file, result)
        print(json.dumps({"status": "succeeded", "result_file": str(args.result_file)}))
        return 0 if result["aggregate"]["failed"] == 0 else 20  # type: ignore[index]
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "objective": OBJECTIVE,
            "executor": args.internal_worker,
            "case": case.payload(),
            "status": "worker_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "worker_process_started_at_ms": started_at_ms,
            "worker_process_finished_at_ms": int(time.time() * 1_000),
        }
        _write_json(args.result_file, failure)
        print(json.dumps({"status": "failed", "error": failure["error"]}))
        return 99


def _container_output_path(host_path: Path, state_root: Path) -> Path:
    node_root = (state_root / "node-0").resolve()
    try:
        relative = host_path.resolve().relative_to(node_root)
    except ValueError as exc:
        raise PerformancePilotError(
            f"output directory must be below the node-0 state mount: {node_root}"
        ) from exc
    return Path("/workspace/state") / relative


def _git_environment() -> dict[str, object]:
    def run(*argv: str) -> str:
        return subprocess.check_output(argv, cwd=REPO_ROOT, text=True).strip()

    try:
        revision = run("git", "rev-parse", "HEAD")
        status = run("git", "status", "--short")
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"revision": revision, "dirty": bool(status), "status": status.splitlines()}


def _control_environment(state_root: Path) -> dict[str, object]:
    config_path = state_root / "node-0" / "control-plane" / "controller.toml"
    catalog_path = state_root / "node-0" / "control-plane" / "model_catalog.toml"
    if not config_path.is_file() or not catalog_path.is_file():
        raise PerformancePilotError("logical-cluster control configuration is missing")
    config = tomllib.loads(config_path.read_text(encoding="ascii"))
    catalog = tomllib.loads(catalog_path.read_text(encoding="ascii"))
    return {
        "profile": config.get("profile"),
        "controller_config_path": str(config_path),
        "controller_config_sha256": _sha256(config_path),
        "config": config,
        "model_catalog_path": str(catalog_path),
        "model_catalog_sha256": _sha256(catalog_path),
        "model_catalog": catalog,
    }


def _worker_command(
    *,
    args: argparse.Namespace,
    executor: str,
    container_case_path: Path,
    container_result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(
            Path(
                "/home/user2/workplace/Ascend-Maze/tools/logical_cluster_performance.py"
            )
        ),
        "--internal-worker",
        executor,
        "--case-file",
        str(container_case_path),
        "--result-file",
        str(container_result_path),
        "--control-socket",
        str(DEFAULT_CONTROL_SOCKET),
        "--data-root",
        str(args.data_root),
        "--text-model-path",
        str(args.text_model_path),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--ray-task-num-cpus",
        str(args.ray_task_num_cpus),
    ]


def _run_container_worker(
    *,
    args: argparse.Namespace,
    executor: str,
    case: WorkloadCase,
    output_dir: Path,
    container_output_dir: Path,
) -> dict[str, object]:
    case_dir = output_dir / "cases" / case.case_id / executor
    container_case_dir = container_output_dir / "cases" / case.case_id / executor
    case_file = case_dir / "case.json"
    result_file = case_dir / "runner.json"
    resource_path = case_dir / "resource_samples.jsonl"
    _write_json(case_file, case.payload())
    worker_argv = _worker_command(
        args=args,
        executor=executor,
        container_case_path=container_case_dir / "case.json",
        container_result_path=container_case_dir / "runner.json",
    )
    shell_command = (
        f"source {shlex.quote(str(CONTAINER_ENV))}; exec {shlex.join(worker_argv)}"
    )
    docker_command = ["docker", "exec", CONTAINER_NAME, "bash", "-lc", shell_command]
    monitor = HostResourceMonitor(
        output_path=resource_path,
        interval_seconds=float(args.resource_sample_interval_seconds),
    )
    monitor.start()
    time.sleep(float(args.resource_baseline_seconds))
    started_at_ms = int(time.time() * 1_000)
    case_dir.mkdir(parents=True, exist_ok=True)
    with (
        (case_dir / "stdout.log").open("w", encoding="utf-8") as stdout_handle,
        (case_dir / "stderr.log").open("w", encoding="utf-8") as stderr_handle,
    ):
        try:
            completed = subprocess.run(
                docker_command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=float(args.case_timeout_seconds),
                check=False,
            )
            exit_code = completed.returncode
            timeout_error = None
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            timeout_error = str(exc)
    samples = monitor.stop()
    finished_at_ms = int(time.time() * 1_000)
    result = (
        _read_json(result_file)
        if result_file.is_file()
        else {
            "schema_version": SCHEMA_VERSION,
            "executor": executor,
            "status": "missing_runner_result",
            "error": timeout_error or f"worker exited with code {exit_code}",
        }
    )
    workload_start = result.get("workload_started_at_ms")
    workload_finish = result.get("workload_finished_at_ms")
    if not isinstance(workload_start, int):
        workload_start = started_at_ms
    if not isinstance(workload_finish, int):
        workload_finish = finished_at_ms
    result["process"] = {
        "docker_command": docker_command,
        "exit_code": exit_code,
        "timeout_error": timeout_error,
        "started_at_ms": started_at_ms,
        "finished_at_ms": finished_at_ms,
        "stdout_path": str(case_dir / "stdout.log"),
        "stderr_path": str(case_dir / "stderr.log"),
    }
    result["resource_samples_path"] = str(resource_path)
    result["resources"] = _aggregate_resources(
        samples,
        started_at_ms=workload_start,
        finished_at_ms=workload_finish,
    )
    _write_json(case_dir / "result.json", result)
    return result


def _fmt(value: object, digits: int = 2) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _render_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Ascend-Maze / Ray 八逻辑节点性能 Pilot",
        "",
        "> 本报告是单台 8 卡主机上的八容器逻辑集群结果，不代表真实跨机网络性能。",
        "",
        "## 实验契约",
        "",
        "- Workflow：`tbench.retail_cancel`",
        "- 模型：`Qwen3-4B`，Transformers `manual_greedy`",
        "- 生成参数：`max_tokens=4096`、`temperature=0`、`max_model_len=10240`",
        "- 模型加载：计入每个请求 E2E；模型 Task 进程一次性使用",
        "- Ray：每个 Task 请求逻辑节点全部 20 CPU，保证每节点同时至多一个 Task；`max_calls=1`",
        "- Maze：performance 配置启用 HACS、static anchor、Standby 和多副本；当前全局 Worker 复用上限仍为 1",
        "- E2E：客户端开始准备并提交请求，到终态结果返回；`DestroyRun` 不计入",
        "",
        "## 汇总",
        "",
        "| Case | 执行器 | 成功/总数 | E2E P95 (ms) | 吞吐 (req/s) | CPU 均值 (%) | CPU 增量 (%) | NPU 八卡均值 (%) | 单卡 NPU 峰值 (%) | HBM 增量峰值 (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    results = summary.get("results", [])
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, Mapping):
                continue
            aggregate = result.get("aggregate")
            resources = result.get("resources")
            if not isinstance(aggregate, Mapping):
                aggregate = {}
            if not isinstance(resources, Mapping):
                resources = {}
            cpu = resources.get("cluster_cpu_utilization_pct")
            npu = resources.get("cluster_npu_utilization_pct")
            max_npu = resources.get("max_device_npu_utilization_pct")
            cpu_mean = cpu.get("mean") if isinstance(cpu, Mapping) else None
            npu_mean = npu.get("mean") if isinstance(npu, Mapping) else None
            max_npu_peak = max_npu.get("max") if isinstance(max_npu, Mapping) else None
            case = result.get("case")
            case_id = case.get("case_id") if isinstance(case, Mapping) else "unknown"
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(case_id),
                        str(result.get("executor")),
                        f"{aggregate.get('succeeded', 0)}/{aggregate.get('request_count', 0)}",
                        _fmt(aggregate.get("p95_e2e_ms")),
                        _fmt(aggregate.get("throughput_requests_per_second"), 4),
                        _fmt(cpu_mean),
                        _fmt(resources.get("incremental_cluster_cpu_utilization_pct")),
                        _fmt(npu_mean),
                        _fmt(max_npu_peak),
                        _fmt(resources.get("peak_incremental_hbm_mb"), 0),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "P95 在单请求或双请求 Pilot 中仅用于验证统计链路，不具有稳定分位数意义。正式结论需要增加重复次数和负载点。",
            "",
            "## 到达负载",
            "",
            "Arrival ratio 定义为 `arrival_rate × average_workflow_seconds`。报告同时保留到达率、准入窗口内完成量以及排空后的完整请求 E2E。",
            "",
            "## 可审计文件",
            "",
            "- `plan.json`：实验顺序、负载计划和冻结配置",
            "- `summary.json`：全部请求、聚合指标、环境和资源统计",
            "- `cases/*/*/runner.json`：容器内原始执行记录",
            "- `cases/*/*/resource_samples.jsonl`：宿主机 CPU/NPU/HBM 时间序列",
            "- `cases/*/*/stdout.log`、`stderr.log`：每个执行器的控制台证据",
            "",
            "## 解释边界",
            "",
            "该 Pilot 验证统一口径和执行链路，不用于宣称最终性能优劣。文件系统页缓存、运行顺序和单机容器网络均已保留在证据中，正式实验需用分块交替重复控制这些因素。",
            "",
        )
    )
    return "\n".join(lines)


def _run_orchestrator(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / f"pilot-{timestamp}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(args)
    order = _execution_order(cases, str(args.executor))
    control_environment = _control_environment(args.state_root)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "objective": OBJECTIVE,
        "created_at_ms": int(time.time() * 1_000),
        "executor": args.executor,
        "cases": [case.payload() for case in cases],
        "execution_order": [
            {
                "ordinal": ordinal,
                "case_id": case.case_id,
                "executor": executor,
                "pair_position": pair_position,
            }
            for ordinal, (case, executor, pair_position) in enumerate(order, start=1)
        ],
        "contract": {
            "dataset": "tbench",
            "workflow": "retail_cancel",
            "model_id": TEXT_MODEL_ID,
            "model_path": str(args.text_model_path),
            "inference_backend": "transformers",
            "generation_method": "manual_greedy",
            "max_tokens": 4096,
            "temperature": 0.0,
            "max_model_len": 10240,
            "model_load_in_request_e2e": True,
            "destroy_in_request_e2e": False,
            "ray_worker_max_calls": ray_smoke.RAY_TASK_MAX_CALLS,
            "ray_task_num_cpus": float(args.ray_task_num_cpus),
        },
        "control_environment": control_environment,
        "git": _git_environment(),
        "command": sys.argv,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "plan.json", plan)
    if args.plan_only:
        summary = {
            **plan,
            "result": "plan_only_succeeded",
            "results": [],
        }
        _write_json(output_dir / "summary.json", summary)
        (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
        print(
            json.dumps({"result": "plan_only_succeeded", "output_dir": str(output_dir)})
        )
        return 0
    if control_environment.get("profile") != "performance":
        raise PerformancePilotError(
            "logical Controller is not using the performance profile; restart with "
            "deploy/logical_cluster/logical_cluster.sh control-up performance"
        )
    if not args.text_model_path.is_dir():
        raise PerformancePilotError(
            f"Qwen3-4B model path is missing: {args.text_model_path}"
        )
    container_output_dir = _container_output_path(output_dir, args.state_root)
    results: list[dict[str, object]] = []
    for ordinal, (case, executor, pair_position) in enumerate(order, start=1):
        print(
            json.dumps(
                {
                    "event": "case_start",
                    "ordinal": ordinal,
                    "case_id": case.case_id,
                    "executor": executor,
                }
            ),
            flush=True,
        )
        result = _run_container_worker(
            args=args,
            executor=executor,
            case=case,
            output_dir=output_dir,
            container_output_dir=container_output_dir,
        )
        result["execution_ordinal"] = ordinal
        result["pair_position"] = pair_position
        results.append(result)
        _write_json(output_dir / "partial_summary.json", {**plan, "results": results})
        print(
            json.dumps(
                {
                    "event": "case_finish",
                    "ordinal": ordinal,
                    "case_id": case.case_id,
                    "executor": executor,
                    "aggregate": result.get("aggregate"),
                }
            ),
            flush=True,
        )
    failed = [
        result
        for result in results
        if not isinstance(result.get("aggregate"), Mapping)
        or result["aggregate"].get("failed") != 0  # type: ignore[index]
        or result.get("process", {}).get("exit_code") != 0  # type: ignore[union-attr]
    ]
    summary = {
        **plan,
        "completed_at_ms": int(time.time() * 1_000),
        "result": "succeeded" if not failed else "failed",
        "result_count": len(results),
        "failed_result_count": len(failed),
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": summary["result"],
                "output_dir": str(output_dir),
                "report": str(output_dir / "report.md"),
            }
        )
    )
    return 0 if not failed else 20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired Maze/Ray performance cases on the logical cluster."
    )
    parser.add_argument(
        "--executor", choices=("paired", "maze", "ray"), default="paired"
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("batch", "arrival"),
        default=None,
    )
    parser.add_argument("--batch-size", action="append", type=int, default=None)
    parser.add_argument("--arrival-ratio", action="append", type=float, default=None)
    parser.add_argument("--average-workflow-seconds", type=float, default=30.0)
    parser.add_argument("--arrival-window-seconds", type=float, default=130.0)
    parser.add_argument("--resource-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--resource-baseline-seconds", type=float, default=3.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--case-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--ray-task-num-cpus", type=float, default=20.0)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--data-root", type=Path, default=qwen_smoke.DEFAULT_DATA_ROOT)
    parser.add_argument("--text-model-path", type=Path, default=TEXT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--internal-worker",
        choices=("maze", "ray"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--case-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--result-file", type=Path, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--control-socket",
        type=Path,
        default=DEFAULT_CONTROL_SOCKET,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = ["batch", "arrival"]
    if args.batch_size is None:
        args.batch_size = [1, 2]
    if args.arrival_ratio is None:
        args.arrival_ratio = [0.25]
    if any(item < 1 for item in args.batch_size):
        parser.error("--batch-size must be positive")
    if any(item <= 0 for item in args.arrival_ratio):
        parser.error("--arrival-ratio must be positive")
    for name in (
        "average_workflow_seconds",
        "arrival_window_seconds",
        "resource_sample_interval_seconds",
        "resource_baseline_seconds",
        "request_timeout_seconds",
        "case_timeout_seconds",
        "ray_task_num_cpus",
    ):
        if float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.state_root = args.state_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.text_model_path = args.text_model_path.expanduser().resolve()
    if args.case_file is not None:
        args.case_file = args.case_file.expanduser().resolve()
    if args.result_file is not None:
        args.result_file = args.result_file.expanduser().resolve()
    args.control_socket = args.control_socket.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.internal_worker is not None:
        return _run_internal_worker(args)
    try:
        return _run_orchestrator(args)
    except PerformancePilotError as exc:
        print(f"logical-cluster performance preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
