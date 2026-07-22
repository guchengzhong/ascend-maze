#!/usr/bin/env python3
"""Manual Qwen3-4B vLLM-Ascend smoke test.

This script is intentionally outside the normal pytest path because it requires
an Ascend NPU, local Qwen3-4B weights, CANN/ATB, and vLLM-Ascend.

It verifies the end-to-end Ascend-Maze path:

    InMemoryRuntimeClient
      -> qwen3_4b workflow
      -> invoke_qwen()
      -> ascend_maze.inference.chat()
      -> VllmAscendInferenceEngineAdapter
      -> vLLM-Ascend OpenAI-compatible service

Typical usage from the repository root:

    PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}" \
      /home/user2/workplace/miniconda3/envs/ascend-maze/bin/python \
      tools/qwen3_4b_vllm_ascend_smoke.py

Use ``--check-only`` for a quick dependency/model/device/preload audit without
starting the model service.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import importlib
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_MODEL_PATH = Path("/home/user2/workplace/model_weight/model_from_hf/Qwen3-4B")
DEFAULT_CATALOG_PATH = REPO_ROOT / "experiments/c14e/model_catalog.toml"
DEFAULT_CONDA_PYTHON = Path(
    "/home/user2/workplace/miniconda3/envs/ascend-maze/bin/python"
)
DEFAULT_PROMPT = "请只回答一个数字：2+3等于多少？ /no_think"
DEFAULT_MODULES = (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "transformers",
    "ray",
    "httpx",
    "prometheus_client",
    "acl",
)


class SmokePreflightError(RuntimeError):
    """Environment is not ready for this hardware smoke test."""


def _install_repo_path() -> None:
    for path in (str(SRC_ROOT), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(SRC_ROOT), str(REPO_ROOT)]
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def _default_python() -> Path:
    if "ASCEND_MAZE_PYTHON" in os.environ:
        return Path(os.environ["ASCEND_MAZE_PYTHON"]).expanduser()
    if DEFAULT_CONDA_PYTHON.is_file():
        return DEFAULT_CONDA_PYTHON
    return Path(sys.executable)


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if hasattr(value, "items_tuple"):
        return {
            str(key): _jsonable(item)
            for key, item in value.items_tuple()  # type: ignore[attr-defined]
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def emit(name: str, value: object) -> None:
    if isinstance(value, str):
        print(f"{name} {value}", flush=True)
    else:
        print(
            f"{name} "
            + json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True),
            flush=True,
        )


def _module_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:
        from importlib import metadata

        return metadata.version(module_name.replace("_", "-"))
    except Exception:
        return "unknown"


def check_current_python_modules(
    modules: tuple[str, ...] = DEFAULT_MODULES,
) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for name in modules:
        try:
            version = _module_version(name)
        except Exception as exc:
            results[name] = {
                "status": "missing_or_import_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            missing.append(name)
        else:
            results[name] = {"status": "ok", "version": version}
    if missing:
        raise SmokePreflightError(
            "required Python modules are unavailable: " + ", ".join(missing)
        )
    return results


def check_service_python_modules(
    python_executable: Path,
    modules: tuple[str, ...] = DEFAULT_MODULES,
) -> dict[str, dict[str, str]]:
    code = """
import importlib
import json
import sys

modules = sys.argv[1:]
results = {}
for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
    except Exception as exc:
        results[name] = {
            "status": "missing_or_import_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        results[name] = {"status": "ok", "version": str(version)}
print("MODULE_CHECK_JSON " + json.dumps(results, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python_executable), "-c", code, *modules],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    payload: dict[str, dict[str, str]] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("MODULE_CHECK_JSON "):
            payload = json.loads(line.removeprefix("MODULE_CHECK_JSON "))
    if completed.returncode != 0 or payload is None:
        raise SmokePreflightError(
            "service Python module check failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr[-1000:]}"
        )
    missing = [
        name
        for name, result in payload.items()
        if result.get("status") != "ok"
    ]
    if missing:
        raise SmokePreflightError(
            "service Python modules are unavailable: " + ", ".join(sorted(missing))
        )
    return payload


def _git_revision() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "0" * 40
    return value or "0" * 40


def _tail_logs(log_dir: Path, lines: int = 120) -> dict[str, str]:
    result: dict[str, str] = {}
    if not log_dir.exists():
        return result
    for path in sorted(log_dir.glob("*.log")):
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            result[str(path)] = f"<cannot read: {exc}>"
        else:
            result[str(path)] = "\n".join(content[-lines:])
    return result


def _residual_vllm_processes(model_path: Path, ports: tuple[int, ...]) -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid,ppid,pgid,stat,cmd"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    residual: list[str] = []
    for line in completed.stdout.splitlines():
        if "grep" in line or "rg " in line:
            continue
        if "vllm.entrypoints.openai.api_server" in line and (
            str(model_path) in line
            or any(f"--port {port}" in line for port in ports)
        ):
            residual.append(line.strip())
    return residual


def _device_summary(device_adapter: Any) -> list[dict[str, object]]:
    return [
        {
            "physical_device_id": device.physical_device_id,
            "chip_type": device.chip_type,
            "health": device.health,
            "used_hbm_mb": device.used_hbm_mb,
            "total_hbm_mb": device.total_hbm_mb,
            "processes": [
                {"pid": process.pid, "hbm_mb": process.hbm_mb}
                for process in device.processes
            ],
        }
        for device in device_adapter.devices()
    ]


def _processes_on_device(
    devices: list[dict[str, object]],
    device_id: str,
) -> list[dict[str, int]]:
    for device in devices:
        if device["physical_device_id"] == device_id:
            return list(device["processes"])  # type: ignore[arg-type]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual Qwen3-4B vLLM-Ascend smoke through Ascend-Maze.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--python-executable", type=Path, default=_default_python())
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--first-port", type=int, default=31180)
    parser.add_argument("--last-port", type=int, default=31220)
    parser.add_argument("--startup-timeout-ms", type=int, default=600_000)
    parser.add_argument("--request-timeout-ms", type=int, default=120_000)
    parser.add_argument("--run-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--hbm-recovery-tolerance-mb", type=int, default=1024)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Default: /tmp/ascend-maze-qwen3-4b-smoke-logs-<timestamp>",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run preflight checks without launching vLLM or submitting a run.",
    )
    parser.add_argument(
        "--allow-busy-device",
        action="store_true",
        help="Do not fail preflight when the selected NPU already has processes.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the final structured smoke summary.",
    )
    return parser.parse_args()


async def run_smoke(args: argparse.Namespace) -> int:
    _install_repo_path()
    try:
        from ascend_maze.ascend.contracts import AscendCorrectnessConfig
        from ascend_maze.ascend.dcmi import DcmiDeviceAdapter
        from ascend_maze.ascend.discovery import (
            build_ascend_node_capacity,
            discover_ascend_environment,
            discover_atb_runtime_library_preloads,
        )
        from ascend_maze.benchmark.workloads.qwen3_4b import build as build_workflow
        from ascend_maze.config.model_catalog import load_model_catalog
        from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
        from ascend_maze.control.service_process import NodeServiceProcessManager
        from ascend_maze.core.canonical import canonical_digest
        from ascend_maze.inference import InferenceCoordinator, ModelCatalog
        from ascend_maze.inference.adapters.vllm_ascend import (
            VllmAscendInferenceEngineAdapter,
        )
        from ascend_maze.placement import PlacementManager
    except Exception as exc:
        emit(
            "SMOKE_PREFLIGHT_FAILED",
            "failed to import Ascend-Maze runtime modules: "
            f"{type(exc).__name__}: {exc}",
        )
        emit("SMOKE_EXCEPTION_TRACEBACK", traceback.format_exc())
        return 2

    class _LoggingProcessManager(NodeServiceProcessManager):
        async def launch(self, request: Any, lease: Any) -> Any:
            payload = {
                "argv": list(request.argv),
                "working_directory": request.working_directory,
                "environment": dict(request.environment.items_tuple()),
                "node_id": lease.node_id,
                "boot_id": lease.boot_id,
                "npu_device_id": lease.npu_device_id,
                "log_path": str(
                    self.log_directory
                    / f"{request.instance_id}.{request.generation}.log"
                ),
            }
            launched_services.append(payload)
            emit("SERVICE_LAUNCH_JSON", payload)
            return await super().launch(request, lease)

    class _PortLeaseWrapper:
        def __init__(self, manager: NodeServiceProcessManager) -> None:
            self.manager = manager
            self._leases: dict[str, Any] = {}

        async def acquire(
            self,
            *,
            node_id: str,
            boot_id: str,
            owner_instance_id: str,
            generation: int,
        ) -> Any:
            lease = await self.manager.acquire_port(
                node_id=node_id,
                boot_id=boot_id,
                owner_instance_id=owner_instance_id,
                generation=generation,
            )
            self._leases[lease.port_lease_id] = lease
            return lease

        async def release(self, lease: Any) -> bool:
            released = await self.manager.release_port(lease)
            if released:
                self._leases.pop(lease.port_lease_id, None)
            return released

        def active_count(self) -> int:
            return len(self._leases)

    model_path = args.model_path.expanduser().resolve(strict=False)
    catalog_path = args.catalog_path.expanduser().resolve(strict=False)
    python_executable = args.python_executable.expanduser().resolve(strict=False)
    log_dir = (
        args.log_dir.expanduser().resolve(strict=False)
        if args.log_dir is not None
        else Path("/tmp") / f"ascend-maze-qwen3-4b-smoke-logs-{int(time.time())}"
    )
    node_id = "local_ascend_smoke"
    boot_id = f"boot_{int(time.time())}"
    launched_services: list[dict[str, object]] = []
    controller: Any = None
    service_manager: Any = None
    run_id: str | None = None
    destroyed = False
    result_code = 99
    cleanup_errors: list[str] = []
    summary: dict[str, object] = {
        "schema_version": 1,
        "result": "not_started",
        "model_path": str(model_path),
        "catalog_path": str(catalog_path),
        "python_executable": str(python_executable),
        "device_id": args.device_id,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "log_dir": str(log_dir),
    }

    try:
        emit("SMOKE_MODEL_PATH", str(model_path))
        emit("SMOKE_MODEL_EXISTS", model_path.is_dir())
        emit("SMOKE_CATALOG_PATH", str(catalog_path))
        emit("SMOKE_PYTHON", str(python_executable))
        emit("SMOKE_LOG_DIR", str(log_dir))
        emit("SMOKE_INPUT_JSON", {"prompt": args.prompt, "max_tokens": args.max_tokens})
        emit(
            "SMOKE_RUNTIME_PATH",
            "ascend_maze.benchmark.workloads.qwen3_4b:build -> "
            "invoke_qwen -> ascend_maze.inference.chat()",
        )

        if not model_path.is_dir():
            raise SmokePreflightError(f"model path does not exist: {model_path}")
        if not catalog_path.is_file():
            raise SmokePreflightError(f"catalog path does not exist: {catalog_path}")
        if not python_executable.is_file():
            raise SmokePreflightError(
                f"python executable does not exist: {python_executable}"
            )

        current_modules = check_current_python_modules()
        service_modules = check_service_python_modules(python_executable)
        emit("CURRENT_PYTHON_MODULES_JSON", current_modules)
        emit("SERVICE_PYTHON_MODULES_JSON", service_modules)
        summary["current_python_modules"] = current_modules
        summary["service_python_modules"] = service_modules

        device_adapter = DcmiDeviceAdapter()
        initial_devices = _device_summary(device_adapter)
        emit("ASCEND_DEVICES_JSON", initial_devices)
        summary["initial_devices"] = initial_devices
        selected_processes = _processes_on_device(initial_devices, args.device_id)
        if selected_processes and not args.allow_busy_device:
            raise SmokePreflightError(
                f"NPU {args.device_id} already has processes: {selected_processes}"
            )

        environment = discover_ascend_environment(device_adapter)
        preloads = discover_atb_runtime_library_preloads()
        emit("ASCEND_ENVIRONMENT_FINGERPRINT", environment.environment_fingerprint)
        emit(
            "ASCEND_ENVIRONMENT_VERSIONS_JSON",
            dict(environment.versions.items_tuple()),
        )
        emit("ATB_RUNTIME_PRELOADS_JSON", dict(preloads.items_tuple()))
        summary["environment_fingerprint"] = environment.environment_fingerprint
        summary["environment_versions"] = dict(environment.versions.items_tuple())
        summary["atb_runtime_preloads"] = dict(preloads.items_tuple())
        if not preloads:
            raise SmokePreflightError("ATB runtime preload libmki.so was not found")

        if args.check_only:
            summary["result"] = "check_only_succeeded"
            emit("SMOKE_RESULT", "check_only_succeeded")
            result_code = 0
            return result_code

        correctness = AscendCorrectnessConfig(
            task_slots_total=1,
            allow_colocation=False,
            max_tasks_per_worker=1,
            standby_min_idle=0,
            npu_system_reserved_hbm_mb=4096,
            npu_hbm_headroom_mb=1024,
            host_mem_headroom_mb=1024,
            io_slots_total=8,
            hbm_recovery_tolerance_mb=args.hbm_recovery_tolerance_mb,
        )
        node = build_ascend_node_capacity(
            node_id=node_id,
            boot_id=boot_id,
            node_ip="127.0.0.1",
            adapter=device_adapter,
            environment=environment,
            config=correctness,
        )
        selected_npus = tuple(
            npu for npu in node.npus if npu.device_id == args.device_id
        )
        if not selected_npus:
            raise SmokePreflightError(
                f"NPU {args.device_id} is not visible through DCMI"
            )
        node = replace(node, npus=selected_npus)
        emit(
            "SMOKE_NODE_CAPACITY_JSON",
            {
                "node_id": node.node_id,
                "boot_id": node.boot_id,
                "cpu_total": node.cpu_total,
                "mem_total_mb": node.mem_total_mb,
                "npus": [asdict(npu) for npu in node.npus],
            },
        )

        catalog_doc = load_model_catalog(
            catalog_path,
            environment_fingerprint=environment.environment_fingerprint,
        )
        base_spec = catalog_doc.specs[0]
        launch_options = dict(base_spec.launch_options.items_tuple())
        launch_options.update(
            {
                "block_size": 128,
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
                "log_level": "INFO",
                "max_num_seqs": 1,
            }
        )
        spec = replace(
            base_spec,
            artifact_path=str(model_path),
            tokenizer_path=str(model_path),
            request_capacity=1,
            min_replicas=0,
            max_replicas=1,
            max_parallel_starts=1,
            target_route_utilization=1.0,
            scale_up_sustain_ms=0,
            scale_down_idle_ms=0,
            scale_cooldown_ms=600_000,
            startup_timeout_ms=int(args.startup_timeout_ms),
            drain_timeout_ms=120_000,
            max_model_len=int(args.max_model_len),
            launch_options=launch_options,
            warmup_request={
                "messages": [
                    {"role": "user", "content": "Reply with exactly: ready"}
                ],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        emit(
            "MODEL_SPEC_JSON",
            {
                "model_id": spec.model_id,
                "artifact_path": spec.artifact_path,
                "backend": spec.backend,
                "dtype": spec.dtype,
                "max_model_len": spec.max_model_len,
                "instance_hbm_mb": spec.instance_hbm_mb,
                "request_capacity": spec.request_capacity,
                "launch_options": dict(spec.launch_options.items_tuple()),
                "startup_timeout_ms": spec.startup_timeout_ms,
                "scale_cooldown_ms": spec.scale_cooldown_ms,
            },
        )

        service_manager = _LoggingProcessManager(
            node_id=node_id,
            boot_id=boot_id,
            device_monitor=device_adapter,
            allowed_executables=(str(python_executable),),
            log_directory=log_dir,
            first_port=int(args.first_port),
            last_port=int(args.last_port),
            port_bind_host="127.0.0.1",
            hbm_recovery_tolerance_mb=args.hbm_recovery_tolerance_mb,
            poll_interval_ms=500,
        )
        port_wrapper = _PortLeaseWrapper(service_manager)
        adapter = VllmAscendInferenceEngineAdapter(
            process_backend=service_manager,
            python_executable=str(python_executable),
            endpoint_host_resolver=lambda lease: "127.0.0.1",
            bind_host="127.0.0.1",
            runtime_library_preloads=preloads,
            request_timeout_ms=int(args.request_timeout_ms),
            probe_timeout_ms=int(args.startup_timeout_ms),
            probe_interval_ms=1_000,
        )
        placement = PlacementManager(
            host_mem_headroom_mb=correctness.host_mem_headroom_mb,
            npu_hbm_headroom_mb=correctness.npu_hbm_headroom_mb,
            required_environment_fingerprint=environment.environment_fingerprint,
        )
        catalog = ModelCatalog(
            (spec,),
            adapters={"vllm_ascend": adapter},
            environment_capabilities=("ascend", "vllm_ascend"),
            max_single_npu_hbm_mb=max(
                npu.total_hbm_mb - npu.system_reserved_hbm_mb
                for npu in node.npus
            ),
        )
        inference = InferenceCoordinator(
            catalog=catalog,
            placement=placement,
            service_backend=service_manager,
            port_leases=port_wrapper,
            reconcile_interval_ms=500,
        )
        config_fingerprint = canonical_digest(
            {
                "profile": "qwen3-4b-smoke",
                "environment_fingerprint": environment.environment_fingerprint,
                "model_catalog_digest": catalog.content_digest,
                "device": args.device_id,
                "launch_options": dict(spec.launch_options.items_tuple()),
                "runtime_preloads": dict(preloads.items_tuple()),
            }
        )
        controller = InMemoryController(
            config_fingerprint=config_fingerprint,
            environment_fingerprint=environment.environment_fingerprint,
            build_revision=_git_revision(),
            node_capacities=(node,),
            placement=placement,
            inference=inference,
            dispatch_timeout_ms=int(args.startup_timeout_ms),
            shutdown_drain_timeout_ms=5_000,
            shutdown_cleanup_timeout_ms=120_000,
        )
        await controller.start()

        workflow = build_workflow()
        compiled = workflow.compile()
        task_ids = {
            node.task_name: task_id
            for task_id, node in compiled.tasks.items()
        }
        emit("WORKFLOW_FINGERPRINT", compiled.workflow_fingerprint)
        emit("WORKFLOW_TASK_IDS_JSON", task_ids)
        summary["workflow_fingerprint"] = compiled.workflow_fingerprint
        summary["workflow_task_ids"] = task_ids

        client = InMemoryRuntimeClient(controller)
        outcome = await client.submit(
            workflow,
            inputs={"prompt": args.prompt, "max_tokens": int(args.max_tokens)},
            submission_id=f"qwen3-4b-smoke-{int(time.time())}",
            run_deadline_ms=int(args.run_timeout_seconds * 1_000),
        )
        emit(
            "SUBMISSION_OUTCOME_JSON",
            {
                "submission_id": outcome.submission_id,
                "state": outcome.state.value,
                "run_id": outcome.run_id,
                "payload_hash": outcome.submission_payload_hash,
                "replayed": outcome.replayed,
                "error": outcome.error,
            },
        )
        if outcome.run_id is None:
            raise RuntimeError("submission did not produce a run_id")
        run_id = outcome.run_id
        summary["run_id"] = run_id

        deadline = time.monotonic() + float(args.run_timeout_seconds)
        last_state: object = None
        snapshot = controller.snapshot(run_id)
        while not snapshot.terminal:
            states = [
                (
                    instance.instance_id,
                    instance.generation,
                    instance.state.value,
                    instance.failure_reason,
                )
                for instance in inference.model_instances()
            ]
            if states != last_state:
                emit("MODEL_INSTANCE_STATES_JSON", states)
                last_state = states
            failed_or_stopped = [
                instance
                for instance in inference.model_instances()
                if instance.failure_reason
                or instance.state.value in {"failed", "stopped"}
            ]
            if failed_or_stopped:
                emit(
                    "MODEL_INSTANCE_FAILURE_JSON",
                    [
                        {
                            "instance_id": instance.instance_id,
                            "generation": instance.generation,
                            "state": instance.state.value,
                            "failure_reason": instance.failure_reason,
                        }
                        for instance in failed_or_stopped
                    ],
                )
                snapshot = await controller.cancel_run(
                    run_id,
                    reason="smoke_model_instance_failed",
                )
                break
            if time.monotonic() >= deadline:
                snapshot = await controller.cancel_run(run_id, reason="smoke_timeout")
                break
            await asyncio.sleep(1.0)
            snapshot = controller.snapshot(run_id)

        run_payload = {
            "run_id": snapshot.run_id,
            "status": snapshot.status.value,
            "failure": None
            if snapshot.failure is None
            else snapshot.failure.error_code,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "pending_reason": task.pending_reason,
                    "last_error": None
                    if task.last_error is None
                    else {
                        "error_code": task.last_error.error_code,
                        "message": task.last_error.message,
                        "phase": task.last_error.execution_phase,
                        "origin": task.last_error.origin,
                        "category": task.last_error.category,
                    },
                }
                for task in snapshot.task_states
            ],
        }
        emit("RUN_TERMINAL_JSON", run_payload)
        summary["run_terminal"] = run_payload

        model_events = [
            {
                "event_type": event.event_type,
                "model_id": event.model_id,
                "instance_id": event.instance_id,
                "run_id": event.run_id,
                "task_id": event.task_id,
                "route_lease_id": event.route_lease_id,
                "payload": dict(event.payload.items_tuple()),
            }
            for event in inference.events()
        ]
        model_instances = [
            {
                "instance_id": instance.instance_id,
                "model_id": instance.model_id,
                "generation": instance.generation,
                "state": instance.state.value,
                "node_id": instance.node_id,
                "npu_device_id": instance.npu_device_id,
                "endpoint_id": instance.endpoint_id,
                "route_occupancy": instance.route_occupancy,
                "actual_request_inflight": instance.actual_request_inflight,
                "failure_reason": instance.failure_reason,
            }
            for instance in inference.model_instances()
        ]
        inference_records = [asdict(record) for record in inference.request_records()]
        emit("MODEL_EVENTS_JSON", model_events)
        emit("MODEL_INSTANCES_JSON", model_instances)
        emit("INFERENCE_RECORDS_JSON", inference_records)
        summary["service_launches"] = launched_services
        summary["model_events"] = model_events
        summary["model_instances"] = model_instances
        summary["inference_records"] = inference_records

        if snapshot.status.value == "succeeded":
            invoke = controller.result(run_id, task_ids["invoke_qwen"])
            final = controller.result(run_id, task_ids["finalize_response"])
            emit("TASK_INVOKE_RESULT_JSON", invoke)
            emit("TASK_FINAL_RESULT_JSON", final)
            summary["invoke_result"] = invoke
            summary["final_result"] = final
            summary["result"] = "succeeded"
            emit("SMOKE_RESULT", "succeeded")
            result_code = 0
        else:
            log_tails = _tail_logs(log_dir)
            emit("SERVICE_LOG_TAILS_JSON", log_tails)
            summary["service_log_tails"] = log_tails
            summary["result"] = f"failed:run_status={snapshot.status.value}"
            emit("SMOKE_RESULT", summary["result"])
            result_code = 10

        destroy = await controller.destroy_run(run_id, force=True)
        destroyed = True
        emit("DESTROY_RESULT_JSON", destroy)
        summary["destroy_result"] = _jsonable(destroy)
    except SmokePreflightError as exc:
        summary["result"] = "preflight_failed"
        summary["error"] = str(exc)
        emit("SMOKE_PREFLIGHT_FAILED", str(exc))
        result_code = 2
    except Exception as exc:
        summary["result"] = "unexpected_exception"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        emit("SMOKE_EXCEPTION_TRACEBACK", summary["traceback"])
        emit("SERVICE_LOG_TAILS_JSON", _tail_logs(log_dir))
        result_code = 99
    finally:
        if controller is not None:
            if run_id is not None and not destroyed:
                try:
                    await controller.cancel_run(run_id, reason="smoke_cleanup")
                except Exception as exc:
                    cleanup_errors.append(f"cancel_run:{type(exc).__name__}:{exc}")
                try:
                    await controller.destroy_run(run_id, force=True)
                except Exception as exc:
                    cleanup_errors.append(f"destroy_run:{type(exc).__name__}:{exc}")
            try:
                shutdown = await controller.shutdown(force=True, drain_timeout_ms=0)
                emit("CONTROLLER_SHUTDOWN_JSON", shutdown)
                summary["controller_shutdown"] = _jsonable(shutdown)
            except Exception as exc:
                cleanup_errors.append(
                    f"controller_shutdown:{type(exc).__name__}:{exc}"
                )
                emit("CONTROLLER_SHUTDOWN_ERROR", traceback.format_exc())
        elif service_manager is not None:
            try:
                await service_manager.close(timeout_ms=120_000)
            except Exception as exc:
                cleanup_errors.append(
                    f"service_manager_close:{type(exc).__name__}:{exc}"
                )
                emit("SERVICE_MANAGER_CLOSE_ERROR", traceback.format_exc())

        try:
            final_devices = _device_summary(DcmiDeviceAdapter())
        except Exception as exc:
            final_devices = []
            cleanup_errors.append(f"final_dcmi_audit:{type(exc).__name__}:{exc}")
            emit("FINAL_ASCEND_AUDIT_FAILED", traceback.format_exc())
        else:
            emit("FINAL_ASCEND_DEVICES_JSON", final_devices)
            summary["final_devices"] = final_devices

        ports = tuple(range(int(args.first_port), int(args.last_port) + 1))
        residual = _residual_vllm_processes(model_path, ports)
        emit("FINAL_RESIDUAL_VLLM_PROCESSES_JSON", residual)
        summary["residual_vllm_processes"] = residual
        final_selected_processes = _processes_on_device(final_devices, args.device_id)
        summary["cleanup_errors"] = cleanup_errors
        if result_code == 0 and (
            cleanup_errors or residual or final_selected_processes
        ):
            summary["result"] = "cleanup_failed"
            result_code = 11

        emit("SMOKE_SUMMARY_JSON", summary)
        if args.output_json is not None:
            output = args.output_json.expanduser().resolve(strict=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(_jsonable(summary), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            emit("SMOKE_SUMMARY_PATH", str(output))
    return result_code


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.max_model_len < 1:
        raise SystemExit("--max-model-len must be positive")
    if not 0 < args.gpu_memory_utilization <= 0.9:
        raise SystemExit("--gpu-memory-utilization must be within (0, 0.9]")
    if args.first_port > args.last_port:
        raise SystemExit("--first-port cannot exceed --last-port")
    exit_code = asyncio.run(run_smoke(args))
    emit("SMOKE_EXIT_CODE", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
