from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
from threading import get_ident
from types import ModuleType

import pytest

from ascend_maze.contracts.resources import PlacementLease, ReservationVector
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.inference.adapters.vllm_ascend import (
    HttpxVllmTransport,
    VllmAscendInferenceEngineAdapter,
    VllmHttpResponse,
)
from ascend_maze.inference.contracts import (
    ChatRequest,
    InferenceCallError,
    ModelRouteContext,
    ModelSpec,
    PortLease,
    ServiceHandle,
    ServiceProcessProbe,
    ServiceStopResult,
)


def _spec(model_dir: Path, **changes: object) -> ModelSpec:
    model_dir.mkdir()
    values: dict[str, object] = {
        "model_id": "qwen3-4b",
        "catalog_revision": "catalog_1",
        "artifact_path": str(model_dir),
        "tokenizer_path": None,
        "artifact_revision": "a" * 64,
        "backend": "vllm_ascend",
        "dtype": "bfloat16",
        "quantization": None,
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "instance_cpu_num": 4,
        "instance_host_mem_mb": 8192,
        "weight_hbm_mb": 8000,
        "runtime_hbm_mb": 4000,
        "kv_cache_hbm_mb": 20000,
        "instance_hbm_mb": 36000,
        "npu_slots": 1,
        "allow_colocation": False,
        "request_capacity": 4,
        "required_capabilities": ("vllm_ascend",),
        "environment_fingerprint": "e" * 64,
        "launch_options": {
            "block_size": 128,
            "enable_prefix_caching": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.5,
            "log_level": "INFO",
            "max_num_batched_tokens": 1024,
        },
        "warmup_request": {
            "messages": [{"role": "user", "content": "Say ready."}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
        "min_replicas": 0,
        "max_replicas": 1,
    }
    values.update(changes)
    return ModelSpec(**values)  # type: ignore[arg-type]


def _lease() -> PlacementLease:
    return PlacementLease(
        lease_id="lease_1",
        reservation_kind="model_instance",
        run_id=None,
        task_id=None,
        attempt=None,
        node_id="node_a",
        boot_id="boot_1",
        npu_device_id="7",
        resources=ReservationVector(4, 8192, 0, 36000, 1),
        snapshot_version=1,
        created_at_ms=10,
        dispatch_deadline_ms=310_000,
        allow_npu_colocation=False,
        model_instance_id="instance_1",
    )


def _port() -> PortLease:
    return PortLease(
        port_lease_id="port_1",
        node_id="node_a",
        boot_id="boot_1",
        port=25000,
        owner_instance_id="instance_1",
        generation=1,
    )


class _ProcessBackend:
    def __init__(self) -> None:
        self.probe = ServiceProcessProbe(
            process_alive=True,
            port_open=True,
            binding_verified=True,
            physical_device_id="7",
            process_hbm_mb=34000,
        )

    async def launch(self, request, lease):
        raise AssertionError((request, lease))

    async def probe_process(self, handle, *, timeout_ms):
        del handle, timeout_ms
        return self.probe

    async def stop(self, handle, *, timeout_ms):
        del handle, timeout_ms
        return ServiceStopResult(True, True, True)


class _Transport:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.calls: list[tuple[str, str, object | None, int]] = []
        self.closed = False
        self.fail_health_status: int | None = None
        self.model_response: object | None = None
        self.chat_response: object | None = None

    async def request(self, method, url, *, json_body, timeout_ms):
        self.calls.append((method, url, json_body, timeout_ms))
        if url.endswith("/health"):
            return VllmHttpResponse(self.fail_health_status or 200, b"", {})
        if url.endswith("/v1/models"):
            if self.model_response is not None:
                return self._json(self.model_response)
            return self._json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen3-4b",
                            "root": str(self.model_dir),
                            "max_model_len": 2048,
                        }
                    ],
                }
            )
        if url.endswith("/v1/chat/completions"):
            if self.chat_response is not None:
                return self._json(self.chat_response)
            return self._json(
                {
                    "choices": [
                        {
                            "message": {"content": "ready"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                }
            )
        if url.endswith("/metrics"):
            return VllmHttpResponse(
                200,
                (
                    "# TYPE vllm:num_requests_waiting gauge\n"
                    "vllm:num_requests_waiting 2\n"
                    "# TYPE vllm:num_requests_running gauge\n"
                    "vllm:num_requests_running 1\n"
                ).encode(),
                {},
            )
        raise AssertionError(url)

    async def close(self) -> None:
        self.closed = True

    @staticmethod
    def _json(value: object) -> VllmHttpResponse:
        return VllmHttpResponse(200, json.dumps(value).encode(), {})


def _adapter(
    model_dir: Path,
    *,
    runtime_library_preloads: dict[str, str] | None = None,
    runtime_library_paths: tuple[str, ...] | None = None,
) -> tuple[
    VllmAscendInferenceEngineAdapter,
    _ProcessBackend,
    _Transport,
]:
    backend = _ProcessBackend()
    transport = _Transport(model_dir)
    adapter = VllmAscendInferenceEngineAdapter(
        process_backend=backend,
        python_executable=sys.executable,
        endpoint_host_resolver=lambda lease: "127.0.0.1",
        transport=transport,
        runtime_library_preloads=runtime_library_preloads,
        runtime_library_paths=runtime_library_paths,
        request_timeout_ms=1_000,
        probe_timeout_ms=2_000,
        probe_interval_ms=1,
    )
    return adapter, backend, transport


def test_httpx_transport_reuses_sync_pool_across_event_loops(monkeypatch) -> None:
    clients: list[object] = []
    request_threads: list[int] = []

    class Response:
        status_code = 200
        content = b"ok"
        headers = {"content-type": "text/plain"}

    class Client:
        def __init__(self, **kwargs: object) -> None:
            clients.append((self, kwargs))

        def request(self, *args: object, **kwargs: object) -> Response:
            request_threads.append(get_ident())
            return Response()

        def close(self) -> None:
            return None

    fake_httpx = ModuleType("httpx")
    fake_httpx.Client = Client  # type: ignore[attr-defined]
    fake_httpx.Limits = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fake_httpx.TimeoutException = TimeoutError  # type: ignore[attr-defined]
    fake_httpx.RequestError = OSError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    transport = HttpxVllmTransport()

    for _ in range(2):
        response = asyncio.run(
            transport.request(
                "GET",
                "http://127.0.0.1/health",
                json_body=None,
                timeout_ms=1_000,
            )
        )
        assert response.status_code == 200
    asyncio.run(transport.close())

    assert len(clients) == 1
    assert len(request_threads) == 2
    assert all(thread_id != get_ident() for thread_id in request_threads)


def test_vllm_launch_request_is_deterministic_and_lease_bound(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "model")
    preload = tmp_path / "libmki.so"
    preload.write_bytes(b"pinned test runtime library")
    preload_digest = hashlib.sha256(preload.read_bytes()).hexdigest()
    adapter, _, _ = _adapter(
        Path(spec.artifact_path),
        runtime_library_preloads={str(preload): preload_digest},
    )
    request = adapter.build_launch_request(spec, _lease(), _port())

    assert request.endpoint_id == "http://127.0.0.1:25000"
    assert request.port_lease_id == "port_1"
    assert request.environment["ASCEND_RT_VISIBLE_DEVICES"] == "7"
    assert request.environment["ASCEND_MAZE_ARTIFACT_REVISION"] == "a" * 64
    assert request.environment["LD_PRELOAD"] == str(preload.resolve())
    assert len(request.environment["ASCEND_MAZE_RUNTIME_LIBRARY_PRELOAD_DIGEST"]) == 64
    assert request.argv[:3] == (
        str(Path(sys.executable).resolve()),
        "-m",
        "vllm.entrypoints.openai.api_server",
    )
    assert request.argv[request.argv.index("--model") + 1] == spec.artifact_path
    assert request.argv[request.argv.index("--gpu-memory-utilization") + 1] == "0.5"
    assert request.argv[request.argv.index("--block-size") + 1] == "128"
    assert request.argv[request.argv.index("--max-num-batched-tokens") + 1] == "1024"
    assert "--enforce-eager" in request.argv
    assert "--enable-prefix-caching" in request.argv


def test_vllm_launch_request_injects_aicpu_runtime_paths(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model"
    aicpu = tmp_path / "aicpu"
    inherited = tmp_path / "inherited"
    aicpu.mkdir()
    inherited.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(inherited))
    adapter, _, _ = _adapter(
        model,
        runtime_library_paths=(str(aicpu), str(aicpu)),
    )

    request = adapter.build_launch_request(_spec(model), _lease(), _port())

    paths = request.environment["LD_LIBRARY_PATH"].split(":")
    assert paths[:2] == [str(aicpu.resolve()), str(inherited)]


def test_vllm_adapter_rejects_unpinned_runtime_library_preloads(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    preload = tmp_path / "libmki.so"
    preload.write_bytes(b"installed runtime library")

    with pytest.raises(ContractValidationError, match="digest mismatch"):
        _adapter(model, runtime_library_preloads={str(preload): "0" * 64})
    with pytest.raises(ContractValidationError, match="does not exist"):
        _adapter(
            model,
            runtime_library_preloads={str(tmp_path / "missing.so"): "0" * 64},
        )


def test_vllm_model_spec_rejects_unverified_or_unbounded_options(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    spec = _spec(model)
    adapter, _, _ = _adapter(model)
    adapter.validate_model_spec(spec)

    with pytest.raises(ContractValidationError, match="allow_colocation"):
        adapter.validate_model_spec(
            _spec(tmp_path / "colocated", allow_colocation=True)
        )
    with pytest.raises(ContractValidationError, match="unsupported"):
        adapter.validate_model_spec(
            _spec(
                tmp_path / "arbitrary",
                launch_options={
                    "gpu_memory_utilization": 0.5,
                    "arbitrary_cli": "--download-dir=/tmp",
                },
            )
        )
    with pytest.raises(ContractValidationError, match="max_num_batched_tokens"):
        adapter.validate_model_spec(
            _spec(
                tmp_path / "bad_batched_tokens",
                launch_options={
                    "gpu_memory_utilization": 0.5,
                    "max_num_batched_tokens": 0,
                },
            )
        )
    with pytest.raises(ContractValidationError, match="quantized"):
        adapter.validate_model_spec(_spec(tmp_path / "quantized", quantization="awq"))


def test_vllm_probe_warmup_chat_metrics_and_close(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec = _spec(tmp_path / "model")
        adapter, _, transport = _adapter(Path(spec.artifact_path))
        launch = adapter.build_launch_request(spec, _lease(), _port())
        handle = ServiceHandle(
            service_handle_id="service_1",
            instance_id="instance_1",
            generation=1,
            endpoint_id=launch.endpoint_id,
            node_id="node_a",
            boot_id="boot_1",
            npu_device_id="7",
            process_id=123,
            port_lease_id="port_1",
            port=25000,
        )
        probe = await adapter.probe(handle, spec)
        assert probe.model_id == spec.model_id
        assert probe.artifact_revision == spec.artifact_revision
        assert probe.physical_device_id == "7"
        assert probe.process_hbm_mb == 34000

        warmup = await adapter.warmup(handle, spec)
        assert warmup.succeeded
        assert warmup.response_digest is not None
        assert len(warmup.response_digest) == 64
        warmup_payload = transport.calls[-1][2]
        assert isinstance(warmup_payload, dict)
        assert warmup_payload["frequency_penalty"] == 0.0
        assert warmup_payload["presence_penalty"] == 0.0
        assert warmup_payload["repetition_penalty"] == 1.0

        response = await adapter.invoke_chat(
            ModelRouteContext(
                route_lease_id="route_1",
                model_id=spec.model_id,
                adapter_name=adapter.name,
                endpoint_id=handle.endpoint_id,
                instance_id=handle.instance_id,
                instance_generation=handle.generation,
            ),
            ChatRequest.create([{"role": "user", "content": "hello"}]),
        )
        assert response.text == "ready"
        assert response.finish_reason == "stop"
        assert response.input_tokens == 4
        assert response.output_tokens == 1
        assert response.engine_queue_depth is None
        chat_payload = transport.calls[-1][2]
        assert isinstance(chat_payload, dict)
        assert chat_payload["frequency_penalty"] == 0.0
        assert chat_payload["presence_penalty"] == 0.0
        assert chat_payload["repetition_penalty"] == 1.0

        metrics = await adapter.read_metrics(handle)
        assert metrics.queue_depth == 2
        assert metrics.actual_request_inflight == 1
        await adapter.close()
        assert transport.closed

    asyncio.run(scenario())


def test_vllm_probe_rejects_process_exit_and_model_mismatch(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec = _spec(tmp_path / "model")
        adapter, backend, transport = _adapter(Path(spec.artifact_path))
        handle = ServiceHandle(
            service_handle_id="service_1",
            instance_id="instance_1",
            generation=1,
            endpoint_id="http://127.0.0.1:25000",
            node_id="node_a",
            boot_id="boot_1",
            npu_device_id="7",
            process_id=123,
            port_lease_id="port_1",
            port=25000,
        )
        backend.probe = ServiceProcessProbe(
            process_alive=False,
            port_open=False,
            binding_verified=False,
            physical_device_id="7",
            process_hbm_mb=None,
            exit_code=9,
        )
        with pytest.raises(InferenceCallError) as exited:
            await adapter.probe(handle, spec)
        assert exited.value.error_code == "model_process_exited"

        backend.probe = ServiceProcessProbe(True, True, True, "7", 34000)
        transport.model_dir = tmp_path / "different"
        with pytest.raises(InferenceCallError) as mismatch:
            await adapter.probe(handle, spec)
        assert mismatch.value.error_code == "model_identity_mismatch"

    asyncio.run(scenario())


def test_vllm_rejects_malformed_protocol_without_echoing_response_body(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        spec = _spec(tmp_path / "model")
        adapter, _, transport = _adapter(Path(spec.artifact_path))
        handle = ServiceHandle(
            service_handle_id="service_1",
            instance_id="instance_1",
            generation=1,
            endpoint_id="http://127.0.0.1:25000",
            node_id="node_a",
            boot_id="boot_1",
            npu_device_id="7",
            process_id=123,
            port_lease_id="port_1",
            port=25000,
        )

        transport.model_response = []
        with pytest.raises(InferenceCallError) as malformed_model:
            await adapter.probe(handle, spec)
        assert malformed_model.value.error_code == "model_identity_mismatch"

        transport.chat_response = {"choices": "not-a-list", "usage": {}}
        with pytest.raises(InferenceCallError) as malformed_chat:
            await adapter.invoke_chat(
                ModelRouteContext(
                    route_lease_id="route_1",
                    model_id=spec.model_id,
                    adapter_name=adapter.name,
                    endpoint_id=handle.endpoint_id,
                    instance_id=handle.instance_id,
                    instance_generation=handle.generation,
                ),
                ChatRequest.create(
                    [{"role": "user", "content": "private prompt"}]
                ),
            )
        assert malformed_chat.value.error_code == "model_protocol_failed"

        secret_body = b'{"error":"private prompt and generated response"}'
        with pytest.raises(InferenceCallError) as rejected:
            adapter._raise_http_error(VllmHttpResponse(400, secret_body, {}), "chat")
        assert "private prompt" not in str(rejected.value)
        assert "generated response" not in str(rejected.value)

    asyncio.run(scenario())
