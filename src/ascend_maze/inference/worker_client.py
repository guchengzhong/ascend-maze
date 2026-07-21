"""Attempt-local inference clients used inside service client Workers."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Protocol

from ascend_maze.core.canonical import CanonicalValue, FrozenMap
from ascend_maze.inference.adapters.vllm_ascend import (
    HttpxVllmTransport,
    VllmHttpResponse,
    VllmHttpTransport,
)
from ascend_maze.inference.contracts import (
    ChatRequest,
    ChatResponse,
    InferenceCallError,
    InferenceWorkerConfig,
    ModelRouteContext,
)


class WorkerInferenceClient(Protocol):
    async def invoke_chat(
        self,
        context: ModelRouteContext,
        request: ChatRequest,
    ) -> ChatResponse: ...

    async def close(self) -> None: ...


def create_worker_inference_client(
    config: InferenceWorkerConfig,
) -> WorkerInferenceClient:
    if config.adapter_name == "vllm_ascend":
        return VllmAscendWorkerClient(request_timeout_ms=config.request_timeout_ms)
    if config.adapter_name == "fake":
        return FakeWorkerInferenceClient(config.adapter_options)
    raise InferenceCallError(
        "model_adapter_unsupported",
        f"unsupported Worker inference adapter: {config.adapter_name}",
    )


class VllmAscendWorkerClient:
    def __init__(
        self,
        *,
        request_timeout_ms: int,
        transport: VllmHttpTransport | None = None,
    ) -> None:
        self.request_timeout_ms = request_timeout_ms
        self.transport = transport or HttpxVllmTransport()

    async def invoke_chat(
        self,
        context: ModelRouteContext,
        request: ChatRequest,
    ) -> ChatResponse:
        payload = {
            "model": context.model_id,
            "messages": [_plain(message) for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": float(request.temperature),
        }
        started = monotonic()
        response = await self.transport.request(
            "POST",
            f"{context.endpoint_id}/v1/chat/completions",
            json_body=payload,
            timeout_ms=self.request_timeout_ms,
        )
        return _decode_vllm_response(response, started=started)

    async def close(self) -> None:
        await self.transport.close()


class FakeWorkerInferenceClient:
    def __init__(
        self,
        options: FrozenMap[CanonicalValue, CanonicalValue],
    ) -> None:
        prefix = options.get("response_prefix")
        delay = options.get("invoke_delay_ms", 0)
        failure = options.get("fail_invoke")
        if not isinstance(prefix, str):
            raise InferenceCallError(
                "model_adapter_config_invalid", "fake response prefix is invalid"
            )
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise InferenceCallError(
                "model_adapter_config_invalid", "fake invoke delay is invalid"
            )
        if failure is not None and not isinstance(failure, str):
            raise InferenceCallError(
                "model_adapter_config_invalid", "fake failure plan is invalid"
            )
        self.prefix = prefix
        self.delay_ms = delay
        self.failure = failure

    async def invoke_chat(
        self,
        context: ModelRouteContext,
        request: ChatRequest,
    ) -> ChatResponse:
        del context
        started = monotonic()
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1_000)
        if self.failure is not None:
            raise RuntimeError(self.failure)
        content = str(request.messages[-1]["content"])
        text = f"{self.prefix}:{content}"
        return ChatResponse(
            text=text,
            finish_reason="stop",
            input_tokens=max(1, len(content.split())),
            output_tokens=max(1, len(text.split())),
            engine_queue_depth=0,
            prefix_cache_hit=False,
            ttft_ms=self.delay_ms,
            total_duration_ms=max(0, int((monotonic() - started) * 1_000)),
        )

    async def close(self) -> None:
        return None


def _decode_vllm_response(
    response: VllmHttpResponse,
    *,
    started: float,
) -> ChatResponse:
    if response.status_code != 200:
        code = (
            "model_service_unavailable"
            if response.status_code >= 500
            else "model_protocol_failed"
        )
        raise InferenceCallError(
            code, f"vLLM chat returned HTTP {response.status_code}"
        )
    body = response.json()
    try:
        if not isinstance(body, dict):
            raise TypeError
        choices = body["choices"]
        usage = body["usage"]
        if not isinstance(choices, list) or not choices or not isinstance(usage, dict):
            raise TypeError
        choice = choices[0]
        if not isinstance(choice, dict):
            raise TypeError
        message = choice["message"]
        if not isinstance(message, dict):
            raise TypeError
        text = message["content"]
        finish_reason = choice["finish_reason"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]
        if not isinstance(text, str) or not isinstance(finish_reason, str):
            raise TypeError
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
            raise TypeError
        if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
            raise TypeError
    except (IndexError, KeyError, TypeError) as exc:
        raise InferenceCallError(
            "model_protocol_failed", "vLLM chat response schema is invalid"
        ) from exc
    return ChatResponse(
        text=text,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        engine_queue_depth=None,
        prefix_cache_hit=None,
        ttft_ms=None,
        total_duration_ms=max(0, int((monotonic() - started) * 1_000)),
    )


def _plain(value: CanonicalValue) -> object:
    if isinstance(value, FrozenMap):
        return {str(key): _plain(item) for key, item in value.items_tuple()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value
