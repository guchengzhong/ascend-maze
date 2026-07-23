from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from ascend_maze.core.canonical import FrozenMap
from ascend_maze.inference import ChatRequest, ChatResponse
from ascend_maze.inference.adapters import transformers_local
from ascend_maze.inference.contracts import InferenceWorkerConfig, ModelRouteContext
from ascend_maze.inference import worker_client


def test_transformers_local_worker_client_uses_c11_device_assignment(
    monkeypatch,
) -> None:
    captured = {}

    class GenerationSession:
        def __init__(self, config):  # type: ignore[no-untyped-def]
            captured["config"] = config
            captured["session_count"] = int(captured.get("session_count", 0)) + 1

        def generate(self, request):  # type: ignore[no-untyped-def]
            captured.setdefault("requests", []).append(request)  # type: ignore[union-attr]
            return (
                ChatResponse(
                    text="ok",
                    finish_reason="stop",
                    input_tokens=1,
                    output_tokens=1,
                    engine_queue_depth=0,
                    prefix_cache_hit=False,
                    ttft_ms=None,
                    total_duration_ms=1,
                ),
                {"model_load_ms": 4, "generate_ms": 7, "cleanup_ms": 0},
            )

        def close(self) -> int:
            captured["close_count"] = int(captured.get("close_count", 0)) + 1
            return 3

    monkeypatch.setattr(
        worker_client,
        "TransformersLocalGenerationSession",
        GenerationSession,
    )
    config = InferenceWorkerConfig(
        adapter_name="transformers_local",
        instance_placement_lease_id="model_lease_1",
        request_timeout_ms=1_000,
        adapter_options=FrozenMap(
            (
                ("model_path", "/models/qwen3-4b"),
                ("tokenizer_path", "/models/qwen3-4b"),
                ("dtype", "bfloat16"),
                ("max_model_len", 10_240),
                ("device_id", "6"),
                ("trust_remote_code", False),
                ("enable_thinking", False),
                ("generation_method", "manual_greedy"),
                ("model_kind", "text"),
                ("qwen2_5_vl_cpu_unique_consecutive_workaround", False),
                ("runtime_library_paths", ("/opt/ascend/lib",)),
            )
        ),
    )
    client = worker_client.create_worker_inference_client(config)
    request = ChatRequest.create(
        [{"role": "user", "content": "hello"}],
        max_tokens=16,
    )
    context = ModelRouteContext(
        route_lease_id="route_1",
        model_id="qwen3-4b",
        adapter_name="transformers_local",
        endpoint_id="transformers-local://node/instance/1",
        instance_id="instance_1",
        instance_generation=1,
    )

    first_response = asyncio.run(client.invoke_chat(context, request))
    second_response = asyncio.run(client.invoke_chat(context, request))
    asyncio.run(client.close())

    assert first_response.text == "ok"
    assert second_response.text == "ok"
    assert captured["requests"] == [request, request]
    assert captured["session_count"] == 1
    assert captured["close_count"] == 1
    assert captured["config"].device_id == "6"
    assert captured["config"].generation_method == "manual_greedy"
    assert captured["config"].model_kind == "text"
    assert client.invocation_records() == (
        {
            "adapter": "transformers_local",
            "route_lease_id": "route_1",
            "model_id": "qwen3-4b",
            "instance_id": "instance_1",
            "instance_generation": 1,
            "call_index": 1,
            "model_load_ms": 4,
            "generate_ms": 7,
            "cleanup_ms": 0,
        },
        {
            "adapter": "transformers_local",
            "route_lease_id": "route_1",
            "model_id": "qwen3-4b",
            "instance_id": "instance_1",
            "instance_generation": 1,
            "call_index": 2,
            "model_load_ms": 4,
            "generate_ms": 7,
            "cleanup_ms": 3,
        },
    )


def test_transformers_local_maps_visible_physical_npu_to_logical_zero(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)

    logical_device_id = transformers_local._configure_process_npu_visibility("6")  # noqa: SLF001

    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "6"
    assert logical_device_id == "0"


def test_manual_greedy_reuses_kv_cache_and_stops_at_eos() -> None:
    import torch

    cache = object()

    class Model:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            token_id = 5 if len(self.calls) == 1 else 9
            input_ids = kwargs["input_ids"]
            logits = torch.zeros((1, int(input_ids.shape[1]), 10))
            logits[:, -1, token_id] = 1
            return SimpleNamespace(logits=logits, past_key_values=cache)

    model = Model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated = transformers_local._manual_greedy_generate(  # noqa: SLF001
        torch=torch,
        model=model,
        input_ids=input_ids,
        max_new_tokens=8,
        eos_token_id=9,
        device=torch.device("cpu"),
    )

    assert generated.tolist() == [[1, 2, 3, 5, 9]]
    assert len(model.calls) == 2
    assert tuple(model.calls[0]["input_ids"].shape) == (1, 3)
    assert tuple(model.calls[0]["attention_mask"].shape) == (1, 3)
    assert model.calls[0]["past_key_values"] is None
    assert model.calls[0]["use_cache"] is True
    assert tuple(model.calls[1]["input_ids"].shape) == (1, 1)
    assert tuple(model.calls[1]["attention_mask"].shape) == (1, 4)
    assert model.calls[1]["past_key_values"] is cache


def test_multimodal_messages_translate_openai_image_url_for_processor() -> None:
    request = ChatRequest.create(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aGVsbG8=",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]
    )

    messages, image_count = transformers_local._multimodal_messages(request)  # noqa: SLF001

    assert image_count == 1
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image",
                    "url": "data:image/png;base64,aGVsbG8=",
                    "detail": "low",
                },
            ],
        }
    ]


def test_vision_language_config_accepts_manual_greedy() -> None:
    config = transformers_local.TransformersLocalGenerationConfig(
        model_path="/models/qwen2.5-vl",
        tokenizer_path="/models/qwen2.5-vl",
        dtype="bfloat16",
        max_model_len=8_192,
        device_id="0",
        generation_method="manual_greedy",
        model_kind="vision_language",
        qwen2_5_vl_cpu_unique_consecutive_workaround=True,
    )

    assert config.model_kind == "vision_language"
    assert config.qwen2_5_vl_cpu_unique_consecutive_workaround is True


def test_vision_language_generation_uses_processor_and_image_text_model(
    monkeypatch,
) -> None:
    import torch
    import transformers

    captured: dict[str, object] = {}
    load_counts = {"processor": 0, "model": 0}

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0)

        def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            captured["messages"] = messages
            captured["processor_kwargs"] = kwargs
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                "pixel_values": torch.ones((2, 4), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
            }

        def batch_decode(self, token_ids, **kwargs):  # type: ignore[no-untyped-def]
            captured["decoded_tokens"] = token_ids.tolist()
            captured["decode_kwargs"] = kwargs
            return ["a red square"]

    class Model:
        def to(self, device):  # type: ignore[no-untyped-def]
            captured["device"] = str(device)
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 3, 7, 8]], dtype=torch.long)

    def load_processor(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        load_counts["processor"] += 1
        return Processor()

    def load_model(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        load_counts["model"] += 1
        return Model()

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        load_processor,
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        load_model,
    )
    monkeypatch.setattr(transformers_local, "_set_npu_device", lambda *args: None)
    monkeypatch.setattr(
        transformers_local,
        "_torch_device",
        lambda *args: torch.device("cpu"),
    )
    monkeypatch.setattr(transformers_local, "_synchronize", lambda *args: None)
    monkeypatch.setattr(transformers_local, "_empty_npu_cache", lambda *args: None)
    request = ChatRequest.create(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aGVsbG8=",
                        },
                    },
                ],
            }
        ],
        max_tokens=4,
        temperature=0.0,
    )
    config = transformers_local.TransformersLocalGenerationConfig(
        model_path="/models/qwen2.5-vl",
        tokenizer_path="/models/qwen2.5-vl",
        dtype="bfloat16",
        max_model_len=8_192,
        device_id="6",
        model_kind="vision_language",
    )

    session = transformers_local.TransformersLocalGenerationSession(config)
    response, metrics = session.generate(request)
    second_response, second_metrics = session.generate(request)
    cleanup_ms = session.close()

    assert response.text == "a red square"
    assert second_response.text == "a red square"
    assert response.input_tokens == 3
    assert response.output_tokens == 2
    assert captured["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {
                    "type": "image",
                    "url": "data:image/png;base64,aGVsbG8=",
                },
            ],
        }
    ]
    generate_kwargs = captured["generate_kwargs"]
    assert isinstance(generate_kwargs, dict)
    assert "pixel_values" in generate_kwargs
    assert "image_grid_thw" in generate_kwargs
    assert generate_kwargs["max_new_tokens"] == 4
    assert captured["decoded_tokens"] == [[7, 8]]
    assert metrics["model_kind"] == "vision_language"
    assert metrics["image_count"] == 1
    assert metrics["max_tokens"] == 4
    assert metrics["temperature"] == 0.0
    assert metrics["model_reused"] is False
    assert second_metrics["model_reused"] is True
    assert second_metrics["model_load_ms"] == 0
    assert second_metrics["processor_load_ms"] == 0
    assert load_counts == {"processor": 1, "model": 1}
    assert cleanup_ms >= 0
    assert "processor_load_ms" in metrics
    assert "multimodal_preprocess_ms" in metrics
