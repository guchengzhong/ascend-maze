from __future__ import annotations

from pathlib import Path

from ascend_maze.control import InMemoryController
from ascend_maze.core.clock import Clock
from ascend_maze.inference import (
    InferenceCoordinator,
    ModelCatalog,
    ModelSpec,
    PortLeaseManager,
)
from ascend_maze.inference.adapters.fake import FakeInferenceEngineAdapter
from ascend_maze.placement import NodeCapacity, NpuCapacity, PlacementManager


ENVIRONMENT_FINGERPRINT = "e" * 64
CONFIG_FINGERPRINT = "c" * 64


def make_spec(
    artifact_path: Path,
    *,
    model_id: str = "model_a",
    request_capacity: int = 1,
    min_replicas: int = 0,
    max_replicas: int = 1,
    scale_up_sustain_ms: int = 0,
    scale_down_idle_ms: int = 60_000,
    scale_cooldown_ms: int = 10_000,
    allow_colocation: bool = False,
) -> ModelSpec:
    artifact_path.mkdir(parents=True, exist_ok=True)
    return ModelSpec(
        model_id=model_id,
        catalog_revision="catalog_v1",
        artifact_path=str(artifact_path),
        tokenizer_path=None,
        artifact_revision="artifact_v1",
        backend="fake",
        dtype="float16",
        quantization=None,
        tensor_parallel_size=1,
        max_model_len=2_048,
        instance_cpu_num=1,
        instance_host_mem_mb=256,
        weight_hbm_mb=512,
        runtime_hbm_mb=128,
        kv_cache_hbm_mb=128,
        instance_hbm_mb=1_024,
        npu_slots=1,
        allow_colocation=allow_colocation,
        request_capacity=request_capacity,
        required_capabilities=(),
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        launch_options={"response_prefix": model_id},
        warmup_request={"prompt": "warmup"},
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        target_route_utilization=1.0,
        scale_up_pending_threshold=1,
        scale_up_sustain_ms=scale_up_sustain_ms,
        scale_down_idle_ms=scale_down_idle_ms,
        scale_cooldown_ms=scale_cooldown_ms,
        max_parallel_starts=1,
        startup_timeout_ms=5_000,
        drain_timeout_ms=5_000,
    )


def make_node(
    node_id: str = "node_a",
    *,
    cpu: int = 8,
    memory_mb: int = 8_192,
    npu_slots: int = 4,
    npu_count: int = 1,
) -> NodeCapacity:
    return NodeCapacity(
        node_id=node_id,
        boot_id=f"boot_{node_id}",
        node_ip="127.0.0.1",
        cpu_total=cpu,
        mem_total_mb=memory_mb,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=4,
        npus=tuple(
            NpuCapacity(
                device_id=str(index),
                chip_type="fake_npu",
                total_hbm_mb=8_192,
                system_reserved_hbm_mb=0,
                task_slots_total=npu_slots,
                observed_free_hbm_mb=8_192,
            )
            for index in range(npu_count)
        ),
        observed_free_mem_mb=memory_mb,
        capabilities={"environment_fingerprint": ENVIRONMENT_FINGERPRINT},
    )


def make_inference(
    spec: ModelSpec | tuple[ModelSpec, ...],
    *,
    placement: PlacementManager | None = None,
    adapter: FakeInferenceEngineAdapter | None = None,
    clock: Clock | None = None,
    port_leases: PortLeaseManager | None = None,
    reconcile_interval_ms: int = 100,
) -> tuple[InferenceCoordinator, PlacementManager, FakeInferenceEngineAdapter]:
    resolved_placement = placement or PlacementManager()
    resolved_adapter = adapter or FakeInferenceEngineAdapter()
    specs = (spec,) if isinstance(spec, ModelSpec) else spec
    catalog = ModelCatalog(
        specs,
        adapters={"fake": resolved_adapter},
        max_single_npu_hbm_mb=8_192,
    )
    inference = InferenceCoordinator(
        catalog=catalog,
        placement=resolved_placement,
        service_backend=resolved_adapter,
        clock=clock,
        port_leases=port_leases,
        reconcile_interval_ms=reconcile_interval_ms,
    )
    return inference, resolved_placement, resolved_adapter


def make_controller(
    spec: ModelSpec | tuple[ModelSpec, ...],
    *,
    adapter: FakeInferenceEngineAdapter | None = None,
    clock: Clock | None = None,
    nodes: tuple[NodeCapacity, ...] | None = None,
    dispatch_timeout_ms: int = 5_000,
) -> tuple[
    InMemoryController,
    InferenceCoordinator,
    FakeInferenceEngineAdapter,
]:
    placement = PlacementManager()
    inference, _, resolved_adapter = make_inference(
        spec,
        placement=placement,
        adapter=adapter,
        clock=clock,
    )
    controller = InMemoryController(
        config_fingerprint=CONFIG_FINGERPRINT,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        build_revision="stage6a_test",
        node_capacities=nodes or (make_node(),),
        placement=placement,
        inference=inference,
        clock=clock,
        dispatch_timeout_ms=dispatch_timeout_ms,
    )
    return controller, inference, resolved_adapter
