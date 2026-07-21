from __future__ import annotations

import asyncio
import hashlib

import pytest

from ascend_maze import Workflow
from ascend_maze.ascend.contracts import (
    AscendCorrectnessConfig,
    AscendEnvironmentSnapshot,
    create_ascend_correctness_config_snapshot,
)
from ascend_maze.ascend.discovery import discover_atb_runtime_library_preloads
from ascend_maze.compiler import compile_workflow
from ascend_maze.control.client import InMemoryRuntimeClient
from ascend_maze.control.controller import InMemoryController
from ascend_maze.control.proto_codec import (
    decode_runtime_event,
    encode_runtime_event,
)
from ascend_maze.contracts.resources import (
    PlacementLease,
    ReservationVector,
    ResourceObservation,
    ResourceSpec,
)
from ascend_maze.contracts.runtime import DeviceBinding, RuntimeNodeBinding
from ascend_maze.core.canonical import (
    FrozenMap,
    canonical_bytes,
    decode_canonical_bytes,
)
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.lifecycle import RunStatus
from ascend_maze.inference import AttemptInferenceSummary, InferenceRequestRecord
from ascend_maze.placement import (
    NodeCapacity,
    NodeObservation,
    NodeStatus,
    NpuCapacity,
    NpuObservation,
    PlacementManager,
)
from ascend_maze.resources import DeclaredOnlyAnchorProvider, StaticAnchorProvider
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.runtime.fake import FakeExecutionPlan
from stage4_task_fixtures import (
    impossible_npu,
    no_retry_npu,
    statically_inferred_npu,
    statically_npu,
    statically_parallel,
)


ENVIRONMENT = "e" * 64


def _npu_node(*, environment: str = ENVIRONMENT) -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_a",
        node_ip="127.0.0.1",
        cpu_total=8,
        mem_total_mb=32_768,
        cpu_system_reserved=1,
        mem_system_reserved_mb=1_024,
        io_slots_total=4,
        npus=(NpuCapacity("7", "910B3", 65_536, 4_096, 1, 62_330),),
        observed_free_mem_mb=30_000,
        capabilities=FrozenMap((("environment_fingerprint", environment),)),
    )


def test_canonical_decoder_round_trips_and_rejects_noncanonical_bytes() -> None:
    value = FrozenMap((("b", (1, 2)), ("a", b"value")))
    payload = canonical_bytes(value)
    assert decode_canonical_bytes(payload) == value
    with pytest.raises(Exception, match="canonical"):
        decode_canonical_bytes(b'["int", "01"]')


def test_environment_fingerprint_covers_versions_and_chip_family() -> None:
    first = AscendEnvironmentSnapshot.create(
        machine="aarch64",
        chip_types=("910B3", "910B3"),
        versions={"cann": "9.0", "torch_npu": "2.7.1"},
    )
    same = AscendEnvironmentSnapshot.create(
        machine="aarch64",
        chip_types=("910B3",),
        versions={"torch_npu": "2.7.1", "cann": "9.0"},
    )
    changed = AscendEnvironmentSnapshot.create(
        machine="aarch64",
        chip_types=("910B3",),
        versions={"cann": "9.0", "torch_npu": "2.7.2"},
    )
    assert first.environment_fingerprint == same.environment_fingerprint
    assert first.environment_fingerprint != changed.environment_fingerprint


def test_atb_runtime_preload_discovery_pins_resolved_library_digest(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "atb" / "cxx_abi_1"
    library_directory = home / "lib"
    library_directory.mkdir(parents=True)
    libmki = library_directory / "libmki.so"
    libmki.write_bytes(b"test-mki")
    (library_directory / "libtbe_adapter.so").write_bytes(b"test-tbe")
    monkeypatch.setenv("ATB_HOME_PATH", str(home))

    discovered = discover_atb_runtime_library_preloads()

    assert dict(discovered) == {
        str(libmki.resolve()): hashlib.sha256(b"test-mki").hexdigest(),
    }


def test_correctness_config_snapshot_covers_every_stage_four_setting() -> None:
    environment = AscendEnvironmentSnapshot.create(
        machine="aarch64",
        chip_types=("910B3",),
        versions={"cann": "9.0", "torch_npu": "2.7.1"},
    )
    config = AscendCorrectnessConfig()
    snapshot = create_ascend_correctness_config_snapshot(
        config,
        environment,
        source_path="/etc/ascend-maze/correctness.toml",
        build_revision="stage4-test-build",
        created_at_ms=1,
    )
    assert snapshot.resolved["profile"] == "correctness"
    assert snapshot.resolved["scheduler"]["policy"] == "fcfs"
    assert snapshot.resolved["anchor"]["strategy"] == config.anchor_strategy
    assert snapshot.resolved["placement"] == FrozenMap(
        (
            ("allow_colocation", config.allow_colocation),
            ("host_mem_headroom_mb", config.host_mem_headroom_mb),
            ("io_slots_total", config.io_slots_total),
            ("npu_hbm_headroom_mb", config.npu_hbm_headroom_mb),
            (
                "npu_system_reserved_hbm_mb",
                config.npu_system_reserved_hbm_mb,
            ),
            ("task_slots_total", config.task_slots_total),
        )
    )
    assert snapshot.resolved["worker"]["max_tasks_per_worker"] == 1
    assert snapshot.resolved["worker"]["standby"]["min_idle"] == 0
    assert snapshot.resolved["cleanup"]["hbm_recovery_deadline_ms"] == 30_000
    assert snapshot.resolved["environment_fingerprint"] == (
        environment.environment_fingerprint
    )

    static_snapshot = create_ascend_correctness_config_snapshot(
        AscendCorrectnessConfig(anchor_strategy="static"),
        environment,
        source_path="/etc/ascend-maze/correctness.toml",
        build_revision="stage4-test-build",
        created_at_ms=1,
    )
    assert static_snapshot.config_fingerprint != snapshot.config_fingerprint


def test_device_binding_is_derived_only_from_one_npu_lease() -> None:
    lease = PlacementLease(
        "lease",
        "task",
        "run",
        "task",
        1,
        "node_a",
        "boot_a",
        "7",
        ReservationVector(1, 128, 0, 1_024, 1),
        1,
        1,
        2,
    )
    runtime_binding = RuntimeNodeBinding(
        "node_a", "boot_a", "ray", 3, "agent", "endpoint", "producer"
    )
    binding = DeviceBinding.from_lease(lease, runtime_binding)
    assert binding.physical_device_id == "7"
    assert binding.visible_device_index == 0
    assert binding.environment_variables["ASCEND_RT_VISIBLE_DEVICES"] == "7"
    with pytest.raises(ContractValidationError, match="one leased NPU slot"):
        DeviceBinding.from_lease(
            PlacementLease(
                "cpu_lease",
                "task",
                "run",
                "task",
                1,
                "node_a",
                "boot_a",
                None,
                ReservationVector(1, 128, 0, 0, 0),
                1,
                1,
                2,
            ),
            runtime_binding,
        )


def test_static_anchor_and_one_oom_revision_are_deterministic() -> None:
    workflow = Workflow("stage4-static")
    node = workflow.add_task(statically_parallel, inputs={"value": -1})
    compiled = compile_workflow(workflow)
    definition = compiled.definitions[compiled.tasks[node.task_id].definition_id]
    assert definition.static_inferred.cpu_num == 4
    provider = StaticAnchorProvider(environment_fingerprint=ENVIRONMENT)
    anchor = provider.resolve(run_id="run", compiled=compiled, task_id=node.task_id)
    assert anchor.strategy == "static"
    assert anchor.effective.cpu_num == 4

    inferred_workflow = Workflow("stage4-inferred-npu")
    inferred_node = inferred_workflow.add_task(
        statically_inferred_npu,
        inputs={"value": 1},
        model_anchor={"model": "stage4-placeholder", "mode": "local_worker"},
    )
    inferred_compiled = compile_workflow(inferred_workflow)
    inferred_definition = inferred_compiled.definitions[
        inferred_compiled.tasks[inferred_node.task_id].definition_id
    ]
    assert inferred_definition.task_kind == "npu"
    assert "npu:torch_npu" in inferred_definition.static_signals
    inferred_anchor = provider.resolve(
        run_id="run_inferred_npu",
        compiled=inferred_compiled,
        task_id=inferred_node.task_id,
    )
    assert inferred_anchor.task_kind == "npu"
    assert inferred_anchor.strategy == "static"

    npu_workflow = Workflow("stage4-oom")
    npu_node = npu_workflow.add_task(statically_npu, inputs={"value": 1})
    npu_compiled = compile_workflow(npu_workflow)
    declared = DeclaredOnlyAnchorProvider(environment_fingerprint=ENVIRONMENT)
    first = declared.resolve(
        run_id="run_npu", compiled=npu_compiled, task_id=npu_node.task_id
    )
    reanchored = declared.reanchor_after_oom(
        run_id="run_npu",
        compiled=npu_compiled,
        task_id=npu_node.task_id,
        observed_peak_npu_mem_mb=2_000,
    )
    repeated = declared.reanchor_after_oom(
        run_id="run_npu",
        compiled=npu_compiled,
        task_id=npu_node.task_id,
        observed_peak_npu_mem_mb=3_000,
    )
    assert first.revision == 1
    assert reanchored.created and reanchored.anchor.revision == 2
    assert reanchored.anchor.effective.npu_mem_mb == 2_200
    assert not repeated.created
    assert repeated.anchor.revision == 2


def test_static_anchor_config_drives_end_to_end_reservation() -> None:
    async def scenario() -> None:
        environment = AscendEnvironmentSnapshot.create(
            machine="aarch64",
            chip_types=("910B3",),
            versions={"cann": "9.0", "torch_npu": "2.7.1"},
        )
        snapshot = create_ascend_correctness_config_snapshot(
            AscendCorrectnessConfig(anchor_strategy="static"),
            environment,
            source_path="/etc/ascend-maze/correctness.toml",
            build_revision="stage4-static-test",
            created_at_ms=1,
        )
        anchors = StaticAnchorProvider(
            environment_fingerprint=environment.environment_fingerprint
        )
        controller = InMemoryController(
            config_fingerprint=snapshot.config_fingerprint,
            environment_fingerprint=environment.environment_fingerprint,
            build_revision="test",
            node_capacities=(
                _npu_node(environment=environment.environment_fingerprint),
            ),
            anchors=anchors,
        )
        await controller.start()
        try:
            workflow = Workflow("stage4-static-e2e")
            node = workflow.add_task(statically_parallel, inputs={"value": -1})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="stage4_static_e2e",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.config_fingerprint == snapshot.config_fingerprint
            attempt = terminal.task(node.task_id).attempts[0]
            lease = controller.placement.lease_snapshot(attempt.lease_id).lease
            assert lease.resources.cpu_num == 4
            assert controller.result(outcome.run_id, node.task_id) == {"result": 1}
            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())


def test_placement_enforces_environment_observations_and_permanent_single_card_limit() -> (
    None
):
    manager = PlacementManager(
        npu_hbm_headroom_mb=1_024,
        required_environment_fingerprint=ENVIRONMENT,
    )
    manager.register_node(_npu_node(environment="wrong"))
    assert manager.snapshot().nodes[0].status is NodeStatus.UNSCHEDULABLE
    manager.register_node(_npu_node())
    assert manager.snapshot().nodes[0].status is NodeStatus.HEALTHY
    assert manager.update_observation(
        NodeObservation(
            "node_a",
            "boot_a",
            1,
            1,
            29_000,
            (NpuObservation("7", "healthy", 60_000, 0.0),),
        )
    )
    assert not manager.update_observation(
        NodeObservation(
            "node_a",
            "boot_a",
            1,
            2,
            1,
            (NpuObservation("7", "unhealthy", 1, 100.0),),
        )
    )
    workflow = Workflow("impossible-placement")
    node = workflow.add_task(impossible_npu, inputs={"value": 1})
    compiled = compile_workflow(workflow)
    anchor = DeclaredOnlyAnchorProvider(environment_fingerprint=ENVIRONMENT).resolve(
        run_id="run", compiled=compiled, task_id=node.task_id
    )
    rejected = manager.try_reserve(
        run_id="run",
        task_id=node.task_id,
        attempt=1,
        anchor=anchor,
        now_ms=1,
        dispatch_deadline_ms=2,
    )
    assert not rejected.selected
    assert rejected.rejection_reason == "resource_request_unsatisfiable"


def test_resource_observation_survives_protobuf_boundary() -> None:
    observation = ResourceObservation(
        run_id="run",
        task_id="task",
        definition_id="definition",
        attempt=1,
        code_hash="a" * 64,
        environment_fingerprint=ENVIRONMENT,
        requested=ResourceSpec(1, 128, 1_024, 0),
        status="failed",
        peak_host_rss_mb=32,
        peak_npu_allocated_mb=128,
        peak_npu_reserved_mb=256,
        peak_npu_process_hbm_mb=512,
        npu_metric_source="dcmi",
        npu_metric_quality="sampled",
        error_type="OutOfMemoryError",
        device_id="7",
        worker_pid=123,
        binding_verified=True,
    )
    event = RuntimeEvent.create(
        kind=RuntimeEventKind.TASK_FAILED,
        dispatch_id="dispatch",
        run_id="run",
        task_id="task",
        attempt=1,
        lease_id="lease",
        route_lease_id=None,
        occurred_at_ms=1,
        worker_pid=123,
        device_id="7",
        binding_verified=True,
        resource_observation=observation,
    )
    decoded = decode_runtime_event(encode_runtime_event(event))
    assert decoded == event


def test_inference_lifecycle_survives_worker_rpc_protobuf_boundary() -> None:
    record = InferenceRequestRecord(
        route_lease_id="route_1",
        call_index=1,
        run_id="run",
        task_id="task",
        attempt=1,
        model_id="model",
        instance_id="instance",
        instance_generation=1,
        instance_placement_lease_id="model_placement",
        started_at_ms=10,
        duration_ms=5,
        status="succeeded",
        input_tokens=2,
        output_tokens=3,
        engine_queue_depth=None,
        prefix_cache_hit=None,
        ttft_ms=None,
        error_code=None,
    )
    summary = AttemptInferenceSummary(
        route_lease_id="route_1",
        request_count=1,
        request_inflight=False,
        context_cleared=True,
    )
    event = RuntimeEvent.create(
        kind=RuntimeEventKind.TASK_RESULT,
        dispatch_id="dispatch",
        run_id="run",
        task_id="task",
        attempt=1,
        lease_id="lease",
        route_lease_id="route_1",
        occurred_at_ms=15,
        inference_call_index=1,
        inference_request=record,
        inference_summary=summary,
    )

    assert decode_runtime_event(encode_runtime_event(event)) == event


def test_scheduler_fails_impossible_npu_request_before_attempt() -> None:
    async def scenario() -> None:
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_npu_node(),),
        )
        await controller.start()
        try:
            workflow = Workflow("impossible-run")
            node = workflow.add_task(impossible_npu, inputs={"value": 1})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="impossible_stage4",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
            assert terminal.status is RunStatus.FAILED
            task = terminal.task(node.task_id)
            assert task.attempt_count == 0
            assert task.last_error is not None
            assert task.last_error.error_code == "resource_request_unsatisfiable"
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())


def test_oom_without_retry_budget_does_not_create_anchor_revision() -> None:
    async def scenario() -> None:
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_npu_node(),),
        )
        await controller.start()
        try:
            workflow = Workflow("no-oom-retry")
            node = workflow.add_task(no_retry_npu, inputs={"value": 1})
            controller.runtime.set_plan(
                node.task_id,
                1,
                FakeExecutionPlan(fail_after_start="npu_oom"),
            )
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="no_oom_retry",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=2)
            task = terminal.task(node.task_id)
            assert terminal.status is RunStatus.FAILED
            assert task.attempt_count == 1
            anchor = controller.anchors.resolve(
                run_id=outcome.run_id,
                compiled=controller.state.compiled(outcome.run_id),
                task_id=node.task_id,
            )
            assert anchor.revision == 1
            events = [
                event
                for event in controller.recorder.events(outcome.run_id)
                if event.event_type == "resource_anchor_oom"
            ]
            assert len(events) == 1
            assert not events[0].payload["created"]
            assert events[0].payload["reason"] == "retry_budget_exhausted"
            await controller.destroy_run(outcome.run_id)
        finally:
            await controller.close()

    asyncio.run(scenario())


def test_correctness_config_rejects_stage_five_features() -> None:
    assert AscendCorrectnessConfig().task_slots_total == 1
    with pytest.raises(ContractValidationError, match="stage-four correctness"):
        AscendCorrectnessConfig(task_slots_total=2)
