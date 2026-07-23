from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ascend_maze import Workflow
from ascend_maze.control import (
    ControllerCheckpoint,
    InMemoryController,
    InMemoryControllerRecoveryStore,
    InMemoryRuntimeClient,
    RecoveryIdentity,
    SqliteControllerRecoveryStore,
)
from ascend_maze.core.errors import (
    DataHandleInvalidError,
    RunDataIndexError,
    StateTransitionError,
    SubmissionConflictError,
)
from ascend_maze.data import InMemoryDataStore
from ascend_maze.lifecycle import AttemptStatus, RunStatus, TaskStatus
from ascend_maze.inference import (
    InMemoryPortLeaseManager,
    InferenceCoordinator,
    ModelCatalog,
    ModelInstanceState,
    ModelSpec,
)
from ascend_maze.inference.adapters.fake import FakeInferenceEngineAdapter
from ascend_maze.placement import (
    LeaseStatus,
    NodeCapacity,
    NpuCapacity,
    PlacementManager,
)
from ascend_maze.runtime import FakeExecutionPlan, FakeRuntimeBackend
from task_fixtures import finish, load_text


CONFIG = "c" * 64
ENVIRONMENT = "e" * 64


def _node() -> NodeCapacity:
    return NodeCapacity(
        node_id="node_a",
        boot_id="boot_1",
        node_ip="127.0.0.1",
        cpu_total=2,
        mem_total_mb=1_024,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=2,
        observed_free_mem_mb=1_024,
    )


def _workflow() -> tuple[Workflow, str, str]:
    workflow = Workflow("controller-recovery")
    value = workflow.input("value")
    loaded = workflow.add_task(load_text, inputs={"path": value}, task_name="loaded")
    result = workflow.add_task(
        finish,
        inputs={"summary": loaded.outputs["text"]},
        task_name="result",
    )
    return workflow, loaded.task_id, result.task_id


def _model_spec(tmp_path: Path) -> ModelSpec:
    artifact = tmp_path / "model"
    artifact.mkdir()
    return ModelSpec(
        model_id="model_recovery",
        catalog_revision="catalog_1",
        artifact_path=str(artifact),
        tokenizer_path=None,
        artifact_revision="a" * 64,
        backend="fake",
        dtype="bfloat16",
        quantization=None,
        tensor_parallel_size=1,
        max_model_len=2_048,
        instance_cpu_num=1,
        instance_host_mem_mb=128,
        weight_hbm_mb=1_024,
        runtime_hbm_mb=512,
        kv_cache_hbm_mb=512,
        instance_hbm_mb=2_048,
        npu_slots=1,
        allow_colocation=False,
        request_capacity=1,
        required_capabilities=(),
        environment_fingerprint=ENVIRONMENT,
        launch_options={},
        warmup_request={
            "messages": [{"role": "user", "content": "ready"}],
            "max_tokens": 4,
        },
        min_replicas=0,
        max_replicas=1,
    )


def _model_node() -> NodeCapacity:
    return NodeCapacity(
        node_id="node_model",
        boot_id="boot_model",
        node_ip="127.0.0.1",
        cpu_total=4,
        mem_total_mb=8_192,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=2,
        observed_free_mem_mb=8_192,
        npus=(
            NpuCapacity(
                device_id="0",
                chip_type="910B3",
                total_hbm_mb=65_536,
                system_reserved_hbm_mb=3_200,
                task_slots_total=1,
                observed_free_hbm_mb=62_336,
            ),
        ),
    )


def _inference(
    *,
    spec: ModelSpec,
    adapter: FakeInferenceEngineAdapter,
    placement: PlacementManager,
    ports: InMemoryPortLeaseManager,
) -> InferenceCoordinator:
    catalog = ModelCatalog(
        (spec,),
        adapters={adapter.name: adapter},
        environment_capabilities=(),
        max_single_npu_hbm_mb=60_000,
    )
    return InferenceCoordinator(
        catalog=catalog,
        placement=placement,
        service_backend=adapter,
        port_leases=ports,
        reconcile_interval_ms=10,
    )


async def _wait_for_running(
    controller: InMemoryController,
    run_id: str,
    task_id: str,
) -> None:
    for _ in range(500):
        if controller.snapshot(run_id).task(task_id).status is TaskStatus.RUNNING:
            return
        await asyncio.sleep(0.002)
    raise AssertionError("Task did not enter running before recovery test deadline")


def test_controller_generation_recovery_interrupts_attempt_and_preserves_data() -> None:
    async def scenario() -> None:
        data_store = InMemoryDataStore()
        recovery = InMemoryControllerRecoveryStore()
        workflow, loaded_task_id, result_task_id = _workflow()
        first = InMemoryController(
            cluster_id="cluster_recovery",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_1",
            data_owner_generation="data_owner_1",
            data_store=data_store,
            recovery_store=recovery,
        )
        assert isinstance(first.runtime, FakeRuntimeBackend)
        first.runtime.set_plan(
            result_task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=30_000),
        )
        await first.start()
        first_client = InMemoryRuntimeClient(first)
        outcome = await first_client.submit(
            workflow,
            inputs={"value": "payload"},
            submission_id="submission_recovery",
        )
        assert outcome.run_id is not None
        run_id = outcome.run_id
        await _wait_for_running(first, run_id, result_task_id)
        before = first.snapshot(run_id)
        assert before.task(loaded_task_id).status is TaskStatus.SUCCEEDED
        active_attempt = before.task(result_task_id).attempts[-1]
        assert active_attempt.status is AttemptStatus.RUNNING
        old_index_ref = first.core.run_index_ref(run_id)
        assert first.result(run_id, loaded_task_id) == {"text": "payload"}
        await first.crash()

        second = InMemoryController(
            cluster_id="cluster_recovery",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_2",
            data_store=data_store,
            recovery_store=recovery,
        )
        await second.start()
        recovered = second.snapshot(run_id)
        assert recovered.status is RunStatus.INTERRUPTED
        assert recovered.task(loaded_task_id).status is TaskStatus.SUCCEEDED
        recovered_attempt = recovered.task(result_task_id).attempts[-1]
        assert recovered_attempt.dispatch_id == active_attempt.dispatch_id
        assert recovered_attempt.status is AttemptStatus.CANCELLED
        assert second.placement.active_lease_count() == 0
        assert (
            second.placement.lease_snapshot(active_attempt.lease_id).status
            is LeaseStatus.INVALIDATED
        )
        assert second.result(run_id, loaded_task_id) == {"text": "payload"}

        new_index_ref = second.core.run_index_ref(run_id)
        assert new_index_ref.controller_generation == "controller_2"
        assert new_index_ref.index_generation == old_index_ref.index_generation + 1
        with pytest.raises(RunDataIndexError, match="stale controller"):
            second.indexes.get(run_id).workflow_input_handle(
                "value",
                controller_generation=old_index_ref.controller_generation,
                index_generation=old_index_ref.index_generation,
            )

        replay_client = InMemoryRuntimeClient(second)
        replay = replay_client.get_submission_status("submission_recovery")
        assert replay is not None
        assert replay.run_id == run_id
        conflicting = replay_client.prepare_submission(
            workflow,
            inputs={"value": "different"},
            submission_id="submission_recovery",
        )
        conflicting_handle = conflicting.request.workflow_inputs[0][1]
        with pytest.raises(SubmissionConflictError, match="different payload"):
            await replay_client.submit_prepared(conflicting)
        with pytest.raises(DataHandleInvalidError, match="released"):
            data_store.state_of(conflicting_handle)
        assert isinstance(second.runtime, FakeRuntimeBackend)
        assert second.runtime.active_dispatch_count() == 0

        destroyed = await second.destroy_run(run_id)
        assert destroyed.tombstone.destroy_succeeded
        assert data_store.active_count == 0
        assert second.runtime.code_reference_count() == 0
        await second.close()

    asyncio.run(scenario())


def test_sqlite_recovery_store_persists_and_fences_old_generation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "controller-recovery.sqlite3"
    identity = RecoveryIdentity("cluster", CONFIG, ENVIRONMENT, "build")
    first = SqliteControllerRecoveryStore(path)
    first_claim = first.claim_generation(
        identity=identity,
        controller_generation="controller_1",
    )
    checkpoint = ControllerCheckpoint(
        schema_version=1,
        identity=identity,
        controller_generation="controller_1",
        checkpoint_sequence=1,
        created_at_ms=1,
        data_owner_generation="data_owner_1",
        data_store_descriptor=None,
        submissions=(),
        runs=(),
        leases=(),
    )
    first.save(
        checkpoint,
        controller_generation="controller_1",
        epoch=first_claim.epoch,
    )

    second = SqliteControllerRecoveryStore(path)
    second_claim = second.claim_generation(
        identity=identity,
        controller_generation="controller_2",
    )
    assert second_claim.previous_generation == "controller_1"
    assert second_claim.checkpoint == checkpoint
    with pytest.raises(StateTransitionError, match="fenced"):
        first.save(
            checkpoint,
            controller_generation="controller_1",
            epoch=first_claim.epoch,
        )
    with pytest.raises(StateTransitionError, match="already claimed"):
        second.claim_generation(
            identity=identity,
            controller_generation="controller_2",
        )
    first.close()
    second.close()


def test_terminal_and_destroyed_run_state_survives_generation_switch() -> None:
    async def scenario() -> None:
        data_store = InMemoryDataStore()
        recovery = InMemoryControllerRecoveryStore()
        workflow, _, result_task_id = _workflow()
        first = InMemoryController(
            cluster_id="cluster_terminal_recovery",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_1",
            data_owner_generation="data_owner_1",
            data_store=data_store,
            recovery_store=recovery,
        )
        await first.start()
        submitted = await InMemoryRuntimeClient(first).submit(
            workflow,
            inputs={"value": "payload"},
            submission_id="submission_terminal_recovery",
        )
        assert submitted.run_id is not None
        run_id = submitted.run_id
        completed = await first.wait_run(run_id, timeout_seconds=2)
        assert completed.status is RunStatus.SUCCEEDED
        completed_attempt = completed.task(result_task_id).attempts[-1]
        expected = first.result(run_id, result_task_id)
        await first.crash()

        second = InMemoryController(
            cluster_id="cluster_terminal_recovery",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_2",
            data_store=data_store,
            recovery_store=recovery,
        )
        await second.start()
        restored = second.snapshot(run_id)
        assert restored.status is RunStatus.SUCCEEDED
        assert restored.task(result_task_id).attempts[-1] == completed_attempt
        assert second.result(run_id, result_task_id) == expected
        assert isinstance(second.runtime, FakeRuntimeBackend)
        assert second.runtime.active_dispatch_count() == 0
        replay = InMemoryRuntimeClient(second).get_submission_status(
            "submission_terminal_recovery"
        )
        assert replay is not None and replay.run_id == run_id
        destroyed = await second.destroy_run(run_id, force=True)
        assert destroyed.tombstone.destroy_succeeded
        assert not destroyed.flush_result.recording_complete
        assert data_store.active_count == 0
        await second.crash()

        third = InMemoryController(
            cluster_id="cluster_terminal_recovery",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_3",
            data_store=data_store,
            recovery_store=recovery,
        )
        await third.start()
        assert third.snapshot(run_id).status is RunStatus.SUCCEEDED
        with pytest.raises(RunDataIndexError, match="destroyed"):
            third.result(run_id, result_task_id)
        assert await third.destroy_run(run_id) == destroyed
        assert isinstance(third.runtime, FakeRuntimeBackend)
        assert third.runtime.code_reference_count() == 0
        assert third.runtime.active_dispatch_count() == 0
        await third.close()

    asyncio.run(scenario())


def test_model_instance_is_restored_fenced_and_stopped_after_controller_restart(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        spec = _model_spec(tmp_path)
        adapter = FakeInferenceEngineAdapter()
        ports = InMemoryPortLeaseManager()
        recovery = InMemoryControllerRecoveryStore()
        data_store = InMemoryDataStore()
        first_placement = PlacementManager()
        first_inference = _inference(
            spec=spec,
            adapter=adapter,
            placement=first_placement,
            ports=ports,
        )
        first = InMemoryController(
            cluster_id="cluster_model_recovery",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_model_node(),),
            controller_generation="controller_1",
            data_owner_generation="data_owner_1",
            data_store=data_store,
            placement=first_placement,
            inference=first_inference,
            recovery_store=recovery,
        )
        await first.start()
        requested = first_inference.instances.create_requested(spec.model_id)
        ready = await first_inference.instances.start_instance(requested.instance_id)
        assert ready.state is ModelInstanceState.READY
        assert adapter.is_process_alive(ready.instance_id, ready.generation)
        assert first_placement.active_lease_count() == 1
        acquired = await first_inference.acquire_route(
            run_id="run_with_active_route",
            task_id="task_with_active_route",
            attempt=1,
            model_id=spec.model_id,
            session_key_hash=None,
            dispatch_deadline_ms=first.clock.monotonic_ms() + 10_000,
        )
        assert acquired.lease is not None
        assert first_inference.activate_route(acquired.lease.route_lease_id)
        first_inference.instances.request_started(ready.instance_id, ready.generation)
        active = first_inference.instances.snapshot(ready.instance_id)
        assert active.route_occupancy == 1
        assert active.actual_request_inflight == 1
        await first.crash()

        second_placement = PlacementManager()
        second_inference = _inference(
            spec=spec,
            adapter=adapter,
            placement=second_placement,
            ports=ports,
        )
        second = InMemoryController(
            cluster_id="cluster_model_recovery",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_model_node(),),
            controller_generation="controller_2",
            data_store=data_store,
            placement=second_placement,
            inference=second_inference,
            recovery_store=recovery,
        )
        await second.start()
        await second_inference.replicas.wait_for_background()
        restored = second_inference.instances.snapshot(ready.instance_id)
        assert restored.generation == ready.generation
        assert restored.state is ModelInstanceState.STOPPED
        assert restored.route_occupancy == 0
        assert restored.actual_request_inflight == 0
        assert not adapter.is_process_alive(ready.instance_id, ready.generation)
        assert second_placement.active_lease_count() == 0
        assert ready.placement_lease_id is not None
        assert (
            second_placement.lease_snapshot(ready.placement_lease_id).status
            is LeaseStatus.INVALIDATED
        )
        await second.close()

    asyncio.run(scenario())


def test_commit_gap_recovers_as_one_committed_submission() -> None:
    async def scenario() -> None:
        data_store = InMemoryDataStore()
        recovery = InMemoryControllerRecoveryStore()
        workflow, _, _ = _workflow()
        first = InMemoryController(
            cluster_id="cluster_commit_gap",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_1",
            data_owner_generation="data_owner_1",
            data_store=data_store,
            recovery_store=recovery,
        )
        await first.start()
        prepared = InMemoryRuntimeClient(first).prepare_submission(
            workflow,
            inputs={"value": "payload"},
            submission_id="submission_commit_gap",
        )
        first.inject_submit_failure("after_commit")
        with pytest.raises(RuntimeError, match="after_commit"):
            await first.submit(prepared.request)
        checkpoint = recovery.load()
        assert checkpoint is not None
        assert checkpoint.submissions[0].state.value == "preparing"
        assert len(checkpoint.runs) == 1
        committed_run_id = checkpoint.runs[0].run_id
        await first.crash()

        second = InMemoryController(
            cluster_id="cluster_commit_gap",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_2",
            data_store=data_store,
            recovery_store=recovery,
        )
        await second.start()
        recovered = second.submission_outcome("submission_commit_gap")
        assert recovered.state.value == "committed"
        assert recovered.run_id == committed_run_id
        replay = await second.submit(prepared.request)
        assert replay.replayed
        assert replay.run_id == committed_run_id
        assert len(second.core.recovery_runs()) == 1
        await second.destroy_run(committed_run_id)
        await second.close()

    asyncio.run(scenario())


def test_new_generation_fences_old_controller_writes() -> None:
    async def scenario() -> None:
        data_store = InMemoryDataStore()
        recovery = InMemoryControllerRecoveryStore()
        first = InMemoryController(
            cluster_id="cluster_fence",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_1",
            data_owner_generation="data_owner_1",
            data_store=data_store,
            recovery_store=recovery,
        )
        await first.start()
        workflow, _, result_task_id = _workflow()
        assert isinstance(first.runtime, FakeRuntimeBackend)
        first.runtime.set_plan(
            result_task_id,
            1,
            FakeExecutionPlan(execution_delay_ms=30_000),
        )
        committed = await InMemoryRuntimeClient(first).submit(
            workflow,
            inputs={"value": "committed"},
            submission_id="submission_committed_before_fence",
        )
        assert committed.run_id is not None
        await _wait_for_running(first, committed.run_id, result_task_id)
        prepared = InMemoryRuntimeClient(first).prepare_submission(
            workflow,
            inputs={"value": "payload"},
            submission_id="submission_fenced",
        )
        second = InMemoryController(
            cluster_id="cluster_fence",
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            node_capacities=(_node(),),
            controller_generation="controller_2",
            data_store=data_store,
            recovery_store=recovery,
        )
        with pytest.raises(StateTransitionError, match="fenced"):
            await first.submit(prepared.request)
        with pytest.raises(StateTransitionError, match="fenced"):
            await first.cancel_run(committed.run_id)
        with pytest.raises(StateTransitionError, match="fenced"):
            await first.destroy_run(committed.run_id, force=True)
        await first.crash()
        await second.start()
        assert second.snapshot(committed.run_id).status is RunStatus.INTERRUPTED
        for _, handle in prepared.request.workflow_inputs:
            data_store.release(handle)
        await second.destroy_run(committed.run_id)
        await second.close()

    asyncio.run(scenario())
