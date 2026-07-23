from __future__ import annotations

import asyncio
from pathlib import Path
import stat

import grpc
import pytest

from ascend_maze import Workflow, task
from ascend_maze.control import InMemoryController, InMemoryRuntimeClient
from ascend_maze.control import local_rpc
from ascend_maze.control.local_rpc import (
    ControlRpcError,
    ControllerStatus,
    LocalControlServer,
    UdsRuntimeClient,
)
from ascend_maze.contracts.recording import ParquetRecorderConfig
from ascend_maze.contracts.resources import ReservationVector
from ascend_maze.core.errors import SubmissionConflictError
from ascend_maze.lifecycle import RunStatus
from ascend_maze.placement import (
    NodeCapacity,
    NodeStatus,
    NpuCapacity,
    PlacementManager,
)
from ascend_maze.recording import ParquetRecorder


@task
def _barrier():
    return {}


@task
def _echo(value: str):
    return {"value": value}


@task
def _slow_echo(value: str):
    import time

    time.sleep(0.2)
    return {"value": value}


def test_local_control_status_uses_protected_unix_socket(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime_dir = tmp_path / "runtime"
        socket_path = runtime_dir / "control.sock"
        expected = ControllerStatus(
            controller_generation="controller_1",
            build_revision="test_build",
            environment_fingerprint="e" * 64,
            healthy_node_count=2,
        )
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: expected,
        )
        await server.start()
        assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        assert await UdsRuntimeClient(socket_path).get_controller_status() == expected
        await server.close(grace_seconds=0)
        assert not socket_path.exists()

    asyncio.run(scenario())


def test_cluster_resources_health_and_watch_version_jump(tmp_path: Path) -> None:
    async def scenario() -> None:
        socket_path = (tmp_path / "runtime" / "control.sock").resolve()
        capacity = NodeCapacity(
            node_id="node_a",
            boot_id="boot_a",
            node_ip="127.0.0.1",
            cpu_total=8,
            mem_total_mb=16_384,
            cpu_system_reserved=1,
            mem_system_reserved_mb=1_024,
            io_slots_total=4,
            npus=(
                NpuCapacity(
                    device_id="0",
                    chip_type="Ascend910B3",
                    total_hbm_mb=65_536,
                    system_reserved_hbm_mb=3_200,
                    task_slots_total=2,
                    observed_free_hbm_mb=62_000,
                ),
            ),
            observed_free_mem_mb=15_000,
        )
        placement = PlacementManager(
            host_mem_headroom_mb=512,
            npu_hbm_headroom_mb=1_024,
        )
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(capacity,),
            placement=placement,
        )
        await controller.start()
        reservation = placement.reserve_model_instance(
            instance_id="instance_a",
            generation=1,
            resources=ReservationVector(
                cpu_num=1,
                host_mem_mb=256,
                io_slots=1,
                npu_hbm_mb=4_096,
                npu_slots=1,
            ),
            allow_colocation=False,
            now_ms=controller.clock.monotonic_ms(),
            startup_deadline_ms=controller.clock.monotonic_ms() + 5_000,
        )
        assert reservation.lease is not None
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: ControllerStatus(
                controller_generation=controller.controller_generation,
                build_revision=controller.build_revision,
                environment_fingerprint=controller.environment_fingerprint,
                healthy_node_count=1,
            ),
            control_api=controller,
        )
        await server.start()
        client = UdsRuntimeClient(
            socket_path,
            data_store=controller.data_store,
            data_owner_generation=controller.data_owner_generation,
        )
        try:
            resources = await client.query("GetClusterSnapshot", filter="resources")
            cluster = resources["cluster"]
            assert cluster["host_mem_headroom_mb"] == 512
            assert cluster["npu_hbm_headroom_mb"] == 1_024
            assert cluster["active_lease_count"] == 1
            assert cluster["nodes"][0]["per_npu_reserved"] == [["0", 4_096, 1]]
            assert cluster["active_leases"] == [
                {
                    "lease": {
                        "allow_npu_colocation": False,
                        "attempt": None,
                        "boot_id": "boot_a",
                        "converted_standby_lease_id": None,
                        "created_at_ms": reservation.lease.created_at_ms,
                        "dispatch_deadline_ms": reservation.lease.dispatch_deadline_ms,
                        "lease_id": reservation.lease.lease_id,
                        "model_instance_id": "instance_a",
                        "node_id": "node_a",
                        "npu_device_id": "0",
                        "reservation_kind": "model_instance",
                        "resources": {
                            "cpu_num": 1,
                            "host_mem_mb": 256,
                            "io_slots": 1,
                            "npu_hbm_mb": 4_096,
                            "npu_slots": 1,
                        },
                        "run_id": None,
                        "snapshot_version": reservation.lease.snapshot_version,
                        "standby_worker_id": None,
                        "task_id": None,
                    },
                    "status": "reserved",
                    "finished_at_ms": None,
                    "finish_reason": None,
                }
            ]

            status = await client.query("GetClusterSnapshot", filter="status")
            assert status["lifecycle_state"] == "ready"
            assert status["components"]["ray_runtime"]["backend"] == "FakeRuntimeBackend"
            assert status["components"]["recorder"]["status"] == "ready"
            assert status["components"]["placement"]["active_lease_count"] == 1
            assert status["components"]["worker_pool"]["status"] == "disabled"
            assert status["components"]["inference"]["status"] == "disabled"

            watched_version = resources["meta"]["snapshot_version"]
            placement.set_node_status(
                "node_a", NodeStatus.DRAINING, now_ms=controller.clock.monotonic_ms()
            )
            placement.set_node_status(
                "node_a", NodeStatus.HEALTHY, now_ms=controller.clock.monotonic_ms()
            )
            watch = client.watch_cluster(
                after_snapshot_version=watched_version,
                limit=1,
                timeout_seconds=2,
            )
            jumped = await anext(watch)
            assert jumped["snapshot_required"] is True
            assert jumped["events"] == []
            assert jumped["next_snapshot_version"] == watched_version + 2
            await watch.aclose()
        finally:
            placement.release_lease(
                reservation.lease.lease_id,
                now_ms=controller.clock.monotonic_ms(),
                reason="test_complete",
            )
            await server.close(grace_seconds=0)
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_local_control_service_exposes_authoritative_snapshots_and_c8(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        socket_path = (tmp_path / "runtime" / "control.sock").resolve()
        recorder = ParquetRecorder(
            ParquetRecorderConfig(
                root_directory=str(tmp_path / "records"),
                flush_interval_ms=5,
            ),
            cursor_signing_key=b"x" * 32,
        )
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test_build",
            node_capacities=(
                NodeCapacity(
                    node_id="node_a",
                    boot_id="boot_a",
                    node_ip="127.0.0.1",
                    cpu_total=2,
                    mem_total_mb=512,
                    cpu_system_reserved=0,
                    mem_system_reserved_mb=0,
                    io_slots_total=1,
                    observed_free_mem_mb=512,
                ),
            ),
            recorder=recorder,
        )
        await controller.start()
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: ControllerStatus(
                controller_generation=controller.controller_generation,
                build_revision=controller.build_revision,
                environment_fingerprint=controller.environment_fingerprint,
                healthy_node_count=1,
            ),
            control_api=controller,
        )
        await server.start()
        client = UdsRuntimeClient(
            socket_path,
            data_store=controller.data_store,
            data_owner_generation=controller.data_owner_generation,
        )
        status = await client.get_controller_status()
        assert status.controller_generation == controller.controller_generation

        submitted_workflow = Workflow("local-control-submit")
        submitted_workflow.add_task(_barrier)
        submitted_run = await client.run(
            submitted_workflow,
            inputs={},
            submission_id="uds_submission",
        )
        assert (
            await client.run(
                submitted_workflow,
                inputs={},
                submission_id="uds_submission",
            )
            == submitted_run
        )
        assert (await controller.wait_run(submitted_run, timeout_seconds=2)).status is RunStatus.SUCCEEDED

        result_workflow = Workflow("local-control-result")
        value = result_workflow.input("value")
        result_task = result_workflow.add_task(_echo, inputs={"value": value})
        result_run = await client.run(result_workflow, inputs={"value": "payload"})
        await controller.wait_run(result_run, timeout_seconds=2)
        assert await client.materialize_task_result(
            result_run, result_task.task_id
        ) == {"value": "payload"}

        workflow = Workflow("local-control-query")
        task = workflow.add_task(_barrier)
        run_id = await InMemoryRuntimeClient(controller).run(workflow, inputs={})
        terminal = await controller.wait_run(run_id, timeout_seconds=2)
        assert terminal.status is RunStatus.SUCCEEDED

        system = await client.query("GetSystemSnapshot")
        assert system["run_count"] == 3
        cluster = await client.query("GetClusterSnapshot", filter="resources")
        assert cluster["cluster"]["nodes"][0]["capacity"]["node_id"] == "node_a"
        worker_pools = await client.query("GetWorkerPools")
        assert worker_pools["worker_pool"] == {
            "active_worker_lease_count": 0,
            "mode": "disabled",
            "worker_leases": [],
            "workers": [],
        }
        cluster_version = cluster["meta"]["snapshot_version"]
        assert isinstance(cluster_version, int)
        cluster_watch = client.watch_cluster(
            after_snapshot_version=cluster_version,
            limit=1,
            timeout_seconds=2,
        )
        next_cluster = asyncio.create_task(anext(cluster_watch))
        await asyncio.sleep(0.025)
        controller.record_control_request(
            run_id,
            request_id="cluster_watch_request",
            operation="test_cluster_watch",
        )
        cluster_batch = await next_cluster
        assert cluster_batch["snapshot_required"] is False
        assert cluster_batch["events"] == [
            {
                "event_type": "cluster_snapshot_changed",
                "snapshot_version": cluster_version + 1,
            }
        ]
        runs = await client.query("ListRuns")
        assert {item["run_id"] for item in runs["runs"]} == {
            run_id,
            submitted_run,
            result_run,
        }
        shown = await client.query("GetRun", resource_id=run_id)
        assert shown["run"]["status"] == "succeeded"
        assert shown["recording_complete"] is None

        batches = [item async for item in client.watch_run(run_id)]
        assert batches[-1]["run_terminal"] is True
        assert any(
            event["event_type"] == "run_terminal"
            for batch in batches
            for event in batch["events"]
        )

        flushed = await client.run_action(
            "FlushRun", run_id, request_id="flush_request"
        )
        assert flushed["recording_complete"] is True
        replayed = await client.run_action(
            "FlushRun", run_id, request_id="flush_request"
        )
        assert replayed == flushed
        page = await client.get_run_events(run_id, limit=1)
        assert len(page["events"]) == 1
        assert page["next_cursor"]
        assert "producer_sequence" in page["events"][0]
        committed_events = await client.get_run_events(run_id, limit=100)
        control_events = [
            event
            for event in committed_events["events"]
            if event["event_type"] == "control_request"
        ]
        flush_event = next(
            event
            for event in control_events
            if event["payload"]["operation"] == "flush_run"
        )
        assert flush_event["payload"] == {
            "config_fingerprint": "c" * 64,
            "controller_generation": controller.controller_generation,
            "operation": "flush_run",
            "request_id": "flush_request",
        }

        handles = await client.query(
            "GetTaskResultHandles",
            resource_id=run_id,
            filter=task.task_id,
        )
        assert handles["handles"] == []
        recorder_status = await client.query("GetRecorderStatus")
        assert recorder_status["backend"] == "parquet"

        with pytest.raises(ControlRpcError, match="requires the current boot_id"):
            await client.node_action("DrainNode", "node_a")
        with pytest.raises(ControlRpcError, match="boot_id changed"):
            await client.node_action(
                "DrainNode", "node_a", boot_id="stale_boot"
            )
        drained = await client.node_action(
            "DrainNode",
            "node_a",
            boot_id="boot_a",
            request_id="drain_request",
        )
        assert drained["status"] == "drained"
        assert drained["cleanup_confirmed"] is True
        assert (
            await client.node_action(
                "DrainNode",
                "node_a",
                boot_id="boot_a",
                request_id="drain_request",
            )
            == drained
        )
        resumed = await client.node_action(
            "ResumeNode",
            "node_a",
            boot_id="boot_a",
            request_id="resume_request",
        )
        assert resumed["status"] == "healthy"
        with pytest.raises(ControlRpcError, match="does not support force"):
            await client.node_action(
                "ResumeNode", "node_a", boot_id="boot_a", force=True
            )
        assert (
            await client.node_action(
                "ResumeNode",
                "node_a",
                boot_id="boot_a",
                request_id="resume_request",
            )
            == resumed
        )

        destroyed = await client.run_action("DestroyRun", run_id)
        assert destroyed["tombstone"]["destroy_succeeded"] is True
        with pytest.raises(ControlRpcError, match="different operation or payload"):
            await client.run_action(
                "CancelRun",
                run_id,
                request_id="flush_request",
            )

        await server.close()
        await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_submit_retries_same_prepared_handles_after_committed_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = local_rpc._LocalControlServicer.SubmitWorkflow
    response_lost = False

    async def lose_first_response(self, request, context):
        nonlocal response_lost
        response = await original(self, request, context)
        if not response_lost and response.status_code == "ok":
            response_lost = True
            await context.abort(grpc.StatusCode.UNAVAILABLE, "injected response loss")
        return response

    monkeypatch.setattr(
        local_rpc._LocalControlServicer,
        "SubmitWorkflow",
        lose_first_response,
    )

    async def scenario() -> None:
        socket_path = (tmp_path / "runtime" / "control.sock").resolve()
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(
                NodeCapacity(
                    node_id="node_a",
                    boot_id="boot_a",
                    node_ip="127.0.0.1",
                    cpu_total=2,
                    mem_total_mb=512,
                    cpu_system_reserved=0,
                    mem_system_reserved_mb=0,
                    io_slots_total=1,
                    observed_free_mem_mb=512,
                ),
            ),
        )
        await controller.start()
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: ControllerStatus(
                controller_generation=controller.controller_generation,
                build_revision=controller.build_revision,
                environment_fingerprint=controller.environment_fingerprint,
                healthy_node_count=1,
            ),
            control_api=controller,
        )
        await server.start()
        client = UdsRuntimeClient(
            socket_path,
            data_store=controller.data_store,
            data_owner_generation=controller.data_owner_generation,
        )
        workflow = Workflow("submit-response-loss")
        value = workflow.input("value")
        workflow.add_task(_echo, inputs={"value": value})
        try:
            missing = await client.get_submission_status(
                "response_loss_submission"
            )
            assert missing == {
                "found": False,
                "submission_id": "response_loss_submission",
            }
            outcome = await client.submit(
                workflow,
                inputs={"value": "same staged input"},
                submission_id="response_loss_submission",
            )
            assert response_lost
            assert outcome["state"] == "committed"
            assert outcome["replayed"] is True
            assert len(controller.list_runs()) == 1
            assert client.prepared_submission_count == 0
            status = await client.get_submission_status(
                "response_loss_submission"
            )
            assert status["found"] is True
            assert status["submission"]["run_id"] == outcome["run_id"]

            second_client = UdsRuntimeClient(
                socket_path,
                data_store=controller.data_store,
                data_owner_generation=controller.data_owner_generation,
            )
            conflicting = await second_client.prepare_submission(
                workflow,
                inputs={"value": "same staged input"},
                submission_id="response_loss_submission",
            )
            active_before_conflict = controller.data_store.active_count
            with pytest.raises(ControlRpcError, match="different payload"):
                await second_client.submit_prepared(conflicting)
            assert controller.data_store.active_count == active_before_conflict - 1
        finally:
            await server.close(grace_seconds=0)
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_local_prepared_submission_rejects_payload_conflict_before_reupload(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        socket_path = (tmp_path / "runtime" / "control.sock").resolve()
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(
                NodeCapacity(
                    node_id="node_a",
                    boot_id="boot_a",
                    node_ip="127.0.0.1",
                    cpu_total=2,
                    mem_total_mb=512,
                    cpu_system_reserved=0,
                    mem_system_reserved_mb=0,
                    io_slots_total=1,
                    observed_free_mem_mb=512,
                ),
            ),
        )
        await controller.start()
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: ControllerStatus(
                controller_generation=controller.controller_generation,
                build_revision=controller.build_revision,
                environment_fingerprint=controller.environment_fingerprint,
                healthy_node_count=1,
            ),
            control_api=controller,
        )
        await server.start()
        client = UdsRuntimeClient(
            socket_path,
            data_store=controller.data_store,
            data_owner_generation=controller.data_owner_generation,
        )
        workflow = Workflow("prepared-conflict")
        value = workflow.input("value")
        workflow.add_task(_echo, inputs={"value": value})
        try:
            first_value = {"payload": "first"}
            first = await client.prepare_submission(
                workflow,
                inputs={"value": first_value},
                submission_id="conflicting_submission",
            )
            cached = await client.prepare_submission(
                workflow,
                inputs={"value": first_value},
                submission_id="conflicting_submission",
            )
            assert cached is first
            with pytest.raises(SubmissionConflictError, match="another payload"):
                await client.prepare_submission(
                    workflow,
                    inputs={"value": {"payload": "first"}},
                    submission_id="conflicting_submission",
                )
            assert client.prepared_submission_count == 1
            assert controller.data_store.state_of(
                first.request.workflow_inputs[0][1]
            ) == "staged"
            outcome = await client.submit_prepared(first)
            assert outcome["state"] == "committed"
        finally:
            await server.close(grace_seconds=0)
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_watch_disconnect_does_not_cancel_run_and_reconnects_by_sequence(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        socket_path = (tmp_path / "runtime" / "control.sock").resolve()
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(
                NodeCapacity(
                    node_id="node_a",
                    boot_id="boot_a",
                    node_ip="127.0.0.1",
                    cpu_total=2,
                    mem_total_mb=512,
                    cpu_system_reserved=0,
                    mem_system_reserved_mb=0,
                    io_slots_total=1,
                    observed_free_mem_mb=512,
                ),
            ),
        )
        await controller.start()
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: ControllerStatus(
                controller_generation=controller.controller_generation,
                build_revision=controller.build_revision,
                environment_fingerprint=controller.environment_fingerprint,
                healthy_node_count=1,
            ),
            control_api=controller,
        )
        await server.start()
        client = UdsRuntimeClient(
            socket_path,
            data_store=controller.data_store,
            data_owner_generation=controller.data_owner_generation,
        )
        workflow = Workflow("watch-disconnect")
        value = workflow.input("value")
        workflow.add_task(_slow_echo, inputs={"value": value})
        try:
            run_id = await client.run(workflow, inputs={"value": "payload"})
            watch = client.watch_run(run_id, limit=1)
            first = await anext(watch)
            first_sequence = first["next_sequence"]
            assert isinstance(first_sequence, int) and first_sequence > 0
            await watch.aclose()

            terminal = await controller.wait_run(run_id, timeout_seconds=2)
            assert terminal.status is RunStatus.SUCCEEDED
            resumed = [
                batch
                async for batch in client.watch_run(
                    run_id,
                    after_sequence=first_sequence,
                    limit=100,
                )
            ]
            sequences = [
                event["sequence"]
                for batch in resumed
                for event in batch["events"]
            ]
            assert sequences
            assert all(sequence > first_sequence for sequence in sequences)
            assert resumed[-1]["run_terminal"] is True
        finally:
            await server.close(grace_seconds=0)
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())


def test_watch_snapshot_required_fetches_run_and_continues_automatically(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        socket_path = (tmp_path / "runtime" / "control.sock").resolve()
        controller = InMemoryController(
            config_fingerprint="c" * 64,
            environment_fingerprint="e" * 64,
            build_revision="test",
            node_capacities=(
                NodeCapacity(
                    node_id="node_a",
                    boot_id="boot_a",
                    node_ip="127.0.0.1",
                    cpu_total=2,
                    mem_total_mb=512,
                    cpu_system_reserved=0,
                    mem_system_reserved_mb=0,
                    io_slots_total=1,
                    observed_free_mem_mb=512,
                ),
            ),
            control_event_retention_count=1,
        )
        await controller.start()
        server = LocalControlServer(
            socket_path=socket_path,
            status_provider=lambda: ControllerStatus(
                controller_generation=controller.controller_generation,
                build_revision=controller.build_revision,
                environment_fingerprint=controller.environment_fingerprint,
                healthy_node_count=1,
            ),
            control_api=controller,
        )
        await server.start()
        client = UdsRuntimeClient(
            socket_path,
            data_store=controller.data_store,
            data_owner_generation=controller.data_owner_generation,
        )
        workflow = Workflow("watch-snapshot-recovery")
        value = workflow.input("value")
        workflow.add_task(_slow_echo, inputs={"value": value})
        try:
            run_id = await client.run(workflow, inputs={"value": "payload"})
            batches = [
                batch
                async for batch in client.watch_run(
                    run_id,
                    after_sequence=1,
                    limit=1,
                    timeout_seconds=2,
                )
            ]
            assert batches[0]["snapshot_required"] is True
            assert any(batch["run_terminal"] is True for batch in batches)
            assert controller.snapshot(run_id).status is RunStatus.SUCCEEDED
        finally:
            await server.close(grace_seconds=0)
            await controller.close(force=True, drain_timeout_ms=0)

    asyncio.run(scenario())
