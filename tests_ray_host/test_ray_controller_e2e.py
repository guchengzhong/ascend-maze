from __future__ import annotations

import asyncio
from pathlib import Path

import ray
import pytest

from ascend_maze import Workflow, task
from ascend_maze.control.client import InMemoryRuntimeClient
from ascend_maze.control.local_rpc import UdsRuntimeClient
from ascend_maze.control.node_rpc import NodeAgent, NodeAgentIdentity
from ascend_maze.control.node_rpc import report_worker_event
from ascend_maze.control.ray_controller import RayHostController
from ascend_maze.contracts.submission import SubmissionState
from ascend_maze.core.errors import DataHandleInvalidError
from ascend_maze.core.errors import (
    ResponseLostError,
    SubmissionConflictError,
)
from ascend_maze.lifecycle import RunStatus, TaskStatus
from ascend_maze.placement import LeaseStatus, NodeCapacity, NodeStatus
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind


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


def _logical_node(node_id: str) -> NodeCapacity:
    return NodeCapacity(
        node_id=node_id,
        boot_id=f"boot_{node_id}",
        node_ip="127.0.0.1",
        cpu_total=1,
        mem_total_mb=256,
        cpu_system_reserved=0,
        mem_system_reserved_mb=0,
        io_slots_total=1,
        observed_free_mem_mb=256,
    )


def test_ray_host_controller_completes_submit_result_destroy_closure(
    ray_namespace: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        @task
        def load(value: str):
            return {"text": value}

        @task
        def finish(text: str):
            return {"result": text}

        socket_path = tmp_path / "runtime" / "control.sock"
        controller = RayHostController(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
            control_socket_path=socket_path,
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_1",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(
                identity=identity,
                authorization_token=b"test-token",
                heartbeat_interval_ms=50,
            )
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            status = await UdsRuntimeClient(socket_path).get_controller_status()
            assert status.controller_generation == controller.controller_generation
            assert status.healthy_node_count == 1

            workflow = Workflow("ray-host-e2e")
            value = workflow.input("value")
            loaded = workflow.add_task(load, inputs={"value": value})
            result_task = workflow.add_task(
                finish,
                inputs={"text": loaded.outputs["text"]},
            )
            client = InMemoryRuntimeClient(controller)
            outcome = await client.submit(
                workflow,
                inputs={"value": "payload"},
                submission_id="submission_ray_host_e2e",
            )
            assert outcome.state is SubmissionState.COMMITTED
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=20
            )
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, result_task.task_id) == {
                "result": "payload"
            }
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            assert controller.deadlines.count_for_run(outcome.run_id) == 0
            assert controller.worker_broker.active_count() == 0
            assert controller.ray_data_store.staged_count == 0
            assert controller.ray_runtime.active_dispatch_count(outcome.run_id) == 0

            destroyed = await controller.destroy_run(outcome.run_id)
            repeated = await controller.destroy_run(outcome.run_id)
            assert repeated is destroyed
            assert destroyed.flush_result.recording_complete
            assert destroyed.tombstone.destroy_succeeded
            assert controller.ray_data_store.active_count == 0
            assert controller.ray_runtime.code_reference_count() == 0
            assert controller.indexes.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_node_registration_resource_change_wakes_queued_run(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task
        def echo(value: str):
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_resource_wakeup",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            controller.placement.set_node_status(
                "node_a",
                NodeStatus.UNSCHEDULABLE,
                now_ms=controller.clock.monotonic_ms(),
            )
            workflow = Workflow("ray-resource-change-wakeup")
            node = workflow.add_task(echo, inputs={"value": "ready"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_ray_resource_wakeup",
            )
            assert outcome.run_id is not None
            await controller.core.wake_deadlines()
            queued = controller.snapshot(outcome.run_id).task(node.task_id)
            assert queued.status is TaskStatus.QUEUED
            assert queued.attempt_count == 0

            identity = NodeAgentIdentity(
                cluster_id="cluster_resource_wakeup",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(
                identity=identity,
                authorization_token=b"test-token",
                heartbeat_interval_ms=50,
            )
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)

            terminal = await controller.wait_run(outcome.run_id, timeout_seconds=20)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, node.task_id) == {"result": "ready"}
            assert any(
                event.event_type == "resource_changed"
                and event.payload["reason"] == "node_binding_registered:node_a"
                for event in controller.recorder.events(outcome.run_id)
            )
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            destroyed = await controller.destroy_run(outcome.run_id)
            assert destroyed.flush_result.recording_complete
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_large_payload_is_only_materialized_by_worker_until_result_is_requested(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task
        def echo_payload(payload: bytes):
            return {"result": payload}

        controller = RayHostController(
            cluster_id="cluster_large_payload",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_large_payload",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(identity=identity, authorization_token=b"test-token")
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            payload = b"x" * (12 * 1024 * 1024)
            workflow = Workflow("large-payload")
            payload_input = workflow.input("payload")
            output = workflow.add_task(
                echo_payload, inputs={"payload": payload_input}
            )
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={"payload": payload},
                submission_id="submission_large_payload",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=30
            )
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.ray_data_store.local_get_count == 0
            result = controller.result(outcome.run_id, output.task_id)
            assert result == {"result": payload}
            assert controller.ray_data_store.local_get_count == 1
            destroyed = await controller.destroy_run(outcome.run_id)
            assert destroyed.flush_result.recording_complete
            assert controller.ray_data_store.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_cancelled_ray_attempt_releases_worker_and_late_output_handle(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task(max_retries=0)
        def slow_task(value: str):
            import time

            time.sleep(30)
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_cancel",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_cancel",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(identity=identity, authorization_token=b"test-token")
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            workflow = Workflow("cancel-ray-worker")
            node = workflow.add_task(slow_task, inputs={"value": "value"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_cancel_ray_worker",
            )
            assert outcome.run_id is not None
            attempt = None
            for _ in range(1_000):
                snapshot = controller.snapshot(outcome.run_id)
                task_snapshot = snapshot.task(node.task_id)
                if (
                    task_snapshot.attempts
                    and task_snapshot.attempts[0].worker_started_at_ms is not None
                ):
                    attempt = task_snapshot.attempts[0]
                    break
                await asyncio.sleep(0.01)
            assert attempt is not None
            cancelled = await controller.cancel_run(outcome.run_id)
            assert cancelled.status is RunStatus.CANCELLED
            assert controller.worker_broker.active_count() == 0
            assert controller.placement.active_lease_count(outcome.run_id) == 0

            late_handle = await asyncio.to_thread(
                controller.ray_data_store.put_staged,
                b"late-result",
                controller.controller_generation,
            )
            late_event = RuntimeEvent.create(
                kind=RuntimeEventKind.TASK_RESULT,
                dispatch_id=attempt.dispatch_id,
                run_id=outcome.run_id,
                task_id=node.task_id,
                attempt=attempt.attempt,
                lease_id=attempt.lease_id,
                route_lease_id=None,
                occurred_at_ms=controller.clock.monotonic_ms(),
                output_handles=(("result", late_handle),),
            )
            assert agent.endpoint is not None
            await asyncio.to_thread(
                report_worker_event,
                endpoint=agent.endpoint,
                identity=identity,
                event=late_event,
                timeout_seconds=2,
            )
            for _ in range(200):
                try:
                    controller.ray_data_store.state_of(late_handle)
                except DataHandleInvalidError:
                    break
                await asyncio.sleep(0.01)
            with pytest.raises(DataHandleInvalidError):
                controller.ray_data_store.get(late_handle)
            assert controller.snapshot(outcome.run_id).status is RunStatus.CANCELLED
            destroyed = await controller.destroy_run(outcome.run_id)
            assert destroyed.flush_result.recording_complete
            assert controller.ray_data_store.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_node_agent_disconnect_invalidates_running_worker_and_c6_node(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task(max_retries=0)
        def slow_task(value: str):
            import time

            time.sleep(30)
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_disconnect",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_disconnect",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(identity=identity, authorization_token=b"test-token")
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            workflow = Workflow("disconnect-ray-worker")
            node = workflow.add_task(slow_task, inputs={"value": "value"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_disconnect_ray_worker",
            )
            assert outcome.run_id is not None
            for _ in range(1_000):
                task_snapshot = controller.snapshot(outcome.run_id).task(node.task_id)
                if (
                    task_snapshot.attempts
                    and task_snapshot.attempts[0].worker_started_at_ms is not None
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Ray Worker did not start")
            await agent.close(grace_seconds=0)
            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=20
            )
            assert terminal.status is RunStatus.FAILED
            assert controller.node_registry.status("node_a").value == "stale"
            cluster = controller.placement.snapshot()
            assert cluster.nodes[0].status is NodeStatus.OFFLINE
            assert controller.placement.active_lease_count(outcome.run_id) == 0
            assert controller.worker_broker.active_count() == 0
            with pytest.raises(RuntimeError, match="recording is incomplete"):
                await controller.destroy_run(outcome.run_id)
            forced = await controller.destroy_run(outcome.run_id, force=True)
            assert not forced.flush_result.recording_complete
            assert forced.flush_result.writer_errors
            assert controller.ray_data_store.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_boot_replacement_invalidates_old_binding_lease_worker_and_dispatch(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task(max_retries=0)
        def slow_task(value: str):
            import time

            time.sleep(30)
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_boot_replace",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agents: list[NodeAgent] = []
        try:
            await controller.start()
            ray_node_id = ray.get_runtime_context().get_node_id()
            first_identity = NodeAgentIdentity(
                cluster_id="cluster_boot_replace",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray_node_id,
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            first = NodeAgent(
                identity=first_identity,
                authorization_token=b"test-token",
            )
            await first.start(controller_endpoint=controller.node_rpc_endpoint)
            agents.append(first)
            old_binding = controller.node_registry.binding("node_a")

            workflow = Workflow("boot-replacement")
            node = workflow.add_task(slow_task, inputs={"value": "value"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_boot_replacement",
            )
            assert outcome.run_id is not None
            old_attempt = None
            for _ in range(1_000):
                task_snapshot = controller.snapshot(outcome.run_id).task(node.task_id)
                if (
                    task_snapshot.attempts
                    and task_snapshot.attempts[0].worker_started_at_ms is not None
                ):
                    old_attempt = task_snapshot.attempts[0]
                    break
                await asyncio.sleep(0.01)
            assert old_attempt is not None
            stale_output_handle = await asyncio.to_thread(
                controller.ray_data_store.put_staged_for_runtime_node,
                b"unpublished-old-output",
                controller.controller_generation,
                node_id="node_a",
                boot_id="boot_1",
                runtime_generation=old_binding.runtime_generation,
            )

            second_identity = NodeAgentIdentity(
                cluster_id="cluster_boot_replace",
                node_id="node_a",
                boot_id="boot_2",
                ray_node_id=ray_node_id,
                agent_generation="agent_2",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_2",
            )
            second = NodeAgent(
                identity=second_identity,
                authorization_token=b"test-token",
            )
            await second.start(controller_endpoint=controller.node_rpc_endpoint)
            agents.append(second)

            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=20
            )
            assert terminal.status is RunStatus.FAILED
            new_binding = controller.node_registry.binding("node_a")
            assert new_binding.boot_id == "boot_2"
            assert new_binding.runtime_generation == old_binding.runtime_generation + 1
            placement_node = controller.placement.snapshot().nodes[0]
            assert placement_node.capacity.boot_id == "boot_2"
            assert placement_node.status is NodeStatus.HEALTHY
            old_lease = controller.placement.lease_snapshot(old_attempt.lease_id)
            assert old_lease.status is LeaseStatus.INVALIDATED
            assert controller.ray_runtime.dispatch_invalidated(
                old_attempt.dispatch_id
            )
            assert controller.worker_broker.active_count() == 0
            with pytest.raises(DataHandleInvalidError):
                controller.ray_data_store.get(stale_output_handle)
            with pytest.raises(RuntimeError, match="recording is incomplete"):
                await controller.destroy_run(outcome.run_id)
            forced = await controller.destroy_run(outcome.run_id, force=True)
            assert not forced.flush_result.recording_complete
            assert controller.ray_data_store.active_count == 0
        finally:
            for agent in agents:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_ray_submission_response_loss_replays_and_conflict_releases_upload(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task
        def echo(value: str):
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_submission",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_submission",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(identity=identity, authorization_token=b"test-token")
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            workflow = Workflow("ray-submission-idempotency")
            value = workflow.input("value")
            result_task = workflow.add_task(echo, inputs={"value": value})
            first_client = InMemoryRuntimeClient(controller)
            prepared = first_client.prepare_submission(
                workflow,
                inputs={"value": "first"},
                submission_id="submission_stable_ray",
            )
            committed_input = prepared.request.workflow_inputs[0][1]
            with pytest.raises(ResponseLostError):
                await first_client.submit_prepared(
                    prepared, lose_response_after_commit=True
                )
            replay = await first_client.submit_prepared(prepared)
            assert replay.replayed
            assert replay.run_id is not None
            assert controller.ray_data_store.state_of(committed_input) == "adopted"

            same_client = InMemoryRuntimeClient(controller)
            redundant = same_client.prepare_submission(
                workflow,
                inputs={"value": "first"},
                submission_id="submission_stable_ray",
            )
            redundant_handle = redundant.request.workflow_inputs[0][1]
            same_replay = await same_client.submit_prepared(redundant)
            assert same_replay.replayed
            assert same_replay.run_id == replay.run_id
            with pytest.raises(DataHandleInvalidError):
                controller.ray_data_store.get(redundant_handle)

            conflict_client = InMemoryRuntimeClient(controller)
            conflicting = conflict_client.prepare_submission(
                workflow,
                inputs={"value": "different"},
                submission_id="submission_stable_ray",
            )
            conflicting_handle = conflicting.request.workflow_inputs[0][1]
            with pytest.raises(SubmissionConflictError):
                await conflict_client.submit_prepared(conflicting)
            with pytest.raises(DataHandleInvalidError):
                controller.ray_data_store.get(conflicting_handle)

            terminal = await controller.wait_run(replay.run_id, timeout_seconds=20)
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(replay.run_id, result_task.task_id) == {
                "result": "first"
            }
            await controller.destroy_run(replay.run_id)
            assert controller.ray_data_store.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_partial_ray_output_put_failure_releases_every_unpublished_handle(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task(max_retries=0)
        def two_outputs(value: str):
            return {"left": value, "right": value}

        controller = RayHostController(
            cluster_id="cluster_partial_output",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_partial_output",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(identity=identity, authorization_token=b"test-token")
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            workflow = Workflow("partial-ray-output")
            node = workflow.add_task(two_outputs, inputs={"value": "value"})
            client = InMemoryRuntimeClient(controller)
            prepared = client.prepare_submission(
                workflow,
                inputs={},
                submission_id="submission_partial_ray_output",
            )
            controller.ray_data_store.fail_on_put_number(
                controller.ray_data_store.put_count + 3
            )
            outcome = await client.submit_prepared(prepared)
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=20
            )
            assert terminal.status is RunStatus.FAILED
            assert terminal.task(node.task_id).last_error is not None
            assert (
                terminal.task(node.task_id).last_error.error_code
                == "result_publish_failed"
            )
            assert controller.ray_data_store.staged_count == 0
            assert controller.worker_broker.active_count() == 0
            destroyed = await controller.destroy_run(outcome.run_id)
            assert destroyed.flush_result.recording_complete
            assert controller.ray_data_store.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_ray_submit_prepare_rollback_releases_staged_input_and_code(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task
        def echo(value: str):
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_submit_rollback",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        try:
            await controller.start()
            workflow = Workflow("ray-submit-rollback")
            value = workflow.input("value")
            workflow.add_task(echo, inputs={"value": value})
            client = InMemoryRuntimeClient(controller)
            prepared = client.prepare_submission(
                workflow,
                inputs={"value": "payload"},
                submission_id="submission_ray_rollback",
            )
            controller.inject_submit_failure("after_prepare")
            outcome = await client.submit_prepared(prepared)
            assert outcome.state is SubmissionState.ABORTED
            assert outcome.run_id is None
            assert controller.ray_runtime.code_reference_count() == 0
            assert controller.ray_data_store.active_count == 0
            assert controller.indexes.active_count == 0
        finally:
            await controller.close()

    asyncio.run(scenario())


def test_empty_output_ray_task_publishes_no_data_handle(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task
        def barrier():
            return {}

        controller = RayHostController(
            cluster_id="cluster_empty_output",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_empty_output",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(identity=identity, authorization_token=b"test-token")
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            workflow = Workflow("empty-output-ray-task")
            node = workflow.add_task(barrier)
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_empty_output_ray",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=20
            )
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, node.task_id) == {}
            assert controller.indexes.get(outcome.run_id).handle_count() == 0
            assert controller.ray_data_store.staged_count == 0
            destroyed = await controller.destroy_run(outcome.run_id)
            assert destroyed.tombstone.released_handle_count == 0
            assert controller.ray_data_store.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_missing_node_agent_producer_marks_recording_incomplete(
    ray_namespace: str,
) -> None:
    async def scenario() -> None:
        @task(max_retries=0)
        def never_recorded(value: str):
            return {"result": value}

        controller = RayHostController(
            cluster_id="cluster_missing_producer",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=(_node(),),
        )
        agent: NodeAgent | None = None
        try:
            await controller.start()
            identity = NodeAgentIdentity(
                cluster_id="cluster_missing_producer",
                node_id="node_a",
                boot_id="boot_1",
                ray_node_id=ray.get_runtime_context().get_node_id(),
                agent_generation="agent_1",
                environment_fingerprint=ENVIRONMENT,
                producer_id="node_agent:node_a:agent_1",
            )
            agent = NodeAgent(identity=identity, authorization_token=b"test-token")
            await agent.start(controller_endpoint=controller.node_rpc_endpoint)
            await agent.stop_worker_event_server()

            workflow = Workflow("missing-node-producer")
            workflow.add_task(never_recorded, inputs={"value": "value"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_missing_producer",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=20
            )
            assert terminal.status is RunStatus.FAILED
            with pytest.raises(RuntimeError, match="recording is incomplete"):
                await controller.destroy_run(outcome.run_id)
            forced = await controller.destroy_run(outcome.run_id, force=True)
            assert not forced.flush_result.recording_complete
            assert forced.flush_result.missing_producer_count == 1
            assert forced.flush_result.writer_errors
            assert controller.ray_data_store.active_count == 0
        finally:
            if agent is not None:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())


def test_two_node_host_dag_matches_every_c6_lease_to_actual_ray_node(
    ray_namespace: str,
    ray_node_ids: tuple[str, str],
) -> None:
    async def scenario() -> None:
        @task
        def left_task(value: str):
            return {"result": value}

        @task(task_kind="io")
        def right_task(value: str):
            return {"result": value}

        capacities = (_logical_node("node_a"), _logical_node("node_b"))
        controller = RayHostController(
            cluster_id="cluster_two_node",
            authorization_token=b"test-token",
            ray_namespace=ray_namespace,
            config_fingerprint=CONFIG,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test_build",
            node_capacities=capacities,
        )
        agents: list[NodeAgent] = []
        try:
            await controller.start()
            node_to_ray = dict(zip(("node_a", "node_b"), ray_node_ids, strict=True))
            for capacity in capacities:
                identity = NodeAgentIdentity(
                    cluster_id="cluster_two_node",
                    node_id=capacity.node_id,
                    boot_id=capacity.boot_id,
                    ray_node_id=node_to_ray[capacity.node_id],
                    agent_generation=f"agent_{capacity.node_id}",
                    environment_fingerprint=ENVIRONMENT,
                    producer_id=f"node_agent:{capacity.node_id}:1",
                )
                agent = NodeAgent(
                    identity=identity,
                    authorization_token=b"test-token",
                    heartbeat_interval_ms=50,
                )
                await agent.start(controller_endpoint=controller.node_rpc_endpoint)
                agents.append(agent)

            workflow = Workflow("ray-host-two-node")
            left = workflow.add_task(left_task, inputs={"value": "left"})
            right = workflow.add_task(right_task, inputs={"value": "right"})
            outcome = await InMemoryRuntimeClient(controller).submit(
                workflow,
                inputs={},
                submission_id="submission_two_node",
            )
            assert outcome.run_id is not None
            terminal = await controller.wait_run(
                outcome.run_id, timeout_seconds=20
            )
            assert terminal.status is RunStatus.SUCCEEDED
            assert controller.result(outcome.run_id, left.task_id) == {
                "result": "left"
            }
            assert controller.result(outcome.run_id, right.task_id) == {
                "result": "right"
            }
            attempt_nodes = {
                attempt.node_id
                for task_snapshot in terminal.task_states
                for attempt in task_snapshot.attempts
            }
            assert attempt_nodes == {"node_a", "node_b"}
            for task_snapshot in terminal.task_states:
                attempt = task_snapshot.attempts[0]
                assert attempt.node_id is not None
                worker_outcome = controller.ray_runtime.worker_outcome(
                    attempt.dispatch_id
                )
                assert worker_outcome is not None
                assert worker_outcome.ray_node_id == node_to_ray[attempt.node_id]

            node_producers = {
                event.producer_id
                for event in controller.ray_recorder.events(outcome.run_id)
                if event.node_id is not None
            }
            assert node_producers == {
                "node_agent:node_a:1",
                "node_agent:node_b:1",
            }
            destroyed = await controller.destroy_run(outcome.run_id)
            assert destroyed.flush_result.recording_complete
            assert controller.ray_data_store.active_count == 0
        finally:
            for agent in agents:
                await agent.close(grace_seconds=0)
            await controller.close()

    asyncio.run(scenario())
