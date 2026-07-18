from __future__ import annotations

import asyncio

import pytest

from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.contracts.recording import RunRecordingContext
from ascend_maze.core.canonical import FrozenMap
from ascend_maze.recording import InMemoryRecorder
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.runtime.ray_node_registry import (
    RayNodeRegistry,
    RuntimeNodeStatus,
)

from ascend_maze.control.node_rpc import (
    NodeAgent,
    NodeAgentIdentity,
    NodeControlServer,
    report_worker_event,
)
from ascend_maze.control.proto_codec import decode_runtime_event, encode_runtime_event


ENVIRONMENT = "e" * 64


def _identity(*, generation: str = "agent_1") -> NodeAgentIdentity:
    return NodeAgentIdentity(
        cluster_id="cluster_1",
        node_id="node_a",
        boot_id="boot_1",
        ray_node_id="ray_node_a",
        agent_generation=generation,
        environment_fingerprint=ENVIRONMENT,
        producer_id=f"node_agent:node_a:{generation}",
    )


def _event() -> RuntimeEvent:
    handle = DataHandle(
        owner_generation="controller_1",
        staged_handle_id="data_1",
        stable_digest="a" * 64,
        size_bytes=7,
        metadata=FrozenMap((("backend", "ray"),)),
    )
    error = ErrorInfo(
        schema_version=1,
        error_code="worker_lost",
        category="worker",
        origin="worker",
        message="worker exited",
        retryable_hint=True,
        classification_confidence="exact",
        execution_phase="running",
        run_id="run_1",
        task_id="task_1",
        attempt=1,
        dispatch_id="dispatch_1",
        lease_id="lease_1",
        node_id="node_a",
        boot_id="boot_1",
        worker_id="worker_1",
        exception_type="RuntimeError",
        occurred_at_ms=11,
    )
    return RuntimeEvent(
        event_id="runtime_event_1",
        kind=RuntimeEventKind.TASK_FAILED,
        dispatch_id="dispatch_1",
        run_id="run_1",
        task_id="task_1",
        attempt=1,
        lease_id="lease_1",
        route_lease_id=None,
        occurred_at_ms=11,
        output_handles=(("result", handle),),
        error=error,
    )


def test_runtime_event_protobuf_round_trip_preserves_control_identity() -> None:
    event = _event()
    restored = decode_runtime_event(encode_runtime_event(event))
    assert restored == event


def test_node_agent_stream_forwards_worker_events_and_marks_disconnect_stale() -> None:
    async def scenario() -> None:
        recorder = InMemoryRecorder()
        identity = _identity()
        recorder.open_run(
            RunRecordingContext(
                schema_version=1,
                experiment_id="run_1",
                run_id="run_1",
                workflow_fingerprint="w" * 64,
                config_fingerprint="c" * 64,
                environment_fingerprint=ENVIRONMENT,
                build_revision="test",
                started_wall_time_ms=1,
                initial_expected_producer_ids=(identity.producer_id,),
            )
        )
        registry = RayNodeRegistry()
        events: list[RuntimeEvent] = []
        event_received = asyncio.Event()

        def sink(event: RuntimeEvent) -> None:
            events.append(event)
            event_received.set()

        controller = NodeControlServer(
            cluster_id=identity.cluster_id,
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=recorder,
            event_sink=sink,
        )
        controller_endpoint = await controller.start()
        agent = NodeAgent(
            identity=identity,
            authorization_token=b"test-token",
            heartbeat_interval_ms=20,
        )
        agent_endpoint = await agent.start(controller_endpoint=controller_endpoint)
        assert agent.runtime_generation == 1
        assert registry.binding("node_a").agent_endpoint == agent_endpoint

        event = _event()
        await asyncio.to_thread(
            report_worker_event,
            endpoint=agent_endpoint,
            identity=identity,
            event=event,
            timeout_seconds=2,
        )
        await asyncio.wait_for(event_received.wait(), timeout=2)
        assert events == [event]
        node_events = recorder.events("run_1")
        assert len(node_events) == 1
        assert node_events[0].producer_id == identity.producer_id
        assert node_events[0].producer_sequence == 1

        await asyncio.to_thread(
            report_worker_event,
            endpoint=agent_endpoint,
            identity=identity,
            event=event,
            timeout_seconds=2,
        )
        await asyncio.sleep(0.05)
        assert events == [event]
        assert (await recorder.flush_run("run_1", 100)).recording_complete

        await agent.close(grace_seconds=0)
        for _ in range(100):
            if registry.status("node_a") is RuntimeNodeStatus.STALE:
                break
            await asyncio.sleep(0.01)
        assert registry.status("node_a") is RuntimeNodeStatus.STALE
        await controller.close(grace_seconds=0)

    asyncio.run(scenario())


def test_rejected_node_registration_does_not_pollute_runtime_registry() -> None:
    async def scenario() -> None:
        recorder = InMemoryRecorder()
        registry = RayNodeRegistry()

        def validate_node(node_id: str) -> None:
            raise ValueError(f"unknown configured node: {node_id}")

        controller = NodeControlServer(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=recorder,
            event_sink=lambda event: None,
            registration_validator=validate_node,
        )
        endpoint = await controller.start()
        agent = NodeAgent(
            identity=_identity(),
            authorization_token=b"test-token",
        )
        try:
            with pytest.raises(Exception, match="unknown configured node"):
                await agent.start(controller_endpoint=endpoint)
            assert registry.active_bindings() == ()
        finally:
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())
