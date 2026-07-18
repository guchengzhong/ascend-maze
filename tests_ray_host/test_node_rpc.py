from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.errors import ErrorInfo
from ascend_maze.contracts.recording import (
    ParquetRecorderConfig,
    RunRecordingContext,
)
from ascend_maze.core.canonical import FrozenMap
from ascend_maze.placement import (
    NodeCapacity,
    NodeObservation,
    NpuCapacity,
    NpuObservation,
    PlacementManager,
)
from ascend_maze.recording import InMemoryRecorder, ParquetRecorder
from ascend_maze.runtime.events import RuntimeEvent, RuntimeEventKind
from ascend_maze.runtime.ray_node_registry import (
    RayNodeRegistry,
    RuntimeNodeStatus,
)

from ascend_maze.control.node_rpc import (
    NodeAgent,
    NodeAgentIdentity,
    NodeControlServer,
    flush_node_recording,
    open_node_recording,
    report_worker_event,
)
from ascend_maze.control.proto_codec import decode_runtime_event, encode_runtime_event


ENVIRONMENT = "e" * 64


def _identity(
    *,
    generation: str = "agent_1",
    environment: str = ENVIRONMENT,
) -> NodeAgentIdentity:
    return NodeAgentIdentity(
        cluster_id="cluster_1",
        node_id="node_a",
        boot_id="boot_1",
        ray_node_id="ray_node_a",
        agent_generation=generation,
        environment_fingerprint=environment,
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


def test_environment_mismatch_is_registered_but_unschedulable() -> None:
    async def scenario() -> None:
        registry = RayNodeRegistry()
        controller = NodeControlServer(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=InMemoryRecorder(),
            event_sink=lambda event: None,
        )
        endpoint = await controller.start()
        agent = NodeAgent(
            identity=_identity(environment="f" * 64),
            authorization_token=b"test-token",
        )
        try:
            await agent.start(controller_endpoint=endpoint)
            assert registry.binding("node_a").agent_generation == "agent_1"
            assert registry.status("node_a") is RuntimeNodeStatus.UNSCHEDULABLE
            assert registry.active_bindings() == ()
        finally:
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())


def test_node_observation_heartbeat_updates_dynamic_capacity_monotonically() -> None:
    async def scenario() -> None:
        manager = PlacementManager(required_environment_fingerprint=ENVIRONMENT)
        manager.register_node(
            NodeCapacity(
                node_id="node_a",
                boot_id="boot_1",
                node_ip="127.0.0.1",
                cpu_total=8,
                mem_total_mb=32_768,
                cpu_system_reserved=1,
                mem_system_reserved_mb=1_024,
                io_slots_total=4,
                npus=(NpuCapacity("7", "910B3", 65_536, 4_096, 1, 60_000),),
                observed_free_mem_mb=30_000,
                capabilities=FrozenMap(
                    (("environment_fingerprint", ENVIRONMENT),)
                ),
            )
        )
        observed_sequences: list[int] = []

        def update(observation: NodeObservation) -> bool:
            observed_sequences.append(observation.sequence)
            return manager.update_observation(observation)

        def observe(sequence: int, now_ms: int) -> NodeObservation:
            return NodeObservation(
                node_id="node_a",
                boot_id="boot_1",
                sequence=sequence,
                received_at_ms=now_ms,
                observed_free_mem_mb=20_000 + sequence,
                npus=(
                    NpuObservation(
                        "7",
                        "healthy",
                        50_000 + sequence,
                        float(sequence),
                    ),
                ),
            )

        registry = RayNodeRegistry()
        controller = NodeControlServer(
            cluster_id="cluster_1",
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=InMemoryRecorder(),
            event_sink=lambda event: None,
            on_node_observation=update,
        )
        endpoint = await controller.start()
        agent = NodeAgent(
            identity=_identity(),
            authorization_token=b"test-token",
            heartbeat_interval_ms=10,
            node_observation_provider=observe,
        )
        try:
            await agent.start(controller_endpoint=endpoint)
            for _ in range(200):
                if len(observed_sequences) >= 2:
                    break
                await asyncio.sleep(0.01)
            assert len(observed_sequences) >= 2
            assert observed_sequences == sorted(set(observed_sequences))
            snapshot = manager.snapshot().nodes[0]
            assert snapshot.observation_sequence == observed_sequences[-1]
            assert snapshot.capacity.observed_free_mem_mb == (
                20_000 + observed_sequences[-1]
            )
            assert snapshot.capacity.npus[0].observed_free_hbm_mb == (
                50_000 + observed_sequences[-1]
            )
        finally:
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())


def test_observation_failure_degrades_without_publishing_zero_metrics() -> None:
    async def scenario() -> None:
        identity = _identity()
        recorder = InMemoryRecorder()
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
        observations: list[NodeObservation] = []
        received = asyncio.Event()

        def fail_observation(sequence: int, now_ms: int) -> NodeObservation:
            del sequence, now_ms
            raise RuntimeError("injected DCMI sampling failure")

        controller = NodeControlServer(
            cluster_id=identity.cluster_id,
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=recorder,
            event_sink=lambda event: received.set(),
            on_node_observation=observations.append,
        )
        endpoint = await controller.start()
        agent = NodeAgent(
            identity=identity,
            authorization_token=b"test-token",
            heartbeat_interval_ms=10,
            node_observation_provider=fail_observation,
        )
        try:
            agent_endpoint = await agent.start(controller_endpoint=endpoint)
            await asyncio.sleep(0.05)
            assert observations == []
            assert registry.status(identity.node_id) is RuntimeNodeStatus.HEALTHY

            await asyncio.to_thread(
                report_worker_event,
                endpoint=agent_endpoint,
                identity=identity,
                event=_event(),
                timeout_seconds=2,
            )
            await asyncio.wait_for(received.wait(), timeout=2)
            assert observations == []
        finally:
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())


def test_node_local_parquet_recorder_owns_worker_events_and_remote_flush(
    tmp_path,
) -> None:
    async def scenario() -> None:
        identity = _identity()
        context = RunRecordingContext(
            schema_version=1,
            experiment_id="run_1",
            run_id="run_1",
            workflow_fingerprint="w" * 64,
            config_fingerprint="c" * 64,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            started_wall_time_ms=1,
            initial_expected_producer_ids=("controller",),
        )
        controller_recorder = InMemoryRecorder()
        controller_recorder.open_run(context)
        registry = RayNodeRegistry()
        forwarded: list[RuntimeEvent] = []
        received = asyncio.Event()

        def sink(event: RuntimeEvent) -> None:
            forwarded.append(event)
            received.set()

        controller = NodeControlServer(
            cluster_id=identity.cluster_id,
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=controller_recorder,
            event_sink=sink,
        )
        node_recorder = ParquetRecorder(
            ParquetRecorderConfig(
                root_directory=str(tmp_path / "node-records"),
                batch_size=2,
                flush_interval_ms=5,
            ),
            cursor_signing_key=b"n" * 32,
        )
        endpoint = await controller.start()
        agent = NodeAgent(
            identity=identity,
            authorization_token=b"test-token",
            recorder=node_recorder,
        )
        try:
            agent_endpoint = await agent.start(controller_endpoint=endpoint)
            binding = registry.binding(identity.node_id)
            assert binding.records_locally
            controller_recorder.expect_producer("run_1", identity.producer_id)
            await asyncio.to_thread(
                open_node_recording,
                binding=binding,
                cluster_id=identity.cluster_id,
                controller_generation="controller_1",
                authorization_token=b"test-token",
                context=context,
                timeout_seconds=2,
            )
            await asyncio.to_thread(
                report_worker_event,
                endpoint=agent_endpoint,
                identity=identity,
                event=_event(),
                timeout_seconds=2,
            )
            await asyncio.wait_for(received.wait(), timeout=2)
            assert forwarded == [_event()]
            assert controller_recorder.events("run_1") == ()

            remote = await asyncio.to_thread(
                flush_node_recording,
                binding=binding,
                cluster_id=identity.cluster_id,
                controller_generation="controller_1",
                authorization_token=b"test-token",
                run_id="run_1",
                timeout_ms=2_000,
            )
            assert remote.recording_complete
            assert remote.committed_files
            assert all(
                identity.node_id in path and "_controller" not in path
                for path in remote.committed_files
                if ".context." not in path
            )
            page = node_recorder.get_run_events("run_1", limit=10)
            assert [event.event_type for event in page.events] == [
                "recorder_producer_joined",
                RuntimeEventKind.TASK_FAILED.value,
            ]
            assert all(event.producer_id == identity.producer_id for event in page.events)
        finally:
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())


def test_node_telemetry_is_device_level_and_tracks_active_leases() -> None:
    async def scenario() -> None:
        identity = _identity()
        local_recorder = InMemoryRecorder()
        registry = RayNodeRegistry()

        def observe(sequence: int, now_ms: int) -> NodeObservation:
            return NodeObservation(
                node_id=identity.node_id,
                boot_id=identity.boot_id,
                sequence=sequence,
                received_at_ms=now_ms,
                observed_free_mem_mb=20_000,
                npus=(NpuObservation("7", "healthy", 50_000, 25.0),),
            )

        controller = NodeControlServer(
            cluster_id=identity.cluster_id,
            authorization_token=b"test-token",
            controller_generation="controller_1",
            environment_fingerprint=ENVIRONMENT,
            registry=registry,
            recorder=InMemoryRecorder(),
            event_sink=lambda event: None,
        )
        endpoint = await controller.start()
        agent = NodeAgent(
            identity=identity,
            authorization_token=b"test-token",
            heartbeat_interval_ms=10,
            recorder=local_recorder,
            worker_device_verifier=lambda pid, device_id: pid == 123 and device_id == "7",
            node_observation_provider=observe,
        )
        context = RunRecordingContext(
            schema_version=1,
            experiment_id="run_telemetry",
            run_id="run_telemetry",
            workflow_fingerprint="w" * 64,
            config_fingerprint="c" * 64,
            environment_fingerprint=ENVIRONMENT,
            build_revision="test",
            started_wall_time_ms=1,
            initial_expected_producer_ids=("controller",),
        )
        try:
            agent_endpoint = await agent.start(controller_endpoint=endpoint)
            binding = registry.binding(identity.node_id)
            await asyncio.to_thread(
                open_node_recording,
                binding=binding,
                cluster_id=identity.cluster_id,
                controller_generation="controller_1",
                authorization_token=b"test-token",
                context=context,
                timeout_seconds=2,
            )
            started = RuntimeEvent.create(
                kind=RuntimeEventKind.WORKER_STARTED,
                dispatch_id="dispatch_telemetry",
                run_id="run_telemetry",
                task_id="task_telemetry",
                attempt=1,
                lease_id="lease_telemetry",
                route_lease_id=None,
                occurred_at_ms=1,
                worker_pid=123,
                device_id="7",
                binding_verified=True,
            )
            await asyncio.to_thread(
                report_worker_event,
                endpoint=agent_endpoint,
                identity=identity,
                event=started,
                timeout_seconds=2,
            )
            for _ in range(200):
                if any(
                    event.event_type == "device_resource_sample"
                    for event in local_recorder.events("run_telemetry")
                ):
                    break
                await asyncio.sleep(0.01)
            telemetry = tuple(
                event
                for event in local_recorder.events("run_telemetry")
                if event.event_type.endswith("resource_sample")
            )
            assert {event.event_type for event in telemetry} == {
                "node_resource_sample",
                "device_resource_sample",
            }
            assert all(event.task_id is None and event.lease_id is None for event in telemetry)
            device = next(
                event
                for event in telemetry
                if event.event_type == "device_resource_sample"
            )
            assert device.device_id == "7"
            assert device.payload["active_lease_count"] == 1

            terminal = replace(
                _event(),
                event_id="terminal_telemetry",
                dispatch_id="dispatch_telemetry",
                run_id="run_telemetry",
                task_id="task_telemetry",
                lease_id="lease_telemetry",
            )
            await asyncio.to_thread(
                report_worker_event,
                endpoint=agent_endpoint,
                identity=identity,
                event=terminal,
                timeout_seconds=2,
            )
            result = await asyncio.to_thread(
                flush_node_recording,
                binding=binding,
                cluster_id=identity.cluster_id,
                controller_generation="controller_1",
                authorization_token=b"test-token",
                run_id="run_telemetry",
                timeout_ms=1_000,
            )
            assert result.recording_complete
        finally:
            await agent.close(grace_seconds=0)
            await controller.close(grace_seconds=0)

    asyncio.run(scenario())
