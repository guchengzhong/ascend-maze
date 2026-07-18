from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from ascend_maze.ascend import AscendEnvironmentSnapshot
from ascend_maze.contracts.recording import ParquetRecorderConfig
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.experiments import Stage5DConfig, build_stage5d_components
from ascend_maze.recording import NoopRecorder, ParquetRecorder
from ascend_maze.resources import (
    DeclaredOnlyAnchorProvider,
    StaticAnchorProvider,
)
from ascend_maze.scheduler import (
    FcfsPolicy,
    HacsNoTpStaticPolicy,
    HeterogeneousPartitioner,
    UnifiedPartitioner,
)


def _environment() -> AscendEnvironmentSnapshot:
    return AscendEnvironmentSnapshot.create(
        machine="aarch64",
        chip_types=("910B3",),
        versions={
            "cann": "9.0",
            "cloudpickle": "3.1.1",
            "ray": "2.49.2",
            "torch_npu": "2.7.1",
        },
    )


def _config(root: Path) -> Stage5DConfig:
    return Stage5DConfig(
        recorder=ParquetRecorderConfig(
            root_directory=str(root),
            control_queue_capacity=128,
            telemetry_queue_capacity=64,
            batch_size=16,
            flush_interval_ms=25,
            compression="zstd",
            max_page_size=200,
        )
    )


def test_stage5d_mainline_builds_versioned_shared_path_components(
    tmp_path: Path,
) -> None:
    components = build_stage5d_components(
        _config(tmp_path / "records"),
        _environment(),
        source_path="/etc/ascend-maze/stage5d.toml",
        build_revision="stage5d-test",
        cursor_signing_key=b"stage5d-controller-cursor-key",
        created_at_ms=1,
    )
    try:
        assert isinstance(components.policy, HacsNoTpStaticPolicy)
        assert isinstance(components.anchors, StaticAnchorProvider)
        assert isinstance(components.partitioner, HeterogeneousPartitioner)
        assert isinstance(components.recorder, ParquetRecorder)
        assert components.worker_pool.mode == "zero_hbm_standby"
        assert all(profile.min_idle == 2 for profile in components.worker_pool.profiles)

        resolved = components.snapshot.resolved
        assert resolved["profile"] == "stage5d"
        assert resolved["scheduler"]["policy"] == "hacs_no_tp"
        assert resolved["anchor"]["strategy"] == "static"
        assert resolved["queue"]["partitioner"] == "heterogeneous"
        assert resolved["worker_pool"]["standby_enabled"] is True
        assert resolved["placement"]["task_slots_total"] == 2
        assert resolved["placement"]["allow_colocation"] is True
        assert resolved["placement"]["npu_hbm_headroom_mb"] == 1_024
        assert resolved["cleanup"]["hbm_recovery_deadline_ms"] == 30_000
        assert resolved["recording"]["backend"] == "parquet"
        assert resolved["recording"]["control_queue_capacity"] == 128
        assert resolved["recording"]["telemetry_queue_capacity"] == 64
        assert resolved["recording"]["batch_size"] == 16
        assert resolved["recording"]["flush_interval_ms"] == 25
        assert resolved["recording"]["compression"] == "zstd"
        assert resolved["recording"]["max_page_size"] == 200
    finally:
        asyncio.run(components.recorder.close(1_000))


def test_stage5d_ablations_change_one_switch_and_keep_component_boundaries(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path / "records")
    variants = (
        replace(baseline, policy="fcfs"),
        replace(baseline, anchor="declared_only"),
        replace(baseline, partitioner="unified"),
        replace(baseline, standby_enabled=False),
        replace(baseline, recording_backend="noop"),
    )
    built = [
        build_stage5d_components(
            config,
            _environment(),
            source_path="/etc/ascend-maze/stage5d.toml",
            build_revision="stage5d-test",
            cursor_signing_key=b"stage5d-ablation-cursor-key",
            created_at_ms=1,
        )
        for config in (baseline, *variants)
    ]
    try:
        baseline_components, policy, anchor, queue, standby, recording = built
        assert len({item.snapshot.config_fingerprint for item in built}) == len(built)
        assert isinstance(policy.policy, FcfsPolicy)
        assert isinstance(anchor.anchors, DeclaredOnlyAnchorProvider)
        assert isinstance(queue.partitioner, UnifiedPartitioner)
        assert standby.worker_pool.mode == "cold_start"
        assert all(profile.min_idle == 0 for profile in standby.worker_pool.profiles)
        assert isinstance(recording.recorder, NoopRecorder)
        assert all(
            type(item.worker_pool) is type(baseline_components.worker_pool)
            for item in built
        )
    finally:
        for item in built:
            asyncio.run(item.recorder.close(1_000))


def test_stage5d_parquet_requires_complete_recorder_configuration() -> None:
    with pytest.raises(ContractValidationError, match="requires recorder config"):
        Stage5DConfig()
