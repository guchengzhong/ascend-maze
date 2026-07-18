"""Ascend platform adapters kept outside the backend-neutral core."""

from ascend_maze.ascend.contracts import (
    AscendCorrectnessConfig,
    AscendDeviceSnapshot,
    AscendEnvironmentSnapshot,
    AscendProcessSnapshot,
    create_ascend_correctness_config_snapshot,
)
from ascend_maze.ascend.dcmi import DcmiDeviceAdapter, DcmiError
from ascend_maze.ascend.discovery import (
    build_ascend_node_capacity,
    build_ascend_node_observation,
    discover_ascend_environment,
)

__all__ = [
    "AscendCorrectnessConfig",
    "AscendDeviceSnapshot",
    "AscendEnvironmentSnapshot",
    "AscendProcessSnapshot",
    "create_ascend_correctness_config_snapshot",
    "DcmiDeviceAdapter",
    "DcmiError",
    "build_ascend_node_capacity",
    "build_ascend_node_observation",
    "discover_ascend_environment",
]
