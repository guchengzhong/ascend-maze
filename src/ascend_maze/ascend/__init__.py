"""Ascend platform adapters kept outside the backend-neutral core."""

from ascend_maze.ascend.contracts import (
    AscendCorrectnessConfig,
    AscendDeviceSnapshot,
    AscendEnvironmentSnapshot,
    AscendProcessSnapshot,
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
    "DcmiDeviceAdapter",
    "DcmiError",
    "build_ascend_node_capacity",
    "build_ascend_node_observation",
    "discover_ascend_environment",
]
