"""Build homogeneous environment fingerprints and C6 node capacity."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import platform
import re
import sys

from ascend_maze.ascend.contracts import (
    AscendCorrectnessConfig,
    AscendDeviceSnapshot,
    AscendEnvironmentSnapshot,
)
from ascend_maze.ascend.dcmi import DcmiDeviceAdapter
from ascend_maze.core.canonical import FrozenMap
from ascend_maze.placement import (
    NodeCapacity,
    NodeObservation,
    NpuCapacity,
    NpuObservation,
)


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def _version_file(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "absent"
    values: list[str] = []
    for line in content.splitlines():
        match = re.match(r"(?:Version|version|package_version)\s*=\s*[\"']?([^\"']+)", line)
        if match:
            values.append(match.group(1).strip())
    return values[0] if values else "unknown"


def _cann_version() -> str:
    candidates: list[Path] = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidates.append(Path(ascend_home) / "version.info")
        candidates.append(Path(ascend_home) / "opp" / "version.info")
    candidates.extend(
        sorted(Path("/usr/local/Ascend").glob("cann-*/opp/version.info"), reverse=True)
    )
    for candidate in candidates:
        version = _version_file(candidate)
        if version != "absent":
            return version
    return "absent"


def discover_ascend_environment(
    adapter: DcmiDeviceAdapter,
    devices: tuple[AscendDeviceSnapshot, ...] | None = None,
) -> AscendEnvironmentSnapshot:
    inventory = adapter.devices() if devices is None else devices
    versions = {
        "python": platform.python_version(),
        "torch": _distribution_version("torch"),
        "torch_npu": _distribution_version("torch-npu"),
        "ray": _distribution_version("ray"),
        "cloudpickle": _distribution_version("cloudpickle"),
        "driver": _version_file(Path("/usr/local/Ascend/driver/version.info")),
        "firmware": _version_file(Path("/usr/local/Ascend/firmware/version.info")),
        "cann": _cann_version(),
        "executable_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    return AscendEnvironmentSnapshot.create(
        machine=platform.machine(),
        chip_types=tuple(item.chip_type for item in inventory),
        versions=versions,
    )


def _host_memory_mb() -> int:
    return int(
        os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    )


def _host_available_memory_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    raise RuntimeError("cannot read host available memory")


def build_ascend_node_observation(
    *,
    node_id: str,
    boot_id: str,
    sequence: int,
    received_at_ms: int,
    adapter: DcmiDeviceAdapter,
) -> NodeObservation:
    devices = adapter.devices()
    return NodeObservation(
        node_id=node_id,
        boot_id=boot_id,
        sequence=sequence,
        received_at_ms=received_at_ms,
        observed_free_mem_mb=_host_available_memory_mb(),
        npus=tuple(
            NpuObservation(
                device_id=item.physical_device_id,
                health=item.health,
                observed_free_hbm_mb=item.free_hbm_mb,
                utilization=item.utilization,
            )
            for item in devices
        ),
    )


def build_ascend_node_capacity(
    *,
    node_id: str,
    boot_id: str,
    node_ip: str,
    adapter: DcmiDeviceAdapter,
    environment: AscendEnvironmentSnapshot,
    config: AscendCorrectnessConfig,
    cpu_system_reserved: int = 1,
    mem_system_reserved_mb: int = 2_048,
) -> NodeCapacity:
    devices = adapter.devices()
    if tuple(sorted(set(item.chip_type for item in devices))) != environment.chip_types:
        raise ValueError("Ascend inventory changed after environment fingerprinting")
    npus = tuple(
        NpuCapacity(
            device_id=item.physical_device_id,
            chip_type=item.chip_type,
            total_hbm_mb=item.total_hbm_mb,
            system_reserved_hbm_mb=config.npu_system_reserved_hbm_mb,
            task_slots_total=config.task_slots_total,
            observed_free_hbm_mb=item.free_hbm_mb,
            healthy=item.health == "healthy",
        )
        for item in devices
    )
    return NodeCapacity(
        node_id=node_id,
        boot_id=boot_id,
        node_ip=node_ip,
        cpu_total=os.cpu_count() or 1,
        mem_total_mb=_host_memory_mb(),
        cpu_system_reserved=cpu_system_reserved,
        mem_system_reserved_mb=mem_system_reserved_mb,
        io_slots_total=config.io_slots_total,
        npus=npus,
        observed_free_mem_mb=None,
        capabilities=FrozenMap(
            (
                ("platform", "ascend"),
                ("chip_family", ",".join(environment.chip_types)),
                (
                    "environment_fingerprint",
                    environment.environment_fingerprint,
                ),
                ("driver_version", environment.versions["driver"]),
                ("cann_version", environment.versions["cann"]),
                ("torch_npu_version", environment.versions["torch_npu"]),
            )
        ),
    )
