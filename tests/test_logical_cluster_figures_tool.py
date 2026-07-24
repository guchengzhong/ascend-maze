from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "logical_cluster_figures.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


figures = _load(TOOL_PATH, "logical_cluster_figures_test")


def _requests(values: list[int]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "request_count": len(values),
        "succeeded": len(values),
        "failed": 0,
        "p95_e2e_ms": float(max(values)),
        "makespan_ms": max(values),
        "throughput_requests_per_second": len(values) / (max(values) / 1000),
        "e2e_latency_ms": {
            "count": len(values),
            "mean": sum(values) / len(values),
            "p50": ordered[len(ordered) // 2],
            "p95": max(values),
            "max": max(values),
        },
    }


def _result(
    tmp_path: Path,
    executor: str,
    durations: tuple[int, int],
) -> dict[str, object]:
    started = 1_000_000
    records = [
        {
            "request_index": 1,
            "sample_id": "gaia.file.sample-1",
            "dataset": "gaia",
            "workflow": "file",
            "family": "text",
            "status": "succeeded",
            "client_e2e_ms": durations[0],
            "client_e2e_started_at_ms": started,
            "client_e2e_finished_at_ms": started + durations[0],
        },
        {
            "request_index": 2,
            "sample_id": "gaia.vision.sample-2",
            "dataset": "gaia",
            "workflow": "vision",
            "family": "vision",
            "status": "succeeded",
            "client_e2e_ms": durations[1],
            "client_e2e_started_at_ms": started,
            "client_e2e_finished_at_ms": started + durations[1],
        },
    ]
    resource_path = tmp_path / f"{executor}-resources.jsonl"
    samples = []
    for offset, process_count in ((0, 0), (1000, 2), (2000, 0)):
        samples.append(
            {
                "timestamp_ms": started + offset,
                "cluster_cpu_utilization_pct": 10.0 + process_count,
                "cluster_npu_utilization_pct": 20.0 * process_count,
                "cluster_hbm_used_mb": 1000 + 5000 * process_count,
                "npus": [
                    {
                        "physical_device_id": str(device),
                        "processes": [{}] * (process_count if device == 0 else 0),
                    }
                    for device in range(8)
                ],
            }
        )
    resource_path.write_text(
        "".join(json.dumps(item) + "\n" for item in samples),
        encoding="utf-8",
    )
    overall = _requests(list(durations))
    text = _requests([durations[0]])
    vision = _requests([durations[1]])
    return {
        "executor": executor,
        "records": records,
        "resource_samples_path": str(resource_path),
        "resources": {
            "window_started_at_ms": started,
            "window_finished_at_ms": started + 2000,
            "baseline_hbm_mb": 1000,
        },
        "breakdowns": {
            "overall": {"requests": overall, "timings": {}},
            "families": {
                "text": {"requests": text, "timings": {}},
                "vision": {"requests": vision, "timings": {}},
            },
            "workflows": {
                "gaia.file": {"requests": text, "timings": {}},
                "gaia.vision": {"requests": vision, "timings": {}},
            },
        },
    }


def test_writes_comparison_and_resource_svg_figures(tmp_path: Path) -> None:
    summary = {
        "results": [
            _result(tmp_path, "maze", (1000, 4000)),
            _result(tmp_path, "ray", (2000, 8000)),
        ]
    }
    output_dir = tmp_path / "report"
    rendered = figures.write_figures(summary, output_dir)

    assert [item["id"] for item in rendered] == [
        "overall",
        "family",
        "paired_speedup",
        "workflow_latency",
        "e2e_ecdf",
        "request_timeline",
        "resource_timeline",
        "device_concurrency",
    ]
    for item in rendered:
        path = output_dir / item["path"]
        assert path.stat().st_size > 1000
        root = ET.parse(path).getroot()
        assert root.attrib["viewBox"]
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
    manifest = json.loads(
        (output_dir / "figures" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == rendered

    paired_root = ET.parse(output_dir / "figures" / "paired_speedup.svg").getroot()
    assert int(paired_root.attrib["width"]) <= 1_080
    assert int(paired_root.attrib["height"]) <= 700
    paired_text = " ".join(
        node.text or ""
        for node in paired_root.findall("{http://www.w3.org/2000/svg}text")
    )
    assert "Ray faster" in paired_text
    assert "Maze faster" in paired_text
    assert "median 2.00x" in paired_text

    overall_root = ET.parse(output_dir / "figures" / "overall.svg").getroot()
    assert overall_root.attrib["width"] == "1080"
    assert overall_root.attrib["height"] == "430"
    overall_text = " ".join(
        node.text or ""
        for node in overall_root.findall("{http://www.w3.org/2000/svg}text")
    )
    assert "bars normalized per metric" in overall_text
    assert overall_text.count("Maze 50.0% lower") == 2
    assert "Maze 2.00x higher" in overall_text
