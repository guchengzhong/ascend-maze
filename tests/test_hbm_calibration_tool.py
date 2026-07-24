from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "hbm_calibration.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("hbm_calibration_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_theoretical_kv_cache_matches_qwen_architectures() -> None:
    tool = _load_tool()
    args = tool.parse_args([])
    specs = tool._family_specs(args)  # noqa: SLF001

    assert tool.theoretical_kv_cache_mb(specs["text"]) == 1440
    assert tool.theoretical_kv_cache_mb(specs["vision"]) == 432


def test_recommendation_adds_measured_safety_and_checks_two_instances() -> None:
    tool = _load_tool()

    assert tool.recommended_instance_hbm_mb(16_518) == 19_456
    assert tool.two_instances_fit(19_456, total_hbm_mb=65_536)
    assert not tool.two_instances_fit(31_000, total_hbm_mb=65_536)


def test_default_scenarios_cover_single_double_and_mixed() -> None:
    tool = _load_tool()

    scenarios = tool._scenarios(("text", "vision"))  # noqa: SLF001
    assert [(item.scenario_id, item.families) for item in scenarios] == [
        ("text-single", ("text",)),
        ("text-double", ("text", "text")),
        ("vision-single", ("vision",)),
        ("vision-double", ("vision", "vision")),
        ("text-vision-double", ("text", "vision")),
    ]


def test_scenarios_can_select_only_mixed_calibration() -> None:
    tool = _load_tool()

    scenarios = tool._scenarios(  # noqa: SLF001
        ("text", "vision"),
        ("text-vision-double",),
    )

    assert [(item.scenario_id, item.families) for item in scenarios] == [
        ("text-vision-double", ("text", "vision"))
    ]
