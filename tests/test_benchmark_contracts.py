from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from ascend_maze.benchmark.canonical import (
    canonical_json_bytes,
    canonical_json_digest,
    derive_seed,
)
from ascend_maze.benchmark.contracts import (
    ExternalAdapterSpec,
    TrialManifest,
    measurement_id,
)
from ascend_maze.benchmark.loader import load_study_plan
from ascend_maze.core.errors import ExperimentValidationError
from benchmark_fixtures import write_experiment_spec


def test_canonical_json_and_seed_derivation_are_order_independent() -> None:
    left = {"nested": {"b": 2, "a": 1}, "name": "e\u0301"}
    right = {"name": "\u00e9", "nested": {"a": 1, "b": 2}}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_digest(left) == canonical_json_digest(right)
    assert derive_seed(7, "arrival") == derive_seed(7, "arrival")
    assert derive_seed(7, "arrival") != derive_seed(7, "inputs")
    assert 0 <= derive_seed(7, "arrival") < 2**63


def test_trial_manifest_and_measurement_identity_are_immutable(tmp_path: Path) -> None:
    plan = load_study_plan(write_experiment_spec(tmp_path))
    trial = plan.trials[0]
    manifest = TrialManifest.planned(trial)
    assert manifest.trial_attempt_id.startswith("trial_attempt_")
    assert measurement_id(manifest.trial_attempt_id, "dct_ms", 1) == measurement_id(
        manifest.trial_attempt_id, "dct_ms", 1
    )
    assert measurement_id(manifest.trial_attempt_id, "dct_ms", 1) != measurement_id(
        manifest.trial_attempt_id, "queue_ms", 1
    )
    with pytest.raises(FrozenInstanceError):
        manifest.state = "valid"  # type: ignore[misc]


def test_direct_contract_construction_cannot_reintroduce_mutability(
    tmp_path: Path,
) -> None:
    plan = load_study_plan(write_experiment_spec(tmp_path))
    argv = ["baseline", "describe"]
    adapter = ExternalAdapterSpec("ray-native", argv, "e" * 64)  # type: ignore[arg-type]
    argv.append("mutated")
    assert adapter.argv == ("baseline", "describe")

    runs = ["run_a"]
    manifest = TrialManifest(
        schema_version=1,
        trial_attempt_id=TrialManifest.planned(plan.trials[0]).trial_attempt_id,
        trial_id=plan.trials[0].trial_id,
        attempt_index=0,
        state="planned",
        run_ids=runs,  # type: ignore[arg-type]
    )
    runs.append("run_b")
    assert manifest.run_ids == ("run_a",)
    with pytest.raises(ExperimentValidationError, match="schema_version"):
        replace(plan.spec, schema_version=True)
    with pytest.raises(ExperimentValidationError, match="one Trial per Cell"):
        replace(plan, trials=plan.trials[:-1])
