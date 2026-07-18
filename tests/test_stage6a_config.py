from __future__ import annotations

from dataclasses import replace

import pytest

from ascend_maze.core.errors import ContractValidationError
from ascend_maze.experiments import Stage6AConfig, create_stage6a_config_snapshot
from ascend_maze.inference import ModelCatalog
from ascend_maze.inference.adapters.fake import FakeInferenceEngineAdapter
from inference_helpers import ENVIRONMENT_FINGERPRINT, make_spec


def _catalog(tmp_path):
    adapter = FakeInferenceEngineAdapter()
    return ModelCatalog(
        (make_spec(tmp_path / "model"),),
        adapters={"fake": adapter},
    )


def test_stage6a_snapshot_covers_complete_catalog_and_control_thresholds(
    tmp_path,
) -> None:
    catalog = _catalog(tmp_path)
    config = Stage6AConfig()
    snapshot = create_stage6a_config_snapshot(
        config,
        catalog,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        source_path=str(tmp_path / "stage6a.toml"),
        build_revision="build_a",
        runtime_versions={"cloudpickle": "3.1.2"},
        created_at_ms=100,
    )

    assert snapshot.model_catalog_revision == catalog.catalog_revision
    assert snapshot.resolved["profile"] == "stage6a-correctness"
    inference = snapshot.resolved["inference"]
    assert inference["catalog_content_digest"] == catalog.content_digest
    assert inference["affinity_ttl_ms"] == config.affinity_ttl_ms

    changed = create_stage6a_config_snapshot(
        replace(config, affinity_capacity=config.affinity_capacity + 1),
        catalog,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        source_path=str(tmp_path / "stage6a.toml"),
        build_revision="build_a",
        runtime_versions={"cloudpickle": "3.1.2"},
        created_at_ms=100,
    )
    assert changed.config_fingerprint != snapshot.config_fingerprint

    changed_model = replace(
        catalog.specs[0],
        request_capacity=catalog.specs[0].request_capacity + 1,
    )
    changed_catalog = ModelCatalog(
        (changed_model,),
        adapters={"fake": FakeInferenceEngineAdapter()},
    )
    changed_catalog_snapshot = create_stage6a_config_snapshot(
        config,
        changed_catalog,
        environment_fingerprint=ENVIRONMENT_FINGERPRINT,
        source_path=str(tmp_path / "stage6a.toml"),
        build_revision="build_a",
        runtime_versions={"cloudpickle": "3.1.2"},
        created_at_ms=100,
    )
    assert changed_catalog_snapshot.config_fingerprint != snapshot.config_fingerprint


def test_stage6a_profile_is_strict() -> None:
    with pytest.raises(ContractValidationError, match="requires FCFS"):
        Stage6AConfig(scheduler_policy="hacs_no_tp")
    with pytest.raises(ContractValidationError, match="disables Standby"):
        Stage6AConfig(standby_enabled=True)
    with pytest.raises(ContractValidationError, match="disables colocation"):
        Stage6AConfig(allow_colocation=True)


def test_stage6a_snapshot_rejects_catalog_environment_mismatch(tmp_path) -> None:
    catalog = _catalog(tmp_path)
    with pytest.raises(ContractValidationError, match="environment mismatch"):
        create_stage6a_config_snapshot(
            Stage6AConfig(),
            catalog,
            environment_fingerprint="wrong_environment",
            source_path=str(tmp_path / "stage6a.toml"),
            build_revision="build_a",
        )
