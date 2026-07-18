from __future__ import annotations

from dataclasses import replace

import pytest

from ascend_maze import Workflow
from ascend_maze.core.errors import ContractValidationError
from ascend_maze.inference import ModelCatalog, ModelSpec
from ascend_maze.inference.adapters.fake import FakeInferenceEngineAdapter
from inference_helpers import make_controller, make_spec
from task_fixtures import service_task


def test_catalog_is_immutable_offline_validated_and_deterministic(tmp_path) -> None:
    adapter = FakeInferenceEngineAdapter()
    first = make_spec(tmp_path / "model_a", model_id="model_a")
    second = make_spec(tmp_path / "model_b", model_id="model_b")
    catalog = ModelCatalog(
        (second, first),
        adapters={"fake": adapter},
        max_single_npu_hbm_mb=8_192,
    )

    assert tuple(spec.model_id for spec in catalog.specs) == ("model_a", "model_b")
    assert catalog.get("model_a") is first
    assert catalog.content_digest == ModelCatalog(
        (first, second),
        adapters={"fake": adapter},
        max_single_npu_hbm_mb=8_192,
    ).content_digest
    assert adapter.launch_count == 0

    with pytest.raises(ContractValidationError, match="not registered"):
        catalog.get("missing")


def test_catalog_rejects_invalid_resources_backend_path_and_revision(tmp_path) -> None:
    valid = make_spec(tmp_path / "valid")
    adapter = FakeInferenceEngineAdapter()

    with pytest.raises(ContractValidationError, match="tensor_parallel_size=1"):
        replace(valid, tensor_parallel_size=2)
    with pytest.raises(ContractValidationError, match="cover weight"):
        replace(valid, instance_hbm_mb=700)
    with pytest.raises(ContractValidationError, match="does not exist"):
        ModelCatalog(
            (replace(valid, artifact_path=str(tmp_path / "missing")),),
            adapters={"fake": adapter},
        )
    with pytest.raises(ContractValidationError, match="unsupported model backend"):
        ModelCatalog((valid,), adapters={})
    with pytest.raises(ContractValidationError, match="catalog revision"):
        ModelCatalog(
            (
                valid,
                replace(
                    make_spec(tmp_path / "other", model_id="other"),
                    catalog_revision="catalog_v2",
                ),
            ),
            adapters={"fake": adapter},
        )
    with pytest.raises(ContractValidationError, match="single-NPU"):
        ModelCatalog(
            (valid,),
            adapters={"fake": adapter},
            max_single_npu_hbm_mb=512,
        )


def test_catalog_validates_workflow_model_ids_before_execution(tmp_path) -> None:
    spec = make_spec(tmp_path / "model_a")
    catalog = ModelCatalog(
        (spec,),
        adapters={"fake": FakeInferenceEngineAdapter()},
    )
    valid = Workflow("known-model")
    valid.add_task(
        service_task,
        inputs={"prompt": "hello"},
        model_anchor={"model": "model_a", "mode": "service"},
    )
    catalog.validate_workflow(valid.compile())

    unknown = Workflow("unknown-model")
    unknown.add_task(
        service_task,
        inputs={"prompt": "hello"},
        model_anchor={"model": "missing", "mode": "service"},
    )
    with pytest.raises(ContractValidationError, match="not registered"):
        catalog.validate_workflow(unknown.compile())


def test_model_spec_rejects_unknown_fake_launch_options(tmp_path) -> None:
    spec: ModelSpec = replace(
        make_spec(tmp_path / "model"),
        launch_options={"arbitrary_cli": "--unsafe"},
    )
    with pytest.raises(ContractValidationError, match="unsupported Fake"):
        ModelCatalog(
            (spec,),
            adapters={"fake": FakeInferenceEngineAdapter()},
        )


def test_controller_rejects_model_catalog_from_another_environment(tmp_path) -> None:
    spec = replace(
        make_spec(tmp_path / "model"),
        environment_fingerprint="different_environment",
    )
    with pytest.raises(ValueError, match="environment fingerprint"):
        make_controller(spec)
