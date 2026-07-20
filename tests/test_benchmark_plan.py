from __future__ import annotations

import json
from pathlib import Path

import pytest

from ascend_maze.benchmark.loader import load_experiment_spec, load_study_plan
from ascend_maze.config import load_config
from ascend_maze.core.errors import ContractValidationError, ExperimentValidationError
from benchmark_fixtures import (
    sha256,
    write_experiment_spec,
    write_performance_config,
)


def test_internal_ablation_plan_expands_paired_blocks(tmp_path: Path) -> None:
    plan = load_study_plan(write_experiment_spec(tmp_path))
    assert plan.spec.study_id.startswith("study_")
    assert len(plan.cells) == 5
    assert len(plan.trials) == 50
    assert [cell.name for cell in plan.cells] == [
        "fcfs",
        "maze_full",
        "no_heterogeneous_queue",
        "no_resource_anchor",
        "no_standby",
    ]
    baseline = next(cell for cell in plan.cells if cell.name == "maze_full")
    assert baseline.differences == ()
    assert {
        difference.path for cell in plan.cells for difference in cell.differences
    } == {
        "scheduler.policy",
        "scheduler.partitioner",
        "placement.anchor_strategy",
        "worker.standby_min_idle",
        "worker.standby_max_idle",
    }
    assert len({cell.config_snapshot.config_fingerprint for cell in plan.cells}) == 5
    for block_index in range(10):
        trials = [trial for trial in plan.trials if trial.block_index == block_index]
        assert len(trials) == 5
        assert len({trial.cell_id for trial in trials}) == 5
        assert len({trial.pairing_seed for trial in trials}) == 1
        assert {trial.position_in_block for trial in trials} == set(range(5))
        assert len({trial.trial_seed for trial in trials}) == 5
    decoded = json.loads(plan.canonical_bytes)
    assert decoded["study_id"] == plan.spec.study_id
    assert len(decoded["schema_digests"]) == 10
    assert str(tmp_path) not in plan.spec.canonical_bytes.decode("utf-8")


def test_logical_spec_order_does_not_change_identity_or_plan(tmp_path: Path) -> None:
    config = write_performance_config(tmp_path)
    first = load_study_plan(
        write_experiment_spec(tmp_path, base_config=config, reverse_matrix=False)
    )
    second = load_study_plan(
        write_experiment_spec(tmp_path, base_config=config, reverse_matrix=True)
    )
    assert first.spec.canonical_bytes == second.spec.canonical_bytes
    assert first.spec.study_id == second.spec.study_id
    assert first.canonical_bytes == second.canonical_bytes


def test_base_config_comments_do_not_change_logical_study_identity(
    tmp_path: Path,
) -> None:
    first_config = write_performance_config(tmp_path)
    first = load_study_plan(write_experiment_spec(tmp_path, base_config=first_config))
    second_config = tmp_path / "equivalent-performance.toml"
    second_config.write_text(
        "# formatting is not logical configuration\n"
        + first_config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    second = load_study_plan(
        write_experiment_spec(tmp_path, base_config=second_config, reverse_matrix=True)
    )
    assert first.spec.base_config_source_digest != second.spec.base_config_source_digest
    assert (
        first.spec.base_config_snapshot.config_fingerprint
        == second.spec.base_config_snapshot.config_fingerprint
    )
    assert first.spec.study_id == second.spec.study_id
    assert first.canonical_bytes == second.canonical_bytes


def test_unknown_field_and_correctness_profile_are_rejected(tmp_path: Path) -> None:
    spec_path = write_experiment_spec(tmp_path)
    spec_path.write_text(
        "unknown_field = true\n" + spec_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="unknown ExperimentSpec field"):
        load_experiment_spec(spec_path)

    correctness_root = tmp_path / "correctness"
    correctness_root.mkdir()
    config = write_performance_config(correctness_root, profile="correctness")
    spec_path = write_experiment_spec(correctness_root, base_config=config)
    with pytest.raises(ExperimentValidationError, match="base_config"):
        load_experiment_spec(spec_path)


def test_workload_environment_and_catalog_must_match_base_snapshot(
    tmp_path: Path,
) -> None:
    spec_path = write_experiment_spec(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            'model_catalog_revision = "no-model-catalog"',
            'model_catalog_revision = "another-catalog"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="model_catalog_revision"):
        load_experiment_spec(spec_path)

    spec_path = write_experiment_spec(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            f'required_environment_fingerprint = "{"d" * 64}"',
            f'required_environment_fingerprint = "{"e" * 64}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentValidationError, match="required_environment_fingerprint"
    ):
        load_experiment_spec(spec_path)


def test_config_overrides_are_explicit_frozen_snapshots(tmp_path: Path) -> None:
    config = write_performance_config(tmp_path)
    baseline = load_config(config, build_revision="a" * 40, created_at_ms=0)
    changed = load_config(
        config,
        build_revision="a" * 40,
        created_at_ms=0,
        config_overrides=(("scheduler.policy", "fcfs"),),
    )
    assert baseline.config.scheduler.policy == "hacs_no_tp"
    assert changed.config.scheduler.policy == "fcfs"
    assert baseline.source_bytes_digest == changed.source_bytes_digest
    assert baseline.snapshot.config_fingerprint != changed.snapshot.config_fingerprint
    with pytest.raises(ContractValidationError, match="duplicated"):
        load_config(
            config,
            config_overrides=(
                ("scheduler.policy", "fcfs"),
                ("scheduler.policy", "hacs_no_tp"),
            ),
        )


def test_digests_and_missing_inputs_are_rejected(tmp_path: Path) -> None:
    spec_path = write_experiment_spec(tmp_path)
    text = spec_path.read_text(encoding="utf-8")
    spec_path.write_text(
        text.replace(
            f'base_config_sha256 = "{sha256(tmp_path / "performance.toml")}"',
            f'base_config_sha256 = "{"0" * 64}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="base_config_sha256"):
        load_experiment_spec(spec_path)

    spec_path = write_experiment_spec(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            'path = "dataset.json"', 'path = "missing.json"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="file does not exist"):
        load_experiment_spec(spec_path)

    spec_path = write_experiment_spec(tmp_path)
    dataset_digest = sha256(tmp_path / "dataset.json")
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            f'sha256 = "{dataset_digest}"', f'sha256 = "{"f" * 64}"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="digest mismatch"):
        load_experiment_spec(spec_path)


def test_duplicate_override_unknown_factor_and_noop_are_rejected(
    tmp_path: Path,
) -> None:
    spec_path = write_experiment_spec(tmp_path)
    text = spec_path.read_text(encoding="utf-8")
    needle = 'path = "scheduler.policy"\nvalue = "fcfs"'
    duplicate = needle + "\n\n[[matrix.cells.overrides]]\n" + needle
    spec_path.write_text(text.replace(needle, duplicate), encoding="utf-8")
    with pytest.raises(ExperimentValidationError, match="duplicate override"):
        load_experiment_spec(spec_path)

    spec_path = write_experiment_spec(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            'factors = ["ordering"]', 'factors = ["unknown_factor"]', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="unknown factor"):
        load_experiment_spec(spec_path)

    spec_path = write_experiment_spec(tmp_path)
    needle = 'path = "scheduler.policy"\nvalue = "fcfs"'
    hidden = (
        needle
        + '\n\n[[matrix.cells.overrides]]\npath = "recording.batch_size"\nvalue = 64'
    )
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(needle, hidden),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="outside factor boundary"):
        load_experiment_spec(spec_path)

    spec_path = write_experiment_spec(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8")
        .replace('kind = "internal_ablation_v1"', 'kind = "custom_v1"')
        .replace('value = "fcfs"', 'value = "hacs_no_tp"'),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="ineffective overrides"):
        load_study_plan(spec_path)


def test_source_mutation_after_loading_is_rejected(tmp_path: Path) -> None:
    spec_path = write_experiment_spec(tmp_path)
    spec = load_experiment_spec(spec_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(dataset.read_bytes() + b"changed")
    from ascend_maze.benchmark.planning import build_study_plan

    with pytest.raises(ExperimentValidationError, match="workload input changed"):
        build_study_plan(spec)


def test_formal_block_minimum_and_frozen_internal_matrix_are_enforced(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExperimentValidationError, match="formal requires at least 10"):
        load_experiment_spec(write_experiment_spec(tmp_path, block_count=9))

    spec_path = write_experiment_spec(tmp_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            'name = "no_standby"', 'name = "standby_disabled"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExperimentValidationError, match="five Cell definitions"):
        load_study_plan(spec_path)
