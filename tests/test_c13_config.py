from __future__ import annotations

import json
from pathlib import Path

import pytest

from ascend_maze.cli.main import build_parser, main
from ascend_maze.config import load_config, load_model_catalog
from ascend_maze.core.errors import ContractValidationError


def _write_config(root: Path, extra: str = "") -> Path:
    runtime = root / "runtime"
    runtime.mkdir()
    token = runtime / "cluster.token"
    token.write_text("test-only-token", encoding="utf-8")
    token.chmod(0o600)
    config = root / "ascend-maze.toml"
    config.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'profile = "correctness"',
                "",
                "[control]",
                f'runtime_directory = "{runtime}"',
                f'cluster_token_file = "{token}"',
                "",
                "[runtime.ray]",
                'namespace = "test-maze"',
                "",
                "[recording]",
                'backend = "parquet"',
                'root_directory = "records"',
                "",
                extra,
            )
        ),
        encoding="utf-8",
    )
    return config


def test_strict_config_is_frozen_normalized_and_deterministic(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "[data]\nshared_filesystem_roots = [\"shared\"]",
    )
    first = load_config(path, build_revision="revision", created_at_ms=1)
    second = load_config(path, build_revision="revision", created_at_ms=2)

    assert first.snapshot.config_fingerprint == second.snapshot.config_fingerprint
    assert first.config.control.socket_path == str(
        (tmp_path / "runtime" / "control.sock").resolve()
    )
    assert first.config.recording.root_directory == str(
        (tmp_path / "records").resolve()
    )
    assert first.config.data.shared_filesystem_roots == (
        str((tmp_path / "shared").resolve()),
    )
    assert first.snapshot.resolved["control"]["cluster_token"] == "<redacted>"
    assert "test-only-token" not in repr(first.snapshot.resolved)

    frozen_fingerprint = first.snapshot.config_fingerprint
    path.write_text("schema_version = 1\nprofile = \"performance\"\n", encoding="utf-8")
    assert first.snapshot.config_fingerprint == frozen_fingerprint
    assert first.config.profile == "correctness"


def test_config_rejects_unknown_fields_and_cross_component_conflicts(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path, "[scheduler]\npolciy = \"fcfs\"")
    with pytest.raises(
        ContractValidationError,
        match=r"scheduler\.polciy: unknown configuration field",
    ):
        load_config(path)

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "[scheduler]\npolciy = \"fcfs\"",
            "[placement]\ntask_slots_total = 2",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="task_slots_total"):
        load_config(path)


def test_config_rejects_bad_units_and_corrupt_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "models.toml"
    catalog.write_text("models = []", encoding="utf-8")
    path = _write_config(
        tmp_path,
        f'[inference]\nmodel_catalog_path = "{catalog}"\n\n'
        '[worker]\nbinding_deadline_ms = "30s"',
    )
    with pytest.raises(ContractValidationError, match="binding_deadline_ms"):
        load_config(path)

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '\n[worker]\nbinding_deadline_ms = "30s"', ""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="catalog_revision"):
        load_config(path)


def test_cli_version_render_and_invalid_config_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_config(tmp_path)
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("Ascend-Maze 0.1.0")

    assert main(["--json", "config", "render", "--config", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["resolved"]["control"]["cluster_token"] == "<redacted>"

    path.write_text("unknown = true", encoding="utf-8")
    assert main(["config", "validate", "--config", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown configuration field" in captured.err


def test_cli_has_no_public_server_url_option() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--server-url", "http://example.invalid"])


def test_complete_model_catalog_is_validated_and_fingerprinted(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    catalog = tmp_path / "models.toml"
    catalog.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'catalog_revision = "catalog_1"',
                "[[models]]",
                'model_id = "fake-model"',
                'artifact_path = "model"',
                f'artifact_revision = "{"a" * 64}"',
                'backend = "fake"',
                'dtype = "float32"',
                "tensor_parallel_size = 1",
                "max_model_len = 2048",
                "instance_cpu_num = 1",
                "instance_host_mem_mb = 1024",
                "weight_hbm_mb = 0",
                "runtime_hbm_mb = 1",
                "kv_cache_hbm_mb = 0",
                "instance_hbm_mb = 1",
                "npu_slots = 1",
                "allow_colocation = false",
                "request_capacity = 1",
                'required_capabilities = ["chat"]',
                "[models.launch_options]",
                'response_prefix = "fake"',
                "[models.warmup_request]",
                'prompt = "hello"',
            )
        ),
        encoding="utf-8",
    )
    document = load_model_catalog(
        catalog,
        environment_fingerprint="e" * 64,
    )
    assert document.catalog_revision == "catalog_1"
    assert document.specs[0].artifact_path == str(artifact.resolve())
    assert len(document.content_digest) == 64

    config_path = _write_config(
        tmp_path,
        f'[inference]\nmodel_catalog_path = "{catalog}"',
    )
    loaded = load_config(config_path, build_revision="test", created_at_ms=0)
    assert loaded.snapshot.model_catalog_revision == "catalog_1"
    first_fingerprint = loaded.snapshot.config_fingerprint

    catalog.write_text(
        catalog.read_text(encoding="utf-8").replace(
            'response_prefix = "fake"', 'response_prefix = "changed"'
        ),
        encoding="utf-8",
    )
    changed = load_config(config_path, build_revision="test", created_at_ms=0)
    assert changed.snapshot.config_fingerprint != first_fingerprint

    catalog.write_text(
        catalog.read_text(encoding="utf-8").replace(
            'response_prefix = "changed"', 'unsupported_option = true'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="unsupported_option"):
        load_config(config_path)

    catalog.write_text(
        catalog.read_text(encoding="utf-8")
        .replace('unsupported_option = true', 'response_prefix = "changed"')
        .replace('artifact_path = "model"', 'artifact_path = "missing-model"'),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="directory does not exist"):
        load_model_catalog(catalog, environment_fingerprint="e" * 64)


def test_model_catalog_accepts_transformers_local_backend(tmp_path: Path) -> None:
    artifact = tmp_path / "qwen"
    artifact.mkdir()
    catalog = tmp_path / "transformers-models.toml"
    catalog.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'catalog_revision = "transformers_catalog_1"',
                "[[models]]",
                'model_id = "qwen-local"',
                'artifact_path = "qwen"',
                'tokenizer_path = "qwen"',
                'artifact_revision = "artifact_1"',
                'backend = "transformers_local"',
                'dtype = "bfloat16"',
                "tensor_parallel_size = 1",
                "max_model_len = 10240",
                "instance_cpu_num = 4",
                "instance_host_mem_mb = 16384",
                "weight_hbm_mb = 7500",
                "runtime_hbm_mb = 4000",
                "kv_cache_hbm_mb = 22000",
                "instance_hbm_mb = 36000",
                "npu_slots = 1",
                "allow_colocation = false",
                "request_capacity = 1",
                'required_capabilities = ["transformers_local"]',
                "[models.launch_options]",
                'generation_method = "manual_greedy"',
                'model_kind = "text"',
                "request_timeout_ms = 600000",
                "trust_remote_code = true",
                "[models.warmup_request]",
                'prompt = "ready"',
            )
        ),
        encoding="utf-8",
    )

    document = load_model_catalog(
        catalog,
        environment_fingerprint="e" * 64,
    )

    assert document.specs[0].backend == "transformers_local"
    assert document.specs[0].launch_options["generation_method"] == "manual_greedy"
