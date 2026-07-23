from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from uuid import uuid4

import pytest
import ray

from ascend_maze import Workflow, task
from ascend_maze.control.client import InMemoryRuntimeClient
from ascend_maze.contracts.data import (
    DataHandle,
    DataOwner,
    SharedFileRef,
    shared_file_from_handle,
)
from ascend_maze.core.errors import DataHandleInvalidError, DataOwnershipError
import ascend_maze.data.ray_store as ray_store_module
from ascend_maze.data.ray_store import RayDataStore


@task
def _echo(value: object) -> dict[str, object]:
    return {"result": value}


@pytest.fixture
def ray_store(ray_namespace: str) -> Iterator[RayDataStore]:
    store = RayDataStore.start(
        owner_generation=f"controller-{uuid4().hex}",
        namespace=ray_namespace,
    )
    try:
        yield store
    finally:
        store.close(kill_owner=True)


def test_ray_data_store_staged_adopt_release_lifecycle(ray_store: RayDataStore) -> None:
    generation = ray_store.descriptor.owner_generation
    handle = ray_store.put_staged({"payload": "value"}, generation)
    assert ray_store.state_of(handle) == "staged"
    assert ray_store.get(handle) == {"payload": "value"}
    owner = DataOwner("run_index", "run_1:1", generation)
    ray_store.adopt((handle,), owner)
    ray_store.adopt((handle,), owner)
    assert ray_store.state_of(handle) == "adopted"
    assert ray_store.owner_of(handle) == owner

    ray_store.release(handle)
    ray_store.release(handle)
    assert ray_store.state_of(handle) == "released"
    with pytest.raises(DataHandleInvalidError):
        ray_store.get(handle)


def test_ray_data_store_resolves_owner_refs_without_materializing_payload(
    ray_store: RayDataStore,
) -> None:
    generation = ray_store.descriptor.owner_generation
    first = ray_store.put_staged({"payload": "first"}, generation)
    second = ray_store.put_staged({"payload": "second"}, generation)
    before = ray_store.stats()

    refs = ray_store.resolve_refs((first, second))

    assert all(isinstance(ref, ray.ObjectRef) for ref in refs)
    assert refs[0].hex() == first.metadata["ray_object_ref_id"]
    assert refs[1].hex() == second.metadata["ray_object_ref_id"]
    assert ray_store.local_get_count == 0
    assert ray.get(list(refs)) == [
        {"payload": "first"},
        {"payload": "second"},
    ]
    after = ray_store.stats()
    assert after["resolve_batch_count"] == before["resolve_batch_count"] + 1
    assert after["resolve_count"] == before["resolve_count"] + 2

    ray_store.release(first)
    with pytest.raises(DataHandleInvalidError):
        ray_store.resolve_ref(first)
    ray_store.release(second)


def test_ray_data_store_preserves_explicit_shared_file_identity(
    ray_store: RayDataStore,
    tmp_path,
) -> None:
    path = (tmp_path / "shared.txt").resolve()
    path.write_text("payload", encoding="utf-8")
    file_ref = SharedFileRef(str(path), "2" * 64, 7)
    handle = ray_store.put_staged(
        file_ref,
        ray_store.descriptor.owner_generation,
    )
    assert handle.stable_digest is None
    assert shared_file_from_handle(handle) == file_ref
    assert ray_store.get(handle) == file_ref


def test_submission_input_skips_canonicalization_and_uses_handle_identity(
    ray_store: RayDataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_canonical_bytes(value: object) -> bytes:
        del value
        raise AssertionError("submission input must not be canonicalized")

    monkeypatch.setattr(ray_store_module, "canonical_bytes", unexpected_canonical_bytes)
    controller = SimpleNamespace(
        data_store=ray_store,
        data_owner_generation=ray_store.descriptor.owner_generation,
        config_fingerprint="c" * 64,
        environment_fingerprint="e" * 64,
    )
    client = InMemoryRuntimeClient(controller)  # type: ignore[arg-type]
    workflow = Workflow("single-canonical-submission")
    value_input = workflow.input("value")
    workflow.add_task(_echo, inputs={"value": value_input})
    value = {"payload": ["one", "two", "three"]}
    before = ray_store.stats()

    prepared = client.prepare_submission(
        workflow,
        inputs={"value": value},
        submission_id="single-canonical-submission",
    )

    handle = prepared.request.workflow_inputs[0][1]
    assert handle.stable_digest is None
    assert handle.size_bytes is None
    assert prepared.input_signature == (
        (
            "value",
            ("object", "builtins", "dict", str(id(value))),
        ),
    )
    assert prepared.request.contract.input_identities[0].identity == (
        "handle",
        handle.owner_generation,
        handle.staged_handle_id,
    )
    after = ray_store.stats()
    assert after["canonicalize_count"] == before["canonicalize_count"]
    assert after["submission_input_canonicalize_count"] == (
        before["submission_input_canonicalize_count"]
    )
    assert after["submission_input_put_count"] == (
        before["submission_input_put_count"] + 1
    )
    assert after["submission_input_value_size_unknown_count"] == (
        before["submission_input_value_size_unknown_count"] + 1
    )
    assert float(after["ray_put_ms"]) >= float(before["ray_put_ms"])
    assert float(after["owner_stage_ms"]) >= float(before["owner_stage_ms"])
    ray_store.release(handle)


def test_runtime_output_skips_digest_and_preserves_lifecycle(
    ray_store: RayDataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_canonical_bytes(value: object) -> bytes:
        del value
        raise AssertionError("runtime output must not be canonicalized")

    monkeypatch.setattr(ray_store_module, "canonical_bytes", unexpected_canonical_bytes)
    generation = ray_store.descriptor.owner_generation
    before = ray_store.stats()
    value = {"large_runtime_output": ["payload"] * 1024}

    handle = ray_store.put_staged_for_runtime_node(
        value,
        generation,
        node_id="node_a",
        boot_id="boot_a",
        runtime_generation=1,
    )

    assert handle.stable_digest is None
    assert handle.size_bytes is None
    assert ray_store.get(handle) == value
    owner = DataOwner("attempt", "run_1:task_1:1", generation)
    ray_store.adopt((handle,), owner)
    assert ray_store.owner_of(handle) == owner
    after = ray_store.stats()
    assert after["canonicalize_count"] == before["canonicalize_count"]
    assert after["runtime_output_canonicalize_count"] == (
        before["runtime_output_canonicalize_count"]
    )
    assert after["runtime_output_put_count"] == (
        before["runtime_output_put_count"] + 1
    )
    assert after["runtime_output_value_size_unknown_count"] == (
        before["runtime_output_value_size_unknown_count"] + 1
    )
    ray_store.release(handle)
    assert ray_store.state_of(handle) == "released"


def test_ray_data_store_adopt_is_atomic_and_generation_checked(
    ray_store: RayDataStore,
) -> None:
    generation = ray_store.descriptor.owner_generation
    first = ray_store.put_staged("first", generation)
    second = ray_store.put_staged("second", generation)
    owner = DataOwner("run_index", "run_1:1", generation)
    other_owner = DataOwner("run_index", "run_2:1", generation)
    ray_store.adopt((second,), other_owner)
    with pytest.raises(DataOwnershipError):
        ray_store.adopt((first, second), owner)
    assert ray_store.state_of(first) == "staged"
    assert ray_store.owner_of(first) is None
    assert ray_store.owner_of(second) == other_owner

    stale = DataHandle(
        owner_generation="old-controller",
        staged_handle_id=first.staged_handle_id,
        stable_digest=first.stable_digest,
        size_bytes=first.size_bytes,
        metadata=first.metadata,
    )
    with pytest.raises(DataHandleInvalidError):
        ray_store.get(stale)
