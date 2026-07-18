from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest

from ascend_maze.contracts.data import DataHandle, DataOwner
from ascend_maze.core.errors import DataHandleInvalidError, DataOwnershipError
from ascend_maze.data.ray_store import RayDataStore


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
