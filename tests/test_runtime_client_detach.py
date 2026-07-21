from __future__ import annotations

from pathlib import Path

from ascend_maze.control.local_rpc import UdsRuntimeClient
from ascend_maze.data import ray_store
from ascend_maze.data.ray_store import RayDataStore, RayDataStoreDescriptor


def test_client_owned_ray_connection_detaches_once(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    descriptor = RayDataStoreDescriptor("owner", "namespace", "generation")
    initialized = False
    init_calls: list[tuple[str, str]] = []
    shutdown_calls: list[None] = []

    def is_initialized() -> bool:
        return initialized

    def init(*, address: str, namespace: str) -> None:
        nonlocal initialized
        initialized = True
        init_calls.append((address, namespace))

    def shutdown() -> None:
        nonlocal initialized
        initialized = False
        shutdown_calls.append(None)

    monkeypatch.setattr(ray_store.ray, "is_initialized", is_initialized)
    monkeypatch.setattr(ray_store.ray, "init", init)
    monkeypatch.setattr(ray_store.ray, "shutdown", shutdown)
    monkeypatch.setattr(
        RayDataStore,
        "connect",
        classmethod(lambda cls, value: cls(value, object())),
    )

    store = RayDataStore.connect_client(descriptor)
    client = UdsRuntimeClient(
        Path("/tmp/ascend-maze-client-detach.sock"),
        data_store=store,
        data_owner_generation=descriptor.owner_generation,
    )
    client.close()
    client.close()

    assert init_calls == [("auto", "namespace")]
    assert shutdown_calls == [None]
    assert client.data_store is None
    assert client.data_owner_generation is None


def test_preexisting_ray_connection_is_not_detached(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    descriptor = RayDataStoreDescriptor("owner", "namespace", "generation")
    shutdown_calls: list[None] = []

    monkeypatch.setattr(ray_store.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_store.ray, "shutdown", lambda: shutdown_calls.append(None))
    monkeypatch.setattr(
        RayDataStore,
        "connect",
        classmethod(lambda cls, value: cls(value, object())),
    )

    store = RayDataStore.connect_client(descriptor)
    store.close(kill_owner=False)

    assert shutdown_calls == []
