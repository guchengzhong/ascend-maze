"""Ray Object Store adapter backed by one long-lived owner actor."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

import ray

from ascend_maze.contracts.data import (
    DataHandle,
    DataOwner,
    SharedFileRef,
    shared_file_metadata,
)
from ascend_maze.core.canonical import FrozenMap, canonical_bytes, canonical_digest
from ascend_maze.core.errors import (
    CanonicalizationError,
    DataHandleInvalidError,
    DataOwnershipError,
    DataStoreWriteError,
)
from ascend_maze.core.identifiers import new_id


@dataclass(frozen=True, slots=True)
class RayDataStoreDescriptor:
    owner_actor_name: str
    owner_namespace: str
    owner_generation: str

    def __post_init__(self) -> None:
        for name in ("owner_actor_name", "owner_namespace", "owner_generation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} is required")


@dataclass(slots=True)
class _OwnerEntry:
    handle: DataHandle
    object_ref: ray.ObjectRef
    state: str
    owner: DataOwner | None


class _DataStoreOwner:
    def __init__(self, owner_generation: str) -> None:
        self.owner_generation = owner_generation
        self._entries: dict[str, _OwnerEntry] = {}
        self._tombstones: set[str] = set()
        self._stage_count = 0
        self._fail_stage_numbers: set[int] = set()

    def stage(
        self,
        handle: DataHandle,
        boxed_ref: list[ray.ObjectRef],
    ) -> None:
        self._stage_count += 1
        if self._stage_count in self._fail_stage_numbers:
            self._fail_stage_numbers.remove(self._stage_count)
            raise RuntimeError(f"injected stage failure at call {self._stage_count}")
        self._validate_handle_generation(handle)
        if len(boxed_ref) != 1 or not isinstance(boxed_ref[0], ray.ObjectRef):
            raise TypeError("stage requires one nested ObjectRef")
        existing = self._entries.get(handle.staged_handle_id)
        if existing is not None:
            if existing.handle != handle or existing.object_ref.hex() != boxed_ref[0].hex():
                raise RuntimeError("staged_handle_id payload conflict")
            return
        self._entries[handle.staged_handle_id] = _OwnerEntry(
            handle=handle,
            object_ref=boxed_ref[0],
            state="staged",
            owner=None,
        )
        self._tombstones.discard(handle.staged_handle_id)

    def resolve(self, handle: DataHandle) -> ray.ObjectRef:
        return self._require_entry(handle).object_ref

    def state_of(self, handle: DataHandle) -> str:
        entry = self._entries.get(handle.staged_handle_id)
        if entry is None:
            if handle.staged_handle_id in self._tombstones:
                return "released"
            raise RuntimeError("data handle is unknown")
        if entry.handle != handle:
            raise RuntimeError("data handle metadata does not match")
        return entry.state

    def owner_of(self, handle: DataHandle) -> DataOwner | None:
        return self._require_entry(handle).owner

    def adopt(self, handles: tuple[DataHandle, ...], owner: DataOwner) -> None:
        if owner.owner_generation != self.owner_generation:
            raise RuntimeError("owner generation does not match DataStoreOwner")
        entries: list[_OwnerEntry] = []
        seen: set[str] = set()
        for handle in handles:
            if handle.staged_handle_id in seen:
                raise RuntimeError("adopt contains a duplicate handle")
            seen.add(handle.staged_handle_id)
            entry = self._require_entry(handle)
            if entry.state == "adopted" and entry.owner != owner:
                raise RuntimeError("handle is already adopted by another owner")
            entries.append(entry)
        for entry in entries:
            entry.state = "adopted"
            entry.owner = owner

    def release(self, handle: DataHandle) -> bool:
        self._validate_handle_generation(handle)
        entry = self._entries.get(handle.staged_handle_id)
        if entry is None:
            return False
        if entry.handle != handle:
            raise RuntimeError("data handle metadata does not match")
        del self._entries[handle.staged_handle_id]
        self._tombstones.add(handle.staged_handle_id)
        del entry.object_ref
        return True

    def release_staged_for_runtime_node(
        self,
        node_id: str,
        boot_id: str,
        runtime_generation: int,
    ) -> int:
        handles = tuple(
            entry.handle
            for entry in self._entries.values()
            if entry.state == "staged"
            and entry.handle.metadata.get("source_node_id") == node_id
            and entry.handle.metadata.get("source_boot_id") == boot_id
            and entry.handle.metadata.get("source_runtime_generation")
            == runtime_generation
        )
        for handle in handles:
            self.release(handle)
        return len(handles)

    def release_staged_for_node(self, node_id: str, boot_id: str) -> int:
        handles = tuple(
            entry.handle
            for entry in self._entries.values()
            if entry.state == "staged"
            and entry.handle.metadata.get("source_node_id") == node_id
            and entry.handle.metadata.get("source_boot_id") == boot_id
        )
        for handle in handles:
            self.release(handle)
        return len(handles)

    def release_owner(self, owner_kind: str, owner_id: str) -> int:
        handles = tuple(
            entry.handle
            for entry in self._entries.values()
            if entry.owner is not None
            and entry.owner.owner_kind == owner_kind
            and entry.owner.owner_id == owner_id
        )
        for handle in handles:
            self.release(handle)
        return len(handles)

    def stats(self) -> dict[str, int | str]:
        return {
            "owner_generation": self.owner_generation,
            "active_count": len(self._entries),
            "staged_count": sum(entry.state == "staged" for entry in self._entries.values()),
            "adopted_count": sum(
                entry.state == "adopted" for entry in self._entries.values()
            ),
            "tombstone_count": len(self._tombstones),
            "stage_count": self._stage_count,
        }

    def fail_on_stage_number(self, stage_number: int) -> None:
        if stage_number < 1:
            raise ValueError("stage_number must be positive")
        self._fail_stage_numbers.add(stage_number)

    def _require_entry(self, handle: DataHandle) -> _OwnerEntry:
        self._validate_handle_generation(handle)
        try:
            entry = self._entries[handle.staged_handle_id]
        except KeyError as exc:
            state = (
                "released"
                if handle.staged_handle_id in self._tombstones
                else "unknown"
            )
            raise RuntimeError(f"data handle is {state}") from exc
        if entry.handle != handle:
            raise RuntimeError("data handle metadata does not match")
        return entry

    def _validate_handle_generation(self, handle: DataHandle) -> None:
        if handle.owner_generation != self.owner_generation:
            raise RuntimeError("data handle owner generation mismatch")


_DATA_STORE_OWNER_ACTOR: Any = ray.remote(
    num_cpus=0,
    max_restarts=0,
    max_task_retries=0,
)(_DataStoreOwner)


class RayDataStore:
    """Synchronous DataStore facade; Scheduler invokes it outside its event loop."""

    def __init__(
        self,
        descriptor: RayDataStoreDescriptor,
        owner_actor: ray.actor.ActorHandle,
    ) -> None:
        self.descriptor = descriptor
        self._owner_actor = owner_actor
        self._local_get_count = 0
        self._local_lock = RLock()

    @classmethod
    def start(
        cls,
        *,
        owner_generation: str,
        namespace: str,
        actor_name: str | None = None,
    ) -> "RayDataStore":
        name = actor_name or new_id("data_store_owner")
        actor = _DATA_STORE_OWNER_ACTOR.options(
            name=name,
            namespace=namespace,
            lifetime="detached",
        ).remote(owner_generation)
        descriptor = RayDataStoreDescriptor(name, namespace, owner_generation)
        ray.get(actor.stats.remote())
        return cls(descriptor, actor)

    @classmethod
    def connect(cls, descriptor: RayDataStoreDescriptor) -> "RayDataStore":
        actor = ray.get_actor(
            descriptor.owner_actor_name,
            namespace=descriptor.owner_namespace,
        )
        stats = ray.get(actor.stats.remote())
        if stats["owner_generation"] != descriptor.owner_generation:
            raise DataHandleInvalidError("DataStoreOwner generation changed")
        return cls(descriptor, actor)

    @classmethod
    def connect_client(cls, descriptor: RayDataStoreDescriptor) -> "RayDataStore":
        """Join the managed Head cluster before resolving its detached owner."""

        if not ray.is_initialized():
            ray.init(address="auto", namespace=descriptor.owner_namespace)
        return cls.connect(descriptor)

    def put_staged(self, value: Any, owner_generation: str) -> DataHandle:
        return self._put_staged(value, owner_generation, FrozenMap((("backend", "ray"),)))

    def put_staged_for_runtime_node(
        self,
        value: Any,
        owner_generation: str,
        *,
        node_id: str,
        boot_id: str,
        runtime_generation: int,
    ) -> DataHandle:
        if not node_id or not boot_id:
            raise DataStoreWriteError("runtime node and boot identity are required")
        if (
            isinstance(runtime_generation, bool)
            or not isinstance(runtime_generation, int)
            or runtime_generation < 1
        ):
            raise DataStoreWriteError("runtime generation must be positive")
        return self._put_staged(
            value,
            owner_generation,
            FrozenMap(
                (
                    ("backend", "ray"),
                    ("source_boot_id", boot_id),
                    ("source_node_id", node_id),
                    ("source_runtime_generation", runtime_generation),
                )
            ),
        )

    def _put_staged(
        self,
        value: Any,
        owner_generation: str,
        metadata: FrozenMap[Any, Any],
    ) -> DataHandle:
        if owner_generation != self.descriptor.owner_generation:
            raise DataStoreWriteError("owner generation does not match DataStoreOwner")
        stable_digest: str | None = None
        size_bytes: int | None = None
        if isinstance(value, SharedFileRef):
            metadata = FrozenMap(
                (*metadata.items_tuple(), *shared_file_metadata(value))
            )
        else:
            try:
                stable_digest = canonical_digest(value)
                size_bytes = len(canonical_bytes(value))
            except CanonicalizationError:
                pass
        handle = DataHandle(
            owner_generation=owner_generation,
            staged_handle_id=new_id("data"),
            stable_digest=stable_digest,
            size_bytes=size_bytes,
            metadata=metadata,
        )
        try:
            object_ref = ray.put(value, _owner=self._owner_actor)
            ray.get(self._owner_actor.stage.remote(handle, [object_ref]))
        except Exception as exc:
            raise DataStoreWriteError(f"Ray put_staged failed: {exc}") from exc
        return handle

    def get(self, handle: DataHandle) -> Any:
        self._validate_generation(handle)
        with self._local_lock:
            self._local_get_count += 1
        try:
            object_ref = ray.get(self._owner_actor.resolve.remote(handle))
            return ray.get(object_ref)
        except Exception as exc:
            raise DataHandleInvalidError(f"Ray data handle cannot be read: {exc}") from exc

    def adopt(self, handles: tuple[DataHandle, ...], owner: DataOwner) -> None:
        if not isinstance(handles, tuple):
            raise DataOwnershipError("adopt handles must be a tuple")
        try:
            ray.get(self._owner_actor.adopt.remote(handles, owner))
        except Exception as exc:
            raise DataOwnershipError(f"Ray adopt failed: {exc}") from exc

    def release(self, handle: DataHandle) -> None:
        self._validate_generation(handle)
        try:
            ray.get(self._owner_actor.release.remote(handle))
        except Exception as exc:
            raise DataHandleInvalidError(f"Ray release failed: {exc}") from exc

    def release_many(self, handles: tuple[DataHandle, ...]) -> None:
        for handle in handles:
            self.release(handle)

    def state_of(self, handle: DataHandle) -> str:
        self._validate_generation(handle)
        try:
            return str(ray.get(self._owner_actor.state_of.remote(handle)))
        except Exception as exc:
            raise DataHandleInvalidError(f"Ray data handle state is unavailable: {exc}") from exc

    def owner_of(self, handle: DataHandle) -> DataOwner | None:
        self._validate_generation(handle)
        try:
            value = ray.get(self._owner_actor.owner_of.remote(handle))
        except Exception as exc:
            raise DataHandleInvalidError(f"Ray data owner is unavailable: {exc}") from exc
        return value if isinstance(value, DataOwner) else None

    def stats(self) -> dict[str, int | str]:
        return dict(ray.get(self._owner_actor.stats.remote()))

    @property
    def active_count(self) -> int:
        return int(self.stats()["active_count"])

    @property
    def staged_count(self) -> int:
        return int(self.stats()["staged_count"])

    @property
    def adopted_count(self) -> int:
        return int(self.stats()["adopted_count"])

    @property
    def local_get_count(self) -> int:
        with self._local_lock:
            return self._local_get_count

    @property
    def put_count(self) -> int:
        return int(self.stats()["stage_count"])

    def fail_on_put_number(self, put_number: int) -> None:
        if (
            isinstance(put_number, bool)
            or not isinstance(put_number, int)
            or put_number < 1
        ):
            raise ValueError("put_number must be a positive integer")
        try:
            ray.get(self._owner_actor.fail_on_stage_number.remote(put_number))
        except Exception as exc:
            raise DataStoreWriteError(f"failed to inject Ray put failure: {exc}") from exc

    def release_staged_for_runtime_node(
        self,
        *,
        node_id: str,
        boot_id: str,
        runtime_generation: int,
    ) -> int:
        try:
            return int(
                ray.get(
                    self._owner_actor.release_staged_for_runtime_node.remote(
                        node_id, boot_id, runtime_generation
                    )
                )
            )
        except Exception as exc:
            raise DataHandleInvalidError(
                f"failed to release stale runtime handles: {exc}"
            ) from exc

    def release_staged_for_node(self, *, node_id: str, boot_id: str) -> int:
        try:
            return int(
                ray.get(
                    self._owner_actor.release_staged_for_node.remote(
                        node_id, boot_id
                    )
                )
            )
        except Exception as exc:
            raise DataHandleInvalidError(
                f"failed to release recovered runtime handles: {exc}"
            ) from exc

    def release_owner(self, *, owner_kind: str, owner_id: str) -> int:
        try:
            return int(
                ray.get(
                    self._owner_actor.release_owner.remote(owner_kind, owner_id)
                )
            )
        except Exception as exc:
            raise DataHandleInvalidError(
                f"failed to release recovered data owner: {exc}"
            ) from exc

    def close(self, *, kill_owner: bool) -> None:
        if kill_owner:
            ray.kill(self._owner_actor, no_restart=True)

    def _validate_generation(self, handle: DataHandle) -> None:
        if handle.owner_generation != self.descriptor.owner_generation:
            raise DataHandleInvalidError("data handle owner generation mismatch")
