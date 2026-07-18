"""Thread-safe in-memory DataStore with explicit logical ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any

from ascend_maze.contracts.data import DataHandle, DataOwner
from ascend_maze.core.canonical import FrozenMap, canonical_bytes, canonical_digest
from ascend_maze.core.errors import (
    CanonicalizationError,
    DataHandleInvalidError,
    DataOwnershipError,
    DataStoreWriteError,
)
from ascend_maze.core.identifiers import new_id


class _EntryState(str, Enum):
    STAGED = "staged"
    ADOPTED = "adopted"


@dataclass(slots=True)
class _Entry:
    handle: DataHandle
    value: Any
    state: _EntryState
    owner: DataOwner | None


class InMemoryDataStore:
    """Model staged/adopt/release semantics without a data proxy service."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._put_count = 0
        self._fail_put_numbers: set[int] = set()
        self._lock = RLock()

    @staticmethod
    def _key(handle: DataHandle) -> tuple[str, str]:
        return (handle.owner_generation, handle.staged_handle_id)

    def fail_on_put_number(self, put_number: int) -> None:
        if isinstance(put_number, bool) or not isinstance(put_number, int) or put_number < 1:
            raise ValueError("put_number must be a positive integer")
        with self._lock:
            self._fail_put_numbers.add(put_number)

    def fail_next_put(self) -> int:
        with self._lock:
            number = self._put_count + 1
            self._fail_put_numbers.add(number)
            return number

    def put_staged(self, value: Any, owner_generation: str) -> DataHandle:
        if not isinstance(owner_generation, str) or not owner_generation:
            raise DataStoreWriteError("owner_generation is required")
        with self._lock:
            self._put_count += 1
            if self._put_count in self._fail_put_numbers:
                self._fail_put_numbers.remove(self._put_count)
                raise DataStoreWriteError(
                    f"injected put failure at call {self._put_count}"
                )
            try:
                stored_value = deepcopy(value)
            except Exception as exc:
                raise DataStoreWriteError("value cannot be copied into DataStore") from exc
            stable_digest: str | None = None
            size_bytes: int | None = None
            try:
                stable_digest = canonical_digest(stored_value)
                size_bytes = len(canonical_bytes(stored_value))
            except CanonicalizationError:
                pass
            handle = DataHandle(
                owner_generation=owner_generation,
                staged_handle_id=new_id("data"),
                stable_digest=stable_digest,
                size_bytes=size_bytes,
                metadata=FrozenMap((("backend", "memory"),)),
            )
            key = self._key(handle)
            self._entries[key] = _Entry(
                handle=handle,
                value=stored_value,
                state=_EntryState.STAGED,
                owner=None,
            )
            return handle

    def get(self, handle: DataHandle) -> Any:
        with self._lock:
            entry = self._require_entry(handle)
            try:
                return deepcopy(entry.value)
            except Exception as exc:
                raise DataHandleInvalidError("stored value cannot be copied") from exc

    def adopt(self, handles: tuple[DataHandle, ...], owner: DataOwner) -> None:
        if not isinstance(handles, tuple):
            raise DataOwnershipError("adopt handles must be a tuple")
        if not isinstance(owner, DataOwner):
            raise DataOwnershipError("adopt owner must be DataOwner")
        with self._lock:
            entries: list[_Entry] = []
            seen: set[tuple[str, str]] = set()
            for handle in handles:
                if not isinstance(handle, DataHandle):
                    raise DataOwnershipError("adopt requires DataHandle values")
                if handle.owner_generation != owner.owner_generation:
                    raise DataOwnershipError("owner generation does not match handle")
                key = self._key(handle)
                if key in seen:
                    raise DataOwnershipError("adopt contains a duplicate handle")
                seen.add(key)
                entry = self._require_entry(handle)
                if entry.state is _EntryState.ADOPTED and entry.owner != owner:
                    raise DataOwnershipError("handle is already adopted by another owner")
                entries.append(entry)
            for entry in entries:
                entry.state = _EntryState.ADOPTED
                entry.owner = owner

    def release(self, handle: DataHandle) -> None:
        key = self._key(handle)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return
            entry.value = None

    def release_many(self, handles: tuple[DataHandle, ...]) -> None:
        for handle in handles:
            self.release(handle)

    def state_of(self, handle: DataHandle) -> str:
        with self._lock:
            entry = self._require_entry(handle)
            return entry.state.value

    def owner_of(self, handle: DataHandle) -> DataOwner | None:
        with self._lock:
            return self._require_entry(handle).owner

    def handles_for_owner(self, owner: DataOwner) -> tuple[DataHandle, ...]:
        with self._lock:
            return tuple(
                entry.handle
                for entry in self._entries.values()
                if entry.owner == owner
            )

    def _require_entry(self, handle: DataHandle) -> _Entry:
        key = self._key(handle)
        entry = self._entries.get(key)
        if entry is None:
            raise DataHandleInvalidError("data handle is unknown or released")
        if entry.handle != handle:
            raise DataHandleInvalidError("data handle metadata does not match")
        return entry

    @property
    def put_count(self) -> int:
        with self._lock:
            return self._put_count

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def staged_count(self) -> int:
        with self._lock:
            return sum(
                entry.state is _EntryState.STAGED
                for entry in self._entries.values()
            )

    @property
    def adopted_count(self) -> int:
        with self._lock:
            return sum(
                entry.state is _EntryState.ADOPTED
                for entry in self._entries.values()
            )
