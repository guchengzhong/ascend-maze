from __future__ import annotations

from threading import Event, Thread

import pytest

from ascend_maze.contracts.data import DataHandle, DataOwner
from ascend_maze.core.errors import (
    DataHandleInvalidError,
    DataOwnershipError,
    RunDataIndexError,
)
from ascend_maze.data import InMemoryDataStore, RunDataIndexRegistry, RunDataState


def test_in_memory_data_store_adopt_is_atomic_and_release_is_idempotent() -> None:
    store = InMemoryDataStore()
    first = store.put_staged({"value": 1}, "controller_1")
    second = store.put_staged({"value": 2}, "controller_1")
    unknown = DataHandle("controller_1", "missing")
    owner = DataOwner("run_index", "run_1:1", "controller_1")
    with pytest.raises(DataHandleInvalidError):
        store.adopt((first, unknown), owner)
    assert store.state_of(first) == "staged"
    assert store.state_of(second) == "staged"

    store.adopt((first, second), owner)
    store.adopt((first, second), owner)
    assert store.owner_of(first) == owner
    with pytest.raises(DataOwnershipError, match="another owner"):
        store.adopt(
            (first,),
            DataOwner("run_index", "run_2:1", "controller_1"),
        )
    store.release(first)
    store.release(first)
    with pytest.raises(DataHandleInvalidError, match="released"):
        store.get(first)


def test_run_data_index_publishes_atomically_and_destroy_keeps_tombstone() -> None:
    store = InMemoryDataStore()
    input_handle = store.put_staged("input", "controller_1")
    registry = RunDataIndexRegistry(
        controller_generation="controller_1",
        data_store=store,
    )
    index = registry.create_and_adopt(
        run_id="run_1",
        workflow_inputs={"value": input_handle},
    )
    ref = index.reference
    assert index.read_workflow_input(
        "value",
        controller_generation=ref.controller_generation,
        index_generation=ref.index_generation,
    ) == "input"

    left = store.put_staged("left", "controller_1")
    right = store.put_staged("right", "controller_1")
    with pytest.raises(RunDataIndexError, match="do not match"):
        index.publish_outputs(
            task_id="task",
            output_handles={"left": left},
            expected_output_names=("left", "right"),
            controller_generation=ref.controller_generation,
            index_generation=ref.index_generation,
        )
    assert store.state_of(left) == "staged"
    index.publish_outputs(
        task_id="task",
        output_handles={"left": left, "right": right},
        expected_output_names=("left", "right"),
        controller_generation=ref.controller_generation,
        index_generation=ref.index_generation,
    )
    assert index.read_task_result(
        "task",
        ("left", "right"),
        controller_generation=ref.controller_generation,
        index_generation=ref.index_generation,
    ) == {"left": "left", "right": "right"}

    tombstone = registry.destroy("run_1", completed_at_ms=100)
    assert tombstone.destroy_succeeded
    assert tombstone.released_handle_count == 3
    assert registry.destroy("run_1", completed_at_ms=200) is tombstone
    assert index.state is RunDataState.DESTROYED
    assert index.handle_count() == 0
    assert store.active_count == 0
    with pytest.raises(RunDataIndexError, match="destroyed"):
        index.read_task_result(
            "task",
            ("left",),
            controller_generation=ref.controller_generation,
            index_generation=ref.index_generation,
        )


def test_old_index_generation_cannot_read_a_recreated_index() -> None:
    store = InMemoryDataStore()
    registry = RunDataIndexRegistry(
        controller_generation="controller_1",
        data_store=store,
    )
    first_handle = store.put_staged("first", "controller_1")
    first = registry.create_and_adopt(
        run_id="run_reused",
        workflow_inputs={"value": first_handle},
    )
    old_ref = first.reference
    registry.destroy("run_reused", completed_at_ms=1)

    second_handle = store.put_staged("second", "controller_1")
    second = registry.create_and_adopt(
        run_id="run_reused",
        workflow_inputs={"value": second_handle},
    )
    assert second.reference.index_generation == old_ref.index_generation + 1
    with pytest.raises(RunDataIndexError, match="stale"):
        second.read_workflow_input(
            "value",
            controller_generation=old_ref.controller_generation,
            index_generation=old_ref.index_generation,
        )
    assert second.read_workflow_input(
        "value",
        controller_generation=second.reference.controller_generation,
        index_generation=second.reference.index_generation,
    ) == "second"


def test_read_and_destroy_are_linearized_by_the_index_lock() -> None:
    class BlockingGetStore(InMemoryDataStore):
        def __init__(self) -> None:
            super().__init__()
            self.block_reads = False
            self.read_entered = Event()
            self.allow_read = Event()

        def get(self, handle):
            if self.block_reads:
                self.read_entered.set()
                assert self.allow_read.wait(timeout=2)
            return super().get(handle)

    store = BlockingGetStore()
    handle = store.put_staged("value", "controller_1")
    registry = RunDataIndexRegistry(
        controller_generation="controller_1",
        data_store=store,
    )
    index = registry.create_and_adopt(
        run_id="run_race",
        workflow_inputs={"value": handle},
    )
    ref = index.reference
    store.block_reads = True
    read_result: list[object] = []
    destroy_result = []

    reader = Thread(
        target=lambda: read_result.append(
            index.read_workflow_input(
                "value",
                controller_generation=ref.controller_generation,
                index_generation=ref.index_generation,
            )
        )
    )
    reader.start()
    assert store.read_entered.wait(timeout=2)
    destroyer = Thread(
        target=lambda: destroy_result.append(
            registry.destroy("run_race", completed_at_ms=10)
        )
    )
    destroyer.start()
    destroyer.join(timeout=0.02)
    assert destroyer.is_alive()
    store.allow_read.set()
    reader.join(timeout=2)
    destroyer.join(timeout=2)
    assert read_result == ["value"]
    assert destroy_result[0].destroy_succeeded
    with pytest.raises(RunDataIndexError, match="destroyed"):
        index.read_workflow_input(
            "value",
            controller_generation=ref.controller_generation,
            index_generation=ref.index_generation,
        )
