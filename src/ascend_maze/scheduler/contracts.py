"""Immutable views shared by SchedulerCore and scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ascend_maze.resources.anchors import ResourceAnchor


@dataclass(frozen=True, slots=True, order=True)
class TaskKey:
    run_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class QueueToken:
    task_key: TaskKey
    queue_generation: int


@dataclass(frozen=True, slots=True)
class SchedulableTaskView:
    queue_token: QueueToken
    task_kind: str
    ready_at_ms: int
    queued_at_ms: int
    enqueue_sequence: int
    depth_from_entry: int
    depth_to_exit: int
    resource_anchor: ResourceAnchor


@dataclass(frozen=True, slots=True)
class PolicyCapabilities:
    requires_prediction: bool
    requires_static_topology: bool
    supports_incremental_dag: bool
    uses_cluster_snapshot: bool


@dataclass(frozen=True, slots=True)
class DispatchProposal:
    task_key: TaskKey
    queue_generation: int
    policy_metadata: tuple[tuple[str, object], ...] = ()


class QueuePartitioner(Protocol):
    def partition(self, task: SchedulableTaskView) -> str: ...


class SchedulingPolicy(Protocol):
    name: str
    version: str
    capabilities: PolicyCapabilities

    def enqueue(self, partition: str, task: SchedulableTaskView) -> None: ...

    def depart(self, token: QueueToken) -> None: ...

    def propose(self, partition: str, limit: int) -> tuple[DispatchProposal, ...]: ...


@runtime_checkable
class RunLifecycleAwarePolicy(Protocol):
    """Optional lifecycle input for policies whose priority spans a whole Run."""

    def register_run(
        self,
        *,
        run_id: str,
        submitted_at_ms: int,
        total_value_tasks: int,
    ) -> None: ...

    def unregister_run(self, run_id: str) -> None: ...

    def task_succeeded(self, *, run_id: str, task_id: str, task_kind: str) -> None: ...

    def run_terminal(
        self,
        *,
        run_id: str,
        status: str,
        finished_at_ms: int,
    ) -> None: ...
