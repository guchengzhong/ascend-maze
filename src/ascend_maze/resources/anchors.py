"""C5 resource anchors used by the stage-two correctness path."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from ascend_maze.compiler.ir import CompiledWorkflow, TaskDefinition, TaskNode
from ascend_maze.contracts.resources import ExecutionTarget, ResourceSpec
from ascend_maze.core.canonical import canonical_digest


_ZERO_RESOURCES = ResourceSpec(cpu_num=0, mem_mb=0, npu_mem_mb=0, io_num=0)


@dataclass(frozen=True, slots=True)
class ResourceAnchor:
    definition_id: str
    task_kind: str
    execution_target: ExecutionTarget
    declared: ResourceSpec
    static_inferred: ResourceSpec
    learned: ResourceSpec | None
    effective: ResourceSpec
    model_id: str | None
    profile_key: str
    revision: int
    strategy: str


class DeclaredOnlyAnchorProvider:
    """Resolve immutable per-run anchors from compiled declarations only."""

    strategy = "declared_only"

    def __init__(self, *, environment_fingerprint: str) -> None:
        if not isinstance(environment_fingerprint, str) or not environment_fingerprint:
            raise ValueError("environment_fingerprint is required")
        self.environment_fingerprint = environment_fingerprint
        self._anchors: dict[tuple[str, str], ResourceAnchor] = {}
        self._lock = RLock()

    def resolve(
        self,
        *,
        run_id: str,
        compiled: CompiledWorkflow,
        task_id: str,
    ) -> ResourceAnchor:
        key = (run_id, task_id)
        with self._lock:
            cached = self._anchors.get(key)
            if cached is not None:
                return cached
            node: TaskNode = compiled.tasks[task_id]
            definition: TaskDefinition = compiled.definitions[node.definition_id]
            target = (
                ExecutionTarget.MODEL_SERVICE
                if node.model_anchor is not None
                and node.model_anchor.mode == "service"
                else ExecutionTarget.LOCAL_WORKER
            )
            model_id = None if node.model_anchor is None else node.model_anchor.model
            profile_key = canonical_digest(
                {
                    "definition_id": definition.definition_id,
                    "code_hash": definition.code_hash,
                    "environment_fingerprint": self.environment_fingerprint,
                    "execution_target": target.value,
                    "model_id": model_id,
                    "strategy": self.strategy,
                }
            )
            anchor = ResourceAnchor(
                definition_id=definition.definition_id,
                task_kind=definition.task_kind,
                execution_target=target,
                declared=definition.resources,
                static_inferred=_ZERO_RESOURCES,
                learned=None,
                effective=definition.resources,
                model_id=model_id,
                profile_key=profile_key,
                revision=1,
                strategy=self.strategy,
            )
            self._anchors[key] = anchor
            return anchor

    def destroy_run(self, run_id: str) -> int:
        with self._lock:
            keys = [key for key in self._anchors if key[0] == run_id]
            for key in keys:
                del self._anchors[key]
            return len(keys)

    def count_for_run(self, run_id: str) -> int:
        with self._lock:
            return sum(key[0] == run_id for key in self._anchors)
