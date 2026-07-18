"""Local RuntimeClient preserving staged input ownership across disconnects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ascend_maze.api.workflow import Workflow
from ascend_maze.compiler.ir import CompiledWorkflow
from ascend_maze.contracts.data import DataHandle
from ascend_maze.contracts.submission import (
    RunInputIdentity,
    SubmissionContract,
    SubmissionOptions,
    SubmissionState,
    hash_session_key,
)
from ascend_maze.core.canonical import FrozenMap, canonical_digest, freeze_canonical
from ascend_maze.core.errors import (
    CanonicalizationError,
    DataHandleInvalidError,
    SubmissionAbortedError,
    SubmissionConflictError,
)
from ascend_maze.core.identifiers import new_id
from ascend_maze.runtime.packaging import build_code_packages

from ascend_maze.control.controller import (
    InMemoryController,
    SubmissionOutcome,
    SubmitRequest,
)


@dataclass(frozen=True, slots=True)
class PreparedSubmission:
    request: SubmitRequest
    input_signature: tuple[tuple[str, tuple[str, ...]], ...]


class InMemoryRuntimeClient:
    def __init__(self, controller: InMemoryController) -> None:
        self.controller = controller
        self._prepared: dict[str, PreparedSubmission] = {}

    def prepare_submission(
        self,
        workflow: Workflow | CompiledWorkflow,
        *,
        inputs: dict[str, object],
        submission_id: str | None = None,
        session_key: str | None = None,
        run_deadline_ms: int | None = None,
        execution_options: dict[str, object] | None = None,
    ) -> PreparedSubmission:
        if isinstance(workflow, Workflow):
            compiled = workflow.compile()
            callables_by_definition: dict[str, Callable[..., object]] = {}
            for draft in workflow._draft_tasks:
                definition_id = compiled.tasks[draft.task_id].definition_id
                callables_by_definition.setdefault(definition_id, draft.template.func)
        else:
            compiled = workflow
            callables_by_definition = {}
        if set(inputs) != set(compiled.workflow_inputs):
            missing = sorted(set(compiled.workflow_inputs) - set(inputs))
            extra = sorted(set(inputs) - set(compiled.workflow_inputs))
            raise ValueError(f"workflow input mismatch; missing={missing}, extra={extra}")
        resolved_submission_id = submission_id or new_id("submission")
        signature = tuple(
            (name, self._value_identity(inputs[name]))
            for name in sorted(inputs)
        )
        frozen_execution_options = freeze_canonical(execution_options or {})
        if not isinstance(frozen_execution_options, FrozenMap):
            raise TypeError("execution_options must freeze to a mapping")
        options = SubmissionOptions(
            run_deadline_ms=run_deadline_ms,
            execution_options=frozen_execution_options,
        )
        existing = self._prepared.get(resolved_submission_id)
        if existing is not None:
            old = existing.request
            if (
                old.compiled.workflow_fingerprint != compiled.workflow_fingerprint
                or existing.input_signature != signature
                or old.contract.session_key_hash != hash_session_key(session_key)
                or old.contract.options != options
                or old.contract.config_fingerprint
                != self.controller.config_fingerprint
            ):
                raise SubmissionConflictError(
                    "local submission_id is already prepared with another payload"
                )
            return existing

        handles: list[tuple[str, DataHandle]] = []
        try:
            for name in sorted(inputs):
                handles.append(
                    (
                        name,
                        self.controller.data_store.put_staged(
                            inputs[name], self.controller.controller_generation
                        ),
                    )
                )
        except Exception:
            for _, handle in handles:
                self.controller.data_store.release(handle)
            raise
        identities = tuple(
            RunInputIdentity.from_data_handle(name, handle)
            for name, handle in handles
        )
        contract = SubmissionContract.create(
            submission_id=resolved_submission_id,
            workflow_fingerprint=compiled.workflow_fingerprint,
            input_identities=identities,
            session_key_hash=hash_session_key(session_key),
            options=options,
            config_fingerprint=self.controller.config_fingerprint,
        )
        request = SubmitRequest(
            compiled=compiled,
            code_packages=build_code_packages(
                compiled,
                environment_fingerprint=self.controller.environment_fingerprint,
                callables_by_definition=callables_by_definition,
            ),
            workflow_inputs=tuple(handles),
            contract=contract,
        )
        prepared = PreparedSubmission(request=request, input_signature=signature)
        self._prepared[resolved_submission_id] = prepared
        return prepared

    async def submit_prepared(
        self,
        prepared: PreparedSubmission,
        *,
        lose_response_after_commit: bool = False,
    ) -> SubmissionOutcome:
        try:
            outcome = await self.controller.submit(
                prepared.request,
                lose_response_after_commit=lose_response_after_commit,
            )
        except SubmissionConflictError:
            self._release_staged_inputs(prepared)
            self._prepared.pop(prepared.request.contract.submission_id, None)
            raise
        if outcome.state is SubmissionState.ABORTED:
            self._release_staged_inputs(prepared)
        elif outcome.replayed:
            self._release_staged_inputs(prepared)
        self._prepared.pop(prepared.request.contract.submission_id, None)
        return outcome

    @property
    def prepared_submission_count(self) -> int:
        return len(self._prepared)

    async def submit(
        self,
        workflow: Workflow | CompiledWorkflow,
        *,
        inputs: dict[str, object],
        submission_id: str | None = None,
        session_key: str | None = None,
        run_deadline_ms: int | None = None,
        execution_options: dict[str, object] | None = None,
        lose_response_after_commit: bool = False,
    ) -> SubmissionOutcome:
        prepared = self.prepare_submission(
            workflow,
            inputs=inputs,
            submission_id=submission_id,
            session_key=session_key,
            run_deadline_ms=run_deadline_ms,
            execution_options=execution_options,
        )
        return await self.submit_prepared(
            prepared,
            lose_response_after_commit=lose_response_after_commit,
        )

    async def run(
        self,
        workflow: Workflow | CompiledWorkflow,
        *,
        inputs: dict[str, object],
        submission_id: str | None = None,
        session_key: str | None = None,
        run_deadline_ms: int | None = None,
        execution_options: dict[str, object] | None = None,
        lose_response_after_commit: bool = False,
    ) -> str:
        outcome = await self.submit(
            workflow,
            inputs=inputs,
            submission_id=submission_id,
            session_key=session_key,
            run_deadline_ms=run_deadline_ms,
            execution_options=execution_options,
            lose_response_after_commit=lose_response_after_commit,
        )
        if outcome.state is SubmissionState.ABORTED or outcome.run_id is None:
            raise SubmissionAbortedError(outcome.error or "submission aborted")
        return outcome.run_id

    def _release_staged_inputs(self, prepared: PreparedSubmission) -> None:
        for _, handle in prepared.request.workflow_inputs:
            try:
                if self.controller.data_store.state_of(handle) == "staged":
                    self.controller.data_store.release(handle)
            except DataHandleInvalidError:
                pass

    @staticmethod
    def _value_identity(value: object) -> tuple[str, ...]:
        try:
            return ("digest", canonical_digest(value))
        except CanonicalizationError:
            return (
                "object",
                type(value).__module__,
                type(value).__qualname__,
                str(id(value)),
            )
