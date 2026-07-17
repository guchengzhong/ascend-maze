"""Stable exceptions raised before distributed execution begins."""


class AscendMazeError(Exception):
    """Base class for Ascend-Maze errors."""


class CanonicalizationError(AscendMazeError, ValueError):
    """Raised when a value cannot be represented deterministically."""


class LiteralSizeError(CanonicalizationError):
    """Raised when a literal exceeds a configured canonical byte limit."""


class TaskDefinitionError(AscendMazeError, ValueError):
    """Raised when a callable violates the phase-one task contract."""


class TaskOutputInferenceError(TaskDefinitionError):
    """Raised when static output names cannot be proven."""


class WorkflowValidationError(AscendMazeError, ValueError):
    """Raised when a workflow cannot be compiled."""


class WorkflowFrozenError(AscendMazeError, RuntimeError):
    """Raised when a compiled workflow is modified."""


class ContractValidationError(AscendMazeError, ValueError):
    """Raised when a cross-component contract object is invalid."""
