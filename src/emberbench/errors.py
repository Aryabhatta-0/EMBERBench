"""Exception hierarchy. Every rejected input raises one of these explicitly."""


class EmberBenchError(Exception):
    """Base class for all EMBERBench errors."""


class ScenarioValidationError(EmberBenchError):
    """A scenario document is malformed or violates a scenario invariant."""


class InvalidActionError(EmberBenchError):
    """A submitted action set is malformed (missing, unknown or bad values)."""


class SimulationError(EmberBenchError):
    """The simulator was used incorrectly (for example, stepped after termination)."""


class ReplayMismatchError(EmberBenchError):
    """A recorded run does not match a fresh replay of the same inputs."""


class MetricsInputError(EmberBenchError):
    """A scenario and run record cannot be scored: they do not match, or the record is malformed."""


class SuiteValidationError(EmberBenchError):
    """A suite manifest is malformed, or it does not match the scenario files it lists."""


class EvaluationError(EmberBenchError):
    """An evaluation manifest or result artifact is malformed, inconsistent or incompatible."""
