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


class ReplayAuditError(MetricsInputError):
    """A replay audit or animation export cannot be built or verified: the record is not authentic for the scenario."""


class SuiteValidationError(EmberBenchError):
    """A suite manifest is malformed, or it does not match the scenario files it lists."""


class EvaluationError(EmberBenchError):
    """An evaluation manifest or result artifact is malformed, inconsistent or incompatible."""


class StatisticsInputError(EvaluationError):
    """An evaluation result cannot be summarized: it fails authentication against its suite."""


class ScoringInputError(EvaluationError):
    """An evaluation result cannot be scored: it fails authentication, or a failure declaration is invalid."""


class EnvironmentPackError(EmberBenchError):
    """A frozen environment pack is malformed, incomplete, tampered with or of another benchmark version."""


class GenerationError(EmberBenchError):
    """A generator config is invalid or unsatisfiable, or a frozen generated suite does not verify."""


class BaselineError(EmberBenchError):
    """A reference baseline ID or configuration is unknown or invalid."""


class ModelOutputError(InvalidActionError):
    """A decision model returned an action set that the model interface rejects."""


class ObservationFormatError(EmberBenchError):
    """A serialized observation is malformed, not canonical or of another observation version."""


class IntegrityError(EmberBenchError):
    """Held-out integrity check failed: provenance, submission identity, workspace layout or a leak guard."""
