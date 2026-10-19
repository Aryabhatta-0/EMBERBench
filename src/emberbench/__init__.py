"""EMBERBench: a deterministic benchmark for decision-making in fire evacuation."""

# Bump this whenever any rule changes simulator output for the same inputs.
# Run records from another version are rejected by ``verify_record``.
SIMULATOR_VERSION = "0.2.0"

from .errors import (  # noqa: E402
    EmberBenchError,
    InvalidActionError,
    MetricsInputError,
    ReplayMismatchError,
    ScenarioValidationError,
    SimulationError,
)
from .events import Casualty, CasualtyCause, Evacuation, RunRecord, StepRecord  # noqa: E402
from .grid import Action, Cell, Pos  # noqa: E402
from .movement import MoveOutcome, MoveResolution  # noqa: E402
from .observation import ALLOWED_ACTIONS, OBSERVATION_VERSION, MapView, Observation, RulesView  # noqa: E402
from .replay import replay, verify_record  # noqa: E402
from .scenario import (  # noqa: E402
    SCHEMA_VERSION,
    EnterFirePolicy,
    ExitSpec,
    FireConfig,
    FireModelName,
    MovementConfig,
    PersonSpec,
    Scenario,
    TerminationConfig,
    load_scenario,
    save_scenario,
)
from .simulator import Simulator, run_episode  # noqa: E402
from .state import PersonState, PersonStatus, SimState, TerminationReason, render  # noqa: E402
from .metrics import METRICS_VERSION, EpisodeMetrics, compute_episode_metrics  # noqa: E402

__all__ = [
    "ALLOWED_ACTIONS",
    "METRICS_VERSION",
    "OBSERVATION_VERSION",
    "SCHEMA_VERSION",
    "SIMULATOR_VERSION",
    "Action",
    "Casualty",
    "CasualtyCause",
    "Cell",
    "EmberBenchError",
    "EnterFirePolicy",
    "EpisodeMetrics",
    "Evacuation",
    "ExitSpec",
    "FireConfig",
    "FireModelName",
    "InvalidActionError",
    "MapView",
    "MetricsInputError",
    "MoveOutcome",
    "MoveResolution",
    "MovementConfig",
    "Observation",
    "PersonSpec",
    "PersonState",
    "PersonStatus",
    "Pos",
    "ReplayMismatchError",
    "RulesView",
    "RunRecord",
    "Scenario",
    "ScenarioValidationError",
    "SimState",
    "SimulationError",
    "Simulator",
    "StepRecord",
    "TerminationConfig",
    "TerminationReason",
    "compute_episode_metrics",
    "load_scenario",
    "render",
    "replay",
    "run_episode",
    "save_scenario",
    "verify_record",
]
