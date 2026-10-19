"""Local AgentFactory targets for the CLI tests (``--agent cli_agents:<name>``)."""

from emberbench import Action, ActionOutput, Agent, Observation, PolicyAgent

from conftest import greedy_exit_policy

# The widest public v1 scenario that ``crash_on_wide_maps`` still plays.
NARROW_WIDTH = 14


def greedy() -> Agent:
    return PolicyAgent(greedy_exit_policy)


def stay() -> Agent:
    return PolicyAgent(lambda obs: {p.id: Action.STAY for p in obs.active})


class _CrashOnWideMaps:
    def act(self, observation: Observation) -> ActionOutput:
        if observation.map.width > NARROW_WIDTH:
            raise RuntimeError("wide map")
        return greedy_exit_policy(observation)


def crash_on_wide_maps() -> Agent:
    """Plays like ``greedy`` on narrow maps and raises on wider ones (an interrupted evaluation)."""
    return _CrashOnWideMaps()


class _Raises:
    def act(self, observation: Observation) -> ActionOutput:
        raise RuntimeError("model exploded")


def raises() -> Agent:
    return _Raises()


class _BadOutput:
    def act(self, observation: Observation) -> ActionOutput:
        return {"nobody": "north"}


def bad_output() -> Agent:
    return _BadOutput()


def factory_raises() -> Agent:
    raise ValueError("cannot build")


def not_an_agent() -> object:
    return object()


_SHARED = PolicyAgent(greedy_exit_policy)


def same_instance() -> Agent:
    return _SHARED


NOT_CALLABLE = 42


class Namespace:
    @staticmethod
    def greedy() -> Agent:
        return PolicyAgent(greedy_exit_policy)
