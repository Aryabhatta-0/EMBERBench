# Model interface

This document specifies the provider-independent boundary between EMBERBench and
a decision model. Planners, LLMs, RL agents and future external adapters all run
through the same boundary:

```text
Observation  ->  {person_id: Action}
```

The code is in [`src/emberbench/agent.py`](../src/emberbench/agent.py). The tests
are in [`tests/test_agent.py`](../tests/test_agent.py).

The interface adds no runtime dependency, no async API and no network API. It
does not change the simulator rules, the `Observation` fields, the evaluation
manifest, paired seed semantics or the result artifact. No version is bumped.

Out of scope here: provider integrations (OpenAI, Anthropic, Gemini and others),
agent frameworks, RL libraries and production baseline agents (Issue #10).

## 1. Model-visible input

A model receives exactly one value per timestep: an
[`Observation`](../src/emberbench/observation.py) (observation version 1.0, see
[v1-fairness-spec.md](v1-fairness-spec.md)). It contains:

| Field             | Content                                                              |
| ----------------- | -------------------------------------------------------------------- |
| `step`            | the current step number, from 0                                      |
| `map`             | grid `width` and `height`, `obstacles`, `exits` (ID and position)    |
| `rules`           | fire model, spread interval and probability, exit flammability, enter-fire policy, conflict resolution, `max_steps` |
| `people`          | every person: ID, position and status (`active`, `evacuated`, `casualty`) |
| `burning`         | every burning cell                                                   |
| `allowed_actions` | `stay`, `north`, `south`, `east`, `west`                             |

The model never receives:

- the `Scenario`, its ID, description, content hash or development seed,
- the evaluator seed, the `EvaluationManifest`, an episode entry or index,
- the `Simulator`, `SimState`, `RunRecord` or result artifact,
- RNG state (stream seeds or positions) or future draws,
- the `trapped` diagnostic, metrics or other evaluator metadata.

The current state is fully observable and the rules are known. The data above
is evaluator-private because it either determines the future (seed, RNG state)
or identifies the episode (scenario ID, hash, index). A model that saw it could
memorize or predict the stochastic fire instead of reacting to the observed
state. The `trapped` diagnostic is a computed answer that the model must find
itself. Keeping all of it outside the interface keeps paired comparisons fair.

## 2. Expected output

`act` returns a mapping from person ID to action:

- exactly one entry for every **active** person (`observation.active`),
- no entry for an unknown person or for an evacuated or casualty person,
- each value an `Action` member or its string value (`"north"`), which is what
  the simulator accepts, and a member of `observation.allowed_actions`.

The static types say the same thing:

```python
ActionValue = Action | str                  # Action.NORTH or "north"
ActionOutput = Mapping[str, ActionValue]    # person ID -> action value
```

`Agent.act`, `PolicyAgent.act`, `validate_actions` and `serialize_actions` use
these types. The static type does not weaken the runtime check: a string that
is not an action value (`"up"`, `"NORTH"`) is still rejected.

A move that cannot happen (into a wall, out of the grid, into an occupied
cell) is still a legal action. The simulator resolves it to a `blocked_*`
outcome. The interface does not judge the quality of an action.

## 3. Interface types

```python
class Agent(Protocol):
    def act(self, observation: Observation) -> ActionOutput: ...

AgentFactory = Callable[[], Agent]          # called once per episode, no arguments
```

| Name                                | Purpose                                                          |
| ----------------------------------- | ---------------------------------------------------------------- |
| `Agent`                             | a decision model for one episode                                 |
| `AgentFactory`                      | builds a new agent; receives no arguments                        |
| `PolicyAgent(policy)`               | adapts a stateless `Observation -> actions` callable to `Agent`  |
| `validate_actions(obs, output)`     | checks model output; returns a new `dict[str, Action]`           |
| `agent_policy(agent)`               | binds one agent to one episode; returns a simulator `Policy`     |
| `agent_policy_factory(factory)`     | a `run_evaluation` policy factory that builds a new agent per episode |
| `ExecutionMetadata`, `StepExecution`, `ReportsExecution` | optional accounting (section 7) |
| `serialize_observation`, `deserialize_observation`, `observation_from_dict` | canonical observation text (section 8) |
| `ACTION_SCHEMA_VERSION`, `serialize_actions`, `deserialize_actions` | canonical action payload (section 9) |

The three adapter shapes map onto these types:

- **Synchronous callable policy:** `PolicyAgent(greedy_policy)`. The callable
  is shared between episodes, so it must be stateless.
- **Stateful per-episode agent:** a class with `act`, built by the factory, for
  example `agent_policy_factory(MyPlanner)`.
- **Future external or remote adapter:** a class with `act` that sends
  `serialize_observation(observation)` to the model, parses the reply with
  `deserialize_actions(observation, reply)` and optionally implements
  `last_execution()`. Remote calls, retries and timeouts live inside the
  adapter (section 6.1). The core stays synchronous.

## 4. Episode lifecycle

`run_evaluation(suite, manifest, model_id, agent_policy_factory(factory), path)`
runs this lifecycle for each episode:

1. **Construct.** `factory()` is called with no arguments. Construction is the
   reset: an agent starts clean. There is no separate `reset` method.
2. **Act.** `agent.act(observation)` is called once per timestep, in step order
   from step 0, until the episode terminates. Each result is validated before
   the simulator sees it.
3. **Discard.** After the episode the agent is dropped. It is never used again.

Rules that make this hold:

- A policy from `agent_policy` accepts the steps of one episode only. An
  observation with an unexpected step number (a second episode, or a skipped
  step) raises `EvaluationError`.
- `agent_policy_factory` raises `EvaluationError` if the factory returns an
  object without a callable `act`, or **any** agent that it returned for an
  earlier episode (A -> A and A -> B -> A are both rejected). Reuse is detected
  by identity, not equality: the factory keeps a strong reference to every
  agent it returned, so each `id()` stays unique for its lifetime. Prior agents
  are therefore kept alive until the factory is dropped; correctness comes
  first. Use one `agent_policy_factory(...)` per evaluation run.

## 5. State isolation

A new agent is built for every episode, so state that an agent keeps cannot
carry into the next episode. This is the same fresh-policy-per-episode
guarantee as in [evaluation.md](evaluation.md) section 4, and it keeps resume
correct: an episode gives the same result in a fresh process or after other
episodes.

The tests prove:

- `test_new_stateful_agent_per_episode`: one distinct agent per episode, each
  factory call has no arguments, and each agent sees its episode's steps
  exactly once from step 0.
- `test_reuse_of_any_earlier_agent_is_rejected`: A -> A, A -> B -> A and
  A -> B -> C -> B each stop the run at the reused agent, before it sees a step.
- `test_distinct_agents_are_accepted`: A -> B -> C is accepted, and agents that
  compare equal but are distinct objects run normally.
- `test_resume_builds_agents_only_for_missing_episodes`: after an interruption,
  resume builds agents only for the episodes that are not in the artifact, and
  the final artifact is byte-identical to an uninterrupted run.
- `test_agent_receives_only_the_observation`: under a distinctive 64-bit seed,
  every `act` call has exactly one positional argument of type `Observation`.
  Every object inside it is a known observation type. Its serialized form holds
  no seed (decimal or hex), manifest hash, suite hash or scenario hash. No
  observation object has a `seed`, `rng_state`, `trapped`, `scenario` or similar
  attribute.
- `test_adapters_reproduce_plain_policy_artifacts`: the always-stay, greedy and
  stateful counting adapters produce artifacts byte-identical to the plain
  function policies. The interface changes nothing in records or metrics.

Limit: this is an interface contract for in-process Python code, not a
sandbox (see [evaluation.md](evaluation.md) section 4). An untrusted model must
run behind an adapter that sends it only the serialized observation.

## 6. Failure semantics

Every failure is fail-closed. Model output is never repaired, filtered,
defaulted or retried by the interface.

| Failure                                  | Result                                         |
| ---------------------------------------- | ---------------------------------------------- |
| missing ID for an active person          | `ModelOutputError`                             |
| unknown ID                               | `ModelOutputError`                             |
| ID of an evacuated or casualty person    | `ModelOutputError`                             |
| invalid action value (`"up"`, `None`, `1`, `"NORTH"`) | `ModelOutputError`                |
| action not in `allowed_actions`          | `ModelOutputError`                             |
| malformed return value (not a mapping, non-string keys) | `ModelOutputError`              |
| exception raised by the model            | the same exception object, unchanged           |
| factory returns a non-agent or any earlier agent | `EvaluationError`                      |
| step out of order for a bound agent      | `EvaluationError`                              |

`ModelOutputError` is a subclass of `InvalidActionError`. Its message is
`step <n>: ` followed by the simulator's own message, and its `__cause__` is
the simulator's `InvalidActionError`. Validation calls the simulator's
`parse_actions`, so the interface and the simulator always accept and reject
the same output (`test_invalid_model_output_fails_closed` checks both for every
case).

Any of these failures stops `run_evaluation`. The simulator is not stepped
with the rejected output, and the artifact keeps only the episodes completed
before the failure. A deterministic model under the same manifest gives the same failure and the
same artifact bytes on every run. A failed episode is not recorded as a
result: a model that fails must be fixed and the run resumed.

### 6.1 Timeouts

The core has no timeout mechanism. The rule is:

- A provider or remote adapter enforces its own timeout inside `act`.
- On expiry it raises (for example `TimeoutError`). The exception propagates
  through `Agent.act` and `run_evaluation` unchanged. It is not wrapped,
  retried or converted into an action.
- The simulator is not stepped for that observation.
- The current episode is not committed to the artifact.
- The episodes completed before it stay in the artifact. A later
  `run_evaluation` call with `resume=True` runs only the missing episodes, each
  with a new agent.

`test_adapter_timeout_propagates_and_the_episode_is_not_committed` checks
each point, and that the resumed artifact is byte-identical to an
uninterrupted run.

## 7. Optional execution accounting

Accounting is evaluator-side and optional. Pass an observer:

```python
log: list[StepExecution] = []
factory = agent_policy_factory(MyAdapter, on_step=log.append)
```

After each accepted `act` call the observer receives a `StepExecution`:

| Field             | Content                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `step`            | the observation step                                             |
| `latency_seconds` | wall-clock time of the `act` call, measured by the evaluator     |
| `reported`        | `ExecutionMetadata` from the agent, or `None`                    |

An agent may implement `last_execution() -> ExecutionMetadata | None`
(`ReportsExecution`). Every `ExecutionMetadata` field is optional and `None`
means unknown:

| Field           | Content                                                   |
| --------------- | --------------------------------------------------------- |
| `provider`      | provider label                                            |
| `model`         | model label                                               |
| `input_tokens`  | input token count, a non-negative integer                 |
| `output_tokens` | output token count, a non-negative integer                |
| `cost`          | a finite non-negative number; needs `cost_currency`       |
| `cost_currency` | currency code, for example `"USD"`; needs `cost`          |

An adapter must report only what its provider returned. It must not estimate
token counts or costs. Invalid values raise `EvaluationError`.

Accounting never reaches the simulator, the run record, the metrics or the
result artifact. It is read only when an observer is given, after the actions
are validated. `test_accounting_is_collected_but_never_changes_outcomes`
checks that an artifact with accounting is byte-identical to one without.
Latency is not deterministic, so it must never be part of a canonical result.

## 8. Serialized observation

`serialize_observation(observation)` returns the canonical JSON text of
`Observation.to_dict()`: sorted keys, no whitespace, ASCII only, no NaN. This is
the payload for a future external adapter.

```json
{"allowed_actions":["stay","north","south","east","west"],"burning":[[0,3]],"map":{"exits":[{"id":"E1","position":[0,4]}],"height":1,"obstacles":[],"width":5},"observation_version":"1.0","people":[{"id":"Pa","position":[0,0],"status":"active"}],"rules":{"conflict_resolution":"lowest_id_wins","enter_fire":"casualty","exits_flammable":true,"fire_model":"deterministic","max_steps":100,"spread_interval":1,"spread_probability":null},"step":0}
```

Order is canonical: people and exits by ID, obstacles and burning cells by
position (row, then column). The simulator produces observations in this order,
so equal observations give equal bytes.

`deserialize_observation(text)` parses the text back. It is strict: exactly the
declared keys, the current `observation_version`, correct types, positions
inside the grid, unique items in canonical order, and text that re-serializes
to the same bytes. Anything else raises `ObservationFormatError`.
`observation_from_dict` applies the same checks to a parsed object without the
byte comparison.

The tests prove that the serialized form:

- round-trips every observation of every public scenario under a greedy policy,
- has exactly the declared key set, with no `seed`, `rng`, `hash`, `trapped`,
  `scenario`, `description`, `manifest`, `episode` or `index` anywhere,
- holds no seed, RNG position, scenario hash or description, even for a
  stochastic scenario with a distinctive seed,
- is identical for two episodes that differ only in seed, scenario ID and
  description,
- does not change when a person is trapped.

## 9. Action payload

`serialize_actions(observation, output)` returns the canonical external form of
one step of model output, for a future external adapter:

```json
{"action_schema_version":"1.0","actions":{"Pa":"east","Pb":"stay"},"step":3}
```

| Key                     | Content                                                   |
| ----------------------- | --------------------------------------------------------- |
| `action_schema_version` | `ACTION_SCHEMA_VERSION`, currently `"1.0"`                |
| `step`                  | the step of the observation the actions answer            |
| `actions`               | person ID -> action string, for exactly the active people |

The text is canonical JSON: sorted keys (so person IDs are sorted), no
whitespace, ASCII only. The output is checked with `validate_actions` before it
is serialized.

`deserialize_actions(observation, text)` parses a payload for that observation
and returns `dict[str, Action]`. It raises `ModelOutputError` (message
`step <n>: ...`) unless all of these hold:

- the text is valid JSON with exactly the three keys above,
- `action_schema_version` is the supported version string,
- `step` is an integer equal to `observation.step`,
- `actions` is an object of string actions for exactly the active people,
  each in `observation.allowed_actions`,
- the text is byte-identical to its canonical form (no extra whitespace, other
  key order, duplicate keys or trailing newline).

The payload holds no seed, RNG state, scenario data, accounting or other
evaluator metadata. It is an exchange format only: run records and evaluation
artifacts are unchanged and do not store it.

## 10. Versioning

This interface wraps existing contracts and changes none of them.
`SIMULATOR_VERSION`, `OBSERVATION_VERSION`, `SCHEMA_VERSION`,
`METRICS_VERSION`, the evaluation manifest and result versions, and all
scenario and suite hashes are unchanged (`test_no_contract_version_changes`).

The two external formats are versioned independently:

- The serialized observation carries `observation_version` (`"1.0"`). A change
  to the observation fields must bump it, and old text is then rejected.
- The action payload carries `action_schema_version` (`ACTION_SCHEMA_VERSION`,
  `"1.0"`). A change to the payload must bump it, and old payloads are then
  rejected.
