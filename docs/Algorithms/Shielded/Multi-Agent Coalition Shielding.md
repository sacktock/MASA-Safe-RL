# Multi-agent coalition safety-game shielding

`CoalitionLTLShield` extends MASA's winning-region shielding to finite PettingZoo
Parallel environments. It protects a selected coalition against every legal action
of agents outside the coalition and every supported stochastic outcome.

The environment stack is:

```text
TabularParallelEnv -> LabelledParallelEnv -> CoalitionLTLShield
```

`TabularParallelEnv` owns the finite game dynamics. `LabelledParallelEnv` supplies
proposition labels. The shield owns the DFA product, safety-game solution, and
runtime enforcement. No separate game-model object is required.

The wrapper supports two independent choices:

| Setting | Options | Meaning |
|---|---|---|
| `mode` | `preemptive`, `postposed` | Reject an unsafe proposal before stepping, or replace it. |
| `execution` | `centralised`, `decentralised` | Select one coalition joint action, or let members choose independently from certified local masks. |

Both execution modes use the same bad-prefix safety DFA and robust coalition
winning region.

## Concurrent safety game

Let $C$ be the focal coalition and $-C$ its complement. At product state
$x=(q,s)$, coalition controllability is

$$
\operatorname{CPre}_C(W)=
\left\{
x:
\exists a_C\;
\forall a_{-C}\;
\forall x'\in\operatorname{Succ}(x,a_C,a_{-C}),
\quad x'\in W
\right\}.
$$

The winning region is the greatest fixed point

$$
W_C=\nu W.\left(\operatorname{Safe}\cap\operatorname{CPre}_C(W)\right).
$$

MASA implements the universal complement quantifier by taking the union of all
successor supports induced by a fixed coalition action and every legal complement
action. The existing `winning_region()` solver can then be reused: its controllable
"action" is a coalition tuple, and its support already contains every outsider
action and every environment outcome.

The quantifier order is important. The coalition chooses one action that works
against every simultaneous complement action. It cannot observe an outsider's
current action and choose a response afterwards.

## Required environment interface

The base environment must subclass `TabularParallelEnv`. Like the existing
single-agent `TabularEnv`, it exposes either a dense transition matrix or sparse
successor/probability dictionaries.

A subclass sets `self._n_states` and one of:

```python
# Dense model
self._transition_matrix[next_state, state, joint_action_index]

# Sparse model
self._successor_states[state]
self._transition_probs[state, joint_action_index]
```

Joint-action indices follow the Cartesian-product order induced by
`possible_agents` and the agents' zero-based `Discrete` action spaces. The class
provides `encode_joint_action()` and `decode_joint_action()`.

The environment also maintains its exact current finite ID in `self._state`, or
overrides `get_state_id()`. Environments with structured or local observations
override `observations_from_state(state)` so the labelled wrapper's existing
labelling functions can be evaluated for every hypothetical model state.

State-dependent action availability is represented by overriding
`legal_actions(state, agent)`. Missing transition support for a legal full joint
action is an error; it cannot silently remove an adversarial action.

```python
from masa.common.multi_agent import Coalition, LabelledParallelEnv
from masa.deterministic_shield import CoalitionLTLShield
from masa.envs.multiagent.matrix.chicken import ChickenMatrix, label_fn
from masa.examples.chicken_safety_game import make_never_crash_dfa

env = CoalitionLTLShield(
    LabelledParallelEnv(ChickenMatrix(), label_fn),
    coalition=Coalition(("player_0",)),
    dfa=make_never_crash_dfa(),
    mode="preemptive",
    execution="centralised",
)
```

By default, the shield unions labels produced for every possible agent. Supply a
`label_combiner` when the shared DFA uses a different alphabet—for example,
agent-namespaced propositions.

## Centralised execution

One controller selects the complete coalition action. The permitted relation is

$$
A_C^{\mathrm{safe}}(x)=
\left\{
a_C:
\forall a_{-C},
\operatorname{Succ}(x,a_C,a_{-C})\subseteq W_C
\right\}.
$$

A central relation need not be Cartesian. It may permit `(left, left)` and
`(right, right)` while excluding both mismatched pairs because one controller
selects the entire tuple.

```python
from masa.deterministic_shield import random_safe

env = CoalitionLTLShield(
    LabelledParallelEnv(ChickenMatrix(), label_fn),
    coalition=("player_0", "player_1"),
    dfa=make_never_crash_dfa(),
    mode="postposed",
    execution="centralised",
    replacement=random_safe(seed=7),
)

observations, infos = env.reset(seed=0)
mask = env.coalition_action_mask()
# mask[i] corresponds to env.coalition_actions[i].
```

A centralised replacement callback receives a coalition-action **index**, not a
primitive action ID. `decode_coalition_action()` and `encode_coalition_action()`
convert between indices and per-agent mappings. Actions of agents outside the
coalition are never changed.

## Decentralised execution

Coalition members choose simultaneously and cannot condition on teammates' current
choices. Teammates are not treated as adversaries: each member may rely on the
others obeying their certified local masks. Agents outside the coalition remain
unrestricted.

For local masks $M_i(x)$, MASA certifies

$$
\varnothing\neq M_i(x)
$$

for each coalition member and

$$
\prod_{i\in C}M_i(x)
\subseteq A_C^{\mathrm{safe}}(x).
$$

Every combination of permitted local actions is therefore safe against every
complement action and supported successor. An action that is safe only through
runtime coordination is not exposed independently.

### Why marginal projection is unsound

Suppose the central relation is

$$
\{(L,L),(R,R)\}.
$$

Projecting it onto each agent gives both agents `{L, R}`, whose Cartesian product
also contains unsafe `(L, R)` and `(R, L)`. A valid decentralised interface must
choose a Cartesian subset, such as `{L} x {L}`.

MASA tries every safe tuple as a singleton seed, greedily expands the local masks
while preserving Cartesian closure, and keeps the candidate admitting the most
joint profiles. The result is sound and deterministic, but is not claimed to be a
globally maximum rectangle. Equally permissive choices may be asymmetric because
ties use canonical agent/action order.

```python
env = CoalitionLTLShield(
    LabelledParallelEnv(ChickenMatrix(), label_fn),
    coalition=("player_0", "player_1"),
    dfa=make_never_crash_dfa(),
    mode="preemptive",
    execution="decentralised",
)

observations, infos = env.reset(seed=0)
player_0_mask = env.local_action_mask("player_0")
player_1_mask = env.local_action_mask("player_1")
```

For postposed decentralised execution, provide a separate replacement callback per
coalition member. A callback receives only the shared product state, that agent's
proposal, and that agent's local mask. It does not receive teammates' simultaneous
proposals.

```python
replacement = {
    "player_0": random_safe(seed=10),
    "player_1": random_safe(seed=11),
}
```

## Temporal and runtime semantics

The reset observation's labels are consumed once. On each environment transition,
the DFA consumes labels of the **successor** state:

$$
(q,s)\xrightarrow{a}
\left(\delta(q,L(s')),s'\right).
$$

The wrapper validates that:

- the live finite state is a supported successor of the executed full joint action;
- reconstructed model observations induce the same labels as live observations;
- the successor product state remains winning;
- all agents remain active until a simultaneous termination or truncation.

A runtime mismatch is detected only after the environment has stepped and cannot
undo an unsafe transition. The safety guarantee therefore depends on a correct
bad-prefix DFA, fixed labels, and transition support containing every possible real
outcome.

`TabularParallelEnv` may represent stochastic dynamics. "Deterministic shielding"
means the property is enforced without an allowed violation probability; a
postposed selector may still randomly choose among already-safe actions.

## Runtime interface

Centralised methods:

```text
coalition_action_mask()
robust_coalition_action_mask()
safe_coalition_actions()
encode_coalition_action(...)
decode_coalition_action(...)
```

Decentralised methods:

```text
local_action_mask(agent)
local_action_masks()
coalition_action_mask()  # the selected Cartesian subset
```

Coalition-agent infos include the product state, DFA state, shared labels, mode,
execution type, and coalition identity. Centralised infos include a joint-action
mask. Decentralised infos include each member's local mask. Postposed steps also
report proposed and executed primitive and coalition actions.

## Limitations

- The implementation targets PettingZoo's Parallel API, not AEC timing.
- State and action spaces must be finite; primitive actions are zero-based
  `Discrete` values.
- The base environment must provide complete transition support.
- Dynamic agent populations and asynchronous per-agent termination are not
  supported.
- Decentralised members must identify the same global tabular and DFA state. This
  is not a partial-observation synthesis method.
- As standard with safety-game based shielding, the DFA accepting states must denote bad prefixes; 
  general liveness objectives are outside this shield.
- Joint-action enumeration grows exponentially with the number of agents.
  - You'd likely want to read further on shield decentralisation techniques if this becomes an issue in practice.

## API reference

```{eval-rst}
.. autoclass:: masa.deterministic_shield.CoalitionLTLShield
   :members:
   :show-inheritance:

.. autoclass:: masa.common.multi_agent.Coalition
   :members:

.. autoclass:: masa.envs.multiagent.TabularParallelEnv
   :members:
   :show-inheritance:
```
