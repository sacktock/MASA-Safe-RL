# Winning-region safety-game shielding

MASA's winning-region shields prevent safety violations by restricting execution
to actions that keep the system inside a computed **winning region**. The region
is computed once, when the wrapper is constructed, from a finite transition model
and a deterministic finite automaton (DFA) recognizing bad prefixes of a safety
property.

There are two interfaces to the same safety analysis:

- `PreemptiveLTLShield` exposes the permitted actions before the policy chooses.
  An unsafe choice is rejected before the environment advances.
- `PostposedLTLShield` checks the policy's proposal and replaces it only when
  necessary. Safe proposals are executed unchanged.

These are environment wrappers, not learning algorithms. They do not optimize
reward, learn a dynamics model, or guarantee eventual task completion.

> **Important**
>
> Winning-region safety does **not** require deterministic dynamics or deterministic
> action replacement. Every positive-probability successor is treated as possible.
> Random replacement is safe when it selects only permitted actions.
>
> The `masa.deterministic_shield` package name distinguishes this support-based,
> zero-risk construction from MASA's risk-budget-based probabilistic shielding; it
> does not imply that every shielded trajectory is deterministic.

For risk budgets and projection onto safe action distributions, see
[Probabilistic Shielding](../Probabilistic/Probabilistic%20Shielding.md). Sampling uniformly among
winning actions is not the probabilistic-shielding algorithm: no probability of
violation is budgeted.

## The safety game

Let the base environment be a finite, fully observed MDP with states $S$, actions
$A$, transition probabilities $P(s' \mid s,a)$, and a fixed state-labelling
function $L$.

A safety DFA has states $Q$, initial state $q_0$, transition function
$\delta$, and bad-prefix accepting states $F$. In MASA's safety monitor,
**accepting means a violation**, not success. Supply a correct DFA for the intended
safety property; these wrappers do not compile LTL strings or solve general
liveness objectives.

The product state $x=(q,s)$ records both the physical state and the monitor's
memory. The game interpretation is:

1. the controller chooses an action;
2. the environment may realize any successor in that action's transition support.

An explicit second player does not need to be implemented in the simulator.

### Monitor timing and state encoding

`LTLSafetyEnv.reset()` consumes the initial state's labels. Its `step()` consumes
the labels of the newly returned state. Therefore $q$ in the current product
observation has **already consumed** $L(s)$, and the shield uses

$$
\operatorname{Succ}((q,s),a)
=
\left\{
(\delta(q,L(s')),s')
:
P(s'\mid s,a)>0
\right\}.
$$

The initial product state is

$$
(\delta(q_0,L(s_0)),s_0).
$$

For discrete observations, MASA encodes the product state as

```text
q_index * n_states + state
```

where `q_index` is the monitor's internal index for a DFA state, not necessarily
the numeric value or name of that DFA state.

The shield's `winning_region` and `safe_actions` arrays use this same encoding.

### Computing the winning region

Initialize the candidate region to the non-rejecting product states:

$$
W_0 = (Q \setminus F) \times S.
$$

Then repeatedly remove states for which there is no enabled action whose entire
successor support remains inside the current candidate region:

$$
W_{k+1}
=
\left\{
x\in W_k
:
\exists a\in A_{\mathrm{enabled}}(x),
\operatorname{Succ}(x,a)\subseteq W_k
\right\}.
$$

Because the product is finite and the sequence only shrinks, this converges to a
greatest fixed point $W$.

For each product state, the permissive safe-action set is

$$
A_{\mathrm{safe}}(x)
=
\begin{cases}
\left\{
a\in A_{\mathrm{enabled}}(x)
:
\operatorname{Succ}(x,a)\subseteq W
\right\},
& x\in W,\\
\varnothing,
& x\notin W.
\end{cases}
$$

An action with no positive transition mass is disabled, not vacuously safe.

A state with no enabled winning action is losing.

No probability cutoff is used: even a very rare unsafe successor disqualifies an
action.

### Why this is stronger than one-step checking

Suppose `start` has two actions:

- one keeps the system at `start`;
- one enters `trap`.

`trap` is not currently unsafe, but its only action leads to `bad`.

A one-step safety filter may allow the transition to `trap`. Winning-region
shielding does not: `trap` is removed from the fixed point, so the action entering
it is also removed from the safe-action mask at `start`.

### Why the restriction preserves safety

Starting in $W$, every permitted action has all modeled successors in $W$.
Applying that property inductively keeps every supported execution inside $W$,
and therefore outside the rejecting DFA states.

Both the preemptive and postposed interfaces enforce the same invariant. Their
difference is only **when** the safe-action set is consulted.

This is a **model-relative guarantee**. It requires:

- the real successor support to be contained in the modeled support;
- the deterministic labelling function to describe the real states correctly;
- the DFA to encode the intended safety property.

A conservative model can remove genuinely safe actions. An incomplete model can
invalidate the guarantee.

## Getting started

Use this wrapper order:

```text
finite base environment
    -> LabelledEnv
    -> LTLSafetyEnv(obs_type="discrete")
    -> PreemptiveLTLShield OR PostposedLTLShield
    -> optional observation transforms / time limit / auto-reset / vectorization
```

Place the shield directly outside `LTLSafetyEnv`.

The base environment must expose:

- a finite transition model;
- zero-based `Discrete` state IDs;
- zero-based `Discrete` action IDs.

The shield keeps the action space and the monitor's discrete product-observation
space unchanged.

The following setup uses the colour-bomb grid world and the safety property

$$
\mathbf{G}\neg \mathit{bomb}.
$$

```python
import numpy as np

from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.labelled_env import LabelledEnv
from masa.common.ltl import Atom, DFA
from masa.deterministic_shield import (
    PreemptiveLTLShield,
    PostposedLTLShield,
    random_safe,
)
from masa.envs.tabular.colour_bomb_grid_world import (
    ColourBombGridWorld,
    label_fn,
)


def make_ltl_env():
    dfa = DFA([0, 1], initial=0, accepting=[1])
    dfa.add_edge(0, 1, Atom("bomb"))

    base = ColourBombGridWorld(slip_prob=0.0)
    return LTLSafetyEnv(
        LabelledEnv(base, label_fn),
        dfa=dfa,
        obs_type="discrete",
    )
```

## Preemptive shielding

Preemptive shielding exposes the safe set **before** action selection.

```python
env = PreemptiveLTLShield(make_ltl_env())
policy_rng = np.random.default_rng(42)

try:
    obs, info = env.reset(seed=17)

    for _ in range(50):
        permitted = np.flatnonzero(info["action_mask"])
        action = int(policy_rng.choice(permitted))

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
finally:
    env.close()
```

The policy may use either

```python
info["action_mask"]
```

or

```python
env.action_masks()
```

to obtain the current safe-action mask.

`action_masks()` returns a copy.

If the caller submits an unsafe action, `PreemptiveLTLShield.step()` raises
`ValueError` **before** the environment is stepped. It does not silently replace
the action, and the caller may retry with a safe choice.

Exposing a mask does not automatically make a reinforcement-learning algorithm
mask-aware. A preemptive learner should apply the mask during:

- exploration;
- exploitation;
- maximization over next-state actions in value targets.

For policy-gradient methods, mask and normalize the action distribution before
sampling and before computing the corresponding log-probability.

## Postposed shielding

Postposed shielding lets the policy propose an action normally.

If the proposal is safe, it is executed unchanged. If it is unsafe, the configured
replacement strategy chooses another safe action.

```python
env = PostposedLTLShield(
    make_ltl_env(),
    replacement=random_safe(seed=23),
)
policy_rng = np.random.default_rng(42)

try:
    obs, info = env.reset(seed=17)

    for _ in range(50):
        proposal = int(policy_rng.integers(env.action_space.n))

        obs, reward, terminated, truncated, info = env.step(proposal)

        if info["shield_intervened"]:
            print(
                info["shield_proposed_action"],
                "->",
                info["shield_executed_action"],
            )

        if terminated or truncated:
            break
finally:
    env.close()
```

Every replacement is checked against the original safe-action mask **before**
the wrapped environment is stepped.

`DeterministicLTLShield` remains a compatibility alias for
`PostposedLTLShield`, including the default lowest-index replacement behavior.

## Replacement strategies

Replacement strategies live in

```text
masa/deterministic_shield/replacement_strategies.py
```

and are also exported from `masa.deterministic_shield`.

| Configuration | Behavior on an unsafe proposal |
|---|---|
| `replacement=None` | Uses the precomputed lowest-index safe action. |
| `replacement=random_safe(seed=23)` | Samples uniformly among the current safe actions. |
| `replacement=highest_score(score_fn)` | Chooses the safe action with the highest score. Ties use the lowest index. |
| `replacement=custom_callback` | Calls `custom_callback(product_obs, proposal, safe_mask)` and validates the result before execution. |

### Random safe replacement

`random_safe(seed)` owns its own NumPy random generator.

```python
replacement = random_safe(seed=42)
```

Important details:

- `env.reset(seed=...)` does **not** seed or rewind the replacement generator;
- the same replacement seed and the same sequence of intervention masks reproduce
  the same replacement sequence;
- construct a fresh `random_safe(seed)` to restart the replacement sequence;
- use a separate replacement callback for each environment;
- safe proposals do not consume replacement random numbers.

Random replacement changes only the selection **among already-safe actions**. It
does not introduce a non-zero acceptable probability of violating the property.

### Highest-score replacement

`highest_score(score_fn)` can use Q-values, policy logits, or another priority
vector.

```python
replacement = highest_score(lambda product_obs: q_values[product_obs])
```

`score_fn` must return one score per action. Scores for safe actions must be
finite. Unsafe entries are ignored.

The callback is invoked only when replacement is required.

### Custom replacement

A custom callback has signature

```python
replacement(product_observation, proposed_action, safe_mask) -> action
```

The mask passed to the callback is a copy. The returned action is validated against
the shield's internal safe set.

A custom callback should not step, reset, or otherwise mutate the wrapped
environment.

## Public interface

Both wrappers expose the same synthesized safety information:

| Attribute or method | Meaning |
|---|---|
| `winning_region` | Read-only Boolean array of shape `(n_dfa * n_states,)`. |
| `safe_actions` | Read-only Boolean array of shape `(n_dfa * n_states, n_actions)`. |
| `action_masks()` | Copy of the safe-action row for the current product state. |
| `reset(...)` | Resets the environment and validates the initial product state. |
| `step(action)` | Executes the preemptive or postposed shielding behavior. |

After a successful step, `info` contains:

```text
action_mask
shield_intervened
shield_proposed_action
shield_executed_action
```

For the preemptive wrapper, the proposed and executed actions are identical and
`shield_intervened` is false.

## Episode boundaries

After true termination, the returned `action_mask` is all false.

After truncation without termination, the returned mask still describes the final
observation. This can be useful for bootstrapping a masked value target.

In both cases, another call to `step()` or `action_masks()` requires `reset()`.

If an outer auto-reset wrapper replaces the final observation, use that wrapper's
final-transition interface rather than combining a reset observation with the
previous episode's mask.

## Postposed shielding and learning semantics

For postposed learning it is important to distinguish the policy's proposal from
the action actually executed by the underlying environment.

If the **shielded wrapper itself** is the learner's environment, the proposal is
the learner's action and replacement is part of the environment dynamics.

If instead you are modelling the **underlying unshielded environment**, the
executed action is the relevant action.

Do not overwrite an on-policy sampled action with the replacement while keeping
the log-probability of the original proposal.

## Failure modes and limits

### Losing initial state

If reset produces a product state outside the winning region, the shield raises
`RuntimeError`.

There is no "least unsafe" fallback.

### Model mismatch

At runtime, the shield checks whether the observed product transition is present
in the modeled support of the executed action.

This check occurs **after** the real environment step. It can diagnose model
disagreement, but cannot undo a transition that has already happened.

The safety guarantee therefore requires the modeled support to cover all real
successors.

### Fixed dynamics and labels

The transition model, deterministic labelling function, and DFA must remain fixed
after shield synthesis.

Reconstruct the shield if any of them changes.

### Infinite-horizon safety

The fixed-point solver reasons about infinite continuation.

It does not solve safety only until a time limit, and it does not implement
finite-trace LTL semantics.

Terminal states therefore need explicit modeled dynamics, normally absorbing.

### Safety only

Remaining in the winning region does not imply:

- reward optimality;
- eventual goal completion;
- food collection;
- fairness;
- satisfaction of arbitrary liveness properties;
- general LTL synthesis.

This component is specifically a **safety-game shield** for bad-prefix properties.

## Model representation and cost

`read_support()` prefers MASA's successor dictionaries when available.

Otherwise it reads a dense transition kernel using

```text
P[next_state, current_state, action]
```

Synthesis uses whether each transition probability is positive. There is no
risk threshold.

Internally, if

- $S$ is the number of base states,
- $A$ is the number of actions,
- $Q$ is the number of DFA states,
- $K$ is the maximum successor-row width,

then the padded support has shape

```text
(S, A, K)
```

and product targets have shape

```text
(Q, S, K)
```

There is no dense product transition tensor.

The current vectorized fixed-point loop has operation count on the order of

$$
O(IQSAK)
$$

for $I$ fixed-point iterations.

Synthesis happens when the wrapper is constructed, not on every step.

## Mini PacMan tutorial

The companion tutorial notebook uses

$$
\mathbf{G}\neg \mathit{ghost}
$$

where `ghost` is Mini PacMan's existing collision label.

It compares:

- mask-aware preemptive action selection;
- postposed random replacement.

It is deliberately a small shielding-interface example rather than a training
benchmark.

[Open the Mini PacMan deterministic-shielding notebook](https://github.com/nightly/MASA-Safe-RL/blob/main/tutorial/05_minipacman_deterministic_shielding.ipynb)

## API reference

::: masa.deterministic_shield.preemptive.PreemptiveLTLShield
    options:
      members: ["reset", "step", "action_masks", "close"]
      inherited_members: true
      show_if_no_docstring: true

::: masa.deterministic_shield.postposed.PostposedLTLShield
    options:
      members: ["reset", "step", "action_masks", "close"]
      inherited_members: true
      show_if_no_docstring: true

::: masa.deterministic_shield.replacement_strategies.random_safe
    options:
      members: false

::: masa.deterministic_shield.replacement_strategies.highest_score
    options:
      members: false

::: masa.deterministic_shield.support.read_support
    options:
      members: false

::: masa.deterministic_shield.winning_region.winning_region
    options:
      members: false
