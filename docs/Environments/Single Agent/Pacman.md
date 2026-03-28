# Pacman

MASA provides four Pacman-style single-agent environments:

- `mini_pacman`
- `pacman`
- `mini_pacman_with_coins`
- `pacman_with_coins`

All four environments model a maze with one controllable agent and one stochastic ghost. They share the same discrete action space and
the same default safety signal, but they differ in observation representation and reward structure.

They also differ in how they expose dynamics for exact methods:

- `mini_pacman` and `mini_pacman_with_coins` provide a full transition matrix.
- `pacman` and `pacman_with_coins` provide successor states and per-action transition probabilities.

## Shared Mechanics

### Actions

All Pacman variants use `Discrete(5)` actions:

- `0`: left
- `1`: right
- `2`: down
- `3`: up
- `4`: stay

The state also tracks the facing direction of both the agent and the ghost, so transitions depend on direction as well as position.

### Ghost Dynamics

The ghost is stochastic and biased toward the agent. The implementation computes a preferred move that points most directly toward the
agent and mixes it with random movement using `ghost_rand_prob = 0.6`.

This produces adversarial but not deterministic pursuit, which is useful for safe RL and probabilistic safety constraints.

### Safety Signal

All Pacman variants use the same default cost convention:

- a ghost collision is labelled `"ghost"`,
- `cost_fn(labels)` returns `1.0` on `"ghost"` and `0.0` otherwise.

Importantly, a ghost collision is a **cost event**, not an automatic episode termination condition.

## Tabular Food-Collection Variants

`mini_pacman` and `pacman` expose the environment state as a single discrete state index.

### Observation and State

The discrete state encodes:

- agent position,
- agent direction,
- ghost position,
- ghost direction,
- a binary food flag.

The food flag indicates whether the unique food item is still available.

### Rewards and Labels

The default `label_fn` returns:

- `{"food"}` when the agent is on the food cell, the food is still present, and the ghost is not on the same cell,
- `{"ghost"}` when the agent and ghost occupy the same cell,
- the empty set otherwise.

Reward is sparse:

- `1.0` when the agent collects the food,
- `0.0` otherwise.

### Termination

Episodes terminate only when the agent reaches the environment's fixed terminal location. They do not terminate on food collection or
ghost collision.

### Variant Sizes

- `mini_pacman` uses a compact `7 x 10` maze and a discrete state space of size `9248`.
- `pacman` uses a larger `15 x 19` maze and a discrete state space of size `262088`.

The larger `pacman` environment is intended as a harder version of the same basic problem, with a substantially larger tabular state
space.

## Coin-Collection Variants

`mini_pacman_with_coins` and `pacman_with_coins` keep the same maze-and-ghost dynamics but change both the observation structure and
the reward.

### Observation Tensor

The observation is a tensor of shape `(H, W, 9)`:

- channel `0`: remaining coins,
- channels `1:5`: one-hot agent direction at the agent position,
- channels `5:9`: one-hot ghost direction at the ghost position.

This gives a spatial observation while keeping the action space discrete.

Concretely, the observation shapes are:

- `mini_pacman_with_coins`: `(7, 10, 9)`
- `pacman_with_coins`: `(15, 19, 9)`

### Rewards

At reset, the coin array is initialized to `1.0` everywhere. In practice, only traversable cells can ever pay out reward. On each
step:

- the agent receives reward equal to the current coin value at its cell,
- the coin at that cell is then removed.

So each cell can contribute reward at most once per episode.

### Labels and Costs

The default labelling function for the coin variants only marks ghost collisions:

- `{"ghost"}` if the agent and ghost share a cell,
- the empty set otherwise.

There is no default label for coin collection.

### Termination

As in the food-based variants, the episode terminates when the agent reaches a fixed terminal location, not when coins are exhausted.

## Safety Abstraction Support

The coin variants also expose helper functions for probabilistic shielding:

- `safety_abstraction(obs)`
- `abstr_label_fn(abs_state)`

`safety_abstraction(obs)` maps the structured observation back to a discrete abstract state. `abstr_label_fn(abs_state)` then labels
that abstract state using the same ghost-collision safety semantics while discarding reward-specific coin information.

This is the setup used in MASA's shielding example for `pacman_with_coins`.
