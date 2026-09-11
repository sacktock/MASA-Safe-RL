# Shielding

MASA provides tutorials for both risk-budget-based probabilistic shielding and
winning-region safety-game shielding.

## Winning-region Mini PacMan example

The
[winning-region safety-game shielding guide](../Shielding/Deterministic/Winning-Region%20Shielding.md)
explains support-based synthesis and the preemptive and postposed interfaces.

The companion Mini PacMan notebook uses the safety property

$$
\mathbf{G}\neg \mathit{ghost}
$$

and compares:

- mask-aware preemptive action selection;
- postposed uniform-random replacement of unsafe proposals.

The notebook is a small shielding-interface demonstration rather than a training
benchmark or a guarantee of task completion.

[Open `tutorial/05_minipacman_deterministic_shielding.ipynb`](https://github.com/nightly/MASA-Safe-RL/blob/main/tutorial/05_minipacman_deterministic_shielding.ipynb)


It uses `replacement_strategies.random_safe`. The replacement random generator is
independent of the environment seed.

<!-- > **Memory note**
>
> The current Mini PacMan simulator constructs a large dense transition matrix on
> import. The notebook uses cached successor dictionaries for shield synthesis,
> but this does not remove the simulator's dense model allocation.
 -->

## Probabilistic shielding tutorials

These tutorials show how MASA's probabilistic shielding wrappers turn risk-bound
analysis into environment interfaces that reinforcement-learning algorithms can
consume.

- [Probabilistic Shielding MiniPacman](Shielding/Probabilistic%20Shielding%20MiniPacman.md)
  builds a PCTL-constrained MiniPacman environment, wraps it with
  `ProbShieldWrapperDisc`, and inspects safety bounds, successor dynamics, and
  projected safe actions.
- [Safety Abstractions Pacman Coins](Shielding/Safety%20Abstractions%20Pacman%20Coins.md)
  shows why structured Pacman coin observations need a discrete safety abstraction
  before probabilistic shielding can compute safety bounds.
- [FrozenLake Shielding](Shielding/FrozenLake%20Shielding.md)
  shows how the same discrete shield-synthesis path applies to Gymnasium-style
  FrozenLake dynamics.

Runnable notebooks:

- [`notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb`](https://github.com/nightly/MASA-Safe-RL/blob/main/notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb)
- [`notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb`](https://github.com/nightly/MASA-Safe-RL/blob/main/notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb)
