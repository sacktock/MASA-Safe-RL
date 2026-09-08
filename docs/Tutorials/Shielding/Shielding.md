# Shielding

MASA provides tutorials for risk-budget-based probabilistic shielding and
winning-region safety-game shielding.

## Single-agent winning-region shielding

The
[winning-region safety-game shielding guide](../Algorithms/Shielded/Winning-Region%20Shielding.md)
explains support-based synthesis and the preemptive and postposed interfaces.

The Mini PacMan notebook uses the safety property

$$
\mathbf{G}\neg \mathit{ghost}
$$

and compares mask-aware preemptive action selection with postposed uniform-random
replacement of unsafe proposals.

[Open `tutorial/05_minipacman_deterministic_shielding.ipynb`](../../tutorial/05_minipacman_deterministic_shielding.ipynb)

## Multi-agent coalition shielding

The
[multi-agent coalition shielding guide](../Algorithms/Shielded/Multi-Agent%20Coalition%20Shielding.md)
explains robust coalition predecessors, centralised action selection, and
coordination-free decentralised masks.

- [`13_coalition_safety_game_shielding.ipynb`](../../notebooks/tutorials/13_coalition_safety_game_shielding.ipynb)
  shields a focal coalition against every legal action of its complement.
- [`14_decentralised_coalition_shielding.ipynb`](../../notebooks/tutorials/14_decentralised_coalition_shielding.ipynb)
  shows why naïve marginal masks are unsafe and constructs a certified Cartesian
  action interface for independent simultaneous execution.

Both use `ChickenMatrix`, now a `TabularParallelEnv`, wrapped by the existing
`LabelledParallelEnv`.

## Probabilistic shielding tutorials

These tutorials show how MASA's probabilistic shielding wrappers turn risk-bound
analysis into environment interfaces that reinforcement-learning algorithms can
consume.

- [Probabilistic Shielding MiniPacman](Shielding/Probabilistic%20Shielding%20MiniPacman)
  builds a PCTL-constrained MiniPacman environment, wraps it with
  `ProbShieldWrapperDisc`, and inspects safety bounds, successor dynamics, and
  projected safe actions.
- [Safety Abstractions Pacman Coins](Shielding/Safety%20Abstractions%20Pacman%20Coins)
  shows why structured Pacman coin observations need a discrete safety abstraction
  before probabilistic shielding can compute safety bounds.
- [FrozenLake Shielding](Shielding/FrozenLake%20Shielding)
  shows how the same discrete shield-synthesis path applies to Gymnasium-style
  FrozenLake dynamics.

Runnable notebooks:

- [`notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb`](../../notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb)
- [`notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb`](../../notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb)

```{toctree}
:maxdepth: 1
:hidden:

Shielding/Probabilistic Shielding MiniPacman
Shielding/Safety Abstractions Pacman Coins
Shielding/FrozenLake Shielding
```
