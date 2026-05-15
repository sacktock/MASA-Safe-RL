# Shielding

These tutorials show how MASA shielding wrappers turn safety analysis into environment interfaces that reinforcement-learning algorithms can consume.

- [Probabilistic Shielding MiniPacman](Shielding/Probabilistic%20Shielding%20MiniPacman) builds a PCTL-constrained MiniPacman environment, wraps it with `ProbShieldWrapperDisc`, and inspects safety bounds, successor dynamics, and projected safe actions.
- [Safety Abstractions Pacman Coins](Shielding/Safety%20Abstractions%20Pacman%20Coins) shows why structured Pacman coin observations need a discrete safety abstraction before probabilistic shielding can compute safety bounds.

Runnable notebooks:

- [notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb](../../notebooks/tutorials/09_probabilistic_shielding_minipacman.ipynb)
- [notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb](../../notebooks/tutorials/10_safety_abstractions_pacman_coins.ipynb)

```{toctree}
:maxdepth: 1
:hidden:

Shielding/Probabilistic Shielding MiniPacman
Shielding/Safety Abstractions Pacman Coins
```
