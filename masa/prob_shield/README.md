# Probabilistic Shielding

This module provides a **Gymnasium-compatible implementation of Probabilistic Shielding** for Safe Reinforcement Learning, based on the state-augmentation framework introduced in:

> **Probabilistic Shielding for Safe Reinforcement Learning**
> Edwin Hamel-De le Court, Francesco Belardinelli, Alexander W. Goodall
> arXiv: [https://arxiv.org/abs/2503.07671](https://arxiv.org/abs/2503.07671) 

The approach guarantees **probabilistic safety during both training and evaluation**, while remaining **optimality-preserving** among all safe policies.

For technical details please see the [docs](https://sacktock.github.io/MASA-Safe-RL/Misc/Probabilistic%20Shielding.html) or the full paper on [arXiv](https://arxiv.org/abs/2503.07671)

## Usage

### 1. Basic Probabilistic Shielding (Discrete MDP, PCTL)

For environments with discrete state spaces and PCTL safety constraints:

```python
env = make_env("pacman", "pctl", 1000, label_fn=label_fn, cost_fn=cost_fn, alpha=0.01)

env = ProbShieldWrapperDisc(
    env,
    init_safety_bound=0.01,
    theta=1e-15,
    max_vi_steps=10_000,
    granularity=20,
)
```

See the full example in
`prob_shield_example.py` .

### 2. Probabilistic Shielding with a Safety Abstraction

For large or combinatorial environments, you can provide a **discrete safety abstraction** that preserves only safety-relevant dynamics.

```python
env = ProbShieldWrapperDisc(
    env,
    label_fn=abstr_label_fn,
    cost_fn=cost_fn,
    safety_abstraction=safety_abstraction,
    init_safety_bound=0.01,
)
```

This enables scalable safety verification even when the full state space is very large.

See
`prob_shield_safety_abstraction_example.py` .

### 3. Probabilistic Shielding for Safety-LTL (DFA-MDP Product)

Safety properties expressed in **Safety-LTL** are handled by constructing the **product MDP with a DFA** internally.

```python
env = make_env(
    "colour_bomb_grid_world_v2",
    "ltl_dfa",
    250,
    label_fn=label_fn,
    dfa=make_dfa(),
)

env = ProbShieldWrapperDisc(env, init_safety_bound=0.01)
```

The shield is built over the **DFA-MDP product**, ensuring probabilistic satisfaction of the LTL safety property.

See
`prob_shield_ltl_example.py` .

## When to Use

Use Probabilistic Shielding when:

* Safety is **non-negotiable**.
* Constraints are **probabilistic**, not expected-cost based.
* You want **formal guarantees**, not penalties or Lagrangians.
* The safety dynamics (or a conservative abstraction) are known.

## Citation

If you use this implementation, please cite:

```
@article{hamel2025probabilistic,
  title={Probabilistic Shielding for Safe Reinforcement Learning},
  author={Hamel-De le Court, Edwin and Belardinelli, Francesco and Goodall, Alexander W.},
  journal={arXiv preprint arXiv:2503.07671},
  year={2025}
}
```