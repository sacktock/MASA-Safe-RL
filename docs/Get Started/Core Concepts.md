# Core Concepts

This page introduces the **core ideas** behind reinforcement learning (RL), safe RL, and how **MASA** builds a *canonical, modular interface* for handling safety constraints - ranging from simple cost budgets to temporal logic specifications.

## Reinforcement Learning: the Big Picture

At its core, **reinforcement learning** studies how an *agent* learns to make decisions by interacting with an *environment*.

### The Agent-Environment Loop

The interaction proceeds in discrete time steps:
```{eval-rst}
1. The environment provides an **observation** :math:`o_t` describing the current situation.
2. The agent chooses an **action** :math:`a_t` based on that observation.
3. The environment transitions to a new state and returns:

   * a **reward** :math:`r_t`, and
   * the next observation :math:`o_{t+1}`.

The agent's goal is to learn a *policy* :math:`\pi(a \mid o)` that maximises expected cumulative reward over time.
```

Informally:

> **Observations tell you what is happening,
> actions let you influence what happens next,
> rewards tell you how well you're doing.**

## The Gymnasium API

Most modern RL code is built around the **Gymnasium** API, which standardises how agents interact with environments.

### Core Methods

A Gymnasium environment exposes two key methods:

* `reset()`
  Starts a new episode and returns an initial observation.
* `step(action)`
  Advances the environment by one step.

### Minimal Example

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # random agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Key points:

* **Observations** are typically arrays, dicts, or tuples.
* **Actions** live in an `action_space`.
* Episodes end when `terminated` (task finished/failure) or `truncated` (time limit).

MASA is designed to *layer safety concepts on top of this exact interaction pattern*, without changing how agents are written.

## Why Safe Reinforcement Learning?

In many applications, **maximising reward is not enough**.

Examples:

* A robot must *never* collide with humans.
* A controller must keep energy consumption below a budget.
* A system must avoid unsafe states *at all times*, not just on average.

**Safe reinforcement learning** augments the RL objective with **constraints**.

## Two Common Safety Paradigms

### 1. Constrained MDPs (CMDPs): Cost Budgets

In a **CMDP**, the agent optimises reward while keeping expected *cost* below a budget:

```{eval-rst}
.. math::

    \mathbb{E}_\pi \left[ \sum^{T-1}_{t=0} c_t \right] \leq B
```

Example:

```python
reward = +1.0          # task progress
cost   = 0.1           # energy usage
```

Here:

* Costs accumulate over time.
* Violations are *soft*: occasional violations may be acceptable if the budget holds.

This is well-suited for:

* Resource limits
* Risk-sensitive control
* Average safety constraints

### 2. Absolute Safety: Unsafe States

Some settings demand **zero tolerance** for failure.

Example:

* Entering an unsafe region ends the episode immediately.
* Any policy that reaches such a state is unacceptable.

```python
if "unsafe" in labels:
    terminated = True
```

Characteristics:

* Binary notion of safety.
* Violations are *hard* and irreversible.
* Often modeled via absorbing failure states.

## MASA's Perspective on Safety

MASA does **not** enforce a single notion of safety.

Instead, it provides a **canonical interface** that allows *many* safety formalisms to coexist.

### Core Design Goals

MASA aims to:

1. **Decouple safety from environment dynamics**
2. **Standardise safety signals across methods**
3. **Support both simple and expressive constraints**
4. **Integrate cleanly with Gymnasium and RL libraries**

## The MASA Abstraction Stack

MASA structures safety around three core ideas:

### 1. Labelling Functions

```text
observation -> set of atomic propositions
```

A labelling function extracts *semantic predicates* from observations, such as:

```python
{"near_obstacle", "speeding"}
```

These labels form the **interface between raw observations and safety logic**.

---

### 2. Cost Functions

```text
labels -> scalar cost
```

Cost functions convert semantic events into numeric signals:

* step-wise costs
* shaped penalties
* violation indicators

This unifies CMDPs, penalties, and safety metrics under one API.

### 3. Constraints and Monitors

Constraints are **stateful objects** that:

* update on each step using labels,
* track safety-relevant internal state,
* expose *step-level* and *episode-level* metrics.

MASA constraints can be:

* simple (running cost sums),
* automaton-based (Safety LTL),
* probabilistic (bounded PCTL).

### 4. Overview 

```{figure} ../_static/img/path355.svg
Overview of the abstraction stack used in MASA
```

## Temporal Logic in MASA

Beyond scalar costs, MASA supports **formal temporal specifications**, including:

* **Safety LTL**

  > "Something bad never happens."

* **Bounded PCTL**

  > "The probability of failure within `T` steps is at most `p`."

These specifications are compiled into **monitors** that:

* evolve alongside the environment,
* emit costs and metrics,
* integrate seamlessly with RL training loops.

## Next Steps

- [Labelling Function](Core%20Concepts/Labelling%20Function) - Learn how observation labelling is handled in MASA.
- [Cost Function](Core%20Concepts/Cost%20Function) - Understand the conventions used for cost functions in MASA.
- [Constraints](../Common/Constraints) - How are different constraints handled in MASA?
- [Wrappers](../Common/Wrappers) - How do environment Wrappers provide a convenient interface for managing constraints?

```{toctree}
:caption: Core Concepts
:hidden:

Core Concepts/Labelling Function
Core Concepts/Cost Function
```