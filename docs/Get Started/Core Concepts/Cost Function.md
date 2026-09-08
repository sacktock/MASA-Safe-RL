# Cost Function

This page describes how **safety costs** are defined and computed in MASA-Safe-RL, and the
conventions used to map **sets of atomic predicates** to **scalar costs**.
Costs provide the basic quantitative signal used by constraints, probabilistic safety
objectives, and logic-based monitors.

## Cost Functions

### Cost Function API

A **cost function** is defined as:

```python
CostFn = Callable[Iterable[str], float]
```

Formally, a cost function is a mapping

$$
c : 2^{\mathcal{AP}} \rightarrow \mathbb{R},
$$

where:

- $\mathcal{AP}$ is the set of atomic predicates,
- the input is the **set of labels** satisfied at the current step,
- the output is a **scalar cost**.

### Semantics

Given a label set $L(s) \subseteq \mathcal{AP}$, the cost function returns the
**instantaneous safety cost** incurred at that step.

Typical interpretations include:

- ``0.0`` - no safety violation,
- ``> 0.0`` - degree of violation or risk,
- ``> 0.5`` - treated as a *violation* by convention in PCTL-based and
  probabilistic-safety components,
- binary costs (``0`` / ``1``) are common but not required.

The meaning of a cost is entirely user-defined, but must be **consistent across episodes**
and evaluation.

### Examples

Minimal binary cost:

```python
def cost_fn(labels):
    return 1.0 if "unsafe" in labels else 0.0
```

Graded cost:

```python
def cost_fn(labels):
    cost = 0.0
    if "collision" in labels:
        cost += 10.0
    if "near_obstacle" in labels:
        cost += 0.1
    return cost
```

## Labels-to-Cost Convention

MASA follows a strict convention:

!!! warning "Important"

    **Costs are computed solely from the current set of atomic predicates.**

This ensures:

- clear and compositional semantics,
- compatibility with abstractions, automata, and shielding,
- independence from hidden environment state.

**Labels to cost summary**

| Input | Output |
| --- | --- |
| `Iterable[str]` (labels) | `float` (cost) |
| Empty label set | Valid input |
| Stateless cost function | Recommended |
| Stateful costs | Supported via constraints (e.g., DFA) |

## Constraints (Conceptual Overview)

A cost function defines a **per-step signal**. A **constraint** builds on top of this to
reason over **trajectories** rather than individual steps.

Conceptually, constraints may:

- accumulate costs over time,
- detect violations or satisfaction events,
- maintain internal state (e.g. DFA states, counters),
- expose step-level and episode-level metrics for logging and evaluation.

In MASA, constraints are implemented as **Gymnasium wrappers** around a
[`masa.common.labelled_env.LabelledEnv`][masa.common.labelled_env.LabelledEnv], and are responsible for calling cost
functions, tracking state, and reporting metrics.

For full details on the constraint interface, lifecycle, and provided implementations,
see the API reference:

!!! info "See also"

    [Constraints API Reference](../../Common/Constraints.md)

## Summary

- Cost functions map **label sets** $L(s) \subseteq \mathcal{AP}$ to **scalar costs**.
- They are pure functions of labels, with no dependence on hidden state.
- Constraints build on cost functions to express trajectory-level safety objectives.
- Detailed constraint APIs and implementations are documented separately.

This separation keeps the cost function API minimal, explicit, and easy to compose with
the broader MASA safety framework.
