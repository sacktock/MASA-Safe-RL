Cost Functions and Constraints
================================

We describe how **safety costs** are defined and computed in MASA-Safe-RL, the API for **constraints**, and the conventions used to map **sets of atomic predicates to scalar costs**. Constraints provide a uniform, extensible mechanism for expressing safety objectives, violations, and logging metrics.

Cost Functions
--------------

## 1. Cost Function API

### Type Signature

A **cost function** is defined as:

```python
CostFn = Callable[Iterable[str], float]
```

Formally, a cost function is a mapping:

[
c : 2^{\mathcal{AP}} \rightarrow \mathbb{R}
]

where:

* (\mathcal{AP}) is the set of atomic predicates.
* The input is the **set of labels** satisfied at the current step.
* The output is a **scalar cost**.

### Semantics

Given a set of labels (L(s) \subseteq \mathcal{AP}), the cost function returns the instantaneous cost incurred at that step.

Typical interpretations:

* `0.0` -> no safety violation.
* `> 0.0` -> degree of violation or risk.
* `>0.5` -> are treated as safety violations by convention for PCTL and ProbabilisticSafety.
* Binary costs (`0` / `1`) are common but not required.

### Example

```python
def cost_fn(labels):
    return 1.0 if "unsafe" in labels else 0.0
```

More expressive costs are also supported:

```python
def cost_fn(labels):
    cost = 0.0
    if "collision" in labels:
        cost += 10.0
    if "near_obstacle" in labels:
        cost += 0.1
    return cost
```

## 2. Convention: Labels -> Float (Cost)

MASA follows a strict convention:

> **Costs are computed solely from the current set of atomic predicates.**

This ensures:

* Clear semantics for safety objectives.
* Compatibility with abstractions, automata, and shielding.
* No dependence on hidden environment state.

#### Summary

| Input                    | Output                              |
| ------------------------ | ----------------------------------- |
| `Iterable[str]` (labels) | `float` (cost)                      |
| Empty label set          | Valid input                         |
| Stateless cost function  | Recommended                         |
| Stateful cost function   | Allowed via constraints (e.g., DFA) |

Constraints
-----------

## 3. Constraints: Conceptual Overview

While a cost function maps **labels -> cost**, a **constraint** may:

* Accumulate cost over time.
* Track violations or satisfaction.
* Maintain internal state (e.g. DFA state, counters).
* Expose metrics for logging and evaluation.

Constraints are implemented as **Gymnasium wrappers** around a `LabelledEnv`.

## 4. Constraint Protocol

All constraints must conform to the following protocol:

```python
class Constraint(Protocol):

    def reset(self):
        pass

    def update(self, labels: Iterable[str]):
        pass

    @property
    def constraint_type(self) -> str:
        ...

    def constraint_step_metrics(self) -> Dict[str, float]:
        ...

    def constraint_episode_metrics(self) -> Dict[str, float]:
        ...
```

### Required Methods

#### `reset()`

* Called automatically on **environment reset**.
* Must reset all internal state.
* Enables **stateful constraints**.

#### `update(labels)`

* Called at **every step**.
* Receives the current set of atomic predicates.
* Updates internal counters, DFA states, cumulative cost, etc.

### Required Properties

#### `constraint_type: str`

* A **stable identifier** for the constraint.
* Used for logging, evaluation, and serialization.
* Example: `"cmdp"`, `"ltl_dfa"`, `"pctl"`.

## 5. Step-Level Metrics

### `constraint_step_metrics()`

Returns metrics that are meaningful **at any step**:

```python
Dict[str, float]
```

Typical examples:

* Running cumulative cost.
* Boolean violation flags (encoded as `0.0 / 1.0`).
* DFA acceptance indicators.
* Risk estimates or belief values.

**Requirements:**

* Must be cheap to compute.
* Must not modify internal state.
* Safe to call multiple times per step.

## 6. Episode-Level Metrics

### `constraint_episode_metrics()`

Returns metrics summarising the **entire episode**, typically used for logging and evaluation.

Examples:

* Total accumulated cost.
* Whether a violation ever occurred.
* Episode satisfaction probability.
* Terminal DFA state indicators.

This method is expected to be called at:

* Episode termination.
* Episode truncation.

## 7. BaseConstraintEnv Wrapper

MASA provides a common wrapper for constraints:

```python
BaseConstraintEnv(env, constraint)
```

### Key Guarantees

* The wrapped environment **must** be a `LabelledEnv` .
* The constraint is:

  * Reset on `env.reset()` .
  * Updated on every `env.step()` .
* Labels are validated to be sets of atomic predicates.

### Execution Flow

1. `env.reset()`

   * Environment reset
   * `constraint.reset()`
   * `constraint.update(labels)`

2. `env.step(action)`

   * Environment transition
   * `constraint.update(labels)`

---

## 8. Stateful Constraints

Constraints **may be stateful**.

This is intentional and supported.

Examples:

* DFA-based LTL constraints.
* Budget or energy constraints.
* Counters (e.g. number of unsafe steps).
* Probabilistic belief updates.

**Important rule:**

> All constraint state must be fully reset in `reset()`.

This ensures correctness across episodes, rollouts, and evaluation runs.

## 9. Defining Your Own Constraint

### Minimal Example

```python
class UnsafeStepConstraint:

    def __init__(self):
        self.cum_cost = 0.0

    def reset(self):
        self.cum_cost = 0.0

    def update(self, labels):
        if "unsafe" in labels:
            self.cum_cost += 1.0

    @property
    def constraint_type(self):
        return "unsafe_step"

    def constraint_step_metrics(self):
        return {"cum_cost": self.cum_cost}

    def constraint_episode_metrics(self):
        return {"total_cost": self.cum_cost}
```

To apply it:

```python
env = LabelledEnv(env, label_fn)
env = BaseConstraintEnv(env, UnsafeStepConstraint())
```