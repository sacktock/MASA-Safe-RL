Labelling Functions and Labelled MDPs
=====================================

We describe the **labelling function API** and underlying **labelled Markov Decision Process (MDP)** formalism, and the conventions used to map environment observations to **sets of atomic predicates**. These components are foundational for safety objectives, probabilistic shielding, and temporal logic specifications (e.g. safety-LTL).

## 1. Atomic Predicates and Labels

We assume a fixed (finite) set of **atomic predicates**:

[
\mathcal{AP} = {p_1, p_2, \dots, p_k}
]

An atomic predicate represents a Boolean property of the environment state, such as:

* `"unsafe"`
* `"goal"`
* `"collision"`
* `"near_obstacle"`

At runtime, an environment state satisfies **zero or more** atomic predicates simultaneously.

## 2. Labelling Function API

### Type Signature

The labelling function has the following type:

```python
LabelFn = Callable[[Any], Iterable[str]]
```

Formally, the labelling function is a map:

[
L : \mathcal{O} \rightarrow 2^{\mathcal{AP}}
]

where:

* (\mathcal{O}) is the observation space of the environment.
* (2^{\mathcal{AP}}) is the power set of atomic predicates.

### Semantics

Given an observation `obs`, the labelling function returns the **set of atomic predicates that hold** in that observation.

**Key requirements:**

* The output must be **iterable** (e.g. `list`, `tuple`, `set`).
* Elements must be **strings**, each corresponding to an atomic predicate.
* The returned collection is interpreted as a **set** (duplicates are ignored).

### Example

```python
def label_fn(obs):
    labels = set()

    if obs["x"] < 0:
        labels.add("unsafe")

    if obs["goal_reached"]:
        labels.add("goal")

    return labels
```

## 3. Observation -> Labels Convention

The MASA convention is:

> **Labels are computed from observations, not from internal environment states.**

This ensures:

* Compatibility with partially observable environments
* Consistent behaviour under wrappers and abstractions
* Clear semantics when composing with automata or abstractions

### Convention Summary

| Concept                 | Convention                                               |
| ----------------------- | -------------------------------------------------------- |
| Input to label function | Raw observation returned by `env.reset()` / `env.step()` |
| Output                  | Set of atomic predicate strings                          |
| Empty output            | Valid (no predicates satisfied)                          |
| Determinism             | Strongly recommended                                     |

## 4. Labelled Environment Wrapper

To standardise access to labels, MASA provides a `LabelledEnv` wrapper.

### Wrapper Semantics

The wrapper:

* Attaches a labelling function to the environment.
* Evaluates it on **every reset and step**.
* Injects the resulting label set into the `info` dictionary under the key `"labels"`.

```python
info["labels"] = set(label_fn(obs))
```

### API

```python
env = LabelledEnv(env, label_fn)
```

After wrapping:

```python
obs, info = env.reset()
labels = info["labels"]

obs, reward, terminated, truncated, info = env.step(action)
labels = info["labels"]
```

## 5. Common Pitfalls

| Pitfall                     | Recommendation                             |
| --------------------------- | ------------------------------------------ |
| Returning non-strings       | Always return strings                      |
| Using environment internals | Derive labels from observations            |
| Stateful label functions    | Prefer pure, stateless functions           |
| Inconsistent predicates     | Define and document (\mathcal{AP}) clearly |


