# Linear Temporal Logic (LTL) Safety Constraint

## Monitor API

```{eval-rst}
.. autoclass:: masa.common.constraints.ltl_safety.LTLSafety
    :members:
    :show-inheritance:

.. autoclass:: masa.common.constraints.ltl_safety.LTLSafetyEnv
    :members:
    :show-inheritance:
```

## Helpers

```{eval-rst}
.. autofunction:: masa.common.constraints.ltl_safety.create_product_transition_matrix
.. autofunction:: masa.common.constraints.ltl_safety.create_product_successor_states_and_probabilities
.. autofunction:: masa.common.constraints.ltl_safety.create_product_label_fn
```


## Monitoring versus enforcement

`LTLSafetyEnv` monitors a safety DFA and augments the observation with the DFA
state.

It does **not** itself restrict the agent's actions.

In MASA's safety interface, accepting DFA states recognize **bad prefixes**:
entering an accepting state means that the safety property has been violated,
rather than that a task has been successfully completed.

For enforcement, wrap

```python
LTLSafetyEnv(..., obs_type="discrete")
```

with either:

```python
PreemptiveLTLShield(...)
```

or:

```python
PostposedLTLShield(...)
```

Both wrappers use the same winning region.

- **Preemptive shielding** exposes a safe-action mask before the policy chooses.
- **Postposed shielding** checks a proposed action and replaces it only when the
  proposal is unsafe.

See
[Winning-region safety-game shielding](../../Shielding/Deterministic/Winning-Region%20Shielding.md)
for the safety-game construction, guarantee assumptions, examples, replacement
strategies, and API reference.

## Monitor timing

The live LTL monitor consumes the initial state's labels during `reset()`.

After each environment transition, it consumes the labels of the newly returned
state.

Therefore, if the current product state is $(q,s)$, the next monitor state after
a physical transition to $s'$ is

$$
q' = \delta(q,L(s')).
$$

The winning-region shield follows this timing when constructing its product
successors.

When using the product-construction helpers below independently, check their
documented label timing rather than assuming that every product representation
uses the same convention.