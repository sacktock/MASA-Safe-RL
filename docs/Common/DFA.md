---
title: DFA
---
# Propositional Formula

```{eval-rst}
.. automodule:: masa.common.ltl
   :members:
   :exclude-members: DFA, DFACostFn, dfa_to_costfn, ShapedCostFn
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

# DFA

```{eval-rst}
.. autoclass:: masa.common.ltl.DFA
```

## Methods

```{eval-rst}
.. automethod:: masa.common.ltl.DFA.add_edge
.. automethod:: masa.common.ltl.DFA.reset
.. automethod:: masa.common.ltl.DFA.has_edge
.. automethod:: masa.common.ltl.DFA.check
.. automethod:: masa.common.ltl.DFA.transition
.. automethod:: masa.common.ltl.DFA.step
.. automethod:: masa.common.ltl.DFA.num_automaton_states
.. automethod:: masa.common.ltl.DFA.automaton_state
```

# Cost Function as a DFA

```{eval-rst}
.. autoclass:: masa.common.ltl.DFACostFn
```

```{eval-rst}
.. automethod:: masa.common.ltl.DFACostFn.__init__
```

```{eval-rst}
.. automethod:: masa.common.ltl.dfa_to_costfn
```

# Shaped Cost Function

```{eval-rst}
.. autoclass:: masa.common.ltl.ShapedCostFn
```

```{eval-rst}
.. automethod:: masa.common.ltl.ShapedCostFn.__init__
```

