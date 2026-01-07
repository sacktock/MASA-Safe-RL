"""
Labelling function type definitions.

This module defines the canonical *labelling function* abstraction used
throughout MASA to map environment observations to sets of atomic
propositions.

A labelling function induces a **labelled Markov decision process (LMDP)**,
where each state (or observation) is augmented with a finite set of atomic
predicates that are assumed to hold at that state.
"""

from __future__ import annotations
from typing import Any, Iterable, Callable

LabelFn = Callable[[Any], Iterable[str]]
