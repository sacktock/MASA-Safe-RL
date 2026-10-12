"""Backward-compatible import for the original postposed shield."""
from .postposed import PostposedLTLShield

DeterministicLTLShield = PostposedLTLShield

__all__ = ["DeterministicLTLShield"]
