from __future__ import annotations
from typing import Callable, Dict, Union, Iterator, Optional
import importlib
import inspect

Factory = Union[Callable, str]

class Registry:
    def __init__(self):
        self._items: Dict[str, Callable] = {}

    def register_module(self, name_or_ctor: Optional[Union[str, Callable]] = None) -> Callable:
        if inspect.isclass(name_or_ctor):
            return self._register_module(name_or_ctor)
        return self._register_module(name=name_or_ctor)

    def _register_module(self, ctor: Optional[Callable] = None, *, name: Optional[str] = None):
        def decorator(cls: Callable) -> Callable:
            if not inspect.isclass(cls):
                raise TypeError(f"module must be a class. Got {type(cls)} instead.")
            key = name or ctor.__name__
            if key in self._items:
                raise KeyError(f"{key} already registered")
            self._items[key] = ctor
            return cls
        
        if ctor is None:
            return decorator

        return decorator(ctor)

    def register(self, name: str, ctor: Callable) -> Callable:
        if name in self._items:
            raise KeyError(f"{name} already registered")
        self._items[name] = ctor
        return ctor

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"{name} not found. Available: {list(self._items)}")
        ctor = self._items[name]
        if isinstance(ctor, str):
            mod, obj = ctor.split(":", 1)
            ctor = getattr(importlib.import_module(mod), obj)
            self._items[name] = ctor
        return ctor

    def keys(self) -> list[str]:
        return list([str(key) for key in self._items.keys()])

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)


ENV_REGISTRY = Registry()
MARL_ENV_REGISTRY = Registry()
ALGO_REGISTRY = Registry()
CONSTRAINT_REGISTRY = Registry()
MARL_CONSTRAINT_REGISTRY = Registry()

get_env = ENV_REGISTRY.get
get_marl_env = MARL_ENV_REGISTRY.get
get_algorithm = ALGO_REGISTRY.get
get_constraint = CONSTRAINT_REGISTRY.get
get_marl_constraint = MARL_CONSTRAINT_REGISTRY.get

# class decorators
register_env = ENV_REGISTRY.register_module
register_marl_env = MARL_ENV_REGISTRY.register_module
register_algorithm = ALGO_REGISTRY.register_module
register_constraint = CONSTRAINT_REGISTRY.register_module
register_marl_constraint = MARL_CONSTRAINT_REGISTRY.register_module