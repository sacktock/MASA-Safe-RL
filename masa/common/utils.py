from typing import Callable, Dict, Union
import importlib

Factory = Union[Callable, str]

class Registry:
    def __init__(self):
        self._items: Dict[str, Callable] = {}


    def register(self, name: str, ctor: Callable):
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


ENV_REGISTRY = Registry()
ALGO_REGISTRY = Registry()
CONSTRAINT_REGISTRY = Registry()