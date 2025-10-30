from typing import Callable, Dict


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
        return self._items[name]


    def list(self):
        return sorted(self._items.keys())

ENV_REGISTRY = Registry()
ALGO_REGISTRY = Registry()
CONSTRAINT_REGISTRY = Registry()