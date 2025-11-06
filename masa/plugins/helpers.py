from importlib import import_module
import pkgutil

def load_plugins(plugin_pkg: str = "masa.plugins"):
    """
    Dynamically import all modules inside masa.plugins/
    so they can self-register to the registries.
    """
    package = import_module(plugin_pkg)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if not ispkg:
            import_module(name)