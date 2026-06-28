"""piLM: Ecpu+Eram collaborative inference engine."""
from . import _abi as abi
from .runtime.device import registry

__version__ = abi.version() if _lib_found() else "0.1.0"
__project__ = "piLM"

def _lib_found():
    try:
        abi._load_lib()
        return True
    except Exception:
        return False

__all__ = ["abi", "registry", "__version__", "__project__"]