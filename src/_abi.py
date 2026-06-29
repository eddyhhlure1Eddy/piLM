"""ABI bindings to libecpu C library."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from .abi.bindings import *  # noqa
except ImportError:
    from abi.bindings import *  # noqa
