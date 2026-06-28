"""ABI bindings to libecpu C library."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abi.bindings import *  # noqa