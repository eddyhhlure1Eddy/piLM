"""Runtime memory maintenance helpers."""
from __future__ import annotations

import ctypes
import gc
import os


def trim_process_working_set() -> bool:
    """Ask the OS to reclaim unused pages from this process working set.

    This does not reduce live tensor memory. It only makes pages freed by Python
    and native allocators available to the OS sooner, which matters after
    replacing large BF16 Linear weights with int8 buffers.
    """
    gc.collect()
    if os.name != "nt":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        return bool(psapi.EmptyWorkingSet(handle))
    except Exception:
        return False
