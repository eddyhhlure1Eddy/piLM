import ctypes
import os
import sys
from pathlib import Path

_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    candidates = []
    here = Path(__file__).resolve().parent
    build_dir = here.parent / "ecpu" / "build"
    candidates.append(build_dir / "libecpu.dll")
    candidates.append(build_dir / "libecpu.so")
    candidates.append(build_dir / "libecpu.dylib")
    for c in candidates:
        if c.exists():
            _lib = ctypes.CDLL(str(c))
            _configure(_lib)
            return _lib
    raise RuntimeError(f"ecpu shared library not found in {build_dir}")

def _configure(lib):
    lib.ecpu_version.restype = ctypes.c_char_p
    lib.ecpu_last_error.restype = ctypes.c_char_p
    lib.ecpu_config_default.restype = EcpuConfig
    lib.ecpu_init.argtypes = [ctypes.POINTER(EcpuConfig)]
    lib.ecpu_init.restype = ctypes.c_int
    lib.ecpu_is_initialized.restype = ctypes.c_int
    lib.ekernel_detect_isa.restype = ctypes.c_int
    lib.ekernel_isa_name.argtypes = [ctypes.c_int]
    lib.ekernel_isa_name.restype = ctypes.c_char_p
    lib.edevice_active.restype = ctypes.POINTER(EdeviceVtable)

class EcpuConfig(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int),
        ("numa_policy", ctypes.c_int),
        ("disable_amx", ctypes.c_int),
        ("disable_avx512", ctypes.c_int),
        ("eram_budget_bytes", ctypes.c_size_t),
        ("kv_cache_budget_bytes", ctypes.c_size_t),
        ("device_type", ctypes.c_int),
    ]

class EdeviceVtable(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("id", ctypes.c_int),
        ("caps", ctypes.c_uint64),
        ("alloc", ctypes.c_void_p),
        ("free", ctypes.c_void_p),
        ("memcpy_h2d", ctypes.c_void_p),
        ("memcpy_d2h", ctypes.c_void_p),
        ("memcpy_d2d", ctypes.c_void_p),
        ("synchronize", ctypes.c_void_p),
        ("mem_total", ctypes.c_void_p),
        ("mem_free", ctypes.c_void_p),
        ("init", ctypes.c_void_p),
        ("shutdown", ctypes.c_void_p),
    ]

EDEV_ECPU = 0
EDEV_CUDA = 1

ISA_NAMES = {0:"scalar",1:"avx2",2:"avx512",3:"amx",4:"neon",5:"sve"}

def init(device_type=EDEV_ECPU, n_threads=0, kv_cache_budget=4*1024**3):
    lib = _load_lib()
    cfg = lib.ecpu_config_default()
    cfg.device_type = device_type
    cfg.n_threads = n_threads
    cfg.kv_cache_budget_bytes = kv_cache_budget
    rc = lib.ecpu_init(ctypes.byref(cfg))
    if rc != 0:
        err = lib.ecpu_last_error()
        raise RuntimeError(f"ecpu_init failed ({rc}): {err.decode() if err else 'unknown'}")
    return lib

def version():
    return _load_lib().ecpu_version().decode()

def detect_isa():
    lib = _load_lib()
    isa = lib.ekernel_detect_isa()
    return ISA_NAMES.get(isa, "unknown")

def active_device_info():
    lib = _load_lib()
    vt = lib.edevice_active()
    if not vt:
        return None
    name = vt.contents.name.decode() if vt.contents.name else "unknown"
    return {"name": name, "id": vt.contents.id, "caps": vt.contents.caps}

def shutdown():
    _load_lib().ecpu_shutdown()