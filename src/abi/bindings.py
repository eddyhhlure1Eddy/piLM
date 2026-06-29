import ctypes
import os
import sys
from pathlib import Path

_lib = None
_lib_path = None

_OPTIONAL_KERNEL_SYMBOLS = [
    "ekernel_gemm",
    "ekernel_linear_w8a32",
    "ekernel_linear_w8a16_bf16",
    "ekernel_linear_w8a16_bf16_argmax",
    "ekernel_linear_w8a16_bf16_i8b8",
    "ekernel_linear_w8a16_bf16_q8",
    "ekernel_linear_w4a16_bf16",
    "ekernel_linear_w4a16g32_bf16",
    "ekernel_linear_w4a16g128_bf16",
    "ekernel_linear_w4a16_bf16_q8",
    "ekernel_linear_w4a16_bf16_i4b8",
    "ekernel_linear_w4a16_bf16_b8",
    "ekernel_swiglu_bf16",
    "ekernel_swiglu_w4a16_bf16",
    "ekernel_rmsnorm_bf16",
    "ekernel_gated_delta_recurrent_f32",
]

def _load_lib():
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    candidates = []
    env_lib = os.environ.get("PILM_ECPU_LIB")
    if env_lib:
        candidates.append(Path(env_lib))
    here = Path(__file__).resolve().parent
    build_dir = here.parent / "ecpu" / "build"
    build_w4_dir = here.parent / "ecpu" / "build_w4"
    build_dirs = [build_dir, build_w4_dir]
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dll_dir in build_dirs + [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]:
            try:
                if dll_dir.exists():
                    os.add_dll_directory(str(dll_dir))
            except OSError:
                pass
    for dll_dir in build_dirs:
        candidates.append(dll_dir / "libecpu.dll")
        candidates.append(dll_dir / "libecpu.so")
        candidates.append(dll_dir / "libecpu.dylib")
    for c in candidates:
        if c.exists():
            _lib = ctypes.CDLL(str(c))
            _lib_path = str(c)
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
    lib.ecpu_shutdown.argtypes = []
    lib.ecpu_shutdown.restype = None
    lib.ekernel_gemm.argtypes = [
        ctypes.POINTER(EkernelGemmDesc),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.ekernel_gemm.restype = ctypes.c_int
    lib.ekernel_linear_w8a32.argtypes = [
        ctypes.POINTER(EkernelLinearW8A32Desc),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.ekernel_linear_w8a32.restype = ctypes.c_int
    lib.ekernel_linear_w8a16_bf16.argtypes = [
        ctypes.POINTER(EkernelLinearW8A32Desc),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.ekernel_linear_w8a16_bf16.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w8a16_bf16_argmax"):
        lib.ekernel_linear_w8a16_bf16_argmax.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.ekernel_linear_w8a16_bf16_argmax.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w8a16_bf16_i8b8"):
        lib.ekernel_linear_w8a16_bf16_i8b8.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w8a16_bf16_i8b8.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w8a16_bf16_q8"):
        lib.ekernel_linear_w8a16_bf16_q8.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w8a16_bf16_q8.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w4a16_bf16"):
        lib.ekernel_linear_w4a16_bf16.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w4a16_bf16.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w4a16g32_bf16"):
        lib.ekernel_linear_w4a16g32_bf16.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w4a16g32_bf16.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w4a16g128_bf16"):
        lib.ekernel_linear_w4a16g128_bf16.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w4a16g128_bf16.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w4a16_bf16_q8"):
        lib.ekernel_linear_w4a16_bf16_q8.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w4a16_bf16_q8.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w4a16_bf16_i4b8"):
        lib.ekernel_linear_w4a16_bf16_i4b8.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w4a16_bf16_i4b8.restype = ctypes.c_int
    if hasattr(lib, "ekernel_linear_w4a16_bf16_b8"):
        lib.ekernel_linear_w4a16_bf16_b8.argtypes = [
            ctypes.POINTER(EkernelLinearW8A32Desc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_linear_w4a16_bf16_b8.restype = ctypes.c_int
    lib.ekernel_swiglu_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.ekernel_swiglu_bf16.restype = ctypes.c_int
    if hasattr(lib, "ekernel_swiglu_w4a16_bf16"):
        lib.ekernel_swiglu_w4a16_bf16.argtypes = [
            ctypes.POINTER(EkernelW4A16SwiGLUDesc),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ekernel_swiglu_w4a16_bf16.restype = ctypes.c_int
    if hasattr(lib, "ekernel_rmsnorm_bf16"):
        lib.ekernel_rmsnorm_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_int,
        ]
        lib.ekernel_rmsnorm_bf16.restype = ctypes.c_int
    if hasattr(lib, "ekernel_gated_delta_recurrent_f32"):
        lib.ekernel_gated_delta_recurrent_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.ekernel_gated_delta_recurrent_f32.restype = ctypes.c_int

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

class EkernelGemmDesc(ctypes.Structure):
    _fields_ = [
        ("M", ctypes.c_size_t),
        ("N", ctypes.c_size_t),
        ("K", ctypes.c_size_t),
        ("a_prec", ctypes.c_int),
        ("b_prec", ctypes.c_int),
        ("out_prec", ctypes.c_int),
        ("transpose_a", ctypes.c_int),
        ("transpose_b", ctypes.c_int),
    ]

class EkernelLinearW8A32Desc(ctypes.Structure):
    _fields_ = [
        ("M", ctypes.c_size_t),
        ("N", ctypes.c_size_t),
        ("K", ctypes.c_size_t),
    ]

class EkernelW4A16SwiGLUDesc(ctypes.Structure):
    _fields_ = [
        ("M", ctypes.c_size_t),
        ("hidden_size", ctypes.c_size_t),
        ("intermediate_size", ctypes.c_size_t),
        ("gate_up_in_features", ctypes.c_size_t),
        ("down_in_features", ctypes.c_size_t),
    ]

EDEV_ECPU = 0
EDEV_CUDA = 1

ECPU_PRECISION_F32 = 0
ECPU_PRECISION_F16 = 1
ECPU_PRECISION_BF16 = 2
ECPU_PRECISION_F8_E4M3 = 3
ECPU_PRECISION_F8_E5M2 = 4
ECPU_PRECISION_I8 = 5
ECPU_PRECISION_I4 = 6

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

def library_path():
    _load_lib()
    return _lib_path

def available_kernel_symbols():
    lib = _load_lib()
    return {name: hasattr(lib, name) for name in _OPTIONAL_KERNEL_SYMBOLS}

def runtime_info():
    return {
        "library_path": library_path(),
        "version": version(),
        "isa": detect_isa(),
        "symbols": available_kernel_symbols(),
        "env_library_path": os.environ.get("PILM_ECPU_LIB"),
    }

def active_device_info():
    lib = _load_lib()
    vt = lib.edevice_active()
    if not vt:
        return None
    name = vt.contents.name.decode() if vt.contents.name else "unknown"
    return {"name": name, "id": vt.contents.id, "caps": vt.contents.caps}

def shutdown():
    _load_lib().ecpu_shutdown()

def gemm_f32_ptr(a_ptr: int, b_ptr: int, c_ptr: int, m: int, n: int, k: int,
                 transpose_a: bool = False, transpose_b: bool = False,
                 isa: int | None = None) -> int:
    lib = _load_lib()
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelGemmDesc(
        m, n, k,
        ECPU_PRECISION_F32, ECPU_PRECISION_F32, ECPU_PRECISION_F32,
        int(transpose_a), int(transpose_b),
    )
    return lib.ekernel_gemm(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w8a32_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                     m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w8a32(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w8a16_bf16_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                          m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w8a16_bf16(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w8a16_bf16_argmax_ptr(a_ptr: int, w_ptr: int, scales_ptr: int,
                                 m: int, n: int, k: int, isa: int | None = None) -> tuple[int, int, float]:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w8a16_bf16_argmax"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w8a16_bf16_argmax")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    out_index = ctypes.c_size_t(0)
    out_value = ctypes.c_float(0.0)
    rc = lib.ekernel_linear_w8a16_bf16_argmax(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.byref(out_index),
        ctypes.byref(out_value),
        int(isa),
    )
    return int(rc), int(out_index.value), float(out_value.value)

def linear_w8a16_bf16_i8b8_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                               m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w8a16_bf16_i8b8"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w8a16_bf16_i8b8")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w8a16_bf16_i8b8(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w8a16_bf16_q8_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                             m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w8a16_bf16_q8"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w8a16_bf16_q8")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w8a16_bf16_q8(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w4a16_bf16_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                          m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w4a16_bf16"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w4a16_bf16")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w4a16_bf16(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w4a16g32_bf16_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                             m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w4a16g32_bf16"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w4a16g32_bf16")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w4a16g32_bf16(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w4a16g128_bf16_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                              m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w4a16g128_bf16"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w4a16g128_bf16")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w4a16g128_bf16(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w4a16_bf16_q8_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                             m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w4a16_bf16_q8"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w4a16_bf16_q8")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w4a16_bf16_q8(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w4a16_bf16_i4b8_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                               m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w4a16_bf16_i4b8"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w4a16_bf16_i4b8")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w4a16_bf16_i4b8(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def linear_w4a16_bf16_b8_ptr(a_ptr: int, w_ptr: int, scales_ptr: int, c_ptr: int,
                             m: int, n: int, k: int, isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_linear_w4a16_bf16_b8"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_linear_w4a16_bf16_b8")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelLinearW8A32Desc(m, n, k)
    return lib.ekernel_linear_w4a16_bf16_b8(
        ctypes.byref(desc),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def swiglu_bf16_ptr(gate_ptr: int, up_ptr: int, out_ptr: int, n: int) -> int:
    lib = _load_lib()
    return lib.ekernel_swiglu_bf16(
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(up_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_size_t(n),
    )

def swiglu_w4a16_bf16_ptr(x_ptr: int, gate_up_w_ptr: int, gate_up_scales_ptr: int,
                          down_w_ptr: int, down_scales_ptr: int, c_ptr: int,
                          m: int, hidden_size: int, intermediate_size: int,
                          gate_up_in_features: int, down_in_features: int,
                          isa: int | None = None) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_swiglu_w4a16_bf16"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_swiglu_w4a16_bf16")
    if isa is None:
        isa = lib.ekernel_detect_isa()
    desc = EkernelW4A16SwiGLUDesc(
        m,
        hidden_size,
        intermediate_size,
        gate_up_in_features,
        down_in_features,
    )
    return lib.ekernel_swiglu_w4a16_bf16(
        ctypes.byref(desc),
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(gate_up_w_ptr),
        ctypes.c_void_p(gate_up_scales_ptr),
        ctypes.c_void_p(down_w_ptr),
        ctypes.c_void_p(down_scales_ptr),
        ctypes.c_void_p(c_ptr),
        int(isa),
    )

def rmsnorm_bf16_ptr(x_ptr: int, out_ptr: int, weight_ptr: int,
                     rows: int, n: int, eps: float, add_one: bool = False) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_rmsnorm_bf16"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_rmsnorm_bf16")
    return lib.ekernel_rmsnorm_bf16(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_size_t(rows),
        ctypes.c_size_t(n),
        ctypes.c_float(float(eps)),
        ctypes.c_int(1 if add_one else 0),
    )

def gated_delta_recurrent_f32_ptr(state_ptr: int, q_ptr: int, k_ptr: int, v_ptr: int,
                                  beta_ptr: int, decay_ptr: int, out_ptr: int,
                                  n_heads: int, k_dim: int, v_dim: int, scale: float) -> int:
    lib = _load_lib()
    if not hasattr(lib, "ekernel_gated_delta_recurrent_f32"):
        raise RuntimeError("loaded eCPU library does not expose ekernel_gated_delta_recurrent_f32")
    return lib.ekernel_gated_delta_recurrent_f32(
        ctypes.c_void_p(state_ptr),
        ctypes.c_void_p(q_ptr),
        ctypes.c_void_p(k_ptr),
        ctypes.c_void_p(v_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_size_t(n_heads),
        ctypes.c_size_t(k_dim),
        ctypes.c_size_t(v_dim),
        ctypes.c_float(float(scale)),
    )
