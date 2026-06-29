from .base import Device, DeviceInfo
try:
    from ... import _abi as abi
except ImportError:
    import _abi as abi

class EcpuDevice(Device):
    def __init__(self):
        self._lib = abi._load_lib()
        self._vt = None

    def init(self) -> int:
        abi.init(device_type=abi.EDEV_ECPU)
        self._vt = self._lib.edevice_active()
        return 0

    def shutdown(self) -> int:
        abi.shutdown()
        return 0

    def alloc(self, bytes: int, alignment: int = 64) -> int:
        alloc_fn = ctypes.cast(self._vt.contents.alloc,
                               ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t))
        return alloc_fn(bytes, alignment)

    def free(self, ptr: int) -> None:
        free_fn = ctypes.cast(self._vt.contents.free,
                              ctypes.CFUNCTYPE(None, ctypes.c_void_p))
        free_fn(ptr)

    def info(self) -> DeviceInfo:
        vt = self._vt.contents
        total_fn = ctypes.cast(vt.mem_total, ctypes.CFUNCTYPE(ctypes.c_size_t))
        free_fn = ctypes.cast(vt.mem_free, ctypes.CFUNCTYPE(ctypes.c_size_t))
        return DeviceInfo(
            name=vt.name.decode() if vt.name else "ecpu",
            caps=vt.caps,
            mem_total=total_fn(),
            mem_free=free_fn(),
        )

    @property
    def is_gpu(self) -> bool:
        return False

import ctypes
