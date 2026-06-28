from .base import Device, DeviceInfo

class CudaDevice(Device):
    """CUDA device placeholder. Future implementation will wrap cuMalloc/cuMemcpy.
    Currently the C-side cuda_device.c vtable returns -1 on init, so this is a stub."""
    def __init__(self):
        self._available = False

    def init(self) -> int:
        return -1

    def shutdown(self) -> int:
        return 0

    def alloc(self, bytes: int, alignment: int = 64) -> int:
        raise NotImplementedError("CUDA device not yet implemented")

    def free(self, ptr: int) -> None:
        raise NotImplementedError("CUDA device not yet implemented")

    def info(self) -> DeviceInfo:
        raise NotImplementedError("CUDA device not yet implemented")

    @property
    def is_gpu(self) -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False