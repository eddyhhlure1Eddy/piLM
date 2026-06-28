from .base import Device
from .ecpu_device import EcpuDevice
from .cuda_device import CudaDevice

_REGISTRY = {
    "ecpu": EcpuDevice,
    "cuda": CudaDevice,
}

def get_device(name: str) -> Device:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"unknown device: {name}, available: {list(_REGISTRY)}")
    return cls()

def available_devices() -> list:
    return list(_REGISTRY)