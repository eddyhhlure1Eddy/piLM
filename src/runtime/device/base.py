from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DeviceInfo:
    name: str
    caps: int
    mem_total: int
    mem_free: int

class Device(ABC):
    @abstractmethod
    def init(self) -> int: ...
    @abstractmethod
    def shutdown(self) -> int: ...
    @abstractmethod
    def alloc(self, bytes: int, alignment: int = 64) -> int: ...
    @abstractmethod
    def free(self, ptr: int) -> None: ...
    @abstractmethod
    def info(self) -> DeviceInfo: ...
    @property
    @abstractmethod
    def is_gpu(self) -> bool: ...