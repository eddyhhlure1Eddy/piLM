"""safetensors file format parser (zero-copy mmap).

Format: [8-byte LE uint64 header_len][JSON metadata][raw tensor data]
Metadata JSON: {"tensor_name": {"dtype": "BF16", "shape": [...], "data_offsets": [start, end]}}
"""
import struct
import json
import mmap
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import ctypes

DTYPES = {
    "F32": (4, ctypes.c_float),
    "F16": (2, None),
    "BF16": (2, None),
    "I64": (8, ctypes.c_int64),
    "I32": (4, ctypes.c_int32),
    "I16": (2, ctypes.c_int16),
    "I8":  (1, ctypes.c_int8),
    "BOOL": (1, ctypes.c_bool),
    "F8_E4M3": (1, None),
    "F8_E5M2": (1, None),
    "U8":  (1, ctypes.c_uint8),
}

class SafetensorInfo:
    __slots__ = ("name", "dtype", "shape", "offset_start", "offset_end", "elem_size")
    def __init__(self, name, dtype, shape, start, end, elem_size):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.offset_start = start
        self.offset_end = end
        self.elem_size = elem_size

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self):
        return self.offset_end - self.offset_start


class SafetensorsFile:
    def __init__(self, path: str):
        self.path = Path(path)
        self.tensors: Dict[str, SafetensorInfo] = {}
        self._data_offset = 0
        self._f = None
        self._mm = None
        self._parse()

    def _parse(self):
        self._f = open(self.path, "rb")
        header_len_bytes = self._f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = self._f.read(header_len)
        header = json.loads(header_json)
        self._data_offset = 8 + header_len

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            elem_size = DTYPES.get(dtype_str, (0, None))[0]
            self.tensors[name] = SafetensorInfo(
                name, dtype_str, shape, start, end, elem_size
            )

        self._f.close()
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def get_tensor_view(self, name: str) -> memoryview:
        info = self.tensors[name]
        start = self._data_offset + info.offset_start
        end = self._data_offset + info.offset_end
        return memoryview(self._mm[start:end])

    def get_tensor_byte_range(self, name: str, byte_start: int, byte_end: int) -> memoryview:
        """Return a memoryview of a byte range within a tensor (relative to tensor start)."""
        info = self.tensors[name]
        abs_start = self._data_offset + info.offset_start + byte_start
        abs_end = self._data_offset + info.offset_start + byte_end
        return memoryview(self._mm[abs_start:abs_end])

    def get_tensor_bytes(self, name: str) -> bytes:
        info = self.tensors[name]
        start = self._data_offset + info.offset_start
        end = self._data_offset + info.offset_end
        return self._mm[start:end]

    def list_tensors(self) -> list:
        return list(self.tensors.keys())

    def tensor_info(self, name: str) -> SafetensorInfo:
        return self.tensors[name]

    def close(self):
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._f:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SafetensorsShardCollection:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.shards: Dict[str, SafetensorsFile] = {}
        self.index: Dict[str, str] = {}
        self.extra_tensors: list[str] = []
        self.bundle_total_bytes: Optional[int] = None
        self._load_index()

    def _load_index(self):
        idx_path = self.model_dir / "model.safetensors.index.json"
        bundle_manifest_path = self.model_dir / "pilm_quant_manifest.json"
        if bundle_manifest_path.exists():
            with open(bundle_manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            for export in manifest.get("residual_exports", []):
                shard_name = export["file"]
                sf = SafetensorsFile(str(self.model_dir / shard_name))
                for name in sf.list_tensors():
                    self.index[name] = shard_name
                self.shards[shard_name] = sf
            self.extra_tensors = list(manifest.get("quantized_linear_sources", []))
            self.bundle_total_bytes = int(manifest.get("bundle_bytes", 0)) or None
        elif idx_path.exists():
            with open(idx_path) as f:
                idx = json.load(f)
            self.index = idx.get("weight_map", {})
        else:
            single = self.model_dir / "model.safetensors"
            if single.exists():
                sf = SafetensorsFile(str(single))
                for name in sf.list_tensors():
                    self.index[name] = "model.safetensors"
                    self.shards["model.safetensors"] = sf
            else:
                raise FileNotFoundError(f"No safetensors in {self.model_dir}")

    def _get_shard(self, shard_name: str) -> SafetensorsFile:
        if shard_name not in self.shards:
            self.shards[shard_name] = SafetensorsFile(
                str(self.model_dir / shard_name)
            )
        return self.shards[shard_name]

    def get_tensor_bytes(self, name: str) -> bytes:
        shard_name = self.index.get(name)
        if not shard_name:
            raise KeyError(f"tensor {name} not in index")
        return self._get_shard(shard_name).get_tensor_bytes(name)

    def tensor_info(self, name: str) -> SafetensorInfo:
        shard_name = self.index[name]
        return self._get_shard(shard_name).tensor_info(name)

    def list_all_tensors(self) -> list:
        return list(self.index.keys()) + list(self.extra_tensors)

    def total_bytes(self) -> int:
        if self.bundle_total_bytes is not None:
            return self.bundle_total_bytes
        total = 0
        for name in self.index:
            total += self.tensor_info(name).nbytes
        return total

    def close_all(self):
        for sf in self.shards.values():
            sf.close()
        self.shards.clear()
