"""Physical KV cache storage: paged torch tensors on CPU (Eram-backed).

Each block stores BLOCK_SIZE tokens of K and V for all KV heads.
Scheduler manages block IDs; this module manages the physical bytes.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class PhysicalKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        per_block = block_size * num_kv_heads * head_dim
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        for _ in range(num_layers):
            self.k_cache.append(torch.empty(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype))
            self.v_cache.append(torch.empty(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype))

        self.used: set = set()

    def write(
        self,
        layer_idx: int,
        block_ids: List[int],
        slot_offsets: List[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Write K/V for given slots. keys: [n_slots, n_kv_heads, head_dim]"""
        k = self.k_cache[layer_idx]
        v = self.v_cache[layer_idx]
        for i, (bid, off) in enumerate(zip(block_ids, slot_offsets)):
            k[bid, off] = keys[i]
            v[bid, off] = values[i]
            self.used.add(bid)

    def read_block(self, layer_idx: int, block_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_idx][block_id], self.v_cache[layer_idx][block_id]

    def read_seq(
        self, layer_idx: int, block_ids: List[int], n_valid: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read full KV for a sequence given its block table.
        Returns [n_valid, n_kv_heads, head_dim] for K and V."""
        bs = self.block_size
        k_parts = []
        v_parts = []
        for bid in block_ids:
            k_parts.append(self.k_cache[layer_idx][bid])
            v_parts.append(self.v_cache[layer_idx][bid])
        k = torch.cat(k_parts, dim=0)[:n_valid]
        v = torch.cat(v_parts, dim=0)[:n_valid]
        return k, v

    def num_used_blocks(self) -> int:
        return len(self.used)

    def free_blocks(self, block_ids: List[int]):
        for bid in block_ids:
            self.used.discard(bid)

    def total_bytes(self) -> int:
        per_layer = self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim
        elem = 2 if self.dtype == torch.bfloat16 or self.dtype == torch.float16 else 4
        return per_layer * 2 * self.num_layers * elem
