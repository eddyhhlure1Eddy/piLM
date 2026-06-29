"""KV cache manager: scheduler-facing facade over BlockPool.

Ported from vLLM v1 (vllm/v1/core/kv_cache_manager.py). Provides
get_computed_blocks (prefix-cache hit lookup) and allocate_slots
(allocate blocks for new tokens, touching cached blocks).
"""
from typing import List, Optional, Tuple
from .block import KVCacheBlock, DEFAULT_BLOCK_SIZE, hash_block_tokens
from .block_pool import BlockPool
from .request import Request


class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        enable_caching: bool = True,
        watermark: float = 0.0,
    ):
        self.block_size = block_size
        self.pool = BlockPool(num_blocks, block_size, enable_caching)
        self.enable_caching = enable_caching
        self.watermark = int(watermark * num_blocks) if watermark > 0 else 0

    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free

    @property
    def num_effective_free(self) -> int:
        return max(0, self.pool.num_free - self.watermark)

    def get_computed_blocks(self, request: Request) -> Tuple[List[KVCacheBlock], int]:
        """Find longest prefix-cache hit. Returns (cached_blocks, num_computed_tokens)."""
        if not self.enable_caching or not request.block_hashes:
            return [], 0
        max_blocks = (request.num_tokens - 1) // self.block_size
        hit = self.pool.find_cache_hit(request.block_hashes, max_blocks)
        return hit, len(hit) * self.block_size

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[List[KVCacheBlock]] = None,
        max_blocks_per_req: int = 512,
        full_sequence_must_fit: bool = False,
    ) -> Optional[List[KVCacheBlock]]:
        """Allocate blocks for new tokens, touching cached prefix blocks.

        Returns the full block list (cached + new) or None if insufficient.
        """
        if new_computed_blocks is None:
            new_computed_blocks = []
        num_cached = len(new_computed_blocks)
        num_cached_tokens = num_cached * self.block_size

        total_tokens_after = request.num_computed_tokens + num_new_tokens
        if num_new_computed_tokens > 0:
            total_tokens_after = num_new_computed_tokens + num_new_tokens

        num_total_blocks_needed = (total_tokens_after + self.block_size - 1) // self.block_size
        num_new_blocks = num_total_blocks_needed - num_cached - len(request.block_ids)

        if num_cached + num_new_blocks + len(request.block_ids) > max_blocks_per_req:
            return None

        if full_sequence_must_fit:
            full_blocks = (request.num_tokens + self.block_size - 1) // self.block_size
            if full_blocks > self.pool.num_free + num_cached + len(request.block_ids):
                return None

        if num_new_blocks > self.num_effective_free:
            return None

        if num_new_blocks > 0:
            new_blocks = self.pool.get_new_blocks(num_new_blocks)
            if len(new_blocks) < num_new_blocks:
                self.pool.free_blocks(new_blocks)
                return None
        else:
            new_blocks = []

        if new_computed_blocks:
            self.pool.touch(new_computed_blocks)

        all_new = new_computed_blocks + new_blocks
        request.block_ids.extend(b.block_id for b in new_blocks)

        self._cache_new_full_blocks(request, all_new, num_cached)

        return all_new

    def _cache_new_full_blocks(
        self,
        request: Request,
        blocks: List[KVCacheBlock],
        num_cached: int,
    ):
        if not self.enable_caching:
            return
        bs = self.block_size
        total = request.num_tokens
        num_full = total // bs
        num_existing = len(request.block_hashes)
        if num_cached >= num_full or num_existing >= num_full:
            return
        request._update_block_hashes(bs)
        parents = [None] + request.block_hashes[:-1] if request.block_hashes else [None]
        self.pool.cache_full_blocks(
            request.all_token_ids, blocks, num_cached, num_full, parents
        )

    def free_request_blocks(self, request: Request):
        blocks = [self.pool.blocks[bid] for bid in request.block_ids if 0 <= bid < self.pool.num_blocks]
        self.pool.free_blocks(blocks)
        request.block_ids = []

    def free_blocks(self, blocks: List[KVCacheBlock]):
        self.pool.free_blocks(blocks)

    def cache_finished_request(self, request: Request):
        if not self.enable_caching:
            return
        bs = self.block_size
        request._update_block_hashes(bs)
        blocks = [self.pool.blocks[bid] for bid in request.block_ids]
        num_full = request.num_tokens // bs
        parents = [None] + request.block_hashes[:-1] if request.block_hashes else [None]
        self.pool.cache_full_blocks(
            request.all_token_ids, blocks, 0, num_full, parents
        )

    def get_block(self, block_id: int) -> Optional[KVCacheBlock]:
        if 0 <= block_id < self.pool.num_blocks:
            return self.pool.blocks[block_id]
        return None