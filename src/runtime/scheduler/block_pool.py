"""Block pool: manages KV cache block allocation, refcount, prefix caching, LRU eviction.

Ported from vLLM v1 (vllm/v1/core/block_pool.py). Owns all KVCacheBlock objects.
Allocates via LRU free-list, supports chained-hash prefix-cache lookup with
ref-counted sharing, evicts cached blocks when reused.
"""
from typing import Dict, List, Optional
from .block import KVCacheBlock, FreeKVCacheBlockQueue, DEFAULT_BLOCK_SIZE, hash_block_tokens


class BlockPool:
    def __init__(
        self,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        enable_caching: bool = True,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching

        self.blocks: List[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_blocks)
        ]
        self.free_queue = FreeKVCacheBlockQueue(self.blocks)

        self.null_block: Optional[KVCacheBlock] = None
        if num_blocks > 0:
            self.null_block = self.free_queue.popleft()
            self.null_block.is_null = True

        self.hash_to_block: Dict[bytes, KVCacheBlock] = {}
        self.used_blocks = 0

    @property
    def num_free(self) -> int:
        return self.free_queue.num_free

    def get_new_blocks(self, n: int) -> List[KVCacheBlock]:
        if n > self.num_free:
            return []
        result = []
        for _ in range(n):
            b = self.free_queue.popleft()
            if b is None:
                break
            if self.enable_caching and b.has_hash:
                self._evict_cached(b)
            b.ref_cnt = 1
            result.append(b)
            self.used_blocks += 1
        return result

    def get_null_block(self) -> Optional[KVCacheBlock]:
        return self.null_block

    def touch(self, blocks: List[KVCacheBlock]):
        for b in blocks:
            if b is None or b.is_null:
                continue
            if b.ref_cnt == 0:
                self.free_queue.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks: List[KVCacheBlock]):
        with_hash: List[KVCacheBlock] = []
        without_hash: List[KVCacheBlock] = []
        for b in blocks:
            if b is None or b.is_null:
                continue
            b.ref_cnt -= 1
            if b.ref_cnt <= 0:
                b.ref_cnt = 0
                if b.has_hash:
                    with_hash.append(b)
                else:
                    without_hash.append(b)
                self.used_blocks -= 1
        self.free_queue.prepend_n(without_hash)
        self.free_queue.append_n(with_hash)

    def cache_full_blocks(
        self,
        all_token_ids: List[int],
        blocks: List[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        parent_hashes: List[Optional[bytes]],
    ):
        if not self.enable_caching:
            return
        bs = self.block_size
        for i in range(num_cached_blocks, num_full_blocks):
            blk = blocks[i]
            if blk is None or blk.is_null:
                continue
            parent = parent_hashes[i] if i < len(parent_hashes) else None
            start = i * bs
            chunk = all_token_ids[start:start + bs]
            if len(chunk) != bs:
                continue
            h = hash_block_tokens(parent, chunk)
            blk.block_hash = h
            blk.hash_num_tokens = (i + 1) * bs
            self.hash_to_block[h] = blk

    def find_cache_hit(
        self,
        block_hashes: List[Optional[bytes]],
        max_blocks: int,
    ) -> List[KVCacheBlock]:
        if not self.enable_caching:
            return []
        hit: List[KVCacheBlock] = []
        for h in block_hashes[:max_blocks]:
            if h is None:
                break
            blk = self.hash_to_block.get(h)
            if blk is None:
                break
            hit.append(blk)
        return hit

    def get_cached_block(self, block_hash: bytes) -> Optional[KVCacheBlock]:
        return self.hash_to_block.get(block_hash)

    def _evict_cached(self, block: KVCacheBlock):
        if block.has_hash:
            self.hash_to_block.pop(block.block_hash, None)
            block.reset_hash()

    def reset_prefix_cache(self) -> int:
        if not self.enable_caching:
            return 0
        count = 0
        for b in self.blocks:
            if b.has_hash and b.ref_cnt == 0:
                self._evict_cached(b)
                count += 1
        return count

    def num_cached_blocks(self) -> int:
        return len(self.hash_to_block)