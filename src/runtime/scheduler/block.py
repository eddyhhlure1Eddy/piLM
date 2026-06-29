"""KV cache block: the fundamental allocation unit for paged attention.

Ported from vLLM v1 (vllm/v1/core/kv_cache_utils.py). Each block holds
BLOCK_SIZE tokens worth of KV cache. Blocks are ref-counted for prefix-cache
sharing and managed via a doubly-linked free list with LRU eviction order.
"""
from dataclasses import dataclass, field
from typing import Optional
import hashlib

DEFAULT_BLOCK_SIZE = 16


@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    is_null: bool = False
    prev_free: Optional["KVCacheBlock"] = None
    next_free: Optional["KVCacheBlock"] = None
    block_hash: Optional[bytes] = None
    hash_num_tokens: Optional[int] = None

    def reset_hash(self):
        self.block_hash = None
        self.hash_num_tokens = None

    @property
    def has_hash(self) -> bool:
        return self.block_hash is not None

    def __repr__(self):
        return f"Block(id={self.block_id}, ref={self.ref_cnt}, hash={'Y' if self.has_hash else 'N'})"


def hash_block_tokens(
    parent_block_hash: Optional[bytes],
    curr_block_token_ids,
    hash_fn=None,
) -> bytes:
    if hash_fn is None:
        hash_fn = hashlib.sha256
    if parent_block_hash is None:
        parent = b"\x00" * 32
    else:
        parent = parent_block_hash
    h = hash_fn()
    h.update(parent)
    h.update(tuple(curr_block_token_ids).__sizeof__().to_bytes(8, "little"))
    for t in curr_block_token_ids:
        h.update(int(t).to_bytes(4, "little"))
    return h.digest()


class FreeKVCacheBlockQueue:
    """Doubly-linked list of free blocks with fake head/tail sentinels.

    LRU at front (popleft = evict LRU), MRU at back (append = recently freed).
    Supports O(1) mid-removal via remove() for touch() operations.
    Ported from vLLM v1 kv_cache_utils.py:179.
    """

    def __init__(self, blocks: list):
        self.head = KVCacheBlock(block_id=-1)
        self.tail = KVCacheBlock(block_id=-2)
        self.head.next_free = self.tail
        self.tail.prev_free = self.head
        self.num_free = 0
        for b in blocks:
            self.append(b)

    def append(self, block: KVCacheBlock):
        block.prev_free = self.tail.prev_free
        block.next_free = self.tail
        self.tail.prev_free.next_free = block
        self.tail.prev_free = block
        self.num_free += 1

    def prepend(self, block: KVCacheBlock):
        block.next_free = self.head.next_free
        block.prev_free = self.head
        self.head.next_free.prev_free = block
        self.head.next_free = block
        self.num_free += 1

    def popleft(self) -> Optional[KVCacheBlock]:
        if self.head.next_free is self.tail:
            return None
        block = self.head.next_free
        self._remove(block)
        return block

    def popleft_n(self, n: int) -> list:
        result = []
        for _ in range(n):
            b = self.popleft()
            if b is None:
                break
            result.append(b)
        return result

    def remove(self, block: KVCacheBlock):
        self._remove(block)

    def _remove(self, block: KVCacheBlock):
        block.prev_free.next_free = block.next_free
        block.next_free.prev_free = block.prev_free
        block.prev_free = None
        block.next_free = None
        self.num_free -= 1

    def append_n(self, blocks: list):
        for b in blocks:
            self.append(b)

    def prepend_n(self, blocks: list):
        for b in blocks:
            self.prepend(b)