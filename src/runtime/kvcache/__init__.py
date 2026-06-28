"""KV cache management: paged block pool + prefix reuse tree.

Design:
  - block: KVCacheBlock dataclass (block_id, ref_cnt, prev/next)
  - block_pool: BlockPool (get_new_blocks, touch, free_blocks, evict)
  - kv_cache_manager: allocate_slots, get_computed_blocks
  - radix_cache: optional prefix reuse tree (TreeNode match/insert)
  - eram_bridge: bridge to C-side eram_kv_cache_t (write/read blocks)

The block pool/refcount layer is pure-Python device-agnostic (operates on block IDs).
Physical KV storage lives in Eram (C-side eram_kv_cache_t).
"""