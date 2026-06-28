"""Scheduler layer: continuous batching and request scheduling.

Design:
  - block_pool: block allocation, refcount, free-list management
  - kv_cache_manager: slot allocation, prefix-cache lookup
  - scheduler: prefill+decode mixed batching, preemption policy
  - request_queue: FCFS / Priority queue
  - block_table: slot_mapping computation for kernel consumption

All device-agnostic pure-Python logic.
The C kernel (Ecpu) consumes the resulting block_table + slot_mapping tensors.
"""