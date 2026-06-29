"""Scheduler: unified continuous batching with chunked prefill + RECOMPUTE preemption.

Ported from vLLM v1 (vllm/v1/core/sched/scheduler.py). Single schedule() loop
advances num_computed_tokens per request; prefill and decode interleave
naturally. When KV cache pressure occurs, preempt newest running request
(RECOMPUTE: free blocks, reset num_computed_tokens=0, requeue to front).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .request import Request, FCFSQueue, RequestStatus
from .kv_cache_manager import KVCacheManager
from .block import DEFAULT_BLOCK_SIZE


@dataclass
class ScheduledRequest:
    req_id: int
    num_tokens: int
    block_ids: List[int]
    positions: List[int]
    is_prefill: bool
    is_prefill_chunk: bool = False
    cached_block_ids: List[int] = field(default_factory=list)


@dataclass
class SchedulerOutput:
    scheduled: List[ScheduledRequest] = field(default_factory=list)
    preempted: List[int] = field(default_factory=list)
    finished: List[int] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled) == 0


class Scheduler:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 128,
        max_model_len: int = 8192,
        max_blocks_per_req: int = 512,
        enable_chunked_prefill: bool = True,
        long_prefill_token_threshold: int = 0,
        policy: str = "fcfs",
    ):
        self.kv_manager = kv_cache_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_blocks_per_req = max_blocks_per_req
        self.enable_chunked_prefill = enable_chunked_prefill
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.policy = policy
        self.block_size = kv_cache_manager.block_size

        self.waiting = FCFSQueue()
        self.running: List[Request] = []
        self.requests: Dict[int, Request] = {}

    def add_request(self, req: Request):
        req.status = RequestStatus.WAITING
        req._update_block_hashes(self.block_size)
        self.waiting.push(req)
        self.requests[req.req_id] = req

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        token_budget = self.max_num_batched_tokens
        num_scheduled: Dict[int, int] = {}

        i = 0
        while i < len(self.running) and token_budget > 0:
            req = self.running[i]
            num_new = req.num_uncomputed
            if num_new <= 0:
                i += 1
                continue
            if self.long_prefill_token_threshold > 0 and num_new > self.long_prefill_token_threshold:
                num_new = self.long_prefill_token_threshold
            num_new = min(num_new, token_budget, self.max_model_len - req.num_computed_tokens - 1)
            block_remaining = self.block_size - (req.num_computed_tokens % self.block_size)
            num_new = min(num_new, block_remaining)
            if num_new <= 0:
                i += 1
                continue

            allocated = self._try_allocate(req, num_new)
            if allocated is None:
                self._preempt_for(req, output)
                break

            is_chunk = req.is_prefill and (num_new < req.num_uncomputed)
            positions = list(range(req.num_computed_tokens, req.num_computed_tokens + num_new))
            output.scheduled.append(ScheduledRequest(
                req_id=req.req_id,
                num_tokens=num_new,
                block_ids=list(req.block_ids),
                positions=positions,
                is_prefill=req.is_prefill,
                is_prefill_chunk=is_chunk,
            ))
            num_scheduled[req.req_id] = num_new
            token_budget -= num_new
            req.num_computed_tokens += num_new
            req.is_prefill_chunk = is_chunk
            i += 1

        while (
            len(self.waiting) > 0
            and len(self.running) < self.max_num_seqs
            and token_budget > 0
        ):
            req = self.waiting.peek()
            if req is None:
                break

            cached_blocks, num_cached_tokens = self.kv_manager.get_computed_blocks(req)
            num_new = req.num_tokens - num_cached_tokens
            if num_new <= 0:
                self.waiting.pop()
                req.status = RequestStatus.FINISHED_STOPPED
                output.finished.append(req.req_id)
                continue

            if not self.enable_chunked_prefill and num_new > token_budget:
                break
            if self.long_prefill_token_threshold > 0 and num_new > self.long_prefill_token_threshold:
                num_new = self.long_prefill_token_threshold
            num_new = min(num_new, token_budget, self.max_model_len - num_cached_tokens - 1)
            block_remaining = self.block_size - (num_cached_tokens % self.block_size)
            num_new = min(num_new, block_remaining)
            if num_new <= 0:
                break

            allocated = self.kv_manager.allocate_slots(
                req,
                num_new,
                num_new_computed_tokens=num_cached_tokens,
                new_computed_blocks=cached_blocks,
                max_blocks_per_req=self.max_blocks_per_req,
            )
            if allocated is None:
                if not self._preempt_one(output):
                    break
                new_head = self.waiting.peek()
                if new_head is None or new_head.status == RequestStatus.PREEMPTED:
                    break
                continue

            req = self.waiting.pop()
            self.running.append(req)
            req.status = RequestStatus.RUNNING
            req.num_computed_tokens = num_cached_tokens
            for b in cached_blocks:
                req.block_ids.insert(0, b.block_id)

            is_chunk = num_new < (req.num_tokens - num_cached_tokens)
            positions = list(range(num_cached_tokens, num_cached_tokens + num_new))
            output.scheduled.append(ScheduledRequest(
                req_id=req.req_id,
                num_tokens=num_new,
                block_ids=list(req.block_ids),
                positions=positions,
                is_prefill=True,
                is_prefill_chunk=is_chunk,
                cached_block_ids=[b.block_id for b in cached_blocks],
            ))
            num_scheduled[req.req_id] = num_new
            token_budget -= num_new
            req.num_computed_tokens += num_new
            req.is_prefill_chunk = is_chunk

        output.total_tokens = sum(num_scheduled.values())
        return output

    def _try_allocate(self, req: Request, num_new: int) -> Optional[object]:
        return self.kv_manager.allocate_slots(
            req, num_new,
            max_blocks_per_req=self.max_blocks_per_req,
        )

    def _preempt_for(self, req: Request, output: SchedulerOutput) -> bool:
        if len(self.running) <= 1:
            return False
        return self._preempt_one(output, exclude=req)

    def _preempt_one(self, output: SchedulerOutput, exclude: Optional[Request] = None) -> bool:
        victim = None
        if self.policy == "priority":
            candidates = [r for r in self.running if r is not exclude]
            if not candidates:
                return False
            victim = max(candidates, key=lambda r: (r.priority, r.arrival_time))
        else:
            for r in reversed(self.running):
                if r is not exclude:
                    victim = r
                    break
        if victim is None:
            return False
        self.running.remove(victim)
        self.kv_manager.free_request_blocks(victim)
        victim.reset_for_preemption()
        victim._update_block_hashes(self.block_size)
        self.waiting.pushleft(victim)
        output.preempted.append(victim.req_id)
        return True

    def update_from_output(
        self,
        output: SchedulerOutput,
        generated_tokens: Dict[int, List[int]],
        release_finished: bool = True,
    ):
        finished = []
        for req in self.running:
            new_tokens = generated_tokens.get(req.req_id, [])
            if new_tokens:
                req.append_output_token_ids(new_tokens, self.block_size)
            if req.check_stop(self.max_model_len):
                finished.append(req)
        for rid in output.finished:
            req = self.requests.get(rid)
            if req and req not in finished and req.is_finished:
                finished.append(req)
        for req in finished:
            if req in self.running:
                self.running.remove(req)
            if release_finished:
                self.kv_manager.cache_finished_request(req)
                self.kv_manager.free_request_blocks(req)

    def finish_request(self, req_id: int):
        req = self.requests.pop(req_id, None)
        if req is None:
            return
        if req in self.running:
            self.running.remove(req)
        self.kv_manager.cache_finished_request(req)
        self.kv_manager.free_request_blocks(req)
        req.status = RequestStatus.FINISHED_STOPPED

    def abort_request(self, req_id: int):
        req = self.requests.pop(req_id, None)
        if req is None:
            return
        self.waiting.remove(req)
        if req in self.running:
            self.running.remove(req)
        self.kv_manager.free_request_blocks(req)
        req.status = RequestStatus.FINISHED_ABORTED

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_free_blocks(self) -> int:
        return self.kv_manager.num_free_blocks
