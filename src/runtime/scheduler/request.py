"""Request, status enum, and queues for the scheduler.

Ported from vLLM v1 (vllm/v1/request.py). Unified Request holds prompt +
output tokens, block table, hash chain, and scheduling state.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
import enum
import heapq
import itertools

from .block import hash_block_tokens, DEFAULT_BLOCK_SIZE


class RequestStatus(enum.IntEnum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    FINISHED_STOPPED = 4
    FINISHED_LENGTH_CAPPED = 5
    FINISHED_ABORTED = 6

    @staticmethod
    def is_finished(s: "RequestStatus") -> bool:
        return s >= RequestStatus.FINISHED_STOPPED

    @property
    def is_running(self) -> bool:
        return self == RequestStatus.RUNNING


@dataclass
class Request:
    req_id: int
    prompt_token_ids: List[int]
    max_new_tokens: int = 256
    arrival_time: float = 0.0
    priority: float = 0.0
    eos_token_id: int = -1
    stop_token_ids: List[int] = field(default_factory=list)

    num_computed_tokens: int = 0
    _output_token_ids: List[int] = field(default_factory=list)
    _all_token_ids: List[int] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)
    block_hashes: List[Optional[bytes]] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    num_preemptions: int = 0
    is_prefill_chunk: bool = False

    def __post_init__(self):
        if not self._all_token_ids:
            self._all_token_ids = list(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_uncomputed(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    @property
    def all_token_ids(self) -> List[int]:
        return self._all_token_ids

    @property
    def output_token_ids(self) -> List[int]:
        return self._output_token_ids

    def next_tokens(self, n: Optional[int] = None) -> List[int]:
        if n is None:
            n = self.num_uncomputed
        n = max(0, min(n, self.num_uncomputed))
        return self._all_token_ids[self.num_computed_tokens:self.num_computed_tokens + n]

    def append_output_token_ids(self, token_ids, block_size: int = DEFAULT_BLOCK_SIZE):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)
        self._update_block_hashes(block_size)

    def _update_block_hashes(self, block_size: int):
        total = self.num_tokens
        num_full = total // block_size
        while len(self.block_hashes) < num_full:
            i = len(self.block_hashes)
            parent = self.block_hashes[i - 1] if i > 0 else None
            chunk = self._all_token_ids[i * block_size:(i + 1) * block_size]
            self.block_hashes.append(hash_block_tokens(parent, chunk))

    def reset_for_preemption(self):
        self.block_ids = []
        self.block_hashes = []
        self.num_computed_tokens = 0
        self.is_prefill_chunk = False
        self.status = RequestStatus.PREEMPTED
        self.num_preemptions += 1

    def check_stop(self, max_model_len: int) -> bool:
        if not self._output_token_ids:
            return False
        last = self._output_token_ids[-1]
        if self.eos_token_id >= 0 and last == self.eos_token_id:
            self.status = RequestStatus.FINISHED_STOPPED
            return True
        if last in self.stop_token_ids:
            self.status = RequestStatus.FINISHED_STOPPED
            return True
        if self.num_tokens >= max_model_len:
            self.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        if self.num_output_tokens >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        return False


class FCFSQueue:
    def __init__(self):
        self._q: deque = deque()

    def push(self, req: Request):
        self._q.append(req)

    def pushleft(self, req: Request):
        self._q.appendleft(req)

    def pop(self) -> Optional[Request]:
        if self._q:
            return self._q.popleft()
        return None

    def peek(self) -> Optional[Request]:
        return self._q[0] if self._q else None

    def remove(self, req: Request):
        try:
            self._q.remove(req)
        except ValueError:
            pass

    def __len__(self):
        return len(self._q)

    def __iter__(self):
        return iter(self._q)


class PriorityHeap:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def push(self, req: Request):
        heapq.heappush(self._heap, (req.priority, next(self._counter), req))

    def pop(self) -> Optional[Request]:
        if self._heap:
            return heapq.heappop(self._heap)[2]
        return None

    def peek(self) -> Optional[Request]:
        return self._heap[0][2] if self._heap else None

    def __len__(self):
        return len(self._heap)