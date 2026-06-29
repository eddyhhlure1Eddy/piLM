"""Async strategy wrapper for the synchronous scheduler.

The core scheduler remains a deterministic synchronous state machine. This
wrapper adds asyncio-safe access and a waitable work signal so HTTP/WebSocket
or future worker loops can enqueue requests concurrently without corrupting
KV/request state.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Dict, List, Optional

from .request import Request
from .scheduler import Scheduler, SchedulerOutput


@dataclass
class AsyncSchedulerConfig:
    wait_timeout: Optional[float] = None
    yield_empty: bool = False


class AsyncScheduler:
    """Async access strategy for `Scheduler`.

    All mutations are serialized through one asyncio lock. `schedule(wait=True)`
    waits until either a new request arrives or `update_from_output` makes a
    running request schedulable again.
    """

    def __init__(self, scheduler: Scheduler, config: Optional[AsyncSchedulerConfig] = None):
        self.scheduler = scheduler
        self.config = config or AsyncSchedulerConfig()
        self._lock = asyncio.Lock()
        self._work_event = asyncio.Event()
        self._refresh_work_event()

    async def add_request(self, req: Request) -> None:
        async with self._lock:
            self.scheduler.add_request(req)
            self._refresh_work_event()

    async def add_requests(self, requests: List[Request]) -> None:
        async with self._lock:
            for req in requests:
                self.scheduler.add_request(req)
            self._refresh_work_event()

    async def schedule(self, wait: bool = False, timeout: Optional[float] = None) -> SchedulerOutput:
        while True:
            async with self._lock:
                output = self.scheduler.schedule()
                self._refresh_work_event()
                if not output.is_empty or not wait:
                    return output

            await self._wait_for_work(timeout if timeout is not None else self.config.wait_timeout)

    async def schedule_loop(self, timeout: Optional[float] = None) -> AsyncIterator[SchedulerOutput]:
        while True:
            output = await self.schedule(wait=True, timeout=timeout)
            if output.is_empty and not self.config.yield_empty:
                continue
            yield output

    async def update_from_output(
        self,
        output: SchedulerOutput,
        generated_tokens: Dict[int, List[int]],
        release_finished: bool = True,
    ) -> None:
        async with self._lock:
            self.scheduler.update_from_output(output, generated_tokens, release_finished=release_finished)
            self._refresh_work_event()

    async def finish_request(self, req_id: int) -> None:
        async with self._lock:
            self.scheduler.finish_request(req_id)
            self._refresh_work_event()

    async def abort_request(self, req_id: int) -> None:
        async with self._lock:
            self.scheduler.abort_request(req_id)
            self._refresh_work_event()

    async def stats(self) -> dict:
        async with self._lock:
            return {
                "num_running": self.scheduler.num_running,
                "num_waiting": self.scheduler.num_waiting,
                "num_free_blocks": self.scheduler.num_free_blocks,
            }

    def _has_schedulable_work_unlocked(self) -> bool:
        if self.scheduler.num_waiting > 0:
            return True
        return any(req.num_uncomputed > 0 for req in self.scheduler.running)

    def _refresh_work_event(self) -> None:
        if self._has_schedulable_work_unlocked():
            self._work_event.set()
        else:
            self._work_event.clear()

    async def _wait_for_work(self, timeout: Optional[float]) -> None:
        if timeout is None:
            await self._work_event.wait()
            return
        try:
            await asyncio.wait_for(self._work_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return
