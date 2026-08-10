from __future__ import annotations

import asyncio
import time


class Clock:
    def now(self) -> float:
        raise NotImplementedError

    async def sleep_until(self, deadline: float) -> None:
        raise NotImplementedError


class RealtimeClock(Clock):
    def now(self) -> float:
        return time.monotonic()

    async def sleep_until(self, deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - self.now()))


class VirtualClock(Clock):
    def __init__(self):
        self.value = 0.0

    def now(self) -> float:
        return self.value

    async def sleep_until(self, deadline: float) -> None:
        self.value = max(self.value, deadline)
        await asyncio.sleep(0)
