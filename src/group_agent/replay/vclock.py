"""Virtual-time asyncio loop: timers fire in order, but no wall-clock time passes.

When no callback is ready the selector jumps the clock to the next scheduled timer
instead of sleeping. `asyncio.sleep`, `asyncio.timeout` and `call_at` all follow the
virtual clock, so production code using them runs unchanged and deterministically.
"""

import asyncio
from contextlib import asynccontextmanager
import selectors
import time as _time


class SimulationStalled(RuntimeError):
    """Nothing runnable and no timer left while the main coroutine is still waiting."""


class _VirtualSelector(selectors.SelectSelector):
    def __init__(self):
        super().__init__()
        self.advance = None
        self.held = None

    def select(self, timeout=None):
        if self.held():
            # Live I/O in flight: wait for real (sockets, executor wake-ups); the loop
            # clock follows wall time while held, so no explicit advance is needed.
            return super().select(timeout)
        ready = super().select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            raise SimulationStalled("replay deadlock: no ready task and no pending timer")
        self.advance(timeout)
        return []


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """Virtual clock that switches to wall time while live work holds it (`real_time()`).

    The live-model replay tier calls real models inside `real_time()`: virtual time jumps
    across idle chat history, but deadlines and latencies of live calls are real seconds.
    """

    def __init__(self, start: float = 0.0):
        self._virtual_now = float(start)
        self._holds = 0
        self._held_since = 0.0
        selector = _VirtualSelector()
        super().__init__(selector)
        selector.advance = self._advance
        selector.held = lambda: self._holds > 0

    def _advance(self, seconds: float) -> None:
        self._virtual_now += max(0.0, seconds)

    def time(self) -> float:
        if self._holds:
            return self._virtual_now + (_time.perf_counter() - self._held_since)
        return self._virtual_now

    def hold(self) -> None:
        if self._holds == 0:
            self._held_since = _time.perf_counter()
        self._holds += 1

    def release(self) -> None:
        if self._holds <= 0:
            raise RuntimeError("release without hold")
        if self._holds == 1:
            self._virtual_now += _time.perf_counter() - self._held_since
        self._holds -= 1


@asynccontextmanager
async def real_time():
    """Run the block on wall time when the current loop is virtual; no-op otherwise."""
    loop = asyncio.get_running_loop()
    if not isinstance(loop, VirtualTimeLoop):
        yield
        return
    loop.hold()
    try:
        yield
    finally:
        loop.release()


def run_virtual(coro, start: float = 0.0):
    loop = VirtualTimeLoop(start)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
