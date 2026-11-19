"""Run admission into bounded execution slots (RUN-04) and busy-notice throttling (D-08).

Pure asyncio primitives; it uses only `loop.time()` and futures, so it works unchanged
under the replay harness's virtual clock.
"""

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field

from .policy import RuntimeLimits


@dataclass(eq=False)
class Ticket:
    user: str
    future: asyncio.Future
    queued: bool
    _held: bool = field(default=False, repr=False)


class RunScheduler:
    """Global and per-user active limits with a bounded FIFO waiting queue.

    `request()` never blocks: it grants a slot, queues the request, or returns None when
    the queue is full (the caller then sends at most one busy notice per user window).
    """

    def __init__(self, limits: RuntimeLimits = RuntimeLimits()):
        self.limits = limits
        self.active: Counter[str] = Counter()
        self.total_active = 0
        self.max_total_active = 0
        self._waiting: deque[Ticket] = deque()

    def update(self, limits):
        self.limits = limits
        for waiting in list(self._waiting):
            if self._can_run(waiting.user):
                self._waiting.remove(waiting)
                self._grant(waiting)

    def _can_run(self, user: str) -> bool:
        return (
            self.total_active < self.limits.global_active
            and self.active[user] < self.limits.user_active
        )

    def _grant(self, ticket: Ticket) -> None:
        self.active[ticket.user] += 1
        self.total_active += 1
        self.max_total_active = max(self.max_total_active, self.total_active)
        ticket._held = True
        ticket.future.set_result(None)

    def waiting_count(self, user: str | None = None) -> int:
        return sum(1 for t in self._waiting if user is None or t.user == user)

    def request(self, user: str) -> Ticket | None:
        future = asyncio.get_running_loop().create_future()
        if self._can_run(user):
            ticket = Ticket(user, future, queued=False)
            self._grant(ticket)
            return ticket
        if (
            len(self._waiting) >= self.limits.global_queue
            or self.waiting_count(user) >= self.limits.user_queue
        ):
            return None
        ticket = Ticket(user, future, queued=True)
        self._waiting.append(ticket)
        return ticket

    async def acquire(self, ticket: Ticket) -> None:
        """Wait for the slot, bounded by the queue wait limit (raises TimeoutError)."""
        try:
            async with asyncio.timeout(self.limits.queue_wait_seconds):
                await asyncio.shield(ticket.future)
        except BaseException:
            # Abandoned (timeout/cancel): give back a slot granted in the meantime too.
            self.release(ticket)
            raise

    def release(self, ticket: Ticket) -> None:
        if not ticket._held:
            if ticket in self._waiting:
                self._waiting.remove(ticket)
            return
        ticket._held = False
        self.active[ticket.user] -= 1
        self.total_active -= 1
        for waiting in list(self._waiting):
            if self._can_run(waiting.user):
                self._waiting.remove(waiting)
                self._grant(waiting)


class BusyNotifier:
    """D-08: at most one busy notice per user per window."""

    def __init__(self, limits: RuntimeLimits = RuntimeLimits()):
        self.window = limits.busy_notice_window_seconds
        self._last: dict[str, float] = {}

    def should_notify(self, user: str, now: float) -> bool:
        last = self._last.get(user)
        if last is not None and now - last < self.window:
            return False
        self._last[user] = now
        return True
