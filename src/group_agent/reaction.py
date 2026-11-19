"""D-20 "processing" reaction: a fixed emoji on each admitted reply request.

`ProcessingReaction` is the only holder of the gateway's single `ReactionChannel`
(`set_msg_emoji_like`, emoji id fixed at issue, `set` always true, rate-limited). Intake
calls `admitted(request)` right after a fresh request was queued (`EventIntake.on_admitted`),
before any scheduling or model call; the reaction then runs as its own task.

Checks before the action: the request must come from this process's intake (`admitted`),
and the store must atomically grant the claim (`Store.claim_reaction`): a queued live
request of kind chat, received within `max_age_s`, never claimed before. The claim is
persistent, so redelivered events and restarts never react twice. Backfilled and synced
messages are not requests; the owner's `#新版` command is not a request either. A failed
reaction is recorded (`failed_<kind>[_<retcode>]`) and logged once; it never touches the run.

Off by default (`[serve] processing_reaction`, `--reaction`): AstrBot reacts during the
shadow trial.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
import time
from typing import Any

from .admission import ReplyRequest
from .store import Store


class ProcessingReaction:
    def __init__(
        self,
        channel: Any,
        store: Store,
        *,
        should_react: Callable[[ReplyRequest], bool] | None = None,
        clock: Callable[[], float] = time.time,
        max_age_s: float = 60.0,
        log: Callable[[str], None] = lambda _: None,
    ):
        self._channel, self._store = channel, store
        self._should_react = should_react
        self.clock, self.max_age_s, self.log = clock, max_age_s, log
        self.counts: Counter[str] = Counter()
        self._tasks: set[asyncio.Task] = set()

    def admitted(self, request: ReplyRequest) -> None:
        """Intake hook (synchronous, on the event loop): claim and schedule one reaction."""
        if not isinstance(request, ReplyRequest):
            self.counts["refused_not_request"] += 1
            return
        if self._should_react is not None and not self._should_react(request):
            self.counts["skipped_policy"] += 1
            return
        key = request.key
        if not self._store.claim_reaction(*key, now=self.clock(), max_age_s=self.max_age_s):
            self.counts["refused_claim"] += 1
            return
        task = asyncio.get_running_loop().create_task(self._react(key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _react(self, key: tuple[str, str, str]) -> None:
        try:
            result = await self._channel.set_msg_emoji_like(key[2])
        except Exception as exc:  # noqa: BLE001 - never affects the run
            outcome = f"failed_{type(exc).__name__}"
        else:
            if result.status == "ok":
                outcome = "ok"
            else:
                outcome = f"failed_{result.error_kind}"
                if result.retcode is not None:
                    outcome += f"_{result.retcode}"
        self.counts[outcome if outcome == "ok" else "failed"] += 1
        if outcome != "ok":
            self.log(f"processing reaction {outcome}")
        try:
            self._store.set_reaction(*key, outcome)
        except Exception:  # noqa: BLE001 - store closed at shutdown
            self.counts["record_errors"] += 1

    async def aclose(self, timeout: float = 5.0) -> None:
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)
        for task in list(self._tasks):
            task.cancel()
