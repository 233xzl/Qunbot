"""Reference lookup for messages missing from the local store (T3.1, MIG-02).

When a stored message has a reply segment pointing to a message the store does not have,
`ReplyBackfill` calls the read-only `get_msg` action and stores the result with
source="backfill". Backfilled messages are ordinary context material: they never pass
through reply admission, so they cannot trigger a run.

Limits: lookup depth (a fetched message may itself reply to a missing one), lookups per
triggering message, in-flight lookups, and a short negative cache for ids that failed.
The fetched message must belong to the same bot and group as the referencing message;
anything else is rejected, so a lookup cannot pull another group's content.
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from collections.abc import Callable
from copy import deepcopy
import time
from typing import Any

from .contracts import Message
from .gateway import GatewayError
from .store import Store


def message_from_get_msg(
    data: Any, bot_id: str, group_id: str, message_id: str
) -> tuple[Message, int | None] | str:
    """Validate a `get_msg` result for the expected group and id.

    Returns:
        `(message, platform_time)`, or a rejection reason string.
    """
    if not isinstance(data, dict):
        return "invalid"
    if data.get("message_type") not in (None, "group"):
        return "not_group"
    got_group = data.get("group_id")
    if isinstance(got_group, bool) or str(got_group) != group_id:
        return "other_group"  # missing group_id also fails closed
    got_id = data.get("message_id", data.get("real_id"))
    if isinstance(got_id, bool) or str(got_id) != message_id:
        return "id_mismatch"
    user_id = data.get("user_id")
    if user_id is None and isinstance(data.get("sender"), dict):
        user_id = data["sender"].get("user_id")
    if isinstance(user_id, bool) or not isinstance(user_id, (str, int)) or not str(user_id):
        return "invalid"
    segments = data.get("message")
    if not isinstance(segments, list) or not all(
        isinstance(s, dict) and isinstance(s.get("type"), str) and isinstance(s.get("data"), dict)
        for s in segments
    ):
        return "invalid"
    platform_time = data.get("time")
    if isinstance(platform_time, bool) or not isinstance(platform_time, int):
        platform_time = None
    message = Message(bot_id, group_id, message_id, str(user_id), tuple(deepcopy(segments)))
    return message, platform_time


class ReplyBackfill:
    def __init__(
        self,
        store: Store,
        api: Any,
        bot_id: str,
        blocked_users: frozenset[str] | Callable[[], frozenset[str]] = frozenset(),
        *,
        max_depth: int = 3,
        max_lookups: int = 4,
        max_inflight: int = 8,
        failure_ttl: float = 600.0,
        clock: Callable[[], float] = time.time,
        on_stored: Callable[[int, Message], Any] | None = None,
    ):
        self.store, self.api, self.bot_id = store, api, bot_id
        # A callable reads the current blocklist (runtime updates via AccessPolicy).
        self.blocked_users = blocked_users
        self.on_stored = on_stored  # e.g. image collection for the fetched message
        self.max_depth, self.max_lookups = max_depth, max_lookups
        self.max_inflight, self.failure_ttl = max_inflight, failure_ttl
        self.clock = clock
        self.counts: Counter[str] = Counter()
        self._failed: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._inflight: set[tuple[str, str]] = set()
        self._tasks: set[asyncio.Task] = set()

    def schedule(self, message: Message) -> bool:
        """Start a background lookup for `message`'s missing references, if any."""
        if not message.reply_ids:
            return False
        if len(self._tasks) >= self.max_inflight:
            self.counts["dropped_busy"] += 1
            return False
        task = asyncio.get_running_loop().create_task(self.resolve(message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def resolve(self, message: Message) -> list[str]:
        """Fetch missing referenced messages of `message`; returns one status per reference."""
        group_id = message.group_id
        pending = [(rid, 1) for rid in message.reply_ids]
        statuses, lookups, seen = [], 0, set()
        while pending:
            rid, depth = pending.pop(0)
            if rid in seen:
                continue
            seen.add(rid)
            status, fetched = await self._one(group_id, rid, lookups)
            if status not in ("known", "recent_failure", "in_flight", "limit"):
                lookups += 1
            statuses.append(status)
            self.counts[status] += 1
            if fetched is not None and depth < self.max_depth:
                pending.extend((r, depth + 1) for r in fetched.reply_ids)
        return statuses

    async def _one(self, group_id: str, rid: str, lookups: int) -> tuple[str, Message | None]:
        key = (group_id, rid)
        if self.store.lookup(self.bot_id, group_id, rid) is not None:
            return "known", None
        failed_at = self._failed.get(key)
        if failed_at is not None and self.clock() - failed_at < self.failure_ttl:
            return "recent_failure", None
        if key in self._inflight:
            return "in_flight", None
        if lookups >= self.max_lookups:
            return "limit", None
        self._inflight.add(key)
        try:
            data = await self.api.get_msg(rid)
        except GatewayError as exc:
            self._remember_failure(key)
            return f"failed_{exc.kind}", None
        finally:
            self._inflight.discard(key)
        parsed = message_from_get_msg(data, self.bot_id, group_id, rid)
        if isinstance(parsed, str):
            self._remember_failure(key)
            return f"rejected_{parsed}", None
        fetched, platform_time = parsed
        blocked = self.blocked_users() if callable(self.blocked_users) else self.blocked_users
        if fetched.user_id in blocked:
            self._remember_failure(key)  # not collected; do not ask again soon
            return "blocked", None
        seq, fresh = self.store.ingest(fetched, platform_time, self.clock(), source="backfill")
        if fresh and self.on_stored is not None:
            self.on_stored(seq, fetched)
        return ("stored" if fresh else "duplicate"), fetched

    def _remember_failure(self, key: tuple[str, str]) -> None:
        self._failed[key] = self.clock()
        self._failed.move_to_end(key)
        while len(self._failed) > 1024:
            self._failed.popitem(last=False)
