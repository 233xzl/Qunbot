"""Reply admission (T3.2, MIG-01, CTX-02, RUN-01).

Collection and reply authorisation are separate decisions (MIG-01): every allowed group
message may be stored as context, but only a live message whose own top-level segments
mention the bot may become a reply request. Refused as triggers: private chats, CQ-code
or nested mentions (inside a reply/forward), @all, blocked users and groups, the bot's
own messages and `message_sent` echoes, duplicates, and messages fetched by reference
lookup (`source='backfill'`, which never pass through admission at all).

Identity data (bot id, owner, blocklists, groups with collection switched off) comes from
configuration through `AccessPolicy`; `update()` swaps the snapshot at runtime, so a
later blacklist change applies to the next event without a restart.

Dedup is persistent: the request row is written together with the message in one SQLite
transaction (`Store.ingest_request`), keyed by (bot, group, message id), so reconnect
replays, redelivered events and process restarts cannot request twice.

Accepted requests go to a `RequestSink`. The scheduler (T3.4) is not implemented yet:
`RequestQueue` is the hand-off point for it, `RecordOnly` is the shadow mode sink that
only records and never starts a run.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .contracts import Message, has_top_level_at


@dataclass(frozen=True)
class AccessSnapshot:
    bot_id: str
    owner_id: str | None = None
    blocked_users: frozenset[str] = frozenset()
    blocked_groups: frozenset[str] = frozenset()
    # Groups whose ordinary messages are not collected (CTX-05). Reply requests there
    # are still admitted; the request message itself is stored for dedup and context.
    collection_off_groups: frozenset[str] = frozenset()

    def is_blocked(self, group_id: str, user_id: str | None = None) -> bool:
        return group_id in self.blocked_groups or (
            user_id is not None and user_id in self.blocked_users
        )

    def is_owner(self, user_id: str) -> bool:
        return self.owner_id is not None and user_id == self.owner_id


class AccessPolicy:
    """Current identity and permission snapshot; `update()` replaces it atomically."""

    def __init__(self, snapshot: AccessSnapshot):
        self._snapshot = snapshot
        self.version = 1

    @classmethod
    def from_settings(cls, settings: Any) -> AccessPolicy:
        return cls(snapshot_from_settings(settings))

    @property
    def snapshot(self) -> AccessSnapshot:
        return self._snapshot

    @property
    def bot_id(self) -> str:
        return self._snapshot.bot_id

    def update(self, **changes: Any) -> AccessSnapshot:
        """Replace fields, e.g. `update(blocked_users=frozenset({...}))`."""
        if "bot_id" in changes:
            raise ValueError("bot identity cannot change at runtime")
        for name in ("blocked_users", "blocked_groups", "collection_off_groups"):
            if name in changes:
                changes[name] = frozenset(map(str, changes[name]))
        self._snapshot = replace(self._snapshot, **changes)
        self.version += 1
        return self._snapshot

    def replace_snapshot(self, snapshot: AccessSnapshot) -> None:
        if snapshot.bot_id != self._snapshot.bot_id:
            raise ValueError("bot identity cannot change at runtime")
        self._snapshot = snapshot
        self.version += 1


def snapshot_from_settings(settings: Any) -> AccessSnapshot:
    collection = getattr(settings, "collection", None)
    return AccessSnapshot(
        bot_id=settings.bot_id,
        owner_id=settings.owner_id,
        blocked_users=frozenset(settings.blocked_users),
        blocked_groups=frozenset(settings.blocked_groups),
        collection_off_groups=frozenset(getattr(collection, "disabled_groups", frozenset())),
    )


def reply_candidate(message: Message, self_sent: bool) -> bool:
    """Trigger rule only (blocklists are checked by the caller): a real top-level @bot in a
    message from someone other than the bot, not a `message_sent` echo."""
    return not self_sent and message.user_id != message.bot_id and has_top_level_at(message)


@dataclass(frozen=True)
class ReplyRequest:
    """One admitted reply request, handed to the scheduler (T3.4)."""

    message: Message
    seq: int  # run boundary: messages after it do not enter this run's input
    platform_time: int | None
    received_at: float | None
    is_owner: bool  # for the cost gate (D-04); never derived from group roles

    @property
    def key(self) -> tuple[str, str, str]:
        m = self.message
        return m.bot_id, m.group_id, m.message_id


class RequestSink(Protocol):
    def submit(self, request: ReplyRequest) -> str:
        """Return the resulting state: "recorded", "queued" or "dropped"."""
        ...


class RecordOnly:
    """Shadow mode: keep the request row and a count; never starts a run."""

    def __init__(self):
        self.counts: Counter[str] = Counter()

    def submit(self, request: ReplyRequest) -> str:
        self.counts["recorded"] += 1
        if request.is_owner:
            self.counts["owner"] += 1
        return "recorded"


class RequestQueue:
    """Bounded hand-off to the future scheduler; a full queue drops (no blocking)."""

    def __init__(self, maxsize: int = 32):
        self.queue: asyncio.Queue[ReplyRequest] = asyncio.Queue(maxsize)
        self.counts: Counter[str] = Counter()

    def submit(self, request: ReplyRequest) -> str:
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull:
            self.counts["dropped"] += 1
            return "dropped"
        self.counts["queued"] += 1
        return "queued"

    async def get(self) -> ReplyRequest:
        return await self.queue.get()
