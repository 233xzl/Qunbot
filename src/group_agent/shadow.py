"""Shadow ingestion from SnowLuma's OneBot v11 forward WebSocket.

By default the shadow never sends a frame: no API calls, no replies, no reactions. It only
records group messages and group recall notices into the local store, so the production
AstrBot link (reverse WebSocket) is unaffected. Transport is `gateway.OneBotGateway`.

Explicit opt-ins (each enables only the read-only action it needs):

- `backfill_replies=True`: `get_msg` for referenced messages missing from the store (T3.1);
- `media_factory`: immediate image collection with `get_image` (T3.3, `media.py`);
- `ShadowIngestor(admit=True, sink=RecordOnly())`: reply admission is evaluated and each
  accepted request is recorded (T3.2); no run is started and nothing is sent.
"""

import asyncio
from collections import Counter
from collections.abc import Callable
import time
from typing import Any

from .admission import AccessPolicy, AccessSnapshot, RequestSink
from .backfill import ReplyBackfill
from .contracts import Message
from .gateway import OneBotApi, OneBotGateway, ReceiveOnly
from .intake import EventIntake
from .store import Store

__all__ = ["ReceiveOnly", "ShadowIngestor", "run_shadow"]


class ShadowIngestor(EventIntake):
    """Event intake with shadow defaults: collection only, admission off unless asked."""

    def __init__(
        self,
        store: Store,
        bot_id: str,
        blocked_users: frozenset[str] = frozenset(),
        blocked_groups: frozenset[str] = frozenset(),
        clock: Callable[[], float] = time.time,
        on_stored: Callable[[Message], Any] | None = None,
        *,
        access: AccessPolicy | None = None,
        admit: bool = False,
        sink: RequestSink | None = None,
        media: Any | None = None,
    ):
        if access is None:
            access = AccessPolicy(
                AccessSnapshot(bot_id, None, frozenset(blocked_users), frozenset(blocked_groups))
            )
        elif access.bot_id != bot_id:
            raise ValueError("access policy is for another bot")
        super().__init__(
            store, access, admit=admit, sink=sink, media=media, on_stored=on_stored, clock=clock
        )

    @property
    def blocked_users(self) -> frozenset[str]:
        return self.access.snapshot.blocked_users

    @property
    def blocked_groups(self) -> frozenset[str]:
        return self.access.snapshot.blocked_groups


async def run_shadow(
    url: str,
    token: str,
    ingestor: ShadowIngestor,
    stop: asyncio.Event,
    *,
    connect: Callable[..., Any] | None = None,
    log: Callable[[str], None] = lambda _: None,
    reconnect_initial: float = 1.0,
    reconnect_max: float = 30.0,
    backfill_replies: bool = False,
    backfill_counts: Counter[str] | None = None,
    media_factory: Callable[[OneBotApi], Any] | None = None,
) -> Counter[str]:
    """Receive events until `stop` is set, reconnecting with capped backoff.

    Without `backfill_replies` and `media_factory` the connection is receive-only. With
    them, the only actions enabled are `get_msg` (references missing locally; stored as
    source=backfill, never a reply trigger) and `get_image` (image collection).

    Returns:
        Connection counters (connects, disconnects, frames, bad_frames, actions).
    """
    actions = set()
    if backfill_replies:
        actions.add("get_msg")
    if media_factory is not None:
        actions.add("get_image")
    gateway = OneBotGateway(
        url,
        token,
        ingestor.handle,
        allow_actions=bool(actions),
        actions=frozenset(actions) or None,
        connect=connect,
        log=log,
        reconnect_initial=reconnect_initial,
        reconnect_max=reconnect_max,
    )
    api = OneBotApi(gateway)
    media = None
    if media_factory is not None:
        media = media_factory(api)
        ingestor.media = media
    backfill = None
    if backfill_replies:
        backfill = ReplyBackfill(
            ingestor.store,
            api,
            ingestor.bot_id,
            lambda: ingestor.access.snapshot.blocked_users,
            clock=ingestor.clock,
            on_stored=media.schedule if media is not None else None,
        )
        if backfill_counts is not None:
            backfill.counts = backfill_counts
        ingestor.on_stored = backfill.schedule
    try:
        return await gateway.run(stop)
    finally:
        if backfill is not None:
            await backfill.aclose()
        if media is not None:
            await media.aclose()
