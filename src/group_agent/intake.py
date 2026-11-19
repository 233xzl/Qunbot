"""Event intake: collection (T3.3) and reply admission (T3.2) for one OneBot event.

`EventIntake.intake(event)` is the single place that turns a pushed event into store
writes. It is transport-free (used by the shadow, the replay pipeline and tests) and
synchronous; slow work (image downloads, reference lookups) is only scheduled here.

Order of checks for a group message:

1. parse (fails closed), blocked group/user -> not stored at all (CTX-01, SEC-02);
2. reply candidate? (real top-level @bot, not the bot itself, not `message_sent`);
3. collection switched off for the group and not a candidate -> not stored (CTX-05);
4. store once per (bot, group, message id); a candidate is stored together with its
   reply request in one transaction (persistent dedup, RUN-01);
5. schedule image collection and reference lookup for the fresh message;
6. hand a fresh request to the sink (scheduler queue, or record-only in shadow mode); a
   queued request goes to `on_admitted` at once (D-20 "processing" reaction).

Recall notices remove content and media links (CTX-07, D-02); unreferenced media files
are deleted right away when a media collector is attached.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

from .admission import AccessPolicy, ReplyRequest, RequestSink, reply_candidate
from .contracts import Message
from .gateway import (
    GroupMessageEvent,
    IgnoredEvent,
    MetaEvent,
    NoticeEvent,
    RecallEvent,
    parse_event,
)
from .store import Store


@dataclass(frozen=True)
class IntakeResult:
    """`status` values: stored, trigger, duplicate, ignored_<reason>, meta_<type>,
    notice_<type>, recall_<revoked|tombstone|duplicate>."""

    status: str
    message: Message | None = None
    seq: int | None = None
    request: ReplyRequest | None = None


class EventIntake:
    def __init__(
        self,
        store: Store,
        access: AccessPolicy,
        *,
        admit: bool = True,
        sink: RequestSink | None = None,
        media: Any | None = None,
        on_stored: Callable[[Message], Any] | None = None,
        clock: Callable[[], float | None] = time.time,
        on_admitted: Callable[[ReplyRequest], Any] | None = None,
        control: Callable[[Message], str | None] | None = None,
        paused: Callable[[], bool] | None = None,
    ):
        self.store, self.access, self.clock = store, access, clock
        self.admit, self.sink, self.media = admit, sink, media
        self.on_stored = on_stored
        # Owner remote control (`owner_control.py`): `control(message)` returns the action
        # of a message that is an owner control command (owner only), else None; such a
        # message is stored flagged `owner_command` and never becomes a reply request.
        # `paused()`: while true a reply candidate is stored and recorded but not handed
        # to the sink (no run, no "processing" reaction); collection goes on.
        self.control, self.paused = control, paused
        # D-20: called once per fresh request the sink queued (the "processing" reaction).
        self.on_admitted = on_admitted
        self.counts: Counter[str] = Counter()
        self.request_states: Counter[str] = Counter()

    @property
    def bot_id(self) -> str:
        return self.access.bot_id

    def handle(self, event: Any) -> str:
        return self.intake(event).status

    def intake(self, event: Any) -> IntakeResult:
        result = self._intake(event)
        self.counts[result.status] += 1
        return result

    def _intake(self, event: Any) -> IntakeResult:
        snapshot = self.access.snapshot
        parsed = parse_event(event, snapshot.bot_id)
        if isinstance(parsed, IgnoredEvent):
            return IntakeResult(f"ignored_{parsed.reason}")
        if isinstance(parsed, MetaEvent):
            return IntakeResult(f"meta_{parsed.meta_event_type}")
        if isinstance(parsed, NoticeEvent):
            return IntakeResult(f"notice_{parsed.notice_type}")
        if isinstance(parsed, RecallEvent):
            if parsed.group_id in snapshot.blocked_groups:
                return IntakeResult("ignored_blocked")
            outcome = self.store.revoke(
                snapshot.bot_id, parsed.group_id, parsed.message_id, parsed.user_id, self.clock()
            )
            if outcome == "revoked" and self.media is not None:
                self.media.gc()
            return IntakeResult(f"recall_{outcome}")
        assert isinstance(parsed, GroupMessageEvent)
        message = parsed.message
        if snapshot.is_blocked(message.group_id, message.user_id):
            return IntakeResult("ignored_blocked", message)
        if self.control is not None and not parsed.self_sent and self.control(message):
            # An instruction, not chat: stored flagged (never context material), never a
            # reply request - also when collection is off for the group.
            seq, fresh = self.store.ingest(
                message, parsed.platform_time, self.clock(), kind="owner_command"
            )
            if not fresh:
                return IntakeResult("duplicate", message, seq)
            return IntakeResult("owner_control", message, seq)
        candidate = self.admit and reply_candidate(message, parsed.self_sent)
        if message.group_id in snapshot.collection_off_groups and not candidate:
            return IntakeResult("ignored_collection_off", message)
        received_at = self.clock()
        if candidate:
            sender = event.get("sender")
            names = (sender.get("card"), sender.get("nickname")) if isinstance(sender, dict) else ()
            name = next((v.strip() for v in names if isinstance(v, str) and v.strip()), "")
            seq, fresh, requested = self.store.ingest_request(
                message, parsed.platform_time, received_at, member_name=name
            )
        else:
            seq, fresh = self.store.ingest(message, parsed.platform_time, received_at)
            requested = False
        if not fresh:
            return IntakeResult("duplicate", message, seq)
        if self.media is not None:
            self.media.schedule(seq, message)
        if self.on_stored is not None:
            self.on_stored(message)
        if not candidate:
            return IntakeResult("stored", message, seq)
        if not requested:
            # The message row was gone (retention) but its request is still recorded.
            return IntakeResult("duplicate", message, seq)
        request = ReplyRequest(
            message, seq, parsed.platform_time, received_at, snapshot.is_owner(message.user_id)
        )
        if self.paused is not None and self.paused():
            self.counts["paused_requests"] += 1  # recorded only: no run, no reaction
            return IntakeResult("trigger", message, seq, request)
        if self.sink is not None:
            state = self.sink.submit(request)
            self.request_states[state] += 1
            if state != "recorded":
                self.store.set_request_state(*request.key, state)
            if state == "queued" and self.on_admitted is not None:
                try:
                    self.on_admitted(request)
                except Exception:  # noqa: BLE001 - a hook never affects admission
                    self.counts["admitted_hook_errors"] += 1
        return IntakeResult("trigger", message, seq, request)
