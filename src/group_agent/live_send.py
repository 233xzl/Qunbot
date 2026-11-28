"""Live answer sending after the cutover (T3.7 rest; RUN-07..09, RUN-12, D-10, D-21).

`LiveSender` is the only holder of the gateway's single `AnswerSendChannel`
(`send_group_msg`, issued once, only with `[serve] live_send` / `--live-send`). The run
flow records every outgoing text first (an outbound row, mode `live_send`, idempotency key
per request message and kind, so one request gets at most one answer, also across
restarts), runs the pre-publish check (request recalled, requester/group blocked, used
knowledge revisions, cost tier) before and after the send-rate wait (D-10: same group 2 s
apart, 10 per group and 20 per account per minute), and only then hands the run id to
`deliver`. The orchestrator holds neither gateway nor channel: it gets `deliver` as an
opaque hook that only accepts ids of rows the flow wrote.

`deliver(run_id)`:

1. the row must exist, be a live `answer` / `busy_notice` row still `pending`, with
   1..`MAX_REPLY_CHARS` characters (the orchestrator already cut longer text,
   `outbox.fit_message_text`);
2. a last re-check right before the frame: the request message is stored and not
   recalled, it is the requester's, requester and group are not blocked in the current
   access snapshot, and the final text check passes (no bot/owner/group/requester ids, no
   account numbers, no key-like strings; `send_check.final_text_check`); otherwise the
   row ends `failed` with the reason and nothing is sent;
3. pending -> sending, atomically and once (`Store.claim_outbound`);
4. the old bot's message shape: a reply segment quoting the request plus one text segment,
   no mention (AstrBot `reply_with_quote = true`, `reply_with_mention = false`; the bot's
   replies in SnowLuma's message store are reply + text);
5. `sent`: QQ message id and `sent_at` on the row, the message stored as the bot's own and
   marked `answered_by` the request (the platform does not push it back), so a member
   quoting the answer gets the question through strategy C's `answers` hop; `failed`:
   known not delivered, reason kept; `uncertain`: timeout or connection lost after the frame
   was written - never resent (RUN-12: recorded only).

A run cancelled while its frame is in flight (recall, deadline, shutdown) does not cancel
the frame: the send is shielded and its outcome recorded when it completes. A process that
ends mid-send leaves the row `sending`; the next start marks it `uncertain`
(`Store.interrupt_active_runs`). Logs and counts carry no ids or content.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
import time
from typing import Any

from .admission import AccessPolicy
from .contracts import Message
from .gateway import SendResult
from .outbox import MAX_REPLY_CHARS
from .send_check import final_text_check
from .store import Store

SEND_KINDS = ("answer", "busy_notice", "owner_control")
# Kinds that are replies to a member's request; refused while the owner has paused the bot.
REPLY_KINDS = ("answer", "busy_notice")


class LiveSender:
    def __init__(
        self,
        channel: Any,
        store: Store,
        access: AccessPolicy,
        *,
        clock: Callable[[], float] = time.time,
        max_chars: int = MAX_REPLY_CHARS,
        send_timeout: float = 15.0,
        log: Callable[[str], None] = lambda _: None,
    ):
        self._channel, self._store, self._access = channel, store, access
        self.clock, self.max_chars, self.send_timeout, self.log = (
            clock,
            max_chars,
            send_timeout,
            log,
        )
        self.counts: Counter[str] = Counter()
        self._detached: set[asyncio.Future] = set()
        self._paused: Callable[[], bool] | None = None
        self.enabled: Callable[[], bool] = lambda: True

    def set_pause_gate(self, paused: Callable[[], bool]) -> None:
        """Owner remote pause (`owner_control.py`): while `paused()` is true no answer or
        busy notice row is sent (it ends `failed/paused`); the control confirmation is."""
        self._paused = paused

    @staticmethod
    def segments(reply_to: str, text: str) -> list[dict]:
        rid: int | str = int(reply_to) if reply_to.isdigit() else reply_to
        return [{"type": "reply", "data": {"id": rid}}, {"type": "text", "data": {"text": text}}]

    # -- checks -----------------------------------------------------------------------
    @staticmethod
    def _live_pending(row: dict | None) -> bool:
        return (
            row is not None
            and row["mode"] == "live_send"
            and row["kind"] in SEND_KINDS
            and row["state"] == "pending"
        )

    def refusal(self, row: dict | None) -> str | None:
        """Why `row` must not be sent now, or None."""
        if row is None:
            return "no_outbound_row"
        if not self.enabled():
            return "sending_disabled"
        if row["mode"] != "live_send" or row["kind"] not in SEND_KINDS:
            return "not_live_row"
        if row["state"] != "pending":
            return "already_sent"
        text = row["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > self.max_chars:
            return "bad_length"
        snap = self._access.snapshot
        if row["bot_id"] != snap.bot_id:
            return "bot_mismatch"
        control = row["kind"] == "owner_control"
        if not control and not self._store.allows_group_reply(row["bot_id"], row["group_id"]):
            return "group_stopped"
        if not control and self._paused is not None and self._paused():
            return "paused"
        request = self._store.lookup(row["bot_id"], row["group_id"], row["reply_to"])
        if request is None or request["revoked"]:
            return "request_recalled"
        if request["user_id"] != row["requester"]:
            return "requester_mismatch"
        if control and not snap.is_owner(row["requester"]):
            return "not_owner"  # a control confirmation answers the owner's command only
        if snap.is_blocked(row["group_id"], row["requester"]):
            return "blocked"
        forbidden = {snap.bot_id, snap.owner_id or "", row["group_id"], row["requester"]}
        return final_text_check(text, forbidden)

    # -- one send ---------------------------------------------------------------------
    async def deliver(self, run_id: str) -> SendResult:
        row = self._store.outbound(run_id)
        reason = self.refusal(row)
        if reason is not None:
            if self._live_pending(row):
                self._store.outbound_update(run_id, self.clock(), state="failed", error_kind=reason)
            self.counts[f"refused_{reason}"] += 1
            self.log(f"live send refused: {reason}")
            return SendResult("failed", None, reason)
        if not self._store.claim_outbound(run_id, SEND_KINDS):
            self.counts["refused_already_sent"] += 1
            return SendResult("failed", None, "already_sent")
        assert row is not None
        send = asyncio.ensure_future(
            self._channel.send_group_msg(
                row["group_id"],
                self.segments(row["reply_to"], row["text"]),
                timeout=self.send_timeout,
            )
        )
        try:
            await asyncio.shield(send)
        except asyncio.CancelledError:
            if send.done():
                self._settle(row, self._outcome(send))
            else:  # the run ended, the frame did not: record its outcome when it completes
                self._detached.add(send)
                send.add_done_callback(lambda t, r=row: self._settle_detached(r, t))
            raise
        except Exception:  # noqa: BLE001 - recorded by `_outcome` as uncertain
            pass
        return self._settle(row, self._outcome(send))

    @staticmethod
    def _outcome(task: asyncio.Future) -> SendResult:
        if task.cancelled():
            return SendResult("uncertain", None, "cancelled")
        exc = task.exception()
        if exc is not None:  # claimed already: the frame may have been written
            return SendResult("uncertain", None, f"error_{type(exc).__name__}")
        result = task.result()
        if not isinstance(result, SendResult) or result.status not in (
            "sent",
            "failed",
            "uncertain",
        ):
            return SendResult("uncertain", None, "bad_result")
        return result

    def _settle_detached(self, row: dict, task: asyncio.Future) -> None:
        self._detached.discard(task)
        try:
            self._settle(row, self._outcome(task))
        except Exception:  # noqa: BLE001 - store closed at shutdown: next start -> uncertain
            self.counts["record_errors"] += 1

    def _settle(self, row: dict, result: SendResult) -> SendResult:
        now = self.clock()
        sent = result.status == "sent"
        self._store.outbound_update(
            row["run_id"],
            now,
            state=result.status,
            error_kind=result.error_kind,
            qq_message_id=result.message_id,
            sent_at=now if sent else None,
        )
        self.counts[f"{row['kind']}_{result.status}"] += 1
        if not sent:
            self.log(f"live send {result.status}: {result.error_kind}")
        elif result.message_id and row["kind"] in REPLY_KINDS:
            self._remember(row, result.message_id, now)
        return result

    def _remember(self, row: dict, message_id: str, now: float) -> None:
        """Store the sent message as the bot's own and link it to the request."""
        bot, group = row["bot_id"], row["group_id"]
        message = Message(
            bot, group, message_id, bot, tuple(self.segments(row["reply_to"], row["text"]))
        )
        seq, _ = self._store.ingest(message, int(now), now)
        request = self._store.lookup(bot, group, row["reply_to"])
        if seq is not None and request is not None:
            self._store.mark_related(request["seq"], [seq], "answered_by")

    async def aclose(self, timeout: float = 15.0) -> None:
        """Wait for frames still in flight after their runs ended (shutdown)."""
        if self._detached:
            await asyncio.wait(set(self._detached), timeout=timeout)
