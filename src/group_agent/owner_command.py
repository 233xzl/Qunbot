"""The owner's `#新版` command (D-19): the only path on which the new framework sends a QQ
message during the shadow trial.

Trigger (`parse_command`, checked on event metadata only): a group message from the
configured owner id, not mentioning the bot (and not @all), whose top-level text without
whitespace is exactly `#新版`, with exactly one reply segment quoting a message of the same
group. Anyone else sending the same text is ignored without notice (counted and logged
once per message, no ids). A mention of the bot is never a command: AstrBot would answer
it as well.

Behaviour (`NewVersionCommand.handle`):

1. the quoted message must be stored here, not recalled, not the bot's, its sender and
   group not blocked; an @bot request with a reply run, or an ordinary message;
2. within `reuse_seconds` (30 min) of the question, a `recorded_not_sent` answer of its
   reply run is reused (an active reply run is awaited first); otherwise a new run answers
   the question with the context cut at the question (`Orchestrator.generate_for`: same run
   flow, usage ledger and daily cap; the owner is served at >= 100 %);
3. text = `〔新版〕` + answer; an outbound row (kind `new_version`, mode `live_send`) is
   written first - its idempotency key allows one row per question, so a question is sent
   at most once, also across restarts;
4. per-group/account send rate (D-10, own `SendRateLimiter`), re-check (recall, blocklists,
   final text check: no QQ/account numbers, no key-like strings), then
   `OwnerCommandSender.send` -> sent | failed | uncertain. Uncertain sends (timeout,
   connection lost after writing) are never retried (RUN-12 later). Failures are recorded
   only; nothing is posted about them.
5. a sent message is stored as the bot's message (reply segment + text), its id is kept in
   the outbound row (`answer_map`) and marked `answered_by` the question, so a member
   quoting it later gets the original question back through strategy C's `answers` hop.

The command message itself is flagged `kind='owner_command'` when it is recognised
(`observe`): it never enters any run's context, and retention keeps it only as the anchor
of its own `new_version` run. The same text from anyone else stays an ordinary message.

`OwnerCommandSender` is built inside `NewVersionCommand` from the gateway's single
`OwnerSendChannel`; no other module receives either.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
import json
import re
import time
from typing import Any
from uuid import uuid4

from .admission import AccessPolicy
from .contracts import Message
from .gateway import SendResult
from .outbox import SendRateLimiter
from .policy import RuntimeLimits
from .store import RUN_ACTIVE_STATES, Store, _message

COMMAND_TEXT = "#新版"
PREFIX = "〔新版〕"
KIND = "new_version"
REUSE_SECONDS = 1800.0
MAX_TEXT_CHARS = 3000
_SPACE = re.compile(r"\s+")
# Final text check: account-number wording next to a 5-11 digit number, and key-like strings.
_ACCOUNT_NUMBER = re.compile(
    r"(?:QQ|qq|Qq|扣扣|群号|账号|帐号|号码|uin|UIN)\s*[:：号是为码]{0,3}\s*\d{5,11}(?!\d)"
)
_SECRET = re.compile(
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{16,}"
    r"|(?i:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]"
    r"|(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9+_-])[A-Za-z0-9+_-]{40,}(?![A-Za-z0-9+_-])"
)


def parse_command(message: Message) -> str | None:
    """The quoted message id when `message` has the command shape (sender not checked)."""
    replies: list[str] = []
    text = []
    for seg in message.segments:
        kind, data = seg.get("type"), seg.get("data") or {}
        if kind == "reply":
            rid = data.get("id")
            if isinstance(rid, bool) or not isinstance(rid, (str, int)) or not str(rid):
                return None
            replies.append(str(rid))
        elif kind == "text":
            text.append(str(data.get("text") or ""))
        elif kind == "at":
            # A QQ client may add a mention of the quoted author; the bot or @all is never
            # part of a command (the old bot would answer as well).
            if str(data.get("qq")) in (message.bot_id, "all", "*"):
                return None
        else:
            return None
    if len(replies) != 1 or _SPACE.sub("", "".join(text)) != COMMAND_TEXT:
        return None
    return replies[0]


def final_text_check(text: str, forbidden: set[str]) -> str | None:
    """Reason to refuse the outgoing text, or None. `forbidden`: ids that must not appear
    (bot, owner, group, asker)."""
    for value in forbidden:
        if value and len(value) >= 5 and re.search(rf"(?<!\d){re.escape(value)}(?!\d)", text):
            return "contains_id"
    if _ACCOUNT_NUMBER.search(text):
        return "contains_account_number"
    if _SECRET.search(text):
        return "contains_secret_like"
    return None


class OwnerCommandSender:
    """Narrow send capability around the gateway's `OwnerSendChannel`: one action, the
    command's own group, a reply segment to the question plus one text segment that starts
    with `〔新版〕`, bounded length, and an atomic claim of the question's pending
    `new_version` outbound row (so one question can only be sent once)."""

    def __init__(self, channel: Any, store: Store, *, max_chars: int = MAX_TEXT_CHARS):
        self._channel, self._store, self._max_chars = channel, store, max_chars

    @staticmethod
    def segments(question_id: str, text: str) -> list[dict]:
        rid: int | str = int(question_id) if question_id.isdigit() else question_id
        return [{"type": "reply", "data": {"id": rid}}, {"type": "text", "data": {"text": text}}]

    def validate(
        self, run_id: str, command_group: str, group_id: str, question_id: str, segments
    ) -> str | None:
        if group_id != command_group:
            return "group_mismatch"
        if (
            len(segments) != 2
            or segments[0].get("type") != "reply"
            or str((segments[0].get("data") or {}).get("id")) != question_id
            or segments[1].get("type") != "text"
            or set(segments[1].get("data") or {}) != {"text"}
        ):
            return "bad_segments"
        text = segments[1]["data"]["text"]
        if not isinstance(text, str) or not text.startswith(PREFIX):
            return "bad_prefix"
        if len(text) > self._max_chars or len(text) <= len(PREFIX):
            return "bad_length"
        row = self._store.outbound(run_id)
        if (
            row is None
            or row["kind"] != KIND
            or row["mode"] != "live_send"
            or row["group_id"] != group_id
            or row["reply_to"] != question_id
            or row["text"] != text
        ):
            return "no_outbound_row"
        return None

    async def send(
        self, run_id: str, command_group: str, group_id: str, question_id: str, text: str
    ) -> SendResult:
        segments = self.segments(question_id, text)
        reason = self.validate(run_id, command_group, group_id, question_id, segments)
        if reason:
            return SendResult("failed", None, reason)
        con = self._store.con
        with con:  # claim: pending -> sending, exactly once per question
            claimed = con.execute(
                "UPDATE outbound SET state='sending' WHERE run_id=? AND kind=? AND state='pending'",
                (run_id, KIND),
            ).rowcount
        if claimed != 1:
            return SendResult("failed", None, "already_sent")
        return await self._channel.send_group_msg(group_id, segments)


class NewVersionCommand:
    def __init__(
        self,
        store: Store,
        access: AccessPolicy,
        generator: Any,
        *,
        channel: Any = None,
        limits: RuntimeLimits | None = None,
        clock: Callable[[], float] = time.time,
        reuse_seconds: float = REUSE_SECONDS,
        wait_run_seconds: float = 150.0,
        quote_wait_seconds: float = 8.0,
        poll_seconds: float = 0.5,
        log: Callable[[str], None] = lambda _: None,
    ):
        self.store, self.access, self.generator = store, access, generator
        # The only sender instance; built here, never handed out.
        self._sender = OwnerCommandSender(channel, store) if channel is not None else None
        self.rate = SendRateLimiter(limits or RuntimeLimits())
        self.clock, self.log = clock, log
        self.reuse_seconds, self.wait_run_seconds = reuse_seconds, wait_run_seconds
        self.quote_wait_seconds, self.poll_seconds = quote_wait_seconds, poll_seconds
        self.counts: Counter[str] = Counter()
        self.tasks: set[asyncio.Task] = set()
        self._inflight: set[tuple[str, str]] = set()

    @property
    def send_enabled(self) -> bool:
        return self._sender is not None

    # -- intake side ------------------------------------------------------------------
    def observe(self, result: Any) -> str | None:
        """Look at one intake result; schedule `handle` for a valid owner command.
        Returns "scheduled", "ignored_not_owner" or None (not a command)."""
        m = getattr(result, "message", None)
        if m is None or result.status not in ("stored", "ignored_collection_off"):
            return None
        if m.user_id == m.bot_id or parse_command(m) is None:
            return None
        if not self.access.snapshot.is_owner(m.user_id):
            self.counts["ignored_not_owner"] += 1
            self.log("new-version command from a non-owner ignored")
            return "ignored_not_owner"
        seq = result.seq
        if seq is None:  # collection is off in this group: keep the command for its run
            seq, _ = self.store.ingest(m, None, self.clock())
        # The command is an instruction, not chat: never context material (strategy C and
        # the others) and not pulled into the related set by quotes or context marks.
        self.store.mark_owner_command(seq)
        task = asyncio.get_running_loop().create_task(self.handle(m, seq))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        self.counts["scheduled"] += 1
        return "scheduled"

    async def aclose(self, timeout: float = 10.0) -> None:
        if self.tasks:
            await asyncio.wait(list(self.tasks), timeout=timeout)
        for task in list(self.tasks):
            task.cancel()

    # -- one command ----------------------------------------------------------------------
    async def handle(self, command: Message, command_seq: int) -> str:
        try:
            outcome = await self._handle(command, command_seq)
        except Exception as exc:  # a broken command must not stop intake
            outcome = f"error_{type(exc).__name__}"
        self.counts[outcome] += 1
        self.log(f"new-version command: {outcome}")
        return outcome

    def _blocked(self, group_id: str, user_id: str) -> bool:
        return not self.store.allows_group_reply(
            self.access.bot_id, group_id
        ) or self.access.snapshot.is_blocked(group_id, user_id)

    async def _question_row(self, command: Message, qid: str) -> dict | None:
        loop = asyncio.get_running_loop()
        end = loop.time() + self.quote_wait_seconds
        while True:  # the quoted message may still be fetched by the backfill
            row = self.store.lookup(command.bot_id, command.group_id, qid)
            if row is not None or loop.time() >= end:
                return row
            await asyncio.sleep(self.poll_seconds)

    async def _settled_reply_run(self, seq: int) -> dict | None:
        loop = asyncio.get_running_loop()
        end = loop.time() + self.wait_run_seconds
        while True:
            run = self.store.reply_run(seq)
            if run is None or run["state"] not in RUN_ACTIVE_STATES or loop.time() >= end:
                return run
            await asyncio.sleep(self.poll_seconds)

    async def _handle(self, command: Message, command_seq: int) -> str:
        qid = parse_command(command)
        if qid is None:
            return "not_command"
        if self._sender is None:
            return "send_disabled"  # recorded only
        group = command.group_id
        key = (group, qid)
        if key in self._inflight or self.store.outbound_for(command.bot_id, group, qid, KIND):
            return "duplicate"
        self._inflight.add(key)
        try:
            return await self._answer_and_send(command, command_seq, qid)
        finally:
            self._inflight.discard(key)

    async def _answer_and_send(self, command: Message, command_seq: int, qid: str) -> str:
        group = command.group_id
        row = await self._question_row(command, qid)
        if row is None:
            return "question_missing"
        if row["revoked"]:
            return "question_recalled"
        if row["user_id"] == command.bot_id:
            return "quoted_bot"
        if self._blocked(group, row["user_id"]):
            return "blocked"
        q_seq = row["seq"]
        q_row = self.store.message_row(q_seq)
        question = _message(q_row)
        q_time = q_row["platform_time"] if q_row["platform_time"] is not None else None
        q_time = q_time if q_time is not None else q_row["received_at"]
        reply = await self._settled_reply_run(q_seq)
        now = self.clock()
        fresh = q_time is not None and now - q_time <= self.reuse_seconds
        if (
            reply
            and fresh
            and reply["outbound_state"] == "recorded_not_sent"
            and reply["error_kind"] != "fallback_reply"  # a fixed failure reply: regenerate
        ):
            source, text = "reuse", reply["outbound_text"]
            run_id = uuid4().hex
            if not self.store.create_run(
                run_id,
                command_seq,
                question,
                state="succeeded",
                cost_mode=reply["cost_mode"] or "normal",
                now=now,
                purpose=KIND,
            ):
                return "duplicate"
            detail = {"new_version": {"source": "reuse", "source_run": reply["run_id"]}}
            fields: dict[str, Any] = {"detail": json.dumps(detail)}
            if reply["route"]:
                fields["route"] = reply["route"]
            self.store.update_run(run_id, now, **fields)
        else:
            source = "regenerate"
            run_id, text = await self.generator.generate_for(question, q_seq, run_seq=command_seq)
            if run_id is None:
                return "duplicate"
            if not text:
                return "generation_failed"
        self.counts[f"source_{source}"] += 1
        full = PREFIX + str(text).strip()
        _, created = self.store.outbound_enqueue(
            run_id, KIND, question, full, self.clock(), mode="live_send"
        )
        if not created:
            return "duplicate"
        forbidden = {command.bot_id, command.user_id, group, question.user_id}
        reason = self._recheck(question, full, forbidden)
        if reason is None:
            loop = asyncio.get_running_loop()
            at = self.rate.reserve(group, loop.time())
            if at > loop.time():
                await asyncio.sleep(at - loop.time())
            reason = self._recheck(question, full, forbidden)
        if reason is not None:
            self.store.outbound_update(run_id, self.clock(), state="failed", error_kind=reason)
            return f"refused_{reason}"
        result = await self._sender.send(run_id, group, group, qid, full)
        now = self.clock()
        if result.error_kind == "already_sent":  # claimed by someone else: leave it
            return "duplicate"
        self.store.outbound_update(
            run_id,
            now,
            state=result.status,
            error_kind=result.error_kind,
            qq_message_id=result.message_id,
            sent_at=now if result.status == "sent" else None,
        )
        if result.status == "sent":
            self._remember_sent(question, q_seq, command_seq, result.message_id, full, now)
        return result.status

    def _recheck(self, question: Message, text: str, forbidden: set[str]) -> str | None:
        row = self.store.lookup(question.bot_id, question.group_id, question.message_id)
        if row is None or row["revoked"]:
            return "question_recalled"
        if self._blocked(question.group_id, question.user_id):
            return "blocked"
        return final_text_check(text, forbidden)

    def _remember_sent(
        self,
        question: Message,
        q_seq: int,
        command_seq: int,
        message_id: str | None,
        text: str,
        now: float,
    ) -> None:
        """Store the sent message as the bot's own (the platform does not push it back)
        and link it to the question, so a follow-up quoting it finds the question."""
        self.store.mark_related(q_seq, [q_seq], "new_version_question")
        self.store.mark_related(q_seq, [command_seq], "new_version_command")
        if not message_id:
            return
        sent = Message(
            question.bot_id,
            question.group_id,
            message_id,
            question.bot_id,
            tuple(OwnerCommandSender.segments(question.message_id, text)),
        )
        seq, _ = self.store.ingest(sent, int(now), now)
        self.store.mark_related(q_seq, [seq], "answered_by")
