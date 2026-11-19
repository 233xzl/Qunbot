"""Owner remote control: `#暂停` / `#恢复` / `#状态` from QQ, while away from the computer.

A fully stopped process cannot hear a resume command, so the stop is an in-process PAUSE
that survives restarts: the paused flag lives in the serve store (`control_state`, schema
v10) and is loaded at start.

Recognition (`parse_control`, `OwnerControl.classify`, checked on event metadata only): a
group message in any group the bot is in, from the configured owner id, whose top-level
text without whitespace is exactly `#暂停`, `#恢复` or `#状态` (half-width `#`), with or
without a mention of the bot and nothing else (no quote, no image, no other mention, no
@all). Anyone else sending the same text is an ordinary message: no effect, no notice.
Intake stores a recognised command flagged `kind='owner_command'` (like `#新版`): never a
reply request, never context material, kept out of the related set by retention.

Paused state:

- `EventIntake` still stores every message (collection, quote backfill, self sync and
  retention are untouched) but hands no request to the scheduler, so there is no run, no
  model call, no send and no "processing" reaction - for the owner's ordinary @bot questions
  too; `Orchestrator.submit` refuses requests already queued, the pre-publish check refuses
  an answer ("paused") and `LiveSender` refuses a pending answer or busy-notice row
  (`failed/paused`);
- reply runs in flight at the moment of `#暂停` are cancelled through the normal cancel path
  (`Orchestrator.cancel_replies`); a frame already being written finishes and is recorded;
- the owner's `#新版` regenerations are not affected;
- `#恢复` clears the flag; nothing queued during the pause is answered afterwards.

Every command is recorded once per command message (`control_events`: time, group, action,
effect) and the state change is written in the same transaction (`Store.apply_control`), so a
redelivered event cannot apply twice. A command whose platform time is older than
`STALE_SECONDS` is recorded as `stale` and changes nothing.

Confirmation (`OwnerControl._confirm`): a short fixed text replies to the command message
(reply segment + one text segment, as answers do). The text is written first as an outbound
row of kind `owner_control` (idempotency key = command message, no run behind it), then
rate-limited with the answer limiter and handed by row id to the one answer sender through
the `deliver` hook (`LiveSender.deliver`: it re-checks the row - bot, owner, command message
stored and not recalled, group not blocked, final text check - and claims it once). This
module holds no gateway and no channel. Without live send (shadow mode) the row is kept as
`record_only` / `recorded_not_sent` and nothing is sent, also when `owner_command_send` is on:
that switch belongs to `#新版` only.

An already-paused `#暂停` and an already-running `#恢复` answer with the current state line
(like `#状态`). With `[serve] owner_control = false` the commands are ordinary messages and
a persisted pause is ignored (there would be no way to resume).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
import re
import time
from typing import Any

from .admission import AccessPolicy
from .contracts import Message
from .store import Store
from .policy import COST_DEGRADED, COST_REJECTED

COMMANDS = {"#暂停": "pause", "#恢复": "resume", "#状态": "status"}
KIND = "owner_control"
PAUSE_TEXT = "已暂停：新版不再回复，消息照常记录。发送 #恢复 继续。"
RESUME_TEXT = "已恢复回复。"
STALE_SECONDS = 300.0
_SPACE = re.compile(r"\s+")


def parse_control(message: Message) -> str | None:
    """`pause`, `resume` or `status` when `message` has a control command's shape (the
    sender is not checked), else None."""
    text = []
    for seg in message.segments:
        kind, data = seg.get("type"), seg.get("data") or {}
        if kind == "text":
            text.append(str(data.get("text") or ""))
        elif kind == "at":
            if str(data.get("qq")) != message.bot_id:  # @all or another member: not a command
                return None
        else:
            return None
    return COMMANDS.get(_SPACE.sub("", "".join(text)))


class PauseState:
    """The pause flag, mirrored from the store (`Store.control_state`) at start and after
    each applied command. `enabled=False`: never paused (owner control is switched off)."""

    def __init__(self, store: Store, *, enabled: bool = True, started: float | None = None) -> None:
        self.store, self.enabled = store, enabled
        self.paused, self.since = False, started if started is not None else time.time()
        self.reload()

    def reload(self) -> None:
        paused, since = self.store.control_state()
        self.paused = self.enabled and paused
        if since is not None:
            self.since = since

    def is_paused(self) -> bool:
        return self.paused


class OwnerControl:
    def __init__(
        self,
        store: Store,
        access: AccessPolicy,
        state: PauseState,
        *,
        cancel_replies: Callable[[str], int] = lambda reason: 0,
        deliver: Callable[[str], Awaitable[Any]] | None = None,
        rate: Any = None,
        status_info: Callable[[], dict] = lambda: {},
        clock: Callable[[], float] = time.time,
        stale_seconds: float = STALE_SECONDS,
        log: Callable[[str], None] = lambda _: None,
    ):
        """`deliver`: the live answer sender's hook (None = record only, shadow mode).
        `rate`: a `SendRateLimiter` shared with the answers. `status_info()`: `spent`,
        `cap`, `cost_mode` (non-owner policy), `answer_model`, `search_model` for the status line."""
        self.store, self.access, self.state = store, access, state
        self.cancel_replies, self.deliver, self.rate = cancel_replies, deliver, rate
        self.status_info, self.clock, self.stale_seconds, self.log = (
            status_info,
            clock,
            stale_seconds,
            log,
        )
        self.counts: Counter[str] = Counter()
        self.tasks: set[asyncio.Task] = set()

    # -- intake side ------------------------------------------------------------------
    def classify(self, message: Message) -> str | None:
        """Intake hook: the action of an owner control command, else None (not a command,
        or the same text from anyone else - an ordinary message)."""
        if message.user_id == message.bot_id:
            return None
        action = parse_control(message)
        if action is None:
            return None
        if not self.access.snapshot.is_owner(message.user_id):
            self.counts["ignored_not_owner"] += 1
            return None
        return action

    def observe(self, result: Any) -> str | None:
        """Look at one intake result; apply a fresh owner control command at once (before
        the next event is taken) and schedule its confirmation. Returns the effect or None."""
        message = getattr(result, "message", None)
        if message is None or result.status != "owner_control":
            return None
        action = parse_control(message)
        if action is None:
            return None
        now = self.clock()
        platform_time = self._platform_time(result.seq)
        stale = platform_time is not None and now - platform_time > self.stale_seconds
        effect = self.apply(
            action, message.bot_id, message.group_id, message.message_id, apply=not stale
        )
        if effect is None:
            self.counts["duplicate"] += 1
            return None
        self.state.reload()
        self.counts[f"action_{action}"] += 1
        self.counts[f"effect_{effect}"] += 1
        self.log(f"owner control: {action} -> {effect}")
        if effect != "stale":
            # The text is fixed now, at the state this command produced.
            text = self.text_for(effect)
            task = asyncio.get_running_loop().create_task(self._confirm(message, text))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return effect

    def apply(self, action, bot_id, source, event_id, *, apply=True):
        effect = self.store.apply_control(
            bot_id, source, event_id, action, self.clock(), apply=apply
        )
        self.state.reload()
        if effect == "paused":
            self.counts["cancelled_runs"] += self.cancel_replies("paused")
        return effect

    def _platform_time(self, seq: int | None) -> float | None:
        row = self.store.message_row(seq) if seq is not None else None
        return None if row is None else row["platform_time"]

    async def aclose(self, timeout: float = 10.0) -> None:
        if self.tasks:
            await asyncio.wait(list(self.tasks), timeout=timeout)
        for task in list(self.tasks):
            task.cancel()

    # -- text -------------------------------------------------------------------------
    def status_text(self) -> str:
        info = self.status_info()
        state = "已暂停" if self.state.paused else "运行中"
        parts = [state]
        mode = info.get("cost_mode")
        if mode == COST_REJECTED:
            parts.append("费用已达上限：仅响应主人，已降级（不联网、缩短上下文）")
        elif mode == COST_DEGRADED:
            parts.append("费用降级（不联网、缩短上下文）")
        if "spent" in info and "cap" in info:
            parts.append(f"今日 ¥{info['spent']:.2f}/¥{info['cap']:.2f}")
        if info.get("answer_model"):
            parts.append(f"回答 {info['answer_model']}")
        if info.get("search_model"):
            parts.append(f"搜索 {info['search_model']}")
        return "状态：" + "｜".join(parts)

    def text_for(self, effect: str) -> str:
        if effect == "paused":
            return PAUSE_TEXT
        if effect == "resumed":
            return RESUME_TEXT
        return self.status_text()  # status, already_paused, already_running

    # -- confirmation -------------------------------------------------------------------
    async def _confirm(self, command: Message, text: str) -> str:
        try:
            outcome = await self._send(command, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a broken confirmation must not stop intake or the state
            outcome = f"error_{type(exc).__name__}"
        self.counts[f"confirm_{outcome}"] += 1
        self.log(f"owner control confirmation: {outcome}")
        return outcome

    async def _send(self, command: Message, text: str) -> str:
        run_id = f"control:{command.group_id}:{command.message_id}"
        live = self.deliver is not None
        _, created = self.store.outbound_enqueue(
            run_id, KIND, command, text, self.clock(), mode="live_send" if live else "record_only"
        )
        if not created:
            return "duplicate"
        if not live:
            now = self.clock()
            self.store.outbound_update(run_id, now, state="recorded_not_sent", recorded_at=now)
            return "recorded"
        loop = asyncio.get_running_loop()
        if self.rate is not None:
            at = self.rate.reserve(command.group_id, loop.time())
            if at > loop.time():
                await asyncio.sleep(at - loop.time())
        result = await self.deliver(run_id)
        return str(getattr(result, "status", None) or "failed")
