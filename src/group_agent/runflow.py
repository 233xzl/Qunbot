"""Shared run flow: the deterministic replay and the online orchestrator run the same code.

One run (RUN-02..RUN-09):

1. scheduler slot (`scheduler.RunScheduler`: global/per-user active limits, bounded FIFO
   queue, queue wait limit); a full queue or an expired wait ends the run with at most one
   busy notice per user window (D-08), published like any outbound;
2. total deadline counted from arrival (`asyncio.timeout_at`), cancellation through
   `cancel_run(run_id, reason)` (user recall, blacklist, shutdown, a scripted fault);
3. context selection with the configured strategy (C by default) over the visible history
   (same group, not recalled, not blocked, within the retention hard limit), reasons kept
   per pick, a cap that shrinks in degraded cost mode, material gaps for quotes;
4. teaching commands are parsed by code from the sender's own top-level text and
   authorised in `teach()` by the event's sender id only (SEC-01); everything else goes
   through the agent (`vision`, `decide_route` - sync or async -, tools, `answer`); a
   model-chosen teaching route is a dropped request (recorded in `dropped_requests`,
   never a write); a web route is downgraded to chat in degraded cost mode (D-04);
5. pre-publish check (RUN-07) before and after the send-rate wait (RUN-09), with one
   regeneration when a used knowledge revision changed;
6. outbound through host hooks (idempotent enqueue, transitions, delivery).

Time is the running loop's clock: the replay runs on `replay.vclock`'s virtual loop, the
online mode on a normal loop, so the same deadlines, queue waits and rate limits apply.
Hosts subclass `RunFlow` and implement the methods marked "host hook".
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
from typing import Any

from .cards import message_cards
from .context_strategies import EXPLAINERS, STRATEGIES
from .contracts import Message
from .outbox import OutboundRecord, SendRateLimiter
from .policy import (
    BUSY_NOTICE_TEXT,
    COST_DEGRADED,
    RuntimeLimits,
    is_expired,
    prepublish_check,
)
from .scheduler import BusyNotifier, RunScheduler
from .teaching import parse_teaching

# Run id of the task currently executing a run (usage attribution, `usage.UsageLedger`).
CURRENT_GROUP: ContextVar[str | None] = ContextVar("current_group", default=None)
CURRENT_RUN: ContextVar[str | None] = ContextVar("current_run", default=None)


class AgentFailure(Exception):
    """A run-ending agent error; `reason` (default: class name) is recorded on the trace."""

    reason: str | None = None


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


@dataclass
class RunTrace:
    run_id: str
    message_id: str
    group_id: str
    user_id: str
    arrived_at: float
    cost_mode: str
    is_anchor: bool
    state: str = "created"
    reason: str | None = None
    transitions: list[tuple[float, str]] = field(default_factory=list)
    started_at: float | None = None
    ended_at: float | None = None
    queued: bool = False
    route: str | None = None
    intended_route: str | None = None
    clarify: bool = False
    context_ids: list[str] = field(default_factory=list)
    context_reasons: dict[str, str] = field(default_factory=dict)  # id -> relation/method
    material_gaps: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    model_calls: list[dict] = field(default_factory=list)
    knowledge: dict[str, int] = field(default_factory=dict)  # used as answer evidence
    knowledge_ids: list[str] = field(default_factory=list)  # all referenced (incl. teaching)
    knowledge_filtered: list[str] = field(default_factory=list)
    teaching: dict | None = None
    # MIG-06: write/tool requests from model output that code did not authorise; they are
    # dropped and only recorded here (no content, just source/kind/reason).
    dropped_requests: list[dict] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    moderation: dict | None = None
    # Strategy C `urges` (a bare "回答我" nudge): the asker's earlier unanswered request and
    # what this run did about it: {"earlier_message_id", "earlier": active|answered|open,
    # "outcome": merged|answer_earlier}. None when the run is not a nudge.
    nudge: dict | None = None

    def mark(self, now: float, state: str, reason: str | None = None) -> None:
        self.state = state
        if reason:
            self.reason = reason
        self.transitions.append((now, state))

    def end(self, now: float, state: str, reason: str | None = None) -> None:
        self.ended_at = now
        self.mark(now, state, reason)

    def as_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        data["transitions"] = [[_r(t), s] for t, s in self.transitions]
        for key in ("arrived_at", "started_at", "ended_at"):
            data[key] = _r(data[key])
        return data


class RunFlow:
    """Run flow shared by `replay.runner.CaseReplay` and `orchestrator.Orchestrator`.

    Hosts set `limits`, `store`, `settings` (bot/owner/blocklists), `agent`, `ledger`
    (knowledge revisions for RUN-07) and call `_init_flow()`.
    """

    # Unexpected exceptions inside a run: the replay lets them surface (a broken case must
    # be visible); the online host turns them into a failed run.
    catch_unexpected = False
    embedding_offline = False

    def _init_flow(self, limits: RuntimeLimits, strategy: str) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.limits = limits
        self.strategy_name, self.strategy = strategy, STRATEGIES[strategy]
        self.scheduler = RunScheduler(limits)
        self.busy = BusyNotifier(limits)
        self.rate = SendRateLimiter(limits)
        self.tasks: list[asyncio.Task] = []
        self.cancel_reasons: dict[str, str] = {}
        self._run_tasks: dict[str, asyncio.Task] = {}

    # -- clock -----------------------------------------------------------------------
    def now(self) -> float:
        return asyncio.get_running_loop().time()

    def remaining(self, trace: RunTrace) -> float:
        return trace.arrived_at + self.limits.deadline_seconds - self.now()

    def platform_now(self) -> float:  # host hook
        raise NotImplementedError

    # -- launch / cancel -------------------------------------------------------------
    def capture_run(self, trace: RunTrace) -> None:
        """Host hook: capture configuration before the run enters the scheduler."""

    def launch(self, trace: RunTrace, m: Message, seq: int, event: dict) -> asyncio.Task:
        self.capture_run(trace)

        async def body() -> None:
            CURRENT_RUN.set(trace.run_id)
            CURRENT_GROUP.set(trace.group_id)
            await self._run(trace, m, seq, event)

        task = asyncio.create_task(body())
        self._run_tasks[trace.run_id] = task
        task.add_done_callback(lambda _: self._run_tasks.pop(trace.run_id, None))
        self.tasks.append(task)
        return task

    def cancel_run(self, run_id: str, reason: str) -> bool:
        """Cancel a run that has not ended; its trace ends `cancelled` with `reason`."""
        task = self._run_tasks.get(run_id)
        if task is None or task.done():
            return False
        self.cancel_reasons.setdefault(run_id, reason)
        task.cancel()
        return True

    # -- host hooks --------------------------------------------------------------------
    def run_mocks(self, m: Message) -> Any:  # host hook: replay mocks, None online
        return None

    def cancel_plan(self, trace: RunTrace, mocks: Any) -> tuple[float, str] | None:
        """Scheduled cancellation `(loop time, reason)` (replay faults); None online."""
        return None

    def history_events(self, m: Message, seq: int) -> list[tuple[Message, dict | None]]:
        """Visible history before the boundary as (message, event dict) pairs."""
        raise NotImplementedError  # host hook

    def explain_options(self, m: Message) -> dict:
        return {}

    def context_built(self, trace: RunTrace, m: Message, seq: int, picks: list) -> None:
        """Called after context selection (online: related marks and snapshot)."""

    def earlier_request_status(self, trace: RunTrace, m: Message, message_id: str) -> str:
        """Host hook: what became of the asker's earlier request `message_id` (nudge):
        "active" (its run is queued/running or in any other non-final state), "answered"
        (its answer was recorded or sent; a failed one does not count) or "open" (no run, or a run that ended
        without an answer: failed, expired, rejected, cancelled, absent)."""
        return "open"

    async def prepare(self, trace: RunTrace, m: Message, seq: int) -> None:
        """Before context selection (online: wait briefly for quote backfill / images)."""

    async def prepare_context(self, trace: RunTrace, m: Message, seq: int) -> None:
        """Prepare selected material before any model call."""

    def teach(self, trace: RunTrace, m: Message, command) -> str:  # host hook
        raise NotImplementedError

    async def tools(self, trace: RunTrace, m: Message, mocks: Any) -> None:  # host hook
        raise NotImplementedError

    async def refresh_knowledge(self, trace: RunTrace, m: Message) -> None:
        """Re-read knowledge before regenerating after a revision change."""

    def capture_answer(self, trace: RunTrace, text: str, kind: str) -> bool:
        """True when the host takes this run's text itself instead of an outbound row
        (online: the owner's `#新版` regeneration, D-19); the host ends the trace."""
        return False

    async def before_publish(self, trace: RunTrace, mocks: Any) -> None:
        """Replay: delayed acknowledgement fault."""

    def outbound_enqueue(self, record: OutboundRecord, m: Message) -> tuple[Any, bool]:
        raise NotImplementedError  # host hook

    def outbound_transition(self, record: Any, state: str, reason: str | None = None) -> None:
        raise NotImplementedError  # host hook

    async def deliver(self, trace: RunTrace, record: Any, mocks: Any) -> str:
        """Hand the record over; returns the answer run's final trace state."""
        raise NotImplementedError  # host hook

    def abandon_outbound(self, trace: RunTrace) -> None:
        raise NotImplementedError  # host hook

    # -- one run -------------------------------------------------------------------
    async def _run(self, trace: RunTrace, m: Message, seq: int, event: dict) -> None:
        loop = asyncio.get_running_loop()
        mocks = self.run_mocks(m)
        ticket = self.scheduler.request(m.user_id)
        if ticket is None:
            trace.end(self.now(), "rejected_busy", "queue_full")
            if self.busy.should_notify(m.user_id, self.now()):
                await self._publish(trace, m, BUSY_NOTICE_TEXT, "busy_notice", None)
            return
        trace.queued = ticket.queued
        trace.mark(self.now(), "queued" if ticket.queued else "admitted")
        plan = self.cancel_plan(trace, mocks)
        handle = None
        if plan is not None:
            handle = loop.call_at(plan[0], self.cancel_run, trace.run_id, plan[1])
        try:
            async with asyncio.timeout_at(trace.arrived_at + self.limits.deadline_seconds):
                try:
                    await self.scheduler.acquire(ticket)
                except TimeoutError:
                    trace.end(self.now(), "queue_timeout", "queue_wait_limit")
                    if self.busy.should_notify(m.user_id, self.now()):
                        await self._publish(trace, m, BUSY_NOTICE_TEXT, "busy_notice", None)
                    return
                trace.started_at = self.now()
                trace.mark(self.now(), "running")
                try:
                    text = await self._work(trace, m, seq, event, mocks)
                finally:
                    self.scheduler.release(ticket)
                if text is not None:
                    await self._publish(trace, m, text, "answer", mocks)
        except TimeoutError:
            trace.end(self.now(), "timeout", "deadline")
            self.abandon_outbound(trace)
        except asyncio.CancelledError:
            trace.end(self.now(), "cancelled", self.cancel_reasons.pop(trace.run_id, None))
            self.abandon_outbound(trace)
        except AgentFailure as exc:
            trace.end(self.now(), "failed", exc.reason or type(exc).__name__)
            self.abandon_outbound(trace)
        except Exception as exc:
            if not self.catch_unexpected:
                raise
            trace.end(self.now(), "failed", f"error:{type(exc).__name__}")
            self.abandon_outbound(trace)
        finally:
            self.scheduler.release(ticket)
            if handle is not None:
                handle.cancel()

    async def _work(
        self, trace: RunTrace, m: Message, seq: int, event: dict, mocks: Any
    ) -> str | None:
        degraded = trace.cost_mode == COST_DEGRADED
        if degraded:
            trace.degraded += ["web_disabled", "context_shortened"]
        await self.prepare(trace, m, seq)
        self._build_context(trace, m, seq, event, degraded)
        await self.prepare_context(trace, m, seq)
        reason = self.check(trace)
        if reason:
            trace.end(self.now(), "cancelled", reason)
            return None
        if trace.nudge is not None:
            # A bare nudge repeats the asker's earlier unanswered request. While that
            # request's own run is in flight (or already answered) it answers; this run
            # only ends, so the group never sees two answers to one question.
            earlier = trace.nudge["earlier_message_id"]
            trace.nudge["earlier"] = self.earlier_request_status(trace, m, earlier)
            if trace.nudge["earlier"] in ("active", "answered"):
                trace.nudge["outcome"] = "merged"
                trace.route = "nudge_merged"
                trace.end(self.now(), "succeeded", "nudge_merged")
                return None
            trace.nudge["outcome"] = "answer_earlier"
        quotes_bot = any(
            (row := self.store.lookup(m.bot_id, m.group_id, rid)) and row["user_id"] == m.bot_id
            for rid in m.reply_ids
        )
        command = parse_teaching(m.text, m.group_id, quotes_bot)
        if command is not None:
            text = self.teach(trace, m, command)
        else:
            if trace.images and not await self.agent.vision(trace, mocks):
                trace.material_gaps.append("image_unrecognized")
            route = self.agent.decide_route(trace, mocks)
            if inspect.isawaitable(route):  # live agents decide with a model call
                route = await route
            trace.intended_route = route
            if route == "teaching":
                # SEC-01 / MIG-06: a model judgement is never a write. Only the code
                # parser above, on the real sender's own top-level text, reaches
                # `teach()` (authorised there by the event's sender id). Persona v3: the
                # run is answered as chat (intended_route keeps "teaching"); nothing is
                # written and no fixed teaching text is sent.
                trace.dropped_requests.append(
                    {"source": "model_route", "kind": "teaching", "reason": "no_code_command"}
                )
                route = "chat"
            if degraded and route == "web":
                route = "chat"
            trace.route, trace.clarify = route, route == "clarify"
            if self.embedding_offline:
                trace.degraded.append("embedding_offline")
            await self.tools(trace, m, mocks)
            text = await self.agent.answer(trace, mocks, m, seq)
        for attempt in range(2):
            reason = self.check(trace)
            if reason == "knowledge_changed" and attempt == 0:
                trace.mark(self.now(), "regenerating", reason)
                await self.refresh_knowledge(trace, m)
                text = await self.agent.answer(trace, mocks, m, seq)
                continue
            if reason:
                trace.end(self.now(), "cancelled", reason)
                return None
            return text
        trace.end(self.now(), "cancelled", "knowledge_changed")
        return None

    def check(self, trace: RunTrace) -> str | None:
        """RUN-07 pre-publish check; hosts may add checks (recall, cost)."""
        s = self.settings
        return prepublish_check(
            trace.user_id,
            trace.group_id,
            s.blocked_users,
            s.blocked_groups,
            trace.knowledge,
            self.ledger.revision,
        )

    def _build_context(
        self, trace: RunTrace, m: Message, seq: int, event: dict, degraded: bool
    ) -> None:
        now_p = self.platform_now()
        blocked = self.settings.blocked_users
        visible = []
        for h, ev in self.history_events(m, seq):  # same bot+group, visible, before boundary
            if ev is None or h.user_id in blocked or is_expired(ev.get("time"), now_p, self.limits):
                continue
            visible.append(ev)
        explainer = EXPLAINERS.get(self.strategy_name)
        picks: list = []
        if explainer is not None:
            picks = explainer(visible + [event], m.message_id, **self.explain_options(m))
            chosen = [p.message_id for p in picks]
            reasons = {p.message_id: f"{p.relation}/{p.method}" for p in picks}
        else:
            chosen, reasons = self.strategy(visible + [event], m.message_id), {}
        cap = self.limits.degraded_context_messages if degraded else self.limits.context_messages
        trace.context_ids = chosen[-cap:]
        trace.context_reasons = {cid: reasons[cid] for cid in trace.context_ids if cid in reasons}
        urged = next((p.message_id for p in picks if p.relation == "urges"), None)
        if urged is not None:
            trace.nudge = {"earlier_message_id": urged}
        for rid in m.reply_ids:
            row = self.store.lookup(m.bot_id, m.group_id, rid)
            if row is None or row["seq"] >= seq:
                trace.material_gaps.append("quote_missing")
            elif row["revoked"]:
                trace.material_gaps.append("quote_recalled")
            elif is_expired(row["platform_time"], now_p, self.limits):
                trace.material_gaps.append("quote_expired")
            elif row["user_id"] in blocked:
                trace.material_gaps.append("quote_unavailable")
        by_id = {str(ev.get("message_id")): ev for ev in visible}
        sources = [event] + [by_id[cid] for cid in trace.context_ids]
        for ev in sources:
            segments = ev.get("message") or []
            mid, uid = str(ev.get("message_id")), str(ev.get("user_id"))
            for seg in segments:
                if isinstance(seg, dict) and seg.get("type") == "image":
                    trace.images.append({"message_id": mid, "user_id": uid})
            for card in message_cards(segments):
                trace.cards.append({"message_id": mid, "user_id": uid, **card})
        kept = set(trace.context_ids)
        self.context_built(trace, m, seq, [p for p in picks if p.message_id in kept])

    # -- publish -------------------------------------------------------------------
    async def _publish(self, trace: RunTrace, m: Message, text: str, kind: str, mocks: Any) -> None:
        if self.capture_answer(trace, text, kind):
            return
        await self.before_publish(trace, mocks)
        record, fresh = self.outbound_enqueue(
            OutboundRecord(
                f"{trace.run_id}:{kind}",
                trace.run_id,
                kind,
                m.group_id,
                m.message_id,
                m.user_id,
                str(text),
            ),
            m,
        )
        if not fresh:
            return
        record.scheduled_at = self.rate.reserve(m.group_id, self.now())
        if record.scheduled_at > self.now():
            if kind == "answer":
                trace.mark(self.now(), "rate_limited")
            await asyncio.sleep(record.scheduled_at - self.now())
        if kind == "answer":
            reason = self.check(trace)
            if reason:
                self.outbound_transition(record, "failed", reason)
                trace.end(self.now(), "cancelled", reason)
                return
        state = await self.deliver(trace, record, mocks)
        if kind == "answer":
            trace.end(self.now(), state)
