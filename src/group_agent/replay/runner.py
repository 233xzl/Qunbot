"""Deterministic replay runner (T1.9, deterministic tier).

Feeds a case's events in order on a virtual clock (event `time` offsets; mock delays in
ms), through the production admission/dedup/recall code (`Pipeline.admit`) and the
shared run flow (`group_agent.runflow.RunFlow`: scheduler, deadline, cancellation,
context strategy A/D/B/C, pre-publish checks, rate-limited publish), with scripted
model/tools/faults from the case `mocks`. It returns a structured, JSON-serialisable
trace; scoring lives in `scoring.py`.

Production rules live in `group_agent.{runflow,pipeline,policy,scheduler,outbox,teaching,
cards}`; the online orchestrator (`group_agent.orchestrator`) runs the same `RunFlow`.
This module only holds harness glue: mocks, faults, the in-memory outbox simulation and
the scripted agent. The live-model tier plugs in through `agent_factory` with an agent
implementing the `ScriptedAgent` methods (`vision`, `decide_route` - may be async -,
`answer`) plus an optional `run_tools(trace, message, budget)` that replaces the mock tool
plan (see `group_agent.live_agent`).
"""

import asyncio
from dataclasses import replace
import re
from typing import Any
from uuid import uuid4

from ..cards import text_urls
from ..config import Settings
from ..contracts import Context, FallbackModel, Message
from ..outbox import OutboundRecord, Outbox
from ..pipeline import Pipeline
from ..policy import (
    COST_REJECTED,
    WEB_TOOLS,
    RuntimeLimits,
    ToolBudget,
    cost_mode,
    knowledge_visible,
    moderation_blocks,
    tool_timeout,
)
from ..runflow import AgentFailure, RunFlow, RunTrace, _r
from ..store import Store
from ..teaching import ASKS_USER, KnowledgeLedger, KnowledgeRecord, reply_for
from .mocks import MockBook, MockItem, RunMocks
from .vclock import run_virtual

__all__ = [
    "AgentFailure",
    "CaseReplay",
    "RunTrace",
    "ScriptedAgent",
    "ScriptedError",
    "run_case",
    "scripted_error",
]

GROUP_TOOLS = frozenset({"group_info", "group_member_list"})
P1_TOOLS = frozenset({"headless_fetch"})  # LINK-02 is P1; the P0a agent has no such tool


class ScriptedError(AgentFailure):
    """Base for mock-injected errors; subclasses are named after the mock `error`."""


def scripted_error(name: str) -> ScriptedError:
    clean = re.sub(r"\W", "_", name) or "ScriptedError"
    if not clean[0].isalpha():
        clean = "E_" + clean
    return type(clean, (ScriptedError,), {})(name)


class ScriptedRunner:
    """ModelRunner that replays one mock item: wait `delay_ms`, then answer or raise."""

    def __init__(self, replay: "CaseReplay", trace: RunTrace, item: MockItem):
        self.replay, self.trace, self.item = replay, trace, item

    async def run(self, context: Context) -> str:
        return await self.replay.model_call(self.trace, self.item)


class ScriptedAgent:
    """Deterministic stand-in for the model side of a run, driven by case mocks."""

    def __init__(self, replay: "CaseReplay"):
        self.replay = replay

    async def vision(self, trace: RunTrace, mocks: RunMocks) -> bool:
        item, fault = mocks.take("vision:primary"), mocks.fault("vision_timeout")
        replay = self.replay
        if item is None and fault is None:
            trace.model_calls.append({"name": "vision:primary", "status": "ok", "mocked": False})
            return True
        delay = (item or fault).delay
        error = item.error if item else (fault.error or "TimeoutError")
        start = replay.now()
        status = "ok"
        try:
            async with asyncio.timeout(tool_timeout(replay.remaining(trace), replay.limits)):
                await asyncio.sleep(delay)
            if error:
                status = "error"
        except TimeoutError:
            status = "timeout"
        trace.model_calls.append(
            {
                "name": "vision:primary",
                "status": status,
                "error": error if status != "ok" else None,
                "t_start": _r(start),
                "t_end": _r(replay.now()),
            }
        )
        return status == "ok"

    def decide_route(self, trace: RunTrace, mocks: RunMocks) -> str:
        item = mocks.take("route")
        if item is not None:
            return str(item.response)
        if trace.material_gaps:
            return "clarify"
        plan = mocks.tool_plan()
        if any(i.base in WEB_TOOLS for i in plan):
            return "web"
        if any(i.base == "knowledge_search" for i in plan):
            return "local_knowledge"
        return "chat"

    async def answer(self, trace: RunTrace, mocks: RunMocks, message: Message, seq: int) -> str:
        replay = self.replay
        hang = mocks.fault("hang:model")
        if hang is not None:
            trace.model_calls.append(
                {"name": "model", "status": "hang", "t_start": _r(replay.now())}
            )
            await asyncio.Event().wait()  # ends only by cancellation or deadline
        context = Context(message, (), seq)
        if mocks.has("moderation"):
            draft = await replay.model_call(trace, mocks.take("draft_answer"))
            judgement = await replay.model_call(trace, mocks.take("moderation"))
            blocked = moderation_blocks(judgement, replay.limits)
            trace.moderation = {"result": judgement, "blocked": blocked}
            if blocked:
                neutral = mocks.take("neutral_answer")
                return await replay.model_call(trace, neutral) if neutral else "这个话题我不展开。"
            return draft
        if mocks.has("primary"):
            runners = [
                ScriptedRunner(replay, trace, item)
                for item in (mocks.take("primary"), mocks.take("fallback"))
                if item is not None
            ]
            return await FallbackModel(*runners).run(context)
        item = mocks.take("answer")
        if item is not None:
            return await replay.model_call(trace, item)
        if trace.route == "clarify":
            return "需要的材料现在不可用（" + "、".join(trace.material_gaps) + "），请重新发送。"
        return f"（模拟回答 {message.message_id}）"


class CaseReplay(RunFlow):
    def __init__(
        self,
        case: dict,
        strategy: str = "C",  # first-version default (docs/LIVE-REPLAY.md section 10)
        limits: RuntimeLimits = RuntimeLimits(),
        agent_factory=ScriptedAgent,
    ):
        env = case.get("environment") or {}
        if env.get("config_overrides"):
            raise ValueError("replay config_overrides are not implemented")
        self._init_flow(limits, strategy)
        self.case = case
        self.events = case["events"]
        self.anchor_id = str(case["anchor"]["message_id"])
        owner = env.get("owner_id")
        settings = Settings(
            bot_id=str(env["bot_id"]),
            owner_id=None if owner is None else str(owner),
            blocked_users=frozenset(map(str, env.get("blacklist") or [])),
            deadline_seconds=limits.deadline_seconds,
            context_messages=limits.context_messages,
        )
        self.store = Store()
        self.pipeline = Pipeline(settings, self.store)
        ids = {str(e["message_id"]) for e in self.events if "message_id" in e}
        self.mocks = MockBook(case.get("mocks") or {}, ids, self.anchor_id)
        self.agent = agent_factory(self)
        self.outbox = Outbox()
        self.ledger = KnowledgeLedger()
        self.usage_ratio = 0.0
        self.embedding_offline = False
        self.events_by_key: dict[tuple[str, str], dict] = {}
        self.event_log: list[dict] = []
        self.runs: list[RunTrace] = []
        self.faults_applied: list[dict] = []
        self.warnings: list[str] = []
        self.answer_evidence: dict[str, dict[str, int]] = {}
        self._last_search: dict[str, MockItem] = {}
        times = [e["time"] for e in self.events if isinstance(e.get("time"), int)]
        self.t0 = min(times) if times else 0

    # -- clock helpers -------------------------------------------------------------
    def platform_now(self) -> float:
        return self.t0 + self.now()

    @property
    def settings(self) -> Settings:
        return self.pipeline.settings

    # -- driver --------------------------------------------------------------------
    def _timeline(self) -> list[tuple[float, int, int, str, Any]]:
        items, previous, anchor_at = [], 0.0, None
        for index, event in enumerate(self.events):
            t = event.get("time")
            offset = max(previous, (t - self.t0) if isinstance(t, int) else previous)
            previous = offset
            if anchor_at is None and str(event.get("message_id")) == self.anchor_id:
                if event.get("post_type") != "notice":
                    anchor_at = offset
            items.append((offset, 1, index, "event", event))
        anchor_at = anchor_at or 0.0
        for n, fault in enumerate(self.mocks.global_faults()):
            items.append((anchor_at + fault.delay, 0, n, "fault", fault))
        return sorted(items, key=lambda x: x[:3])

    def _seed_state(self) -> None:
        for item in self.mocks.state("knowledge_store"):
            for kid, rec in (item.response or {}).items():
                self.ledger.seed(_record(kid, rec))
        for item in self.mocks.state("knowledge_search", mark=False):
            response = item.response or {}
            for hit in response.get("hits") or response.get("candidates") or []:
                self.ledger.seed(_record(hit["id"], hit))
        for item in self.mocks.state("answer_evidence"):
            for mid, used in (item.response or {}).items():
                self.answer_evidence[str(mid)] = {str(k): int(v) for k, v in used.items()}

    async def run(self) -> dict:
        self._seed_state()
        for offset, _, index, kind, payload in self._timeline():
            if offset > self.now():
                await asyncio.sleep(offset - self.now())
            if kind == "fault":
                self._apply_fault(payload)
            else:
                self._feed(index, payload)
        await asyncio.gather(*self.tasks)
        return self._result()

    def _apply_fault(self, item: MockItem) -> None:
        item.used = True
        resp = item.response if isinstance(item.response, dict) else {}
        effect: Any = None
        if item.base == "knowledge_update":
            self.ledger.external_update(
                str(resp["id"]),
                int(resp["revision"]),
                str(resp.get("content", "")),
                self.now(),
                resp.get("by_message"),
            )
            effect = {"id": resp["id"], "revision": resp["revision"]}
        elif item.base == "blacklist_update":
            add_users = frozenset(map(str, resp.get("add_users") or []))
            add_groups = frozenset(map(str, resp.get("add_groups") or []))
            self.pipeline.settings = replace(
                self.settings,
                blocked_users=self.settings.blocked_users | add_users,
                blocked_groups=self.settings.blocked_groups | add_groups,
            )
            effect = {"add_users": sorted(add_users), "add_groups": sorted(add_groups)}
        elif item.base == "usage_ledger":
            ratio = resp.get("ratio")
            if ratio is None and resp.get("daily_cap_cny"):
                ratio = resp["daily_cost_cny"] / resp["daily_cap_cny"]
            self.usage_ratio = float(ratio or 0)
            effect = {"ratio": self.usage_ratio}
        elif item.base == "embedding_offline":
            self.embedding_offline = True
            effect = {"embedding": "offline"}
        elif item.base == "event_redelivery":
            # Descriptive: the duplicates are already part of `events`; verify the count.
            mid = str(resp.get("message_id"))
            extra = sum(1 for e in self.events if str(e.get("message_id")) == mid) - 1
            effect = {"verified_in_events": extra == resp.get("extra_deliveries", extra)}
            if not effect["verified_in_events"]:
                self.warnings.append(f"event_redelivery count mismatch for {mid}")
        self.faults_applied.append({"name": item.name, "t": _r(self.now()), "effect": effect})

    def _feed(self, index: int, event: dict) -> None:
        admission = self.pipeline.admit(event)
        m = admission.message
        entry = {
            "index": index,
            "t": _r(self.now()),
            "message_id": None if m is None else m.message_id,
            "group_id": None if m is None else m.group_id,
            "post_type": event.get("post_type"),
            "status": admission.status,
        }
        if m is None and event.get("post_type") == "notice":
            entry["message_id"] = str(event.get("message_id"))
            entry["group_id"] = str(event.get("group_id"))
        self.event_log.append(entry)
        if m is not None and admission.status in ("collected", "trigger"):
            self.events_by_key.setdefault((m.group_id, m.message_id), event)
        if admission.status != "trigger":
            return
        is_owner = self.settings.owner_id is not None and m.user_id == self.settings.owner_id
        mode = cost_mode(self.usage_ratio, is_owner, self.limits)
        if mode == COST_REJECTED:
            entry["status"] = "ignored_cost_cap"  # D-12: no notice to non-owners
            return
        trace = RunTrace(
            uuid4().hex[:12],
            m.message_id,
            m.group_id,
            m.user_id,
            self.now(),
            mode,
            m.message_id == self.anchor_id,
        )
        self.runs.append(trace)
        self.launch(trace, m, admission.seq, event)

    # -- RunFlow hooks -------------------------------------------------------------
    def run_mocks(self, m: Message) -> RunMocks:
        return self.mocks.for_run(m.message_id)

    def cancel_plan(self, trace: RunTrace, mocks: RunMocks) -> tuple[float, str] | None:
        cancel = mocks.fault("cancel")
        if cancel is None:
            return None
        by = (cancel.response or {}).get("by_message")
        return trace.arrived_at + cancel.delay, f"user_cancel:{by}" if by else "user_cancel"

    def history_events(self, m: Message, seq: int) -> list[tuple[Message, dict | None]]:
        return [
            (h, self.events_by_key.get((h.group_id, h.message_id)))
            for h in self.store.history(m, seq)
        ]

    def earlier_request_status(self, trace: RunTrace, m: Message, message_id: str) -> str:
        # Nudge: an earlier request without a run in this replay is simply "open".
        earlier = next(
            (r for r in self.runs if r.message_id == message_id and r.group_id == m.group_id),
            None,
        )
        if earlier is None:
            return "open"
        if earlier.ended_at is None:
            return "active"
        sent = any(
            r.run_id == earlier.run_id and r.kind == "answer" and r.state != "failed"
            for r in self.outbox.records.values()
        )
        return "answered" if sent else "open"

    def abandon_outbound(self, trace: RunTrace) -> None:
        for record in self.outbox.records.values():
            if record.run_id == trace.run_id and record.state == "pending":
                self.outbox.transition(record.key, "failed", self.now(), trace.state)
            elif record.run_id == trace.run_id and record.state == "sending":
                self.outbox.transition(record.key, "uncertain", self.now(), trace.state)

    def outbound_enqueue(self, record: OutboundRecord, m: Message):
        return self.outbox.enqueue(record, self.now())

    def outbound_transition(self, record, state: str, reason: str | None = None) -> None:
        self.outbox.transition(record.key, state, self.now(), reason)

    async def before_publish(self, trace: RunTrace, mocks: RunMocks | None) -> None:
        if mocks is not None:
            slow = mocks.fault("slow_ack")
            if slow is not None and trace.arrived_at + slow.delay > self.now():
                trace.mark(self.now(), "delayed_ack")
                await asyncio.sleep(trace.arrived_at + slow.delay - self.now())

    async def deliver(self, trace: RunTrace, record, mocks: RunMocks | None) -> str:
        # Simulated platform: nothing leaves the process (`sent_to_qq` stays 0).
        self.outbox.transition(record.key, "sending", self.now())
        record.sent_at = self.now()
        lost = mocks.fault("outbound_ack_lost") if mocks is not None else None
        if lost is not None:
            await asyncio.sleep(lost.delay)
            self.outbox.transition(record.key, "uncertain", self.now(), lost.error)
        else:
            self.outbox.transition(record.key, "acked", self.now())
        return "sent" if record.state == "acked" else "sent_uncertain"

    async def refresh_knowledge(self, trace: RunTrace, m: Message) -> None:
        await self._knowledge_search(trace, m, self._last_search.get(trace.run_id))

    def teach(self, trace: RunTrace, m: Message, command) -> str:
        trace.route = trace.intended_route = "teaching"
        target = None
        if command.via_quote:
            for rid in m.reply_ids:
                evidence = self.answer_evidence.get(rid) or {}
                if len(evidence) == 1:
                    target = next(iter(evidence.items()))
        commit = self.mocks.state("knowledge_commit", scope=m.message_id)
        new_id = str(commit[0].response["id"]) if commit else f"kb_{m.message_id}"
        entry = self.ledger.apply(
            command,
            actor=m.user_id,
            owner_id=self.settings.owner_id,
            source_message=m.message_id,
            group_id=m.group_id,
            now=self.now(),
            new_id=new_id,
            target=target,
        )
        entry["run_id"] = trace.run_id
        trace.teaching = {k: v for k, v in entry.items() if k != "t"}
        trace.clarify = entry["outcome"] in ASKS_USER
        if entry["id"]:
            trace.knowledge_ids = [entry["id"]]
        return reply_for(entry, self.ledger.describe)

    # -- tools ---------------------------------------------------------------------
    async def tools(self, trace: RunTrace, m: Message, mocks: RunMocks) -> None:
        budget = ToolBudget(self.limits)
        live_tools = getattr(self.agent, "run_tools", None)
        if live_tools is not None:  # live tier: the agent runs real/stubbed tools itself
            await live_tools(trace, m, budget)
            return
        url = next((c["url"] for c in trace.cards if c.get("url")), None)
        url = url or next(iter(text_urls(m.text)), None)
        for item in mocks.tool_plan():
            if item.base in WEB_TOOLS and trace.route != "web":
                continue
            if item.base == "knowledge_search":
                if trace.route == "local_knowledge":
                    item.used = True
                    self._last_search[trace.run_id] = item
                    await self._knowledge_search(trace, m, item)
                continue
            item.used = True
            if item.base in GROUP_TOOLS:
                args = {"group_id": trace.group_id}
            elif item.base == "web_search":
                args = {"query_chars": len(m.text.strip())}
            else:
                args = {"url": url}
            response = await self._call_tool(trace, item, args, budget)
            if item.base == "short_link_expand" and isinstance(response, dict):
                url = response.get("to") or url

    async def _call_tool(
        self, trace: RunTrace, item: MockItem, args: dict, budget: ToolBudget | None
    ) -> Any:
        call = {"name": item.base, "args": args, "t_start": _r(self.now())}
        trace.tool_calls.append(call)
        if item.base in P1_TOOLS:
            call["status"] = "unavailable_p1"
            return None
        if budget is not None and not budget.try_consume(item.base):
            call["status"] = "budget_exhausted"
            return None
        timeout = tool_timeout(self.remaining(trace), self.limits)
        if timeout <= 0:
            call["status"] = "no_time_left"
            return None
        try:
            async with asyncio.timeout(timeout):
                await asyncio.sleep(item.delay)
        except TimeoutError:
            call.update(status="timeout", t_end=_r(self.now()))
            return None
        call["t_end"] = _r(self.now())
        if item.error:
            call.update(status="error", error=item.error)
            return None
        response = item.response
        if item.base in GROUP_TOOLS:
            # GRP-01: the tool is bound to the request's group; foreign data is refused.
            if not isinstance(response, dict) or str(response.get("group_id")) != trace.group_id:
                call["status"] = "rejected_cross_group"
                return None
        call["status"] = "ok"
        return response

    async def _knowledge_search(self, trace: RunTrace, m: Message, item: MockItem | None):
        if item is None:
            return
        args = {"query_chars": len(m.text.strip())}
        if self.embedding_offline:
            args["mode"] = "keyword_only"
        response = await self._call_tool(trace, item, args, None) or {}
        used, filtered = {}, []
        for hit in response.get("hits") or response.get("candidates") or []:
            kid = str(hit["id"])
            record = self.ledger.records.get(kid)  # KB-03 back-source check
            current = (
                {"scope": record.scope, "active": record.active, "revision": record.revision}
                if record
                else hit
            )
            if not knowledge_visible(current, trace.group_id):
                filtered.append(kid)
                continue
            used[kid] = int(current["revision"])
        trace.knowledge, trace.knowledge_filtered = used, sorted(set(filtered))
        trace.knowledge_ids = sorted(used)

    async def model_call(self, trace: RunTrace, item: MockItem | None) -> Any:
        if item is None:
            return None
        call = {"name": item.name, "t_start": _r(self.now())}
        trace.model_calls.append(call)
        await asyncio.sleep(item.delay)
        call["t_end"] = _r(self.now())
        if item.error:
            call.update(status="error", error=item.error)
            raise scripted_error(item.error)
        call["status"] = "ok"
        return item.response

    # -- result --------------------------------------------------------------------
    def _result(self) -> dict:
        real = self.case.get("source") == "real"
        outbound = []
        for r in self.outbox.records.values():
            outbound.append(
                {
                    "run_id": r.run_id,
                    "kind": r.kind,
                    "group_id": r.group_id,
                    "reply_to": r.reply_to,
                    "requester": r.requester,
                    "text": "<omitted>" if real else r.text,
                    "state": r.state,
                    "error_kind": r.error_kind,
                    "scheduled_at": _r(r.scheduled_at),
                    "sent_at": _r(r.sent_at),
                    "history": [[_r(t), s] for t, s in r.history],
                }
            )
        knowledge_log = [{**e, "t": _r(e["t"])} for e in self.ledger.log]
        return {
            "case_id": self.case.get("case_id"),
            "source": self.case.get("source"),
            "strategy": self.strategy_name,
            "anchor": self.anchor_id,
            "owner_id": self.settings.owner_id,
            "events": self.event_log,
            "triggered": [r.message_id for r in self.runs],
            "runs": [r.as_dict() for r in self.runs],
            "outbound": outbound,
            "knowledge_log": knowledge_log,
            "knowledge_final": {
                k: {"scope": v.scope, "revision": v.revision, "active": v.active, "term": v.term}
                for k, v in self.ledger.records.items()
            },
            "faults_applied": self.faults_applied,
            "max_active_runs": self.scheduler.max_total_active,
            "final_blocked_users": sorted(self.settings.blocked_users),
            "unused_mocks": self.mocks.unused(),
            "unknown_mocks": self.mocks.unknown,
            "warnings": self.warnings,
            "sent_to_qq": 0,
        }


def _record(kid: str, rec: dict) -> KnowledgeRecord:
    source = rec.get("source_message")
    return KnowledgeRecord(
        str(kid),
        str(rec.get("scope", "public")),
        int(rec.get("revision", 1)),
        bool(rec.get("active", True)),
        str(rec.get("content", "")),
        rec.get("term"),
        {str(source)} if source else set(),
    )


def run_case(
    case: dict,
    strategy: str = "C",
    limits: RuntimeLimits = RuntimeLimits(),
    agent_factory=ScriptedAgent,
) -> dict:
    """Replay one case on a fresh virtual-time loop and return its trace.

    Live agents wrap their model/tool awaits in `vclock.real_time()` so those sections
    run on wall time inside the otherwise virtual timeline.
    """
    replay = CaseReplay(case, strategy, limits, agent_factory)

    async def main() -> dict:
        try:
            return await replay.run()
        finally:
            close = getattr(replay.agent, "aclose", None)
            if close is not None:  # live agents own loop-bound HTTP clients
                await close()

    try:
        return run_virtual(main())
    finally:
        replay.store.close()
