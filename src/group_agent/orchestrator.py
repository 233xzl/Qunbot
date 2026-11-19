"""Online orchestrator (T3.4 scheduling, T3.7 record-only sending, T3.9 cost gate).

`Orchestrator` consumes the reply requests that `intake.EventIntake` hands to an
`admission.RequestQueue` and runs each through `runflow.RunFlow`, the same flow the
deterministic replay uses (scheduler limits, queue wait, deadline, cancellation, context
strategy C, pre-publish checks, send-rate limits). The online-specific parts are:

- run state machine persisted in `runs` (queued -> running -> succeeded | failed |
  cancelled | expired | rejected), one run per request message (AC-15); runs left active
  by a previous process are marked failed/interrupted at start-up;
- cost gate from the persistent `usage.UsageLedger`: below 80 % normal, from 80 %
  degraded, from 100 % only the owner gets a run; others get a `rejected/cost_cap` row
  and no notice (D-04, D-12);
- context from the store (`history_window` around the request plus its quote ancestry,
  platform-time order), selection reasons written as related marks (retention keeps the
  related set, including answers/continues) and a `context_snapshots` summary (ids and
  reasons only);
- cancellation when the request is recalled or its sender/group is blocked mid-run;
- pre-publish re-check: blocklists, knowledge revisions, request recalled, cost cap;
- outbound: idempotent row per run (`outbound`, text cut to one QQ message by
  `outbox.fit_message_text`), rate-limited and re-checked, then either `recorded_not_sent`
  (record only, the default) or, with a `publisher` hook (serve with `[serve] live_send`,
  the cutover), handed over by run id: the hook (the answer sender's `deliver`, see
  `live_send.py`) re-checks the row and sends it. This module never calls the gateway and holds no channel; the
  hook only accepts the id of a row this flow wrote. The owner's `#新版` path (D-19,
  `owner_command.py`) gets answers from `generate_for` (a `new_version` run whose answer is
  handed back instead of being recorded) and writes its own outbound rows.
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from .admission import AccessPolicy, ReplyRequest
from .config import Settings
from .contracts import Message
from .outbox import OutboundRecord, fit_message_text
from .policy import COST_REJECTED, RuntimeLimits, ToolBudget, is_expired
from .knowledge_base import KnowledgeBase
from .runflow import CURRENT_RUN, RunFlow, RunTrace
from .store import Store, _message
from .teaching import ASKS_USER, UNAUTHORIZED_TEXT, KnowledgeLedger, reply_for
from .usage import UsageLedger

# Trace states (runflow) -> persisted run states (store.RUN_STATES).
RUN_STATE_OF = {
    "created": "queued",
    "admitted": "queued",
    "queued": "queued",
    "running": "running",
    "fallback_reply": "running",  # every answer attempt failed; the fixed reply follows
    "regenerating": "running",
    "rate_limited": "running",
    "delayed_ack": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "timeout": "expired",
    "queue_timeout": "expired",
    "rejected_busy": "rejected",
}
TEACHING_PENDING_TEXT = "教学写入还没有开放，这条没有记录。"


class OnlineTrace(RunTrace):
    """RunTrace whose state changes are persisted through `on_mark`."""

    def __init__(self, *args, on_mark: Callable[[RunTrace], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_mark = on_mark

    def mark(self, now: float, state: str, reason: str | None = None) -> None:
        super().mark(now, state, reason)
        if self._on_mark is not None:
            self._on_mark(self)


class _Lru(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def put(self, key, value) -> None:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.capacity:
            self.popitem(last=False)


def _compact_record(record: dict) -> dict:
    """Run diagnostics without chat content (no texts, queries or URLs)."""
    keep = {}
    for key in (
        "config_revision",
        "stages",
        "errors",
        "tools",
        "route_rule",
        "clarify_override",  # evidence names only (quote/image/recent_same_user/command_text)
        "answer_source",
        "answer_thinking",
        "answer_attempts",
        "answer_fallback_used",
        "answer_failed",
        "latency_total_s",
        "decision_attempts",
        "decision_first_try",
        "decision_mode",  # code-selected policy names only
        "decision_source",
        "decision_guard",  # code-selected capability / diagnostic reason
        "knowledge_mode",  # candidates vs selected, no entry contents
        "breaker",  # D-20: Gemini breaker state at decision/answer (closed/open/half_open)
        "breaker_skips",
    ):
        if key in record:
            keep[key] = record[key]
    decision = record.get("decision") or {}
    keep["decision"] = {
        k: v
        for k, v in decision.items()
        if k not in ("clarify_question", "reason", "search_terms", "subject_terms")
    }
    keep["calls"] = [
        {
            k: c.get(k)
            for k in (
                "stage",
                "task",
                "model",
                "mode",
                "status",
                "error",
                "latency_s",
                "cost_cny",
                "error_causes",
                "failure_phase",
            )
        }
        for c in record.get("calls") or []
    ]
    web = record.get("web") or {}
    if web:
        keep["web"] = {k: web.get(k) for k in ("searches", "reads", "cost_cny", "statuses")}
        keep["web"]["sources"] = len(web.get("sources") or [])
    if record.get("dropped_fields"):  # model-written names: count only
        keep["dropped_fields"] = len(record["dropped_fields"])
    for key in ("supplement", "subject_check"):
        if record.get(key):
            keep[key] = {"status": record[key].get("status")}
    if record.get("material"):
        keep["material"] = {
            k: v for k, v in record["material"].items() if k not in ("image_labels", "anchor_id")
        }
    return keep


class Orchestrator(RunFlow):
    catch_unexpected = True

    def __init__(
        self,
        store: Store,
        access: AccessPolicy,
        settings: Settings,
        *,
        agent_factory: Callable[[Any], Any],
        usage: UsageLedger,
        limits: RuntimeLimits | None = None,
        strategy: str = "C",
        knowledge: KnowledgeBase | None = None,
        media_root: Path | None = None,
        media: Any = None,
        clock: Callable[[], float] = time.time,
        history_seconds: float = 1800.0,
        history_limit: int = 400,
        ancestry_depth: int = 16,
        ancestry_max: int = 300,
        prepare_wait_seconds: float = 8.0,
        quote_context_seconds: float = 120.0,
        backfill: Any = None,
        self_sync: Any = None,
        publisher: Callable[[str], Awaitable[Any]] | None = None,
        paused: Callable[[], bool] | None = None,
        log: Callable[[str], None] = lambda _: None,
    ):
        """`publisher`: None = record only; else an async hook taking the run id of a live
        outbound row after the pre-publish check and rate wait, returning an object with
        `status` (sent / failed / uncertain) and `error_kind`.

        `paused`: owner remote control (`owner_control.py`). While it returns true `submit`
        starts no run and the pre-publish check refuses an answer ("paused")."""
        self.group_agents = {}
        self._run_agents = {}
        self._run_limits = {}
        self._run_revisions = {}
        self._model_resources = set()
        self._closing = set()
        self.publisher = publisher
        self.paused = paused
        self._init_flow(limits or settings.limits, strategy)
        self.store, self.access, self._base_settings = store, access, settings
        self.usage, self.clock, self.log = usage, clock, log
        self.usage.set_profiles(settings.group_profiles)
        # T4.3: with a knowledge base, owner teaching writes to it and the pre-publish check
        # reads its live revisions; without one, teaching is recognised but not written.
        self.knowledge = knowledge
        self.ledger = knowledge if knowledge is not None else KnowledgeLedger()
        self.media_root = media_root
        self.media = media
        self.history_seconds, self.history_limit = history_seconds, history_limit
        self.ancestry_depth, self.ancestry_max = ancestry_depth, ancestry_max
        self.prepare_wait_seconds = prepare_wait_seconds
        self.quote_context_seconds = quote_context_seconds
        self.backfill = backfill
        # Read-only bot-message sync (self_sync.SelfSync): `catch_up` before the context.
        self.self_sync = self_sync
        self.events_by_key: _Lru = _Lru(8192)
        self.active: dict[str, tuple[OnlineTrace, Message]] = {}
        self.counts: Counter[str] = Counter()
        # D-19 regenerations: run id -> future receiving the answer text instead of an
        # outbound row (the caller, owner_command.NewVersionCommand, publishes it).
        self._captures: dict[str, asyncio.Future] = {}
        self._settings_cache: tuple[int, Settings] | None = None
        self.interrupted = store.interrupt_active_runs(clock())
        self.agent = agent_factory(self)

    @property
    def agent(self):
        return self._run_agents.get(CURRENT_RUN.get(), self._agent)

    @agent.setter
    def agent(self, value):
        self._agent = value
        self._model_resources.update(self.model_resources(value))

    @property
    def limits(self):
        return self._run_limits.get(CURRENT_RUN.get(), self._limits)

    @limits.setter
    def limits(self, value):
        self._limits = value

    @staticmethod
    def model_resources(agent):
        return {
            getattr(agent, name)
            for name in ("gemini", "deepseek", "gemini_decider")
            if hasattr(agent, name)
        }

    def capture_run(self, trace):
        self._run_agents[trace.run_id] = self.group_agents.get(trace.group_id, self._agent)
        profile = self._base_settings.group_profiles.get(trace.group_id, {})
        trace.profile_fields = (
            sorted(
                f"{section}.{key}"
                for section in ("prompts", "model_roles")
                for key in profile.get(section, {})
            )
            if profile.get("enabled")
            else []
        )
        self._run_limits[trace.run_id] = self._limits
        self._run_revisions[trace.run_id] = self._base_settings.revision

    async def retire_models(self):
        used = set()
        for agent in [self._agent, *self.group_agents.values(), *self._run_agents.values()]:
            used.update(self.model_resources(agent))
        unused = self._model_resources - used
        self._model_resources.difference_update(unused)
        for model in unused:
            await model.aclose()

    def update(self, settings, agent):
        self._base_settings, self._settings_cache = settings, None
        self.agent, self.limits = agent, settings.limits
        self.scheduler.update(settings.limits)
        self.rate.limits = settings.limits
        self.busy.window = settings.limits.busy_notice_window_seconds
        self.usage.daily_cap_cny, self.usage.limits = settings.daily_cost_cny, settings.limits
        self.usage.set_profiles(settings.group_profiles)
        self.apply_access(
            owner_id=settings.owner_id,
            blocked_users=settings.blocked_users,
            blocked_groups=settings.blocked_groups,
            collection_off_groups=settings.collection.disabled_groups,
        )

    # -- identity / clock ---------------------------------------------------------
    @property
    def settings(self) -> Settings:
        version = self.access.version
        if self._settings_cache is None or self._settings_cache[0] != version:
            snap = self.access.snapshot
            self._settings_cache = (
                version,
                replace(
                    self._base_settings,
                    owner_id=snap.owner_id,
                    blocked_users=snap.blocked_users,
                    blocked_groups=snap.blocked_groups,
                ),
            )
        return self._settings_cache[1]

    def platform_now(self) -> float:
        return self.clock()

    def _wall(self, loop_time: float | None) -> float | None:
        if loop_time is None:
            return None
        return self.clock() - (self.now() - loop_time)

    # -- intake side ------------------------------------------------------------------
    async def serve(self, queue, stop: asyncio.Event) -> None:
        """Consume `queue` (an `admission.RequestQueue`) until `stop` is set."""
        while not stop.is_set():
            try:
                request = await asyncio.wait_for(queue.get(), 0.5)
            except TimeoutError:
                continue
            try:
                self.submit(request)
            except Exception as exc:  # one bad request must not stop the consumer
                self.counts["submit_errors"] += 1
                self.log(f"submit failed: {type(exc).__name__}")

    def submit(self, request: ReplyRequest) -> str:
        """Start a run for one admitted request. Returns started / rejected / duplicate."""
        m = request.message
        if not self.store.allows_group_reply(m.bot_id, m.group_id, request.seq) or (
            self.paused is not None and self.paused()
        ):
            # Remote pause: no run, no model call, no outbound; the request stays recorded.
            self.counts["paused_skipped"] += 1
            self.store.set_request_state(*request.key, "recorded")
            return "paused"
        mode, scope = self.usage.decision(request.is_owner, m.group_id)
        run_id = uuid4().hex
        now = self.clock()
        if mode == COST_REJECTED:  # D-12: no run, no notice for non-owners
            if self.store.create_run(
                run_id,
                request.seq,
                m,
                state="rejected",
                cost_mode=mode,
                now=now,
                error_kind="cost_cap",
            ):
                self.store.update_run(
                    run_id,
                    now,
                    detail=json.dumps(
                        {"budget_scope": scope, "config_revision": self.settings.revision}
                    ),
                )
                self.counts["rejected_cost_cap"] += 1
                return "rejected"
            self.counts["duplicate"] += 1
            return "duplicate"
        if not self.store.create_run(
            run_id, request.seq, m, state="queued", cost_mode=mode, now=now
        ):
            self.counts["duplicate"] += 1
            return "duplicate"
        lag = 0.0
        if request.received_at is not None:
            lag = min(max(0.0, now - request.received_at), self.limits.deadline_seconds)
        trace = OnlineTrace(
            run_id,
            m.message_id,
            m.group_id,
            m.user_id,
            self.now() - lag,
            mode,
            True,
            on_mark=self._persist,
        )
        trace.budget_scope = scope
        event = self._event(self.store.message_row(request.seq))
        self.events_by_key.put((m.group_id, m.message_id), event)
        self.active[run_id] = (trace, m)
        self.counts["started"] += 1
        task = self.launch(trace, m, request.seq, event)
        task.add_done_callback(lambda t, tr=trace: self._finish(tr, t))
        return "started"

    async def generate_for(
        self, question: Message, question_seq: int, *, run_seq: int
    ) -> tuple[str | None, str | None]:
        """D-19 regeneration for the owner's `#新版` command: one `new_version` run that
        answers `question` with the context cut at the question (`question_seq` is the run
        boundary), billed to the usage ledger and gated as the owner's request (served at
        >= 100 % of the daily cap). The answer is returned, not recorded or sent.

        Returns `(run_id, text)`; `(None, None)` when the command already has a run, text
        None when the run ended without an answer."""
        m = question
        if not self.store.allows_group_reply(m.bot_id, m.group_id, run_seq):
            return None, None
        mode = self.usage.cost_mode(True)
        run_id = uuid4().hex
        if not self.store.create_run(
            run_id,
            run_seq,
            m,
            state="queued",
            cost_mode=mode,
            now=self.clock(),
            purpose="new_version",
        ):
            return None, None
        row = self.store.message_row(question_seq)
        if row is None:
            self.store.update_run(
                run_id, self.clock(), state="failed", error_kind="question_missing"
            )
            return run_id, None
        future = asyncio.get_running_loop().create_future()
        self._captures[run_id] = future
        trace = OnlineTrace(
            run_id, m.message_id, m.group_id, m.user_id, self.now(), mode, True,
            on_mark=self._persist,
        )  # fmt: skip
        event = self._event(row)
        self.events_by_key.put((m.group_id, m.message_id), event)
        self.active[run_id] = (trace, m)
        self.counts["new_version_runs"] += 1
        task = self.launch(trace, m, question_seq, event)
        task.add_done_callback(lambda t, tr=trace: self._finish(tr, t))
        try:
            text = await future
        finally:
            self._captures.pop(run_id, None)
        return run_id, text

    def on_recall(self, group_id: str, message_id: str) -> int:
        """The request message was recalled: cancel its run (RUN-05)."""
        cancelled = 0
        for run_id, (_, m) in list(self.active.items()):
            if m.group_id == group_id and m.message_id == message_id:
                cancelled += self.cancel_run(run_id, "request_recalled")
        return cancelled

    def cancel_group(self, group_id: str) -> int:
        return sum(
            self.cancel_run(run_id, "group_stopped")
            for run_id, (_, message) in list(self.active.items())
            if message.group_id == group_id
        )

    def cancel_replies(self, reason: str) -> int:
        """Cancel every active reply run (remote pause); a frame already being written
        finishes (the send hook shields it) and pending outbound rows end failed.
        The owner's `#新版` regenerations are not touched. Returns the number cancelled."""
        cancelled = 0
        for run_id in list(self.active):
            if run_id not in self._captures:
                cancelled += self.cancel_run(run_id, reason)
        return cancelled

    def apply_access(self, **changes) -> int:
        """Update identity/blocklists and cancel runs whose requester or group is now
        blocked (AC-14); returns the number of cancelled runs."""
        snap = self.access.update(**changes)
        cancelled = 0
        for run_id, (_, m) in list(self.active.items()):
            if m.group_id in snap.blocked_groups:
                cancelled += self.cancel_run(run_id, "group_blocked")
            elif m.user_id in snap.blocked_users:
                cancelled += self.cancel_run(run_id, "requester_blocked")
        return cancelled

    async def aclose(self, timeout: float = 10.0) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for run_id in list(self._run_tasks):
            self.cancel_run(run_id, "shutdown")
        pending = [t for t in self.tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)  # run completion callbacks release their snapshots
        if self._closing:
            await asyncio.gather(*self._closing)
        if self._model_resources:
            for model in self._model_resources:
                await model.aclose()
            self._model_resources.clear()
        else:
            close = getattr(self.agent, "aclose", None)
            if close is not None:
                await close()

    # -- persistence -------------------------------------------------------------------
    def _persist(self, trace: RunTrace) -> None:
        state = RUN_STATE_OF.get(trace.state, "running")
        fields: dict[str, Any] = {"state": state, "queued": int(trace.queued)}
        if trace.reason:
            fields["error_kind"] = trace.reason
        if trace.route:
            fields["route"] = trace.route
        if trace.started_at is not None:
            fields["started_at"] = self._wall(trace.started_at)
        if trace.ended_at is not None:
            fields["ended_at"] = self._wall(trace.ended_at)
        try:
            self.store.update_run(trace.run_id, self.clock(), **fields)
        except Exception as exc:
            self.counts["persist_errors"] += 1
            self.log(f"run state write failed: {type(exc).__name__}")

    def _finish(self, trace: RunTrace, task: asyncio.Task) -> None:
        if task.cancelled() and trace.ended_at is None:
            trace.end(self.now(), "cancelled", self.cancel_reasons.pop(trace.run_id, "cancelled"))
            self.abandon_outbound(trace)
        self.active.pop(trace.run_id, None)
        capture = self._captures.get(trace.run_id)
        if capture is not None and not capture.done():
            capture.set_result(None)  # ended without an answer
        try:
            self.tasks.remove(task)
        except ValueError:
            pass
        self.counts[f"run_{RUN_STATE_OF.get(trace.state, trace.state)}"] += 1
        record = {}
        agent = self._run_agents.pop(trace.run_id, self._agent)
        revision = self._run_revisions.pop(trace.run_id, self._base_settings.revision)
        self._run_limits.pop(trace.run_id, None)
        retire = asyncio.create_task(self.retire_models())
        self._closing.add(retire)
        retire.add_done_callback(self._closing.discard)
        runs = getattr(agent, "runs", None)
        if isinstance(runs, dict):
            state = runs.pop(trace.run_id, None)
            record = getattr(state, "record", None) or {}
        detail = {
            "config_revision": revision,
            "group_overrides": getattr(trace, "profile_fields", []),
            "trace_state": trace.state,
            "cost_mode": trace.cost_mode,
            "budget_scope": getattr(trace, "budget_scope", None),
            "intended_route": trace.intended_route,
            "clarify": trace.clarify,
            "material_gaps": trace.material_gaps,
            "degraded": trace.degraded,
            "images": len(trace.images),
            "cards": len(trace.cards),
            "knowledge": trace.knowledge,
            "tool_calls": [
                {k: v for k, v in c.items() if k in ("name", "status", "phase")}
                for c in trace.tool_calls
            ],
            "teaching": trace.teaching,
            "nudge": trace.nudge,  # earlier request id, its status, merged | answer_earlier
            # MIG-06: unauthorised model write/tool requests (source/kind/reason only).
            "dropped_requests": trace.dropped_requests,
            "live": _compact_record(record) if record else {},
        }
        if trace.dropped_requests:
            self.counts["dropped_requests"] += len(trace.dropped_requests)
        try:
            self.store.update_run(
                trace.run_id, self.clock(), detail=json.dumps(detail, ensure_ascii=False)
            )
        except Exception as exc:
            self.counts["persist_errors"] += 1
            self.log(f"run detail write failed: {type(exc).__name__}")

    # -- context ---------------------------------------------------------------------
    def _event(self, row) -> dict:
        """OneBot-shaped event dict for one stored message (context strategies and the
        live agent's material renderer work on these); image segments carry the local
        media reference."""
        segments = json.loads(row["segments"])
        for slot in self.store.message_media(row["seq"]):
            index = slot["segment_index"]
            if 0 <= index < len(segments) and segments[index].get("type") == "image":
                data = dict(segments[index].get("data") or {})
                status, reason = slot["status"], slot["reason"]
                if status == "available" and self.media_root is not None:
                    if not slot["path"] or not (self.media_root / slot["path"]).is_file():
                        status, reason = "unavailable", "file_missing"
                data["media_ref"] = {"status": status, "path": slot["path"], "reason": reason}
                if self.media is not None:
                    data["media_ref"].update(self.media.diagnostics(row["seq"], index, data))
                segments[index] = {**segments[index], "data": data}
        t = row["platform_time"]
        if t is None and row["received_at"] is not None:
            t = int(row["received_at"])
        return {
            "post_type": "message",
            "message_type": "group",
            "self_id": row["bot_id"],
            "group_id": row["group_id"],
            "message_id": row["message_id"],
            "user_id": row["user_id"],
            "time": t,
            "message": segments,
        }

    def history_events(self, m: Message, seq: int) -> list[tuple[Message, dict | None]]:
        anchor = self.store.message_row(seq)
        t = None
        if anchor is not None:
            t = anchor["platform_time"] if anchor["platform_time"] is not None else None
            t = t if t is not None else anchor["received_at"]
        since = None if t is None else t - self.history_seconds
        rows = {r["seq"]: r for r in self.store.history_window(m, seq, since, self.history_limit)}
        # Quote ancestry may reach past the window (reply_to / answers chains).
        frontier = list(m.reply_ids)
        for r in rows.values():
            frontier += _message(r).reply_ids
        known = {r["message_id"] for r in rows.values()}
        extra = 0
        for _ in range(self.ancestry_depth):
            wanted = [i for i in dict.fromkeys(frontier) if i not in known]
            if not wanted or extra >= self.ancestry_max:
                break
            found = self.store.visible_by_ids(m, seq, wanted[: self.ancestry_max - extra])
            frontier = []
            for r in found:
                rows[r["seq"]] = r
                known.add(r["message_id"])
                frontier += _message(r).reply_ids
                extra += 1
            known.update(wanted)
        # Strategy C `quote_context`: the neighbourhood of the directly quoted message,
        # which may lie outside the history window (an old quote).
        for target in [r for r in rows.values() if r["message_id"] in m.reply_ids]:
            tt = target["platform_time"] if target["platform_time"] is not None else None
            tt = tt if tt is not None else target["received_at"]
            if tt is None:
                continue
            for r in self.store.history_range(
                m, seq, tt - self.quote_context_seconds, tt + self.quote_context_seconds, 24
            ):
                rows.setdefault(r["seq"], r)
        ordered = sorted(
            rows.values(),
            key=lambda r: (
                r["platform_time"] if r["platform_time"] is not None else r["received_at"] or 0,
                r["seq"],
            ),
        )
        out = []
        for r in ordered:
            event = self._event(r)
            self.events_by_key.put((r["group_id"], r["message_id"]), event)
            out.append((_message(r), event))
        return out

    def explain_options(self, m: Message) -> dict:
        answers = self.store.answer_map(m.bot_id, m.group_id)
        return {"answers": answers} if answers else {}

    def earlier_request_status(self, trace: RunTrace, m: Message, message_id: str) -> str:
        """Nudge: the earlier request's run state from the store (every mark is persisted);
        a run of this process that is still tracked counts as active too."""
        for run_id, (_, other) in self.active.items():
            if (
                run_id != trace.run_id
                and run_id not in self._captures
                and other.group_id == m.group_id
                and other.message_id == message_id
            ):
                return "active"
        return self.store.request_run_status(m.bot_id, m.group_id, message_id)

    def context_built(self, trace: RunTrace, m: Message, seq: int, picks: list) -> None:
        by_relation: dict[str, list[int]] = defaultdict(list)
        relation = {p.message_id: p.relation for p in picks}
        for mid in trace.context_ids:
            if relation.get(mid) == "recent_chat":
                continue  # background slice: kept in the snapshot, not retained as related
            row = self.store.lookup(m.bot_id, m.group_id, mid)
            if row is not None:
                by_relation[relation.get(mid, "context")].append(row["seq"])
        for rel, seqs in by_relation.items():
            self.store.mark_related(seq, seqs, rel)
        self.store.save_context_snapshot(
            trace.run_id,
            seq,
            self.strategy_name,
            list(trace.context_ids),
            dict(trace.context_reasons),
            list(trace.material_gaps),
            "context_shortened" in trace.degraded,
            self.clock(),
        )

    async def prepare(self, trace: RunTrace, m: Message, seq: int) -> None:
        """Give a quote lookup (get_msg backfill) and pending image downloads of the
        request and its quote a short, bounded time before the context is frozen. When the
        asker's previous request has no answer here yet, the old bot's answer may still be
        on its way through the self-sync: one immediate read-only sync (at most ~1 s)."""
        deadline = self.now() + min(self.prepare_wait_seconds, max(0.0, self.remaining(trace) - 60))
        trace._prepare_deadline = deadline
        if self.self_sync is not None and self.now() < deadline:
            try:
                async with asyncio.timeout_at(deadline):
                    outcome = await self.self_sync.catch_up(m, seq)
                    if outcome != "skipped":
                        self.counts[f"self_sync_{outcome}"] += 1
            except TimeoutError:
                self.counts["prepare_timeouts"] += 1
        while self.now() < deadline:
            missing = self.backfill is not None and any(
                self.store.lookup(m.bot_id, m.group_id, rid) is None for rid in m.reply_ids
            )
            if not missing:
                return
            await asyncio.sleep(min(0.05, max(0, deadline - self.now())))
        self.counts["prepare_timeouts"] += 1

    async def prepare_context(self, trace: RunTrace, m: Message, seq: int) -> None:
        """Wait only for selected images, then freeze fresh per-run events."""
        deadline = getattr(trace, "_prepare_deadline", self.now())

        def rows():
            access = self.access.snapshot
            if m.group_id in access.blocked_groups or m.group_id in access.collection_off_groups:
                return []
            anchor = self.store.message_row(seq)
            history = self.store.visible_by_ids(m, seq, trace.context_ids)
            by_id = {r["message_id"]: r for r in history}
            ordered = ([anchor] if anchor is not None else []) + [
                by_id[mid]
                for mid in dict.fromkeys([*m.reply_ids, *reversed(trace.context_ids)])
                if mid in by_id
            ]
            return [
                r
                for r in ordered
                if not r["revoked"]
                and r["user_id"] not in access.blocked_users
                and not is_expired(r["platform_time"], self.platform_now(), self.limits)
            ]

        if self.media is not None:
            await self.media.prepare(lambda: [r["seq"] for r in rows()], deadline)
        else:
            while self.now() < deadline and self.store.pending_media(r["seq"] for r in rows()):
                await asyncio.sleep(min(0.05, max(0, deadline - self.now())))
        # Never rely on the shared LRU: another run may refresh or evict its entries.
        events = {(m.group_id, r["message_id"]): self._event(r) for r in rows()}
        events.setdefault(
            (m.group_id, m.message_id),
            {"message_id": m.message_id, "user_id": m.user_id, "message": []},
        )
        trace._material_events = events
        if self.store.pending_media(r["seq"] for r in rows()):
            self.counts["prepare_timeouts"] += 1

    # -- run hooks -----------------------------------------------------------------
    def teach(self, trace: RunTrace, m: Message, command) -> str:
        """Owner teaching (T4.3): authorised by the event's sender id only (SEC-01), written
        to the knowledge base idempotently per source message with revision checks."""
        trace.route = trace.intended_route = "teaching"
        owner = self.settings.owner_id
        is_owner = owner is not None and m.user_id == owner
        if self.knowledge is None:
            outcome = "not_enabled" if is_owner else "unauthorized"
            trace.teaching = {
                "op": command.op,
                "outcome": outcome,
                "scope": command.scope,
                "applied": False,
                "run_id": trace.run_id,
            }
            return UNAUTHORIZED_TEXT if outcome == "unauthorized" else TEACHING_PENDING_TEXT
        target = None
        if command.via_quote:
            for rid in m.reply_ids:
                used = self.store.answer_knowledge(m.bot_id, m.group_id, rid)
                if len(used) == 1:
                    target = next(iter(used.items()))
        images = self._teaching_images(m) if is_owner and command.op in ("add", "correct") else []
        entry = self.knowledge.apply(
            command,
            actor=m.user_id,
            owner_id=owner,
            source_message=m.message_id,
            group_id=m.group_id,
            now=self.clock(),
            target=target,
            images=images,
        )
        trace.teaching = {k: v for k, v in entry.items() if k != "t"} | {"run_id": trace.run_id}
        trace.clarify = entry["outcome"] in ASKS_USER
        if entry["id"]:
            trace.knowledge_ids = [entry["id"]]
        self.counts[f"teaching_{entry['op']}_{entry['outcome']}"] += 1
        return reply_for(entry, self.knowledge.describe)

    def _teaching_images(self, m: Message) -> list[Path | None]:
        """Image files of the teaching message and its quotes (legacy: examples); None for
        an image that is not available locally."""
        if self.media_root is None:
            return []
        out: list[Path | None] = []
        for mid in (m.message_id, *m.reply_ids):
            row = self.store.lookup(m.bot_id, m.group_id, mid)
            if row is None:
                continue
            for slot in self.store.message_media(row["seq"]):
                path = slot.get("path")
                ok = slot.get("status") == "available" and path
                out.append(self.media_root / path if ok else None)
        return out

    async def refresh_knowledge(self, trace: RunTrace, m: Message) -> None:
        """RUN-07: a used knowledge revision changed before publishing; the agent re-reads
        the entries before the one regeneration."""
        refresh = getattr(self.agent, "refresh_knowledge", None)
        if refresh is not None:
            result = refresh(trace)
            if asyncio.iscoroutine(result):
                await result

    async def tools(self, trace: RunTrace, m: Message, mocks: Any) -> None:
        run_tools = getattr(self.agent, "run_tools", None)
        if run_tools is not None:
            await run_tools(trace, m, ToolBudget(self.limits))

    def check(self, trace: RunTrace) -> str | None:
        reason = super().check(trace)
        if reason:
            return reason
        if self.paused is not None and self.paused() and trace.run_id not in self._captures:
            return "paused"
        s = self.settings
        if not self.store.allows_group_reply(s.bot_id, trace.group_id):
            return "group_stopped"
        row = self.store.lookup(s.bot_id, trace.group_id, trace.message_id)
        if row is None or row["revoked"]:
            return "request_recalled"
        is_owner = s.owner_id is not None and trace.user_id == s.owner_id
        is_owner = is_owner or trace.run_id in self._captures  # the owner's `#新版`
        mode, scope = self.usage.decision(is_owner, trace.group_id)
        if scope:
            trace.budget_scope = scope
        if mode == COST_REJECTED:
            return "cost_cap"
        return None

    # -- outbound: record only, or live through the publisher hook ------------------------
    def capture_answer(self, trace: RunTrace, text: str, kind: str) -> bool:
        capture = self._captures.get(trace.run_id)
        if capture is None:
            return False
        # A `#新版` regeneration: hand the answer back; no outbound row, no busy notice.
        # A fallback reply (every answer attempt failed) is not an answer to send.
        fallback = "fallback_reply" in trace.degraded
        if not capture.done():
            capture.set_result(text if kind == "answer" and not fallback else None)
        if kind == "answer":
            if fallback:
                trace.end(self.now(), "failed", "fallback_reply")
            else:
                trace.end(self.now(), "succeeded")
        return True

    def outbound_enqueue(self, record: OutboundRecord, m: Message):
        record.text, truncated = fit_message_text(record.text)
        if truncated:
            self.counts["outbound_truncated"] += 1
        mode = "record_only" if self.publisher is None else "live_send"
        _, fresh = self.store.outbound_enqueue(
            record.run_id, record.kind, m, record.text, self.clock(), mode=mode
        )
        if not fresh:
            self.counts["outbound_duplicate"] += 1
        return record, fresh

    def outbound_transition(self, record, state: str, reason: str | None = None) -> None:
        record.state, record.error_kind = state, reason
        self.store.outbound_update(record.run_id, self.clock(), state=state, error_kind=reason)

    async def deliver(self, trace: RunTrace, record, mocks: Any) -> str:
        now = self.clock()
        if self.publisher is not None:
            # Live (cutover): the row was re-checked after the rate wait; the hook re-checks
            # it once more, sends it once and records sent / failed / uncertain.
            self.store.outbound_update(
                record.run_id, now, scheduled_at=self._wall(record.scheduled_at)
            )
            result = await self.publisher(record.run_id)
            status = getattr(result, "status", None) or "failed"
            record.state = status
            self.counts[f"{status}_{record.kind}"] += 1
            if status == "failed":
                if record.kind == "answer":
                    trace.reason = f"send_{getattr(result, 'error_kind', None) or 'failed'}"
                return "failed"
            return "succeeded"  # sent, or uncertain (never resent; outbound row says so)
        # Record only: the text and target are kept, nothing is handed to the platform.
        self.store.outbound_update(
            record.run_id,
            now,
            state="recorded_not_sent",
            scheduled_at=self._wall(record.scheduled_at),
            recorded_at=now,
        )
        record.state = "recorded_not_sent"
        self.counts[f"recorded_{record.kind}"] += 1
        return "succeeded"

    def abandon_outbound(self, trace: RunTrace) -> None:
        self.store.fail_pending_outbound(trace.run_id, trace.state)
