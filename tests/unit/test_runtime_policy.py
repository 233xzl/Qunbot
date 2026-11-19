"""Production rules reused by the replay runner: policy, scheduler, outbox, teaching, cards."""

import asyncio
import json

import pytest

from group_agent.cards import parse_card
from group_agent.config import Settings
from group_agent.contracts import Context, FallbackModel, Message
from group_agent.outbox import OutboundRecord, Outbox, SendRateLimiter
from group_agent.pipeline import Pipeline
from group_agent.policy import (
    RuntimeLimits,
    ToolBudget,
    cost_mode,
    is_expired,
    knowledge_visible,
    moderation_blocks,
    prepublish_check,
    tool_timeout,
)
from group_agent.replay.vclock import SimulationStalled, run_virtual
from group_agent.scheduler import BusyNotifier, RunScheduler
from group_agent.store import Store
from group_agent.teaching import KnowledgeLedger, parse_teaching


def test_cost_mode_thresholds():
    assert cost_mode(0.79, False) == "normal"
    assert cost_mode(0.8, False) == "degraded"
    assert cost_mode(1.0, False) == "rejected"
    assert cost_mode(1.2, True) == "degraded"


def test_tool_budget_and_timeout():
    budget = ToolBudget(RuntimeLimits(web_searches=1))
    assert budget.try_consume("web_search") and not budget.try_consume("web_search")
    assert budget.try_consume("group_info")
    assert tool_timeout(120) == 25 and tool_timeout(40) == 10 and tool_timeout(10) == 0


def test_scope_retention_moderation_and_prepublish():
    assert knowledge_visible({"scope": "public"}, "g_1")
    assert knowledge_visible({"scope": "group:g_1"}, "g_1")
    assert not knowledge_visible({"scope": "group:g_2"}, "g_1")
    assert not knowledge_visible({"scope": "public", "active": False}, "g_1")
    assert is_expired(0, 7 * 86400 + 1) and not is_expired(0, 7 * 86400)
    assert not is_expired(None, 10**9)
    assert moderation_blocks({"choice": "sensitive", "probability": 0.5})
    assert not moderation_blocks({"choice": "normal", "probability": 0.99})
    revisions = {"kb": 2}.get
    assert prepublish_check("u", "g", frozenset(), frozenset(), {"kb": 2}, revisions) is None
    assert prepublish_check("u", "g", frozenset({"u"}), frozenset(), {}, revisions) == (
        "requester_blocked"
    )
    assert prepublish_check("u", "g", frozenset(), frozenset(), {"kb": 1}, revisions) == (
        "knowledge_changed"
    )


def test_send_rate_limiter_interval_and_minute_caps():
    limiter = SendRateLimiter()
    times = [limiter.reserve("g_1", 0) for _ in range(11)]
    assert times[:3] == [0, 2, 4]
    assert times[10] == 60  # 11th in a group waits for the first to leave the window
    limiter = SendRateLimiter(RuntimeLimits(account_per_minute=3))
    sends = [limiter.reserve(g, 0) for g in ("g_1", "g_2", "g_3", "g_4")]
    assert sends == [0, 0, 0, 60]


def test_outbox_is_idempotent_and_enforces_transitions():
    outbox = Outbox()
    record = OutboundRecord("r1:answer", "r1", "answer", "g", "m", "u", "hi")
    assert outbox.enqueue(record, 0)[1]
    assert not outbox.enqueue(OutboundRecord("r1:answer", "r1", "answer", "g", "m", "u", "x"), 1)[1]
    outbox.transition("r1:answer", "sending", 1)
    outbox.transition("r1:answer", "uncertain", 2)
    with pytest.raises(ValueError):
        outbox.transition("r1:answer", "sending", 3)


def test_scheduler_limits_queue_and_fifo_release():
    async def scenario():
        s = RunScheduler(RuntimeLimits(global_active=2, user_active=1, user_queue=1))
        a = s.request("u1")
        b = s.request("u2")
        c = s.request("u1")  # user limit -> queued
        assert not a.queued and not b.queued and c.queued
        assert s.request("u1") is None  # per-user queue full
        s.release(a)
        await s.acquire(c)
        assert s.active["u1"] == 1 and s.total_active == 2
        d = s.request("u3")
        with pytest.raises(TimeoutError):
            await s.acquire(d)  # waits 30 virtual seconds, then gives up
        assert s.waiting_count() == 0
        return asyncio.get_running_loop().time()

    assert run_virtual(scenario()) == 30


def test_busy_notifier_window():
    notifier = BusyNotifier()
    assert notifier.should_notify("u", 0)
    assert not notifier.should_notify("u", 59)
    assert notifier.should_notify("u", 60)


def test_virtual_loop_detects_deadlock():
    async def stuck():
        await asyncio.Event().wait()

    with pytest.raises(SimulationStalled):
        run_virtual(stuck())


def test_fallback_model_records_attempts():
    class Fail:
        async def run(self, context):
            raise RuntimeError("x")

    class Ok:
        async def run(self, context):
            return "ok"

    model = FallbackModel(Fail(), Ok())
    context = Context(Message("b", "g", "m", "u", ()), (), 1)
    assert asyncio.run(model.run(context)) == "ok"
    assert model.attempts == [(0, "RuntimeError"), (1, None)]


def test_teaching_grammar():
    add = parse_teaching("教学 本群：「开黑」指周五组队。", "g_1")
    assert (add.op, add.scope, add.term, add.content) == (
        "add",
        "group:g_1",
        "开黑",
        "指周五组队。",
    )
    assert parse_teaching("记住：「早八人」指早起的人。", "g_1").scope is None
    fix = parse_teaching("更正 「开黑」：指周六组队。", "g_1")
    assert (fix.op, fix.term, fix.content) == ("correct", "开黑", "指周六组队。")
    quoted = parse_teaching("不对，更正为：指周日。", "g_1", quotes_bot=True)
    assert quoted.via_quote and quoted.content == "指周日。"
    assert parse_teaching("照我上一条说的，把它记进知识库。", "g_1") is None


def test_ledger_authorization_idempotency_and_revision_check():
    ledger = KnowledgeLedger()
    add = parse_teaching("教学 公共：「梗」指某事。", "g_1")
    kwargs = dict(owner_id="owner", group_id="g_1", now=0, new_id="kb1")
    assert ledger.apply(add, actor="other", source_message="m0", **kwargs)["outcome"] == (
        "unauthorized"
    )
    assert ledger.apply(add, actor="owner", source_message="m1", **kwargs)["applied"]
    assert ledger.apply(add, actor="owner", source_message="m1", **kwargs)["outcome"] == "duplicate"
    assert ledger.apply(add, actor="owner", source_message="m2", **kwargs)["outcome"] == "duplicate"
    fix = parse_teaching("更正 「梗」：指另一件事。", "g_1")
    entry = ledger.apply(fix, actor="owner", source_message="m3", target=("kb1", 1), **kwargs)
    assert entry["applied"] and entry["revision"] == 2
    stale = ledger.apply(fix, actor="owner", source_message="m4", target=("kb1", 1), **kwargs)
    assert stale["outcome"] == "conflict" and ledger.records["kb1"].revision == 2


def test_parse_cards():
    mini = {
        "type": "json",
        "data": {
            "data": json.dumps(
                {
                    "app": "com.tencent.miniapp_01",
                    "meta": {
                        "detail_1": {
                            "title": "站",
                            "desc": "标题",
                            "qqdocurl": "https://a.invalid/x",
                        }
                    },
                }
            )
        },
    }
    assert parse_card(mini) == {
        "app": "com.tencent.miniapp_01",
        "kind": "miniapp",
        "title": "标题",
        "desc": None,
        "url": "https://a.invalid/x",
        "source": "站",
    }
    assert parse_card({"type": "json", "data": {"data": "not json"}}) is None
    assert parse_card({"type": "text", "data": {"text": "x"}}) is None


def test_pipeline_admit_handles_recall_before_and_after_message():
    store = Store()
    pipeline = Pipeline(Settings(bot_id="BOT"), store)
    message = {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": "G",
        "message_id": "m1",
        "user_id": "U",
        "message": [{"type": "text", "data": {"text": "hi"}}],
    }
    recall = {
        "post_type": "notice",
        "notice_type": "group_recall",
        "self_id": "BOT",
        "group_id": "G",
        "message_id": "m1",
        "user_id": "U",
    }
    assert pipeline.admit(message).status == "collected"
    assert pipeline.admit(recall).status == "recall_revoked"
    assert store.lookup("BOT", "G", "m1")["revoked"] == 1
    assert pipeline.admit({**recall, "message_id": "m2"}).status == "recall_tombstone"
    assert pipeline.admit({**message, "message_id": "m2"}).status == "duplicate"
    store.close()
