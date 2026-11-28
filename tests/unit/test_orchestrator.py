"""Online orchestrator (T3.4 / T3.7 / T3.9) with fake agent, fake clock and fake gateway.

Nothing here touches the network or credentials; outbound rows end `recorded_not_sent`.
"""

import asyncio
import json
import sqlite3

import pytest

from group_agent.admission import AccessPolicy, AccessSnapshot, RequestQueue
from group_agent.config import Settings
from group_agent.intake import EventIntake
from group_agent.models import CallRecord, Usage
from group_agent.orchestrator import Orchestrator
from group_agent.policy import RuntimeLimits
from group_agent.replay.vclock import run_virtual
from group_agent.runflow import CURRENT_RUN
from group_agent.store import Store
from group_agent.teaching import UNAUTHORIZED_TEXT
from group_agent.usage import UsageLedger, cn_day

BASE = 1_800_000_000  # wall-clock origin of the fake clock (2027-01-15, Asia/Shanghai)


def wall() -> float:
    return BASE + asyncio.get_running_loop().time()


def ev(mid, user="U1", group="G1", text="问一下", at=True, t=None, reply=None):
    segments = []
    if reply:
        segments.append({"type": "reply", "data": {"id": reply}})
    if at:
        segments.append({"type": "at", "data": {"qq": "BOT"}})
    segments.append({"type": "text", "data": {"text": text}})
    event = {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": group,
        "message_id": mid,
        "user_id": user,
        "message": segments,
    }
    if t is not None:
        event["time"] = t
    return event


def recall(mid, user="U1", group="G1"):
    return {
        "post_type": "notice",
        "notice_type": "group_recall",
        "self_id": "BOT",
        "group_id": group,
        "message_id": mid,
        "user_id": user,
        "operator_id": user,
    }


class FakeAgent:
    """Model side stand-in: decide `route`, answer after `delay[message_id]` seconds."""

    def __init__(self, host, delays=None, route="chat", usage=None):
        self.host, self.delays, self.route, self.usage = host, delays or {}, route, usage
        self.answered: list[str] = []
        self.tool_routes: list[str] = []
        self.runs = {}

    async def vision(self, trace, mocks):
        return True

    async def decide_route(self, trace, mocks):
        return self.route

    async def run_tools(self, trace, m, budget):
        self.tool_routes.append(trace.route)

    async def answer(self, trace, mocks, message, seq):
        if self.usage is not None:  # a billed model call inside the run
            record = CallRecord("answer", "gemini", "text", usage=Usage(10, 0, 5, None, 1))
            record.cost_cny = 0.01
            self.usage.reserve(0.01)
            self.usage.settle(0.01, record)
        await asyncio.sleep(self.delays.get(message.message_id, 1.0))
        self.answered.append(message.message_id)
        return f"answer:{message.message_id}"


class Harness:
    def __init__(self, store=None, limits=None, owner="OWNER", daily=0.0, agent_kw=None):
        self.store = store or Store()
        self.access = AccessPolicy(AccessSnapshot("BOT", owner))
        self.settings = Settings(bot_id="BOT", owner_id=owner)
        if daily:
            seed_usage(self.store, daily, cn_day(BASE))
        self.usage = UsageLedger(self.store, daily_cap_cny=3.0, clock=wall)
        self.queue = RequestQueue(64)
        self.intake = EventIntake(self.store, self.access, admit=True, sink=self.queue, clock=wall)
        kw = dict(agent_kw or {})
        self.orch = Orchestrator(
            self.store,
            self.access,
            self.settings,
            agent_factory=lambda host: self._agent(host, kw),
            usage=self.usage,
            limits=limits or RuntimeLimits(),
            clock=wall,
        )

    def _agent(self, host, kw):
        self.agent = FakeAgent(host, usage=kw.pop("usage", None) and self.usage, **kw)
        return self.agent

    def feed(self, event):
        result = self.intake.intake(event)
        if result.status.startswith("recall_") and event.get("message_id"):
            self.orch.on_recall(str(event["group_id"]), str(event["message_id"]))
        while not self.queue.queue.empty():
            self.orch.submit(self.queue.queue.get_nowait())
        return result.status

    async def settle(self):
        while self.orch.tasks:
            await asyncio.gather(*list(self.orch.tasks))

    def runs(self):
        rows = self.store.con.execute("SELECT * FROM runs ORDER BY created_at, rowid").fetchall()
        return {r["message_id"]: dict(r) for r in rows}

    def outbound(self):
        rows = self.store.con.execute("SELECT * FROM outbound ORDER BY scheduled_at").fetchall()
        return [dict(r) for r in rows]


def seed_usage(store, cost, day):
    store.record_usage(
        {
            "run_id": None,
            "day": day,
            "created_at": BASE,
            "provider": "gemini",
            "model": "m",
            "task": "seed",
            "mode": "text",
            "status": "ok",
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": None,
            "requests": 1,
            "cost_cny": cost,
            "latency_s": 0.0,
        }
    )


def virtual(coro_fn):
    return run_virtual(coro_fn())


@pytest.mark.parametrize("quoted", [False, True])
async def test_image_completed_during_prepare_reaches_live_material(tmp_path, quoted):
    """The old anchor event is pending; only the fresh run snapshot has the image."""
    import io
    from PIL import Image
    from group_agent.live_agent import LiveAgent
    from test_live_replay import live_config
    from test_media import make as make_media

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, "PNG")
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def download(*_):
        entered.set()
        await gate.wait()
        return 200, buffer.getvalue(), None

    store, media = make_media(tmp_path, download=download)
    h = Harness(store=store)
    h.intake.media = h.orch.media = media
    h.orch.media_root = media.root
    config = live_config(tmp_path, lambda: None, lambda: None)
    config.media_root = media.root
    live = LiveAgent(h.orch, config)
    seen = []

    async def answer(trace, *_):
        # Poison the shared cache to prove this run uses its own snapshot.
        h.orch.events_by_key.clear()
        state = live._state(trace)
        material = await live._material(trace, state)
        seen.append(material)
        return "读到了"

    h.agent.answer = answer
    image_event = ev("picture", at=not quoted)
    image_event["message"].append(
        {"type": "image", "data": {"file": "same", "url": "https://gchat.qpic.cn/img"}}
    )
    h.feed(image_event)
    if quoted:
        h.feed(ev("question", reply="picture"))
    await entered.wait()
    # Wait until prepare_context is waiting for the still-pending selected media.
    await asyncio.sleep(0.02)
    gate.set()
    await h.settle()
    assert len(seen) == 1
    assert len(seen[0].images) == 1
    assert seen[0].stats["images_unavailable"] == 0
    assert "已附上" in seen[0].text
    await media.aclose()


@pytest.mark.parametrize("change", ["recalled", "blocked", "timeout"])
async def test_prepared_image_snapshot_respects_late_validity_and_deadline(tmp_path, change):
    from test_media import make as make_media, FakeDownload

    gate = asyncio.Event()
    download = FakeDownload(gate=gate)
    store, media = make_media(tmp_path, download=download)
    h = Harness(store=store)
    h.intake.media = h.orch.media = media
    h.orch.media_root = media.root
    h.orch.prepare_wait_seconds = 0.07
    seen = []

    async def answer(trace, *_):
        seen.append(trace._material_events)
        return "完成"

    h.agent.answer = answer
    event = ev("picture", user="PICTURE_AUTHOR", at=False)
    event["message"].append(
        {"type": "image", "data": {"file": "f", "url": "https://gchat.qpic.cn/img"}}
    )
    h.feed(event)
    h.feed(ev("question", reply="picture"))
    while not download.calls:
        await asyncio.sleep(0)
    if change == "recalled":
        h.feed(recall("picture", user="PICTURE_AUTHOR"))
    elif change == "blocked":
        h.access.update(blocked_users={"PICTURE_AUTHOR"})
    # A message arriving while preparing must not slip past the request boundary.
    h.feed(ev("late", at=False))
    await h.settle()
    assert len(seen) == 1
    assert ("G1", "late") not in seen[0]
    if change == "timeout":
        image = seen[0][("G1", "picture")]["message"][-1]
        assert image["data"]["media_ref"]["status"] == "pending"
    else:
        assert ("G1", "picture") not in seen[0]
    gate.set()
    await media.aclose()


# -- T3.4 scheduling ------------------------------------------------------------------


def test_concurrency_and_queue_limits_with_busy_notice_recorded():
    limits = RuntimeLimits(global_active=2, user_active=1, global_queue=2, user_queue=1)

    async def main():
        h = Harness(limits=limits, agent_kw={"delays": {"a1": 10, "a2": 10, "b1": 10}})
        for mid, user in [("a1", "UA"), ("a2", "UA"), ("b1", "UB"), ("a3", "UA"), ("c1", "UC")]:
            h.feed(ev(mid, user=user))
        await h.settle()
        return h

    h = virtual(main)
    runs = h.runs()
    assert h.orch.scheduler.max_total_active == 2
    assert runs["a3"]["state"] == "rejected" and runs["a3"]["error_kind"] == "queue_full"
    assert {runs[m]["state"] for m in ("a1", "a2", "b1", "c1")} == {"succeeded"}
    assert runs["a2"]["queued"] == 1 and runs["c1"]["queued"] == 1
    kinds = {(o["reply_to"], o["kind"], o["state"]) for o in h.outbound()}
    assert ("a3", "busy_notice", "recorded_not_sent") in kinds
    assert len([k for k in kinds if k[1] == "answer"]) == 4
    assert all(o["qq_message_id"] is None for o in h.outbound())
    assert h.store.summary()["sent_to_qq"] == 0


def test_queue_wait_limit_expires_run_with_one_busy_notice():
    limits = RuntimeLimits(global_active=1, queue_wait_seconds=5)

    async def main():
        h = Harness(limits=limits, agent_kw={"delays": {"x1": 20}})
        h.feed(ev("x1", user="UX"))
        h.feed(ev("y1", user="UY"))
        await asyncio.sleep(1)
        h.feed(ev("y2", user="UY", text="再问"))
        await h.settle()
        return h

    h = virtual(main)
    runs = h.runs()
    assert (runs["y1"]["state"], runs["y1"]["error_kind"]) == ("expired", "queue_wait_limit")
    assert (runs["y2"]["state"], runs["y2"]["error_kind"]) == ("expired", "queue_wait_limit")
    notices = [o for o in h.outbound() if o["kind"] == "busy_notice"]
    assert [o["reply_to"] for o in notices] == ["y1"]  # D-08: one per user window


def test_total_deadline_expires_run_without_outbound():
    async def main():
        h = Harness(limits=RuntimeLimits(deadline_seconds=10), agent_kw={"delays": {"s": 100}})
        h.feed(ev("s"))
        h.feed(ev("f", user="U2"))
        await h.settle()
        return h

    h = virtual(main)
    runs = h.runs()
    assert (runs["s"]["state"], runs["s"]["error_kind"]) == ("expired", "deadline")
    assert runs["f"]["state"] == "succeeded"
    assert [o["reply_to"] for o in h.outbound()] == ["f"]


def test_recalled_question_cancels_run_and_releases_slot():
    limits = RuntimeLimits(global_active=1)

    async def main():
        h = Harness(limits=limits, agent_kw={"delays": {"q": 50}})
        h.feed(ev("q"))
        h.feed(ev("next", user="U2"))
        await asyncio.sleep(5)
        h.feed(recall("q"))
        await h.settle()
        return h

    h = virtual(main)
    runs = h.runs()
    assert (runs["q"]["state"], runs["q"]["error_kind"]) == ("cancelled", "request_recalled")
    assert runs["next"]["state"] == "succeeded"
    assert runs["next"]["started_at"] - runs["next"]["created_at"] == pytest.approx(5, abs=0.01)
    assert [o["reply_to"] for o in h.outbound()] == ["next"]


def test_recall_noticed_only_at_publish_withholds_answer():
    async def main():
        h = Harness(agent_kw={"delays": {"q": 5}})
        h.feed(ev("q"))
        await asyncio.sleep(1)
        h.intake.intake(recall("q"))  # store updated, run not told
        await h.settle()
        return h

    h = virtual(main)
    run = h.runs()["q"]
    assert (run["state"], run["error_kind"]) == ("cancelled", "request_recalled")
    assert h.outbound() == []


# -- AC-14 / AC-15 --------------------------------------------------------------------


@pytest.mark.parametrize("notify", [True, False])
def test_blacklisted_during_generation_is_not_published(notify):
    async def main():
        h = Harness(agent_kw={"delays": {"q": 10}})
        h.feed(ev("q", user="U1"))
        await asyncio.sleep(3)
        if notify:
            assert h.orch.apply_access(blocked_users={"U1"}) == 1
        else:
            h.access.update(blocked_users={"U1"})  # e.g. a config reload without notice
        await h.settle()
        return h

    h = virtual(main)
    run = h.runs()["q"]
    assert (run["state"], run["error_kind"]) == ("cancelled", "requester_blocked")
    assert h.outbound() == []


def test_duplicate_events_make_one_run_and_one_outbound():
    async def main():
        h = Harness()
        statuses = [h.feed(ev("q")) for _ in range(3)]
        await h.settle()
        # Even a duplicated hand-off of the same request cannot start a second run.
        row = h.store.message_row(h.store.lookup("BOT", "G1", "q")["seq"])
        from group_agent.admission import ReplyRequest
        from group_agent.store import _message

        again = ReplyRequest(_message(row), row["seq"], None, BASE, False)
        assert h.orch.submit(again) == "duplicate"
        await h.settle()
        return h, statuses

    h, statuses = virtual(main)
    assert statuses == ["trigger", "duplicate", "duplicate"]
    assert len(h.runs()) == 1
    (out,) = h.outbound()
    assert (out["reply_to"], out["state"], out["mode"]) == ("q", "recorded_not_sent", "record_only")
    assert out["idempotency_key"] == "BOT:G1:q:answer"


def test_outbound_enqueue_is_idempotent_per_run():
    from group_agent.contracts import Message

    store = Store()
    m = Message("BOT", "G1", "q", "U1", ())
    seq, _ = store.ingest(m)
    assert store.create_run("r1", seq, m, state="queued", cost_mode="normal", now=1.0)
    assert not store.create_run("r2", seq, m, state="queued", cost_mode="normal", now=1.0)
    _, fresh = store.outbound_enqueue("r1", "answer", m, "a", 2.0)
    _, again = store.outbound_enqueue("r1", "answer", m, "b", 3.0)
    assert fresh and not again
    assert store.outbound("r1")["text"] == "a"
    with pytest.raises(sqlite3.IntegrityError):
        with store.con:
            store.con.execute("UPDATE outbound SET state='acked'")
    # Idempotent across modes too: the key is the request message and kind.
    _, live = store.outbound_enqueue("r1", "answer", m, "c", 3.0, mode="live_send")
    assert not live and store.outbound("r1")["mode"] == "record_only"
    with pytest.raises(ValueError):  # the owner's #新版 rows are always live (D-19)
        store.outbound_enqueue("r1", "new_version", m, "c", 3.0)
    with pytest.raises(ValueError):
        store.outbound_update("r1", 4.0, state="acked")


# -- AC-19 send rate ------------------------------------------------------------------


def test_six_simultaneous_answers_are_rate_limited_and_keep_their_targets():
    async def main():
        h = Harness(limits=RuntimeLimits(global_active=6, user_active=1))
        for n in range(6):
            h.feed(ev(f"q{n}", user=f"U{n}"))
        await h.settle()
        return h

    h = virtual(main)
    out = h.outbound()
    assert len(out) == 6 and {o["state"] for o in out} == {"recorded_not_sent"}
    times = [o["scheduled_at"] for o in out]
    assert all(b - a >= 2 - 1e-6 for a, b in zip(times, times[1:]))  # D-10 min interval
    assert all(sum(1 for t in times if s <= t < s + 60) <= 10 for s in times)
    for o in out:
        assert o["requester"] == "U" + o["reply_to"][1:] and o["group_id"] == "G1"
    assert {r["state"] for r in h.runs().values()} == {"succeeded"}


def test_per_minute_group_cap():
    async def main():
        limits = RuntimeLimits(global_active=12, user_active=1, group_per_minute=3)
        h = Harness(limits=limits)
        for n in range(5):
            h.feed(ev(f"q{n}", user=f"U{n}"))
        await h.settle()
        return h

    h = virtual(main)
    times = sorted(o["scheduled_at"] for o in h.outbound())
    assert all(sum(1 for t in times if s <= t < s + 60) <= 3 for s in times)
    assert times[3] - times[0] >= 60 - 1e-6


# -- AC-20 context: recall, expiry, snapshot, related marks ----------------------------


def test_context_snapshot_excludes_recalled_and_expired_and_marks_related():
    async def main():
        h = Harness()
        old = BASE - 8 * 86400
        h.feed(ev("old", user="U1", at=False, text="很久以前的话", t=old))
        h.feed(ev("img", user="U1", at=False, text="看这个", t=BASE - 30))
        h.feed(ev("gone", user="U1", at=False, text="撤回的话", t=BASE - 20))
        h.feed(recall("gone"))
        h.feed(ev("other", user="U9", at=False, text="无关", t=BASE - 10))
        h.feed(ev("q", user="U1", text="这个什么意思", t=BASE, reply="old"))
        await h.settle()
        return h

    h = virtual(main)
    run = h.runs()["q"]
    snap = h.store.context_snapshot(run["run_id"])
    assert "gone" not in snap["message_ids"] and "old" not in snap["message_ids"]
    assert "img" in snap["message_ids"]
    # A member's unrelated line 10 s earlier is only the background slice (recent_chat).
    assert snap["reasons"]["other"].startswith("recent_chat/")
    assert snap["gaps"] == ["quote_expired"] and snap["strategy"] == "C"
    assert set(snap["reasons"]) == set(snap["message_ids"])
    marks = h.store.con.execute(
        "SELECT m.message_id, r.relation FROM related_marks r JOIN messages m ON m.seq=r.seq"
    ).fetchall()
    assert {tuple(r) for r in marks} == {("img", "asker_recent")}
    # The snapshot keeps ids and reasons only, no message text.
    raw = json.dumps(dict(h.store.con.execute("SELECT * FROM context_snapshots").fetchone()))
    assert "看这个" not in raw


def test_quote_ancestry_outside_the_window_is_loaded():
    async def main():
        h = Harness()
        h.orch.history_seconds = 60
        h.feed(ev("root", user="U2", at=False, text="原帖", t=BASE - 3000))
        h.feed(ev("q", user="U1", text="这说的啥", t=BASE, reply="root"))
        await h.settle()
        return h

    h = virtual(main)
    snap = h.store.context_snapshot(h.runs()["q"]["run_id"])
    assert snap["message_ids"] == ["root"] and snap["reasons"]["root"].startswith("reply_to")


# -- nudge (strategy C `urges`): merge or answer the earlier request ---------------------


def nudge_run(earlier_setup):
    """Feed an asker's request, let `earlier_setup(h)` change its fate, then a bare nudge."""

    async def main():
        h = Harness()
        h.feed(ev("q1", user="U1", text="崩铁17326是什么意思", t=BASE))
        await earlier_setup(h)
        h.feed(ev("nudge", user="U1", text="回答我", t=BASE + 30))
        await h.settle()
        return h

    return virtual(main)


def test_nudge_is_merged_while_the_earlier_run_is_active_or_answered():
    async def keep(h):  # the nudge is admitted while q1's run is still queued/running
        return None

    h = nudge_run(keep)
    runs = h.runs()
    assert runs["nudge"]["state"] == "succeeded" and runs["nudge"]["route"] == "nudge_merged"
    assert runs["nudge"]["error_kind"] == "nudge_merged"
    detail = json.loads(runs["nudge"]["detail"])["nudge"]
    assert detail["earlier_message_id"] == "q1" and detail["outcome"] == "merged"
    assert [o["reply_to"] for o in h.outbound()] == ["q1"]  # no second answer
    assert h.agent.answered == ["q1"]  # the model was never called for the nudge

    async def finish_first(h):  # q1 was answered (recorded) before the nudge arrived
        await h.settle()

    h = nudge_run(finish_first)
    assert h.runs()["nudge"]["route"] == "nudge_merged"
    assert json.loads(h.runs()["nudge"]["detail"])["nudge"]["earlier"] == "answered"
    assert [o["reply_to"] for o in h.outbound()] == ["q1"] and h.agent.answered == ["q1"]


def test_nudge_answers_the_earlier_request_when_its_run_ended_without_answer():
    async def cancel_first(h):  # the earlier run ends cancelled, no answer
        await asyncio.sleep(0.1)  # let the run start
        h.orch.cancel_run(next(iter(h.orch.active)), "scripted")
        await h.settle()

    h = nudge_run(cancel_first)
    runs = h.runs()
    assert runs["q1"]["state"] == "cancelled"
    assert runs["nudge"]["state"] == "succeeded" and runs["nudge"]["route"] == "chat"
    detail = json.loads(runs["nudge"]["detail"])["nudge"]
    assert detail["earlier"] == "open" and detail["outcome"] == "answer_earlier"
    assert [o["reply_to"] for o in h.outbound()] == ["nudge"] and h.agent.answered == ["nudge"]
    snap = h.store.context_snapshot(runs["nudge"]["run_id"])
    assert snap["reasons"]["q1"].startswith("urges/")


def test_request_run_status_covers_every_run_fate():
    async def main():
        h = Harness()
        h.feed(ev("q1", user="U1", t=BASE))
        h.feed(ev("q2", user="U2", t=BASE + 1))
        await asyncio.sleep(0.1)  # let the run start
        h.orch.cancel_run(next(iter(h.orch.active)), "scripted")
        await h.settle()
        return h

    h = virtual(main)
    status = h.store.request_run_status
    states = {m: (status("BOT", "G1", m), h.runs()[m]["state"]) for m in ("q1", "q2")}
    assert states == {"q1": ("open", "cancelled"), "q2": ("answered", "succeeded")}
    assert status("BOT", "G1", "nope") == "open" and status("BOT", "G2", "q2") == "open"


# -- AC-21 cost cap -------------------------------------------------------------------


def test_cost_80_percent_degrades_and_disables_web():
    async def main():
        h = Harness(daily=2.5, agent_kw={"route": "web"})
        h.feed(ev("q"))
        await h.settle()
        return h

    h = virtual(main)
    run = h.runs()["q"]
    detail = json.loads(run["detail"])
    assert run["cost_mode"] == "degraded" and run["route"] == "chat"
    assert detail["intended_route"] == "web" and "web_disabled" in detail["degraded"]
    assert h.agent.tool_routes == ["chat"]


def test_cost_100_percent_serves_only_owner_without_notice():
    async def main():
        h = Harness(daily=3.2)
        h.feed(ev("u", user="U1"))
        h.feed(ev("o", user="OWNER"))
        await h.settle()
        return h

    h = virtual(main)
    runs = h.runs()
    assert (runs["u"]["state"], runs["u"]["error_kind"]) == ("rejected", "cost_cap")
    assert runs["o"]["state"] == "succeeded" and runs["o"]["cost_mode"] == "degraded"
    assert [o["reply_to"] for o in h.outbound()] == ["o"]  # no notice to the non-owner
    assert h.agent.answered == ["o"]


def test_cap_reached_during_generation_withholds_non_owner_answer():
    async def main():
        h = Harness(daily=2.0, agent_kw={"delays": {"q": 10}})
        h.feed(ev("q"))
        await asyncio.sleep(2)
        seed_usage(h.store, 0.0, cn_day(BASE))
        h.usage._daily += 1.5  # another run's spending lands meanwhile
        await h.settle()
        return h

    h = virtual(main)
    run = h.runs()["q"]
    assert (run["state"], run["error_kind"]) == ("cancelled", "cost_cap")
    assert h.outbound() == []


def test_usage_ledger_persists_rows_and_restores_the_day(tmp_path):
    path = tmp_path / "u.sqlite3"

    async def main():
        h = Harness(store=Store(path), agent_kw={"usage": True})
        h.feed(ev("q"))
        await h.settle()
        h.store.close()

    run_virtual(main())
    store = Store(path)
    rows = store.con.execute("SELECT * FROM usage_ledger").fetchall()
    run_id = store.con.execute("SELECT run_id FROM runs").fetchone()[0]
    assert [(r["run_id"], r["provider"], r["cost_cny"]) for r in rows] == [(run_id, "gemini", 0.01)]
    assert rows[0]["input_tokens"] == 10 and rows[0]["output_tokens"] == 5
    seed_usage(store, 2.0, cn_day(BASE - 86400))  # yesterday does not count
    restarted = UsageLedger(store, daily_cap_cny=3.0, clock=lambda: BASE + 5)
    assert restarted.daily_spent() == pytest.approx(0.01)
    assert restarted.restored_cny == pytest.approx(0.01)
    seed_usage(store, 2.5, cn_day(BASE))
    again = UsageLedger(store, daily_cap_cny=3.0, clock=lambda: BASE + 6)
    assert again.cost_mode(False) == "degraded"
    # Next natural day (Asia/Shanghai) starts from zero.
    tomorrow = UsageLedger(store, daily_cap_cny=3.0, clock=lambda: BASE + 86400)
    assert tomorrow.daily_spent() == 0 and tomorrow.cost_mode(False) == "normal"
    store.close()


def test_usage_ledger_day_rolls_over_in_process():
    store = Store()
    now = [BASE]
    ledger = UsageLedger(store, daily_cap_cny=1.0, clock=lambda: now[0])
    token = CURRENT_RUN.set("r9")
    try:
        record = CallRecord("answer", "deepseek", "text")
        record.cost_cny = 0.9
        ledger.settle(0.0, record)
    finally:
        CURRENT_RUN.reset(token)
    assert ledger.cost_mode(False) == "degraded"
    assert store.con.execute("SELECT run_id FROM usage_ledger").fetchone()[0] == "r9"
    now[0] += 86400
    assert ledger.cost_mode(False) == "normal"


def test_process_budget_refuses_calls():
    from group_agent.models import BudgetExceeded

    ledger = UsageLedger(Store(), budget_cny=0.01, clock=lambda: BASE)
    ledger.reserve(0.005)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.01)


# -- restart / teaching ------------------------------------------------------------------


def test_restart_marks_unfinished_runs_interrupted(tmp_path):
    from group_agent.contracts import Message

    path = tmp_path / "r.sqlite3"
    store = Store(path)
    m = Message("BOT", "G1", "q", "U1", ())
    seq, _ = store.ingest(m)
    store.create_run("r1", seq, m, state="running", cost_mode="normal", now=1.0)
    store.outbound_enqueue("r1", "answer", m, "a", 2.0)
    store.close()

    async def main():
        h = Harness(store=Store(path))
        assert h.orch.interrupted == 1
        return h

    h = run_virtual(main())
    assert (h.store.run("r1")["state"], h.store.run("r1")["error_kind"]) == (
        "failed",
        "interrupted",
    )
    assert h.store.outbound("r1")["state"] == "failed"


def test_teaching_without_knowledge_base_is_recognised_but_not_written():
    async def main():
        h = Harness()
        h.feed(ev("t1", user="OWNER", text="记住 本群：「开黑」晚上九点"))
        h.feed(ev("t2", user="U1", text="记住 本群：「开黑」随时"))
        await h.settle()
        return h

    h = virtual(main)
    out = {o["reply_to"]: o["text"] for o in h.outbound()}
    assert "没有记录" in out["t1"] and out["t2"] == UNAUTHORIZED_TEXT
    detail = json.loads(h.runs()["t1"]["detail"])
    assert detail["teaching"]["applied"] is False and h.agent.answered == []


def test_v4_database_with_runs_migrates_to_v5(tmp_path):
    path = tmp_path / "v4.db"
    with sqlite3.connect(path) as con:
        con.executescript(
            "CREATE TABLE messages (seq INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL,"
            " group_id TEXT NOT NULL, message_id TEXT NOT NULL, user_id TEXT NOT NULL,"
            " segments TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,"
            " platform_time INTEGER, received_at REAL,"
            " source TEXT NOT NULL DEFAULT 'live', UNIQUE(bot_id, group_id, message_id));"
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, message_seq INTEGER NOT NULL UNIQUE"
            " REFERENCES messages(seq), state TEXT NOT NULL CHECK(state IN ('running',"
            " 'recorded', 'failed', 'cancelled')), error_kind TEXT);"
            "CREATE TABLE outbound (run_id TEXT PRIMARY KEY REFERENCES runs(run_id),"
            " bot_id TEXT NOT NULL, group_id TEXT NOT NULL, reply_to TEXT NOT NULL,"
            " requester TEXT NOT NULL, text TEXT NOT NULL, state TEXT NOT NULL"
            " CHECK(state = 'pending'), mode TEXT NOT NULL CHECK(mode = 'offline'));"
            "INSERT INTO messages(bot_id,group_id,message_id,user_id,segments)"
            " VALUES ('BOT','G1','m1','U1','[]');"
            "INSERT INTO runs VALUES ('r1', 1, 'recorded', NULL);"
            "INSERT INTO outbound VALUES ('r1','BOT','G1','m1','U1','x','pending','offline');"
            "PRAGMA user_version=4;"
        )
    store = Store(path)
    assert store.run("r1")["state"] == "recorded"
    assert (
        store.outbound("r1")["kind"] == "answer" and store.outbound("r1")["qq_message_id"] is None
    )
    store.update_run("r1", 5.0, state="expired")
    assert store.summary()["run_states"] == {"expired": 1}
    store.close()


@pytest.mark.parametrize("mode,guard", [("lightweight", None), ("model", "group_info")])
def test_persisted_live_diagnostics_keep_speed_policy_without_raw_material(mode, guard):
    from types import SimpleNamespace

    marker = "synthetic-private-material"
    record = {
        "decision_mode": mode,
        "decision_source": mode,
        "knowledge_mode": "candidates",
        "anchor_text": marker,
        "quoted_text": marker,
        "answer": marker,
        "knowledge_candidates": [marker],
        "decision": {
            "route": "chat",
            "reason": marker,
            "search_terms": [marker],
            "subject_terms": [marker],
            "clarify_question": marker,
        },
        "web": {"sources": [{"url": marker}], "calls": [{"query": marker}]},
    }
    if guard:
        record["decision_guard"] = guard

    class DiagnosticsAgent(FakeAgent):
        async def answer(self, trace, mocks, message, seq):
            self.runs[trace.run_id] = SimpleNamespace(record=record)
            return await super().answer(trace, mocks, message, seq)

    async def main():
        h = Harness()
        h.agent = h.orch.agent = DiagnosticsAgent(h.orch)
        h.feed(ev("speed"))
        await h.settle()
        return h

    h = virtual(main)
    saved = json.loads(h.runs()["speed"]["detail"])["live"]
    assert saved["decision_source"] == saved["decision_mode"] == mode
    assert saved["knowledge_mode"] == "candidates"
    assert saved.get("decision_guard") == guard
    assert marker not in json.dumps(saved)


def test_group_budget_reservations_midnight_and_hot_limits():
    from group_agent.models import BudgetExceeded
    from group_agent.runflow import CURRENT_GROUP

    now = [BASE]
    usage = UsageLedger(Store(), daily_cap_cny=18, clock=lambda: now[0])
    usage.set_profiles({"A": {"enabled": True, "daily_cost_cny": 5}})
    token = CURRENT_GROUP.set("A")
    try:
        usage.reserve(4)
        assert usage.decision(False, "A") == ("degraded", "group")
        assert usage.decision(False, "B") == ("normal", None)
        with pytest.raises(BudgetExceeded, match="group"):
            usage.reserve(2)
        usage.set_profiles({"A": {"enabled": True, "daily_cost_cny": 3}})
        assert usage.decision(False, "A") == ("rejected", "group")
        assert usage.decision(True, "A") == ("degraded", "group")
        usage.owner_call = lambda: True
        usage.reserve(1)
        usage.settle(1, CallRecord("answer", "m", "text", cost_cny=1))
        now[0] += 86400
        assert usage.decision(False, "A") == ("rejected", "group")  # outstanding reservation
        usage.settle(4, CallRecord("read", "m", "http", status="cancelled", cost_cny=0))
        assert usage.pending == 0 and not usage.group_pending
        assert usage.decision(False, "A") == ("normal", None)
    finally:
        CURRENT_GROUP.reset(token)


async def test_stop_one_group_during_billed_requests():
    from group_agent import received_groups

    h = Harness(agent_kw={"delays": {"a": 20, "b": 0.05}, "usage": True})
    h.feed(ev("a", group="A"))
    h.feed(ev("b", user="U2", group="B"))
    async with asyncio.timeout(3):
        while not h.store.con.execute("SELECT 1 FROM usage_ledger WHERE group_id='A'").fetchone():
            await asyncio.sleep(0.005)
    pending = h.intake.intake(ev("waiting", group="A")).request
    with h.store.con:
        received_groups.stop_boundary(h.store.con, "BOT", "A")
        assert h.orch.cancel_group("A") == 1
    assert h.intake.intake(ev("silent", group="A")).status == "trigger"
    assert h.store.lookup("BOT", "A", "silent")
    await h.settle()
    assert h.runs()["a"]["state"] == "cancelled"
    assert h.runs()["b"]["state"] == "succeeded"
    assert [row["group_id"] for row in h.outbound()] == ["B"]
    assert (
        h.store.con.execute("SELECT sum(cost_cny) FROM usage_ledger WHERE group_id='A'").fetchone()[
            0
        ]
        == 0.01
    )
    h.store.con.execute("UPDATE received_groups SET enabled=1 WHERE group_id='A'")
    assert h.orch.submit(pending) == "paused"
    h.store.close()


@pytest.mark.parametrize("source", ["anchor", "quote", "previous"])
async def test_forward_nodes_reach_answer_material(tmp_path, source):
    from group_agent.live_agent import LiveAgent
    from test_live_replay import live_config

    h = Harness()
    live = LiveAgent(h.orch, live_config(tmp_path, lambda: None, lambda: None))
    calls, materials = [], []

    async def fetch(key):
        calls.append(key)
        return [
            {
                "sender": {"nickname": "访客"},
                "message": [
                    {"type": "text", "data": {"text": "只有转发里有的原话"}},
                    {"type": "forward", "data": {"id": key}},
                ],
            }
        ]

    async def answer(trace, *_):
        materials.append(await live._material(trace, live._state(trace)))
        return "回答"

    h.orch.forward_fetch = fetch
    h.agent.answer = answer
    stamp = int(wall())
    first = ev("record", at=source == "anchor", text="", t=stamp)
    first["message"].append({"type": "forward", "data": {"id": "resource"}})
    h.feed(first)
    if source != "anchor":
        h.feed(
            ev(
                "ask",
                text="看看我刚发的内容",
                t=stamp,
                reply="record" if source == "quote" else None,
            )
        )
    await h.settle()
    assert calls == ["resource"]
    assert "只有转发里有的原话" in materials[0].text
    assert "转发部分内容未取得" in materials[0].text
    detail = json.loads(next(iter(h.runs().values()))["detail"])
    assert detail["forward"]["nodes_used"] == 1
    assert detail["forward"]["gaps"] == {"repeat_or_depth_limit": 1}


async def test_regeneration_hands_the_answer_back_without_an_outbound_row():
    """`generate_for` (scripts/replay_shadow.py): same run flow, answer returned, nothing
    recorded or sent, and a second regeneration of the same command seq is refused."""
    h = Harness()
    h.feed(ev("m1", at=False, t=int(wall())))
    seq = h.store.lookup("BOT", "G1", "m1")["seq"]
    from group_agent.store import _message

    m = _message(h.store.message_row(seq))
    run_id, text = await h.orch.generate_for(m, seq, run_seq=seq)
    assert text == "answer:m1" and run_id
    assert h.outbound() == []
    assert h.runs()["m1"]["purpose"] == "new_version"
    assert await h.orch.generate_for(m, seq, run_seq=seq) == (None, None)