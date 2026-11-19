"""Owner remote control (`#暂停` / `#恢复` / `#状态`, owner_control.py): recognition, persisted
pause, no runs while paused, cancellation of in-flight runs, confirmation rows. Fake channel,
fake clock and fake connection only; nothing real is sent."""

import asyncio
import inspect
import sqlite3

import pytest

from group_agent.admission import AccessPolicy, AccessSnapshot, RequestQueue
from group_agent.config import Settings
from group_agent.contracts import Message
from group_agent.intake import EventIntake
from group_agent.live_send import LiveSender
from group_agent.orchestrator import Orchestrator
from group_agent.owner_control import (
    PAUSE_TEXT,
    RESUME_TEXT,
    OwnerControl,
    PauseState,
    parse_control,
)
from group_agent.reaction import ProcessingReaction
from group_agent.serve import run_serve
from group_agent.store import SCHEMA_VERSION, Store
from group_agent.usage import UsageLedger

from test_live_send import FakeChannel
from test_orchestrator import BASE, FakeAgent, ev, seed_usage, virtual, wall
from test_serve import FakeConnection, message, settings_for

INFO = {"answer_model": "answer-x", "search_model": "search-y"}


def at_text(text, user="OWNER", extra=(), at=False):
    segments = [{"type": "at", "data": {"qq": "BOT"}}] if at else []
    segments.extend(extra)
    segments.append({"type": "text", "data": {"text": text}})
    return Message("BOT", "G1", "x", user, tuple(segments))


def cmd(mid, text="#暂停", user="OWNER", group="G1", at=False, t=None):
    return ev(mid, user=user, group=group, text=text, at=at, t=t)


class CH:
    """Intake + orchestrator + live sender + owner control, as `serve` wires them."""

    def __init__(
        self, store=None, live=True, channel=None, daily=0.0, delays=None, stale=300.0, cap=3.0
    ):
        self.store = store or Store()
        self.access = AccessPolicy(AccessSnapshot("BOT", "OWNER"))
        if daily:
            seed_usage(self.store, daily, "2027-01-15")
        self.usage = UsageLedger(self.store, daily_cap_cny=cap, clock=wall)
        self.state = PauseState(self.store, started=BASE)
        self.channel = channel or FakeChannel()
        self.sender = LiveSender(self.channel, self.store, self.access, clock=wall)
        self.sender.set_pause_gate(self.state.is_paused)
        self.queue = RequestQueue(64)
        self.admitted: list[str] = []
        self.orch = Orchestrator(
            self.store,
            self.access,
            Settings(bot_id="BOT", owner_id="OWNER"),
            agent_factory=lambda host: FakeAgent(host, delays=delays or {}),
            usage=self.usage,
            clock=wall,
            publisher=self.sender.deliver if live else None,
            paused=self.state.is_paused,
        )
        self.control = OwnerControl(
            self.store,
            self.access,
            self.state,
            cancel_replies=self.orch.cancel_replies,
            deliver=self.sender.deliver if live else None,
            rate=self.orch.rate,
            status_info=lambda: {
                "spent": self.usage.report()["daily_spent_cny"],
                "cap": cap,
                "cost_mode": self.usage.report()["cost_mode"],
                **INFO,
            },
            clock=wall,
            stale_seconds=stale,
        )
        self.intake = EventIntake(
            self.store,
            self.access,
            admit=True,
            sink=self.queue,
            clock=wall,
            on_admitted=lambda r: self.admitted.append(r.message.message_id),
            control=self.control.classify,
            paused=self.state.is_paused,
        )

    def feed(self, event):
        result = self.intake.intake(event)
        self.control.observe(result)
        while not self.queue.queue.empty():
            self.orch.submit(self.queue.queue.get_nowait())
        return result.status

    async def settle(self):
        while self.orch.tasks or self.control.tasks:
            pending = [*self.orch.tasks, *self.control.tasks]
            await asyncio.gather(*pending, return_exceptions=True)
        await self.sender.aclose()

    def runs(self):
        rows = self.store.con.execute("SELECT * FROM runs ORDER BY created_at, rowid").fetchall()
        return {r["message_id"]: dict(r) for r in rows}

    def outbound(self, kind=None):
        sql = "SELECT * FROM outbound" + (" WHERE kind=?" if kind else "") + " ORDER BY rowid"
        rows = self.store.con.execute(sql, (kind,) if kind else ()).fetchall()
        return [dict(r) for r in rows]


# -- recognition --------------------------------------------------------------------------


def test_parse_control_exact_text_only():
    assert parse_control(at_text("#暂停")) == "pause"
    assert parse_control(at_text(" #恢复 \n")) == "resume"
    assert parse_control(at_text("# 状态")) == "status"  # whitespace is stripped
    assert parse_control(at_text("#暂停", at=True)) == "pause"  # @bot is allowed
    assert parse_control(at_text("＃暂停")) is None  # full-width hash
    assert parse_control(at_text("#暂停吧")) is None  # extra text
    assert parse_control(at_text("请 #暂停")) is None
    assert parse_control(at_text("暂停")) is None
    assert parse_control(at_text("#新版")) is None
    other = {"type": "at", "data": {"qq": "U9"}}
    assert parse_control(at_text("#暂停", extra=[other])) is None
    everyone = {"type": "at", "data": {"qq": "all"}}
    assert parse_control(at_text("#暂停", extra=[everyone])) is None
    reply = {"type": "reply", "data": {"id": "m1"}}
    assert parse_control(at_text("#暂停", extra=[reply])) is None
    image = {"type": "image", "data": {"file": "a.png"}}
    assert parse_control(at_text("#暂停", extra=[image])) is None


def test_owner_command_is_flagged_and_never_a_request_or_context():
    async def main():
        h = CH()
        h.feed(cmd("c1", "#暂停", at=True))  # even with @bot: an instruction, not a question
        h.feed(cmd("q0", "你好", user="U1", at=False))
        await h.settle()
        return h

    h = virtual(main)
    row = h.store.lookup("BOT", "G1", "c1")
    assert h.store.message_row(row["seq"])["kind"] == "owner_command"
    assert h.store.con.execute("SELECT count(*) FROM reply_requests").fetchone()[0] == 0
    assert h.runs() == {}
    anchor = Message("BOT", "G1", "q9", "U1", ())
    seq, _ = h.store.ingest(anchor, int(BASE), BASE)
    assert [m.message_id for m in h.store.history(anchor, seq)] == ["q0"]  # not the command


def test_non_owner_same_text_is_an_ordinary_message_without_effect():
    async def main():
        h = CH()
        status = h.feed(cmd("c1", "#暂停", user="U1"))
        await h.settle()
        return h, status

    h, status = virtual(main)
    assert status == "stored" and not h.state.paused
    assert h.store.message_row(h.store.lookup("BOT", "G1", "c1")["seq"])["kind"] == "chat"
    assert h.store.control_events() == [] and h.outbound() == []
    assert h.control.counts["ignored_not_owner"] == 1


def test_command_works_in_any_group_and_with_collection_off():
    async def main():
        h = CH()
        h.access.update(collection_off_groups=frozenset({"G7"}))
        h.feed(cmd("c1", "#暂停", group="G7"))
        await h.settle()
        return h

    h = virtual(main)
    assert h.state.paused
    assert h.outbound("owner_control")[0]["group_id"] == "G7"


# -- persisted state ------------------------------------------------------------------------


def test_pause_is_persisted_across_store_reopen(tmp_path):
    path = tmp_path / "agent.sqlite3"
    store = Store(path)
    assert store.control_state() == (False, None)
    assert store.apply_control("BOT", "G1", "c1", "pause", 100.0) == "paused"
    assert store.apply_control("BOT", "G1", "c1", "pause", 101.0) is None  # same command
    assert store.apply_control("BOT", "G1", "c2", "pause", 102.0) == "already_paused"
    store.close()
    store = Store(path)
    assert store.control_state() == (True, 100.0)
    state = PauseState(store)
    assert state.paused and state.since == 100.0
    assert not PauseState(store, enabled=False).paused  # switched off: never paused
    assert store.apply_control("BOT", "G1", "c3", "resume", 200.0) == "resumed"
    assert store.apply_control("BOT", "G1", "c4", "resume", 201.0) == "already_running"
    assert store.apply_control("BOT", "G1", "c5", "status", 202.0) == "status"
    store.close()
    store = Store(path)
    assert store.control_state() == (False, 200.0)
    assert [(e["group_id"], e["action"], e["effect"]) for e in store.control_events()] == [
        ("G1", "pause", "paused"),
        ("G1", "pause", "already_paused"),
        ("G1", "resume", "resumed"),
        ("G1", "resume", "already_running"),
        ("G1", "status", "status"),
    ]
    assert store.summary()["control_events"] == 5
    store.close()


def test_v9_database_migrates_to_v10_and_keeps_outbound_rows(tmp_path):
    path = tmp_path / "v9.db"
    Store(path).close()
    con = sqlite3.connect(path)
    con.executescript(
        """
        DROP TABLE control_state; DROP TABLE control_events; DROP TABLE outbound;
        CREATE TABLE outbound (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
            bot_id TEXT NOT NULL, group_id TEXT NOT NULL, reply_to TEXT NOT NULL,
            requester TEXT NOT NULL, text TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending', 'recorded_not_sent', 'sending',
                'sent', 'uncertain', 'failed')),
            mode TEXT NOT NULL CHECK(mode IN ('offline', 'record_only', 'live_send')),
            kind TEXT NOT NULL DEFAULT 'answer' CHECK(kind IN ('answer', 'busy_notice',
                'new_version')),
            idempotency_key TEXT UNIQUE,
            error_kind TEXT, created_at REAL, scheduled_at REAL, recorded_at REAL,
            qq_message_id TEXT, sent_at REAL
        );
        INSERT INTO outbound(run_id,bot_id,group_id,reply_to,requester,text,state,mode,kind,
            idempotency_key,sent_at) VALUES ('r1','BOT','G1','m1','U1','答','sent','live_send',
            'answer','BOT:G1:m1:answer',5.0);
        PRAGMA user_version=9;
        """
    )
    con.commit()
    con.close()
    store = Store(path)
    assert store.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    row = store.outbound("r1")
    assert (row["text"], row["state"], row["sent_at"]) == ("答", "sent", 5.0)
    assert store.control_state() == (False, None)
    m = Message("BOT", "G1", "c1", "OWNER", ())
    _, fresh = store.outbound_enqueue("control:G1:c1", "owner_control", m, "x", 1.0)
    assert fresh  # no run row behind it
    store.close()


# -- paused behaviour -------------------------------------------------------------------------


def test_paused_creates_no_runs_or_reactions_for_anyone_and_collection_continues():
    async def main():
        h = CH(live=True)
        h.feed(cmd("c1", "#暂停"))
        h.feed(ev("q1", user="OWNER", text="主人问", at=True))
        h.feed(ev("q2", user="U2", text="路人问", at=True))
        h.feed(ev("chat1", user="U3", text="闲聊", at=False))
        await h.settle()
        return h

    h = virtual(main)
    assert h.runs() == {} and h.admitted == []  # no run, no "processing" reaction hook
    assert h.orch.agent.answered == []
    assert [o["kind"] for o in h.outbound()] == ["owner_control"]
    assert {m for m in ("q1", "q2", "chat1") if h.store.lookup("BOT", "G1", m)} == {
        "q1",
        "q2",
        "chat1",
    }
    assert h.intake.counts["paused_requests"] == 2
    assert [c[1][0]["data"]["id"] for c in h.channel.calls] == ["c1"]  # only the confirmation


def test_paused_reaction_hook_is_not_called_through_a_real_reaction_holder():
    class Reactions:
        def __init__(self):
            self.ids = []

        async def set_msg_emoji_like(self, message_id, *, timeout=5.0):
            self.ids.append(message_id)
            from group_agent.gateway import ReactionResult

            return ReactionResult("ok")

    async def main():
        h = CH(live=True)
        channel = Reactions()
        reaction = ProcessingReaction(channel, h.store, clock=wall)
        h.intake.on_admitted = reaction.admitted
        h.feed(ev("q0", user="U1", text="先问", at=True))  # running: reacts
        h.feed(cmd("c1", "#暂停"))
        h.feed(ev("q1", user="OWNER", text="暂停后", at=True))
        h.feed(cmd("c2", "#恢复"))
        h.feed(ev("q2", user="U1", text="恢复后", at=True))
        await h.settle()
        await reaction.aclose()
        return channel

    assert virtual(main).ids == ["q0", "q2"]


def test_queued_request_is_refused_by_the_orchestrator_while_paused():
    async def main():
        h = CH()
        h.feed(ev("q1", user="U1", at=True))
        request = h.queue  # (already submitted by feed; submit a second one by hand)
        h.state.paused = True
        from group_agent.admission import ReplyRequest

        m = Message("BOT", "G1", "q2", "U1", ())
        seq, _ = h.store.ingest(m)
        state = h.orch.submit(ReplyRequest(m, seq, None, None, False))
        h.state.paused = False
        await h.settle()
        return h, state, request

    h, state, _ = virtual(main)
    assert state == "paused" and "q2" not in h.runs()
    assert h.orch.counts["paused_skipped"] == 1


def test_in_flight_run_is_cancelled_without_a_send():
    async def main():
        h = CH(delays={"q1": 50})
        h.feed(ev("q1", user="U1", text="慢问题", at=True))
        await asyncio.sleep(5)
        h.feed(cmd("c1", "#暂停"))
        await h.settle()
        return h

    h = virtual(main)
    run = h.runs()["q1"]
    assert run["state"] == "cancelled" and run["error_kind"] == "paused"
    assert h.outbound("answer") == [] or all(o["state"] == "failed" for o in h.outbound("answer"))
    assert [c[1][0]["data"]["id"] for c in h.channel.calls] == ["c1"]
    assert h.control.counts["cancelled_runs"] == 1


def test_send_already_in_progress_finishes_and_a_pending_row_does_not_go_out():
    async def main():
        # the answer to q1 is being written when the pause arrives; q2's row is only pending
        channel = FakeChannel(delays={"q1": 6.0})
        h = CH(channel=channel, delays={"q1": 1, "q2": 20})
        h.feed(ev("q1", user="U1", text="一", at=True))
        h.feed(ev("q2", user="U2", text="二", at=True))
        await asyncio.sleep(3)  # q1: frame in flight (it takes 6 s); q2 still thinking
        h.feed(cmd("c1", "#暂停"))
        await h.settle()
        return h

    h = virtual(main)
    answers = {o["reply_to"]: o for o in h.outbound("answer")}
    assert answers["q1"]["state"] == "sent"  # let finish
    assert "q2" not in answers or answers["q2"]["state"] == "failed"
    assert h.runs()["q2"]["state"] == "cancelled"
    sent_ids = [c[1][0]["data"]["id"] for c in h.channel.calls]
    assert sorted(sent_ids) == ["c1", "q1"]


def test_live_sender_refuses_answer_rows_while_paused_but_not_the_confirmation():
    store = Store()
    access = AccessPolicy(AccessSnapshot("BOT", "OWNER"))
    channel = FakeChannel()
    sender = LiveSender(channel, store, access)
    flag = {"paused": True}
    sender.set_pause_gate(lambda: flag["paused"])
    q = Message("BOT", "G1", "q1", "U1", ())
    c = Message("BOT", "G1", "c1", "OWNER", ())
    store.ingest(q)
    store.ingest(c, kind="owner_command")
    store.outbound_enqueue("r1", "answer", q, "答", 1.0, mode="live_send")
    store.outbound_enqueue("control:G1:c1", "owner_control", c, RESUME_TEXT, 1.0, mode="live_send")
    assert sender.refusal(store.outbound("r1")) == "paused"
    assert sender.refusal(store.outbound("control:G1:c1")) is None
    # a control row answers the owner only
    store.outbound_enqueue("control:G1:q1", "owner_control", q, RESUME_TEXT, 1.0, mode="live_send")
    assert sender.refusal(store.outbound("control:G1:q1")) == "not_owner"
    flag["paused"] = False
    assert sender.refusal(store.outbound("r1")) is None


def test_resume_restores_replies():
    async def main():
        h = CH()
        h.feed(cmd("c1", "#暂停"))
        h.feed(ev("q1", user="U1", at=True))
        h.feed(cmd("c2", "#恢复"))
        h.feed(ev("q2", user="U1", at=True))
        await h.settle()
        return h

    h = virtual(main)
    assert not h.state.paused and set(h.runs()) == {"q2"}
    assert h.runs()["q2"]["state"] == "succeeded"
    assert h.store.control_state()[0] is False
    answers = [o for o in h.outbound("answer") if o["state"] == "sent"]
    assert [o["reply_to"] for o in answers] == ["q2"]


# -- confirmation ---------------------------------------------------------------------------


def test_confirmation_is_a_quoted_fixed_text_sent_once_per_command():
    async def main():
        h = CH()
        h.feed(cmd("c1", "#暂停"))
        h.feed(cmd("c1", "#暂停"))  # redelivered event
        h.feed(cmd("c2", "#恢复"))
        await h.settle()
        return h

    h = virtual(main)
    rows = h.outbound("owner_control")
    assert [(r["reply_to"], r["text"], r["state"], r["mode"]) for r in rows] == [
        ("c1", PAUSE_TEXT, "sent", "live_send"),
        ("c2", RESUME_TEXT, "sent", "live_send"),
    ]
    assert PAUSE_TEXT == "已暂停：新版不再回复，消息照常记录。发送 #恢复 继续。"
    assert RESUME_TEXT == "已恢复回复。"
    assert rows[0]["qq_message_id"] and rows[0]["idempotency_key"] == "BOT:G1:c1:owner_control"
    first = h.channel.calls[0][1]
    assert first == [
        {"type": "reply", "data": {"id": "c1"}},
        {"type": "text", "data": {"text": PAUSE_TEXT}},
    ]
    assert len(h.channel.calls) == 2 and len(h.store.control_events()) == 2
    # rate limits: same group, at least the group minimum interval apart
    assert h.channel.calls[1][2] - h.channel.calls[0][2] >= 2.0
    # a confirmation is not an answer: it is not stored as an answered_by link
    assert h.store.answer_map("BOT", "G1") == {}


def test_status_line_and_repeated_commands_answer_with_current_state():
    async def main():
        h = CH(daily=0.45)
        h.feed(cmd("s0", "#状态"))
        h.feed(cmd("c1", "#暂停"))
        h.feed(cmd("c2", "#暂停"))  # already paused: current state
        h.feed(cmd("c3", "#恢复"))
        h.feed(cmd("c4", "#恢复"))  # already running: current state
        await h.settle()
        return h

    h = virtual(main)
    texts = {r["reply_to"]: r["text"] for r in h.outbound("owner_control")}
    assert texts["s0"] == "状态：运行中｜今日 ¥0.45/¥3.00｜回答 answer-x｜搜索 search-y"
    assert texts["c1"] == PAUSE_TEXT and texts["c3"] == RESUME_TEXT
    assert texts["c2"].startswith("状态：已暂停｜") and "answer-x" in texts["c2"]
    assert texts["c4"].startswith("状态：运行中｜") and "\n" not in texts["c4"]
    assert all("（自 " not in text for text in texts.values())
    effects = [e["effect"] for e in h.store.control_events()]
    assert effects == ["status", "paused", "already_paused", "resumed", "already_running"]


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize(
    "daily, expected",
    [
        (14.399, ""),
        (14.4, "费用降级（不联网、缩短上下文）｜"),
        (17.999, "费用降级（不联网、缩短上下文）｜"),
        (18.0, "费用已达上限：仅响应主人，已降级（不联网、缩短上下文）｜"),
    ],
)
def test_status_shows_daily_cost_policy_even_when_paused(daily, expected, paused):
    async def main():
        h = CH(daily=daily, cap=18.0)
        if paused:
            h.feed(cmd("pause", "#暂停"))
        h.feed(cmd("status", "#状态"))
        await h.settle()
        return h

    h = virtual(main)
    text = next(r["text"] for r in h.outbound("owner_control") if r["reply_to"] == "status")
    state = "已暂停" if paused else "运行中"
    assert text == (
        f"状态：{state}｜{expected}今日 ¥{daily:.2f}/¥18.00｜回答 answer-x｜搜索 search-y"
    )


def test_stale_command_is_recorded_but_changes_nothing():
    async def main():
        h = CH()
        old = int(wall()) - 1000
        h.feed(cmd("c1", "#暂停", t=old))
        await h.settle()
        return h

    h = virtual(main)
    assert not h.state.paused and h.outbound() == []
    assert [e["effect"] for e in h.store.control_events()] == ["stale"]


def test_shadow_mode_applies_and_records_but_never_sends():
    async def main():
        h = CH(live=False)
        h.feed(cmd("c1", "#暂停"))
        h.feed(ev("q1", user="OWNER", at=True))
        await h.settle()
        return h

    h = virtual(main)
    assert h.state.paused and h.runs() == {}
    rows = h.outbound("owner_control")
    assert [(r["state"], r["mode"], r["text"]) for r in rows] == [
        ("recorded_not_sent", "record_only", PAUSE_TEXT)
    ]
    assert h.channel.calls == []


def test_owner_control_switched_off_ignores_commands_and_stored_pause():
    store = Store()
    store.apply_control("BOT", "G1", "c0", "pause", 1.0)
    state = PauseState(store, enabled=False)
    assert not state.paused
    access = AccessPolicy(AccessSnapshot("BOT", "OWNER"))
    intake = EventIntake(store, access, admit=True, sink=RequestQueue(4), paused=state.is_paused)
    assert intake.intake(cmd("c1", "#暂停")).status == "stored"  # no control hook: ordinary
    assert intake.intake(ev("q1", user="U1", at=True)).status == "trigger"
    assert intake.counts["paused_requests"] == 0


def test_source_boundary_owner_control_holds_no_send_capability():
    from group_agent import owner_control

    source = inspect.getsource(owner_control)
    for word in ("send_group_msg", "gateway", "Channel", "grant=", "set_msg_emoji_like"):
        assert word not in source.replace("holds no gateway and no channel", ""), word


# -- serve end to end --------------------------------------------------------------------------


def _serve(tmp_path, events, db):
    conn = FakeConnection(events)

    async def connect(*args, **kwargs):
        return conn

    async def scenario():
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(1.5, stop.set)
        return await run_serve(
            settings_for(tmp_path),
            db_path=db,
            media_root=tmp_path / "media",
            duration=None,
            dry_models=True,
            budget_cny=0.01,
            cleanup=False,
            knowledge_path=None,
            connect=connect,
            stop=stop,
            self_sync=False,
        )

    return asyncio.run(scenario()), conn


def test_serve_starts_paused_when_the_pause_was_persisted(tmp_path):
    db = tmp_path / "serve.sqlite3"
    store = Store(db)
    store.apply_control("BOT", "G1", "old", "pause", 1_700_000_000.0)
    store.close()
    summary, conn = _serve(tmp_path, [message("m1", "U1", "你好")], db)
    assert summary["owner_control"]["paused"] is True
    assert summary["session"]["runs"] == {} and summary["sent_to_qq"] == 0
    assert not any(f["action"].startswith("send") for f in conn.sent)
    store = Store(db)
    assert store.lookup("BOT", "G1", "m1") is not None  # still collected
    assert store.con.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    store.close()


def test_serve_pause_resume_sequence_and_summary(tmp_path):
    def owner(mid, text):
        return {
            **message(mid, "OWNER", text),
            "message": [{"type": "text", "data": {"text": text}}],
        }

    events = [
        owner("c1", "#暂停"),
        message("m1", "U1", "暂停中"),
        owner("c2", "#恢复"),
        message("m2", "U1", "恢复后"),
    ]
    db = tmp_path / "serve.sqlite3"
    summary, conn = _serve(tmp_path, events, db)
    assert summary["owner_control"]["paused"] is False
    assert summary["owner_control"]["events"] == 2
    assert summary["owner_control"]["counts"]["action_pause"] == 1
    assert summary["owner_control"]["counts"]["effect_resumed"] == 1
    assert not any(f["action"].startswith("send") for f in conn.sent)  # dry run: no live send
    store = Store(db)
    runs = {r["message_id"]: r["state"] for r in store.con.execute("SELECT * FROM runs")}
    assert runs == {"m2": "succeeded"}
    kinds = [r["kind"] for r in store.con.execute("SELECT kind FROM outbound ORDER BY rowid")]
    assert kinds.count("owner_control") == 2
    store.close()
