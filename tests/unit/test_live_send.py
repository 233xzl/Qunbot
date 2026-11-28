"""Cutover sending (T3.7 rest; RUN-07..09, RUN-12, D-10): `live_send.LiveSender` with a fake
channel or a fake SnowLuma connection. Nothing here touches the network or QQ."""

import asyncio
import json
import sqlite3

import pytest

from group_agent import gateway
from group_agent.admission import AccessPolicy, AccessSnapshot, RequestQueue
from group_agent.app import main
from group_agent.config import ConfigError, Settings, load_settings
from group_agent.contracts import Message
from group_agent.gateway import (
    ActionRefused,
    AnswerSendChannel,
    OneBotGateway,
    ReadOnlyActions,
    SendResult,
)
from group_agent.intake import EventIntake
from group_agent.live_send import LiveSender
from group_agent.orchestrator import Orchestrator
from group_agent.outbox import MAX_REPLY_CHARS, TRUNCATION_MARK, fit_message_text
from group_agent.policy import RuntimeLimits
from group_agent.serve import DRY_ANSWER, run_serve
from group_agent.store import Store
from group_agent.teaching import UNAUTHORIZED_TEXT
from group_agent.usage import UsageLedger, cn_day

from test_orchestrator import BASE, FakeAgent, ev, recall, seed_usage, virtual, wall
from test_serve import FakeConnection, message, settings_for


class FakeChannel:
    """Answer channel stand-in: records calls; `results` maps reply_to -> SendResult,
    `delays` reply_to -> seconds, `errors` reply_to -> exception."""

    def __init__(self, results=None, delays=None, errors=None):
        self.calls: list[tuple[str, list[dict], float]] = []
        self.results, self.delays, self.errors = results or {}, delays or {}, errors or {}
        self.counter = 7000

    async def send_group_msg(self, group_id, segments, *, timeout=15.0):
        rid = str(segments[0]["data"]["id"])
        self.calls.append((group_id, segments, asyncio.get_running_loop().time()))
        await asyncio.sleep(self.delays.get(rid, 0.1))
        if rid in self.errors:
            raise self.errors[rid]
        if rid in self.results:
            return self.results[rid]
        self.counter += 1
        return SendResult("sent", str(self.counter), None)


class LiveHarness:
    def __init__(self, store=None, channel=None, live=True, limits=None, daily=0.0, **agent_kw):
        self.store = store or Store()
        self.access = AccessPolicy(AccessSnapshot("BOT", "OWNER"))
        self.settings = Settings(bot_id="BOT", owner_id="OWNER")
        if daily:
            seed_usage(self.store, daily, cn_day(BASE))
        self.usage = UsageLedger(self.store, daily_cap_cny=3.0, clock=wall)
        self.queue = RequestQueue(64)
        self.intake = EventIntake(self.store, self.access, admit=True, sink=self.queue, clock=wall)
        self.channel = channel or FakeChannel()
        self.sender = LiveSender(self.channel, self.store, self.access, clock=wall)
        self.orch = Orchestrator(
            self.store,
            self.access,
            self.settings,
            agent_factory=lambda host: self._agent(host, agent_kw),
            usage=self.usage,
            limits=limits or RuntimeLimits(),
            clock=wall,
            publisher=self.sender.deliver if live else None,
        )

    def _agent(self, host, kw):
        self.agent = FakeAgent(host, **kw)
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
        await self.sender.aclose()

    def runs(self):
        rows = self.store.con.execute("SELECT * FROM runs ORDER BY created_at, rowid").fetchall()
        return {r["message_id"]: dict(r) for r in rows}

    def outbound(self):
        rows = self.store.con.execute("SELECT * FROM outbound ORDER BY reply_to").fetchall()
        return {r["reply_to"]: dict(r) for r in rows}


# -- the channel ------------------------------------------------------------------------


def test_answer_channel_is_one_shot_only_when_enabled():
    plain = OneBotGateway("ws://x", "t", allow_actions=True)
    with pytest.raises(ActionRefused):
        plain.answer_send_channel()
    receive_only = OneBotGateway("ws://x", "t", answer_send=True)  # actions off
    with pytest.raises(ActionRefused):
        receive_only.answer_send_channel()
    gw = OneBotGateway("ws://x", "t", allow_actions=True, answer_send=True)
    assert isinstance(gw.answer_send_channel(), AnswerSendChannel)
    with pytest.raises(ActionRefused):
        gw.answer_send_channel()  # nobody else can take a second one
    with pytest.raises(ActionRefused):
        AnswerSendChannel(gw, object())  # a forged grant
    with pytest.raises(ActionRefused):
        AnswerSendChannel(gw, gateway._REACTION_GRANT)  # another channel's grant
    with pytest.raises(ActionRefused):
        asyncio.run(gw.call("send_group_msg", {"group_id": 1, "message": []}))

    class Ws:
        async def send(self, frame):
            raise AssertionError("no frame may be written")

    with pytest.raises(ActionRefused):  # the answer grant does not unlock other actions
        asyncio.run(
            ReadOnlyActions(Ws()).send_action("delete_msg", {}, "e", grant=gateway._ANSWER_GRANT)
        )
    with pytest.raises(ActionRefused):
        asyncio.run(
            ReadOnlyActions(Ws()).send_action(
                "set_msg_emoji_like", {}, "e", grant=gateway._ANSWER_GRANT
            )
        )


def test_answer_channel_writes_one_send_group_msg_frame():
    class Ws:
        def __init__(self):
            self.frames, self.incoming = [], asyncio.Queue()

        def __aiter__(self):
            return self

        async def __anext__(self):
            item = await self.incoming.get()
            if item is None:
                raise StopAsyncIteration
            return item

        async def send(self, frame):
            request = json.loads(frame)
            self.frames.append(request)
            response = {"status": "ok", "retcode": 0, "data": {"message_id": 55}}
            self.incoming.put_nowait(json.dumps(response | {"echo": request["echo"]}))

        async def close(self):
            self.incoming.put_nowait(None)

    async def scenario():
        ws = Ws()

        async def connect(*a, **k):
            return ws

        gw = OneBotGateway("ws://x", "t", allow_actions=True, answer_send=True, connect=connect)
        channel = gw.answer_send_channel()
        stop = asyncio.Event()
        runner = asyncio.create_task(gw.run(stop))
        await gw.wait_connected(2)
        result = await channel.send_group_msg("10001", LiveSender.segments("77", "好"))
        stop.set()
        await runner
        return ws.frames, result

    frames, result = asyncio.run(scenario())
    assert [f["action"] for f in frames] == ["send_group_msg"]
    assert frames[0]["params"] == {
        "group_id": 10001,
        "message": [
            {"type": "reply", "data": {"id": 77}},
            {"type": "text", "data": {"text": "好"}},
        ],
    }
    assert (result.status, result.message_id) == ("sent", "55")


# -- sending through the run flow --------------------------------------------------------


def test_sent_answer_keeps_message_id_and_answer_mapping():
    async def scenario():
        h = LiveHarness()
        h.feed(ev("q1", user="U1", text="问一下"))
        h.feed(ev("q1", user="U1", text="问一下"))  # redelivered: still one run, one send
        await h.settle()
        return h

    h = virtual(scenario)
    assert len(h.channel.calls) == 1
    group, segments, _ = h.channel.calls[0]
    # The old bot's shape: quote the request, then the text; no mention.
    assert group == "G1" and segments == [
        {"type": "reply", "data": {"id": "q1"}},
        {"type": "text", "data": {"text": "answer:q1"}},
    ]
    out = h.outbound()["q1"]
    assert (out["state"], out["mode"], out["qq_message_id"]) == ("sent", "live_send", "7001")
    assert out["sent_at"] is not None and out["scheduled_at"] is not None
    assert h.runs()["q1"]["state"] == "succeeded"
    assert h.store.answer_map("BOT", "G1") == {"7001": "q1"}
    bot_row = h.store.lookup("BOT", "G1", "7001")
    assert bot_row["user_id"] == "BOT"
    marks = h.store.con.execute(
        "SELECT q.message_id FROM related_marks r JOIN messages q ON q.seq=r.anchor_seq"
        " WHERE r.relation='answered_by' AND r.seq=?",
        (bot_row["seq"],),
    ).fetchall()
    assert [m[0] for m in marks] == ["q1"]
    assert h.store.summary()["sent_to_qq"] == 1
    assert h.sender.counts["answer_sent"] == 1


def test_follow_up_quoting_the_sent_answer_finds_the_question():
    async def scenario():
        h = LiveHarness()
        h.feed(ev("q1", user="U1", text="原来的问题"))
        await h.settle()
        h.feed(ev("q2", user="U1", text="那后来呢", reply="7001"))
        await h.settle()
        return h

    h = virtual(scenario)
    snap = h.store.context_snapshot(h.runs()["q2"]["run_id"])
    assert "q1" in snap["message_ids"] and "7001" in snap["message_ids"]


def test_failed_and_uncertain_sends_are_recorded_and_never_resent():
    async def scenario():
        channel = FakeChannel(
            results={
                "q1": SendResult("failed", None, "failed_1200"),
                "q2": SendResult("uncertain", None, "timeout"),
            },
            errors={"q3": RuntimeError("boom")},
        )
        h = LiveHarness(channel=channel)
        for mid in ("q1", "q2", "q3"):
            h.feed(ev(mid, user=f"U{mid}"))
        await h.settle()
        again = [await h.sender.deliver(h.runs()[mid]["run_id"]) for mid in ("q1", "q2", "q3")]
        return h, again

    h, again = virtual(scenario)
    out, runs = h.outbound(), h.runs()
    assert (out["q1"]["state"], out["q1"]["error_kind"]) == ("failed", "failed_1200")
    assert (out["q2"]["state"], out["q2"]["error_kind"]) == ("uncertain", "timeout")
    assert (out["q3"]["state"], out["q3"]["error_kind"]) == ("uncertain", "error_RuntimeError")
    assert runs["q1"]["state"] == "failed" and runs["q1"]["error_kind"] == "send_failed_1200"
    assert runs["q2"]["state"] == "succeeded"  # the answer exists; delivery is uncertain
    assert all(o["qq_message_id"] is None for o in out.values())
    assert len(h.channel.calls) == 3  # nothing resent
    assert [r.error_kind for r in again] == ["already_sent"] * 3


def test_process_exit_mid_send_becomes_uncertain_and_is_not_resent(tmp_path):
    db = tmp_path / "live.sqlite3"

    async def first():
        h = LiveHarness(store=Store(db), channel=FakeChannel(delays={"q1": 1000}))
        h.feed(ev("q1"))
        for _ in range(100):  # until the frame is in flight
            await asyncio.sleep(0.5)
            if h.channel.calls:
                break
        state = h.outbound()["q1"]["state"]
        h.store.close()  # the process ends here: nothing is settled
        return state

    assert virtual(first) == "sending"

    async def second():
        h = LiveHarness(store=Store(db))
        h.feed(ev("q1"))  # redelivered after the restart
        await h.settle()
        return h

    h = virtual(second)
    out = h.outbound()["q1"]
    assert (out["state"], out["error_kind"]) == ("uncertain", "interrupted")
    assert h.channel.calls == [] and h.orch.interrupted == 1
    assert h.runs()["q1"]["state"] == "failed"


def test_deliver_is_idempotent_and_refuses_foreign_rows():
    async def scenario():
        store = Store()
        access = AccessPolicy(AccessSnapshot("BOT", "OWNER"))
        channel = FakeChannel()
        sender = LiveSender(channel, store, access)
        q = Message("BOT", "G1", "q1", "U1", ({"type": "text", "data": {"text": "x"}},))
        seq, _ = store.ingest(q, BASE, BASE)
        for run_id, kind, mode in (
            ("r1", "answer", "live_send"),
            ("r2", "new_version", "live_send"),
            ("r3", "busy_notice", "record_only"),
        ):
            m = Message("BOT", "G1", f"q-{run_id}", "U1", ())
            s = seq if run_id == "r1" else store.ingest(m, BASE, BASE)[0]
            target = q if run_id == "r1" else m
            store.create_run(run_id, s, target, state="running", cost_mode="normal", now=BASE)
            store.outbound_enqueue(run_id, kind, target, "好的", BASE, mode=mode)
        results = [
            await sender.deliver("r1"),
            await sender.deliver("r1"),  # the same row twice
            await sender.deliver("r2"),  # the owner's #新版 row is not this sender's
            await sender.deliver("r3"),  # record-only row
            await sender.deliver("nope"),
        ]
        return results, channel, store

    results, channel, store = asyncio.run(scenario())
    assert [r.status for r in results] == ["sent", "failed", "failed", "failed", "failed"]
    assert [r.error_kind for r in results[1:]] == [
        "already_sent",
        "not_live_row",
        "not_live_row",
        "no_outbound_row",
    ]
    assert len(channel.calls) == 1
    assert store.outbound("r2")["state"] == "pending" and store.outbound("r3")["state"] == "pending"


def test_rate_limits_space_real_sends_per_group_and_account():
    limits = RuntimeLimits(global_active=12, user_active=1, account_per_minute=4)

    async def scenario():
        h = LiveHarness(limits=limits)
        for n in range(3):
            h.feed(ev(f"a{n}", user=f"UA{n}", group="G1"))
            h.feed(ev(f"b{n}", user=f"UB{n}", group="G2"))
        await h.settle()
        return h

    h = virtual(scenario)
    calls = h.channel.calls
    assert len(calls) == 6 and {o["state"] for o in h.outbound().values()} == {"sent"}
    for group in ("G1", "G2"):
        times = [t for g, _, t in calls if g == group]
        assert all(b - a >= 2 - 1e-6 for a, b in zip(times, times[1:]))  # D-10 min interval
    times = sorted(t for _, _, t in calls)
    assert all(sum(1 for t in times if s <= t < s + 60) <= 4 for s in times)  # account cap
    for g, segments, _ in calls:  # each answer still goes to its own request
        rid = segments[0]["data"]["id"]
        assert segments[1]["data"]["text"] == f"answer:{rid}" and g == (
            "G1" if rid[0] == "a" else "G2"
        )


def test_prepublish_recheck_blocks_the_send():
    async def scenario():
        h = LiveHarness(delays={"q2": 5, "q3": 5})
        h.feed(ev("q1", user="U1"))
        h.feed(ev("q2", user="U2"))
        h.feed(ev("q3", user="U3"))
        await asyncio.sleep(1)
        h.feed(recall("q2", user="U2"))  # recalled while the run works
        h.orch.apply_access(blocked_users={"U3"})  # requester blocked mid-run
        await h.settle()
        return h

    h = virtual(scenario)
    sent = {segments[0]["data"]["id"] for _, segments, _ in h.channel.calls}
    assert sent == {"q1"}
    runs = h.runs()
    assert runs["q2"]["error_kind"] == "request_recalled"
    assert runs["q3"]["error_kind"] == "requester_blocked"


def test_recheck_after_the_rate_wait_and_cost_cap():
    async def scenario():
        h = LiveHarness(delays={"q1": 1, "q2": 1})
        h.feed(ev("q1", user="U1"))
        h.feed(ev("q2", user="U2"))  # same group: waits 2 s for the send slot
        await asyncio.sleep(1.05)  # both answered; q2 is in its rate wait
        h.feed(recall("q2", user="U2"))
        await h.settle()
        return h

    h = virtual(scenario)
    assert [s[0]["data"]["id"] for _, s, _ in h.channel.calls] == ["q1"]
    assert h.outbound()["q2"]["state"] == "failed"

    async def capped():
        h = LiveHarness(daily=3.5)  # >= 100 %: only the owner is answered (D-04)
        h.feed(ev("q1", user="U1"))
        h.feed(ev("q2", user="OWNER"))
        await h.settle()
        return h

    h = virtual(capped)
    assert [s[0]["data"]["id"] for _, s, _ in h.channel.calls] == ["q2"]
    assert h.runs()["q1"]["state"] == "rejected"


def test_sender_final_check_refuses_ids_and_account_numbers():
    async def scenario():
        store = Store()
        access = AccessPolicy(AccessSnapshot("BOT", "OWNER", blocked_groups=frozenset({"G9"})))
        channel = FakeChannel()
        sender = LiveSender(channel, store, access)
        texts = {
            "r1": "我的QQ号是 123456789",
            "r2": "找 12345678 就行",  # the requester's own id
            "r3": "ok",
            "r4": "",
        }
        out = {}
        for run_id, text in texts.items():
            group = "G9" if run_id == "r3" else "G1"
            m = Message("BOT", group, f"q{run_id}", "12345678", ())
            seq, _ = store.ingest(m, BASE, BASE)
            store.create_run(run_id, seq, m, state="running", cost_mode="normal", now=BASE)
            store.outbound_enqueue(run_id, "answer", m, text, BASE, mode="live_send")
            out[run_id] = (await sender.deliver(run_id)).error_kind
        return out, channel, store

    out, channel, store = asyncio.run(scenario())
    assert out == {
        "r1": "contains_account_number",
        "r2": "contains_id",
        "r3": "blocked",
        "r4": "bad_length",
    }
    assert channel.calls == []
    assert {store.outbound(r)["state"] for r in out} == {"failed"}


def test_long_answers_are_cut_to_one_message():
    long = ("这是一句很长的解释。" * 300).strip()
    text, cut = fit_message_text(long)
    assert cut and len(text) <= MAX_REPLY_CHARS and text.endswith("。" + TRUNCATION_MARK)
    assert fit_message_text("  短回答 \n") == ("短回答", False)
    no_stop = "啊" * 2000
    text, cut = fit_message_text(no_stop)
    assert cut and len(text) == MAX_REPLY_CHARS

    class LongAgent(FakeAgent):
        async def answer(self, trace, mocks, message, seq):
            return long

    async def scenario():
        h = LiveHarness()
        h.orch.agent = LongAgent(h.orch)
        h.feed(ev("q1"))
        await h.settle()
        return h

    h = virtual(scenario)
    ((_, segments, _),) = h.channel.calls
    sent = segments[1]["data"]["text"]
    assert len(sent) <= MAX_REPLY_CHARS and sent.endswith(TRUNCATION_MARK)
    assert h.outbound()["q1"]["text"] == sent and h.orch.counts["outbound_truncated"] == 1


def test_record_only_never_calls_the_channel():
    async def scenario():
        h = LiveHarness(live=False)
        h.feed(ev("q1"))
        await h.settle()
        return h

    h = virtual(scenario)
    out = h.outbound()["q1"]
    assert (out["state"], out["mode"]) == ("recorded_not_sent", "record_only")
    assert h.channel.calls == [] and h.store.summary()["sent_to_qq"] == 0


def test_teaching_refusal_and_busy_notice_are_sent_like_answers():
    limits = RuntimeLimits(global_active=1, user_active=1, global_queue=1, user_queue=1)

    async def scenario():
        h = LiveHarness(limits=limits, delays={"a1": 10})
        h.feed(ev("a1", user="UA"))
        h.feed(ev("t1", user="UB", text="记住 本群：「开黑」随时"))  # queued
        h.feed(ev("c1", user="UC"))  # queue full: busy notice
        await h.settle()
        return h

    h = virtual(scenario)
    texts = {s[0]["data"]["id"]: s[1]["data"]["text"] for _, s, _ in h.channel.calls}
    assert texts["t1"] == UNAUTHORIZED_TEXT
    assert texts["a1"] == "answer:a1" and "c1" in texts
    out = h.outbound()
    assert out["c1"]["kind"] == "busy_notice" and out["c1"]["state"] == "sent"


def test_a_send_in_flight_survives_the_end_of_its_run():
    async def scenario():
        h = LiveHarness(channel=FakeChannel(delays={"q1": 3}))
        h.feed(ev("q1"))
        for _ in range(100):
            await asyncio.sleep(0.25)
            if h.channel.calls:
                break
        h.orch.cancel_run(h.runs()["q1"]["run_id"], "shutdown")
        await h.settle()
        return h

    h = virtual(scenario)
    out = h.outbound()["q1"]
    assert out["state"] == "sent" and out["qq_message_id"] == "7001"
    assert h.runs()["q1"]["state"] == "cancelled"


# -- serve wiring, configuration and command line ----------------------------------------


def _serve_live(tmp_path, events, **kw):
    conn = FakeConnection(events)
    db = tmp_path / "serve.sqlite3"

    async def connect(*args, **kwargs):
        return conn

    async def scenario():
        stop = asyncio.Event()

        async def watcher():
            for _ in range(200):
                await asyncio.sleep(0.05)
                if not db.exists():
                    continue
                probe = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
                try:
                    done = probe.execute(
                        "SELECT count(*) FROM outbound WHERE state IN ('sent','recorded_not_sent')"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    done = 0
                finally:
                    probe.close()
                if done >= 1:
                    await asyncio.sleep(0.3)
                    break
            stop.set()

        watch = asyncio.create_task(watcher())
        summary = await run_serve(
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
            **kw,
        )
        await watch
        return summary

    return asyncio.run(scenario()), conn, db


def test_serve_live_send_sends_answers_and_ignores_the_retired_new_version_command(tmp_path):
    question = message("m1", "U1", "你好")
    command = {
        **message("c1", "OWNER", "#新版"),
        "message": [
            {"type": "reply", "data": {"id": "m1"}},
            {"type": "text", "data": {"text": "#新版"}},
        ],
    }
    summary, conn, db = _serve_live(tmp_path, [question, command], live_send=True)
    sends = [f for f in conn.sent if f["action"] == "send_group_msg"]
    assert [f["params"]["message"] for f in sends] == [
        [{"type": "reply", "data": {"id": "m1"}}, {"type": "text", "data": {"text": DRY_ANSWER}}]
    ]
    assert summary["mode"] == "serve_live_send" and summary["live_send"] is True
    assert "owner_command_send" not in summary
    assert summary["live_sends"] == {"answer_sent": 1} and summary["sent_to_qq"] == 1
    store = Store(db)
    assert store.answer_map("BOT", "G1") == {"9001": "m1"}
    store.close()


def test_serve_live_send_off_by_default(tmp_path):
    summary, conn, _ = _serve_live(tmp_path, [message("m1", "U1", "你好")])
    assert summary["mode"] == "serve_record_only" and summary["live_sends"] is None
    assert not any(f["action"].startswith("send") for f in conn.sent)


def test_serve_stops_when_the_stop_file_appears(tmp_path):
    stop_file = tmp_path / "stop.flag"
    stop_file.write_text("stop", "utf-8")
    conn = FakeConnection([])

    async def connect(*args, **kwargs):
        return conn

    async def scenario():
        return await asyncio.wait_for(
            run_serve(
                settings_for(tmp_path),
                db_path=tmp_path / "s.sqlite3",
                media_root=tmp_path / "media",
                duration=None,
                dry_models=True,
                budget_cny=0.01,
                cleanup=False,
                knowledge_path=None,
                connect=connect,
                stop_file=stop_file,
            ),
            10,
        )

    summary = asyncio.run(scenario())
    assert summary["sent_to_qq"] == 0


def test_settings_and_cli_switches(tmp_path, capsys):
    path = tmp_path / "settings.toml"
    path.write_text("[serve]\nlive_send = true\n", "utf-8")
    assert load_settings(path).live_send is True
    assert Settings().live_send is False
    path.write_text("[serve]\nlive_send = 'yes'\n", "utf-8")
    with pytest.raises(ConfigError):
        load_settings(path)
    for argv, error in (
        (["--shadow", "--live-send"], "--live-send requires --serve"),
        (["--serve", "--live-send", "--no-live-send"], "exclude each other"),
        (["--serve", "--live-send", "--dry-models"], "cannot be combined with --dry-models"),
        (["--serve", "--reaction", "--dry-models"], "cannot be combined with --dry-models"),
        (["--cleanup", "--stop-file", "x"], "--stop-file requires --serve"),
    ):
        with pytest.raises(SystemExit):
            main(argv)
        assert error in capsys.readouterr().err
