"""D-19 owner `#新版` command with fake channel / fake gateway connection; nothing real is sent."""

import asyncio
import json

import pytest

from group_agent.admission import AccessPolicy, AccessSnapshot, RequestQueue
from group_agent.config import Settings
from group_agent.context_strategies import explain_c
from group_agent.gateway import (
    ActionRefused,
    OneBotGateway,
    OwnerSendChannel,
    ReadOnlyActions,
    SendResult,
)
from group_agent.intake import EventIntake
from group_agent.models import CallRecord, Usage
from group_agent.orchestrator import Orchestrator
from group_agent.owner_command import (
    PREFIX,
    NewVersionCommand,
    OwnerCommandSender,
    final_text_check,
    parse_command,
)
from group_agent.contracts import Message
from group_agent.replay.vclock import run_virtual
from group_agent.store import Store
from group_agent.usage import UsageLedger, cn_day

BASE = 1_800_000_000


def wall() -> float:
    return BASE + asyncio.get_running_loop().time()


def ev(mid, user="U1", text="问一下", at=True, t=None, reply=None, group="G1", extra=()):
    segments = []
    if reply:
        segments.append({"type": "reply", "data": {"id": reply}})
    if at:
        segments.append({"type": "at", "data": {"qq": "BOT"}})
    segments.extend(extra)
    segments.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": group,
        "message_id": mid,
        "user_id": user,
        "time": int(t if t is not None else BASE),
        "message": segments,
    }


def command(mid, quoted, user="OWNER", text="#新版", t=None, **kw):
    return ev(mid, user=user, text=text, at=False, t=t, reply=quoted, **kw)


class FakeAgent:
    def __init__(self, host, usage=None, answers=None):
        self.host, self.usage, self.answers = host, usage, answers or {}
        self.calls: list[tuple[str, list[str], str]] = []
        self.runs = {}

    async def vision(self, trace, mocks):
        return True

    async def decide_route(self, trace, mocks):
        return "chat"

    async def run_tools(self, trace, m, budget):
        pass

    async def answer(self, trace, mocks, message, seq):
        if self.usage is not None:
            record = CallRecord("answer", "gemini", "text", usage=Usage(10, 0, 5, None, 1))
            record.cost_cny = 0.01
            self.usage.reserve(0.01)
            self.usage.settle(0.01, record)
        await asyncio.sleep(1.0)
        self.calls.append((message.message_id, list(trace.context_ids), trace.run_id))
        return self.answers.get(message.message_id, f"answer:{message.message_id}")


class FakeChannel:
    def __init__(self, results=None):
        self.calls: list[tuple[str, list[dict]]] = []
        self.results = list(results or [])

    async def send_group_msg(self, group_id, segments, *, timeout=15.0):
        self.calls.append((group_id, segments))
        await asyncio.sleep(0.1)
        if self.results:
            return self.results.pop(0)
        return SendResult("sent", f"nv{len(self.calls)}", None)


class H:
    def __init__(self, send=True, daily=0.0, billed=False, channel=None, answers=None):
        self.store = Store()
        self.access = AccessPolicy(AccessSnapshot("BOT", "OWNER"))
        self._billed, self._answers = billed, answers
        if daily:
            self.store.record_usage(
                {
                    "run_id": None,
                    "day": cn_day(BASE),
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
                    "cost_cny": daily,
                    "latency_s": 0.0,
                }
            )
        self.usage = UsageLedger(self.store, daily_cap_cny=3.0, clock=wall)
        self.queue = RequestQueue(64)
        self.intake = EventIntake(self.store, self.access, admit=True, sink=self.queue, clock=wall)
        self.orch = Orchestrator(
            self.store,
            self.access,
            Settings(bot_id="BOT", owner_id="OWNER"),
            agent_factory=self._agent,
            usage=self.usage,
            clock=wall,
        )
        self.channel = (channel or FakeChannel()) if send else None
        self.cmd = NewVersionCommand(
            self.store,
            self.access,
            self.orch,
            channel=self.channel,
            clock=wall,
            quote_wait_seconds=1.0,
        )

    def _agent(self, host):
        self.agent = FakeAgent(
            host, usage=self.usage if self._billed else None, answers=self._answers
        )
        return self.agent

    def feed(self, event):
        result = self.intake.intake(event)
        while not self.queue.queue.empty():
            self.orch.submit(self.queue.queue.get_nowait())
        return self.cmd.observe(result)

    async def settle(self):
        while self.orch.tasks or self.cmd.tasks:
            await asyncio.gather(*list(self.orch.tasks), *list(self.cmd.tasks))

    def new_version_rows(self):
        return [
            dict(r)
            for r in self.store.con.execute("SELECT * FROM outbound WHERE kind='new_version'")
        ]

    def nv_runs(self):
        return [
            dict(r)
            for r in self.store.con.execute("SELECT * FROM runs WHERE purpose='new_version'")
        ]


def virtual(fn):
    return run_virtual(fn())


# -- parsing ------------------------------------------------------------------------


def msg(event):
    from group_agent.contracts import parse_group_message

    return parse_group_message(event, "BOT")


def test_parse_command_shape():
    assert parse_command(msg(command("c", "q"))) == "q"
    assert parse_command(msg(command("c", "q", text="  #新版 \n"))) == "q"
    assert parse_command(msg(command("c", "q", text="# 新 版"))) == "q"
    # A client-added mention of the quoted author is allowed; the bot or @all is not.
    other = ({"type": "at", "data": {"qq": "U1"}},)
    assert parse_command(msg(command("c", "q", extra=other))) == "q"
    for qq in ("BOT", "all"):
        bad = ({"type": "at", "data": {"qq": qq}},)
        assert parse_command(msg(command("c", "q", extra=bad))) is None
    assert parse_command(msg(ev("c", user="OWNER", text="#新版", at=False))) is None  # no quote
    assert parse_command(msg(command("c", "q", text="#新版 谢谢"))) is None
    assert parse_command(msg(command("c", "q", text="＃新版"))) is None
    image = ({"type": "image", "data": {"file": "x"}},)
    assert parse_command(msg(command("c", "q", extra=image))) is None


def test_final_text_check():
    ids = {"BOT", "123456789", "G1"}
    assert final_text_check(PREFIX + "今天天气不错，2026 年见。", ids) is None
    assert final_text_check(PREFIX + "机器人是 123456789", ids) == "contains_id"
    assert final_text_check(PREFIX + "他的QQ号是 98765432", ids) == "contains_account_number"
    assert final_text_check(PREFIX + "群号：87654321", ids) == "contains_account_number"
    assert final_text_check(PREFIX + "key sk-" + "a" * 24, ids) == "contains_secret_like"
    assert final_text_check(PREFIX + "api_key=abc", ids) == "contains_secret_like"
    assert final_text_check(PREFIX + "见 https://example.org/a/b/c/d/e/f/g/h/i/j/k/l", ids) is None


# -- command flow ----------------------------------------------------------------------


def test_owner_command_reuses_recorded_answer_and_sends_once():
    async def main():
        h = H()
        h.feed(ev("q1", t=BASE))
        await h.settle()
        assert h.feed(command("c1", "q1", t=BASE + 60)) == "scheduled"
        await h.settle()
        return h

    h = virtual(main)
    assert h.channel.calls == [
        (
            "G1",
            [
                {"type": "reply", "data": {"id": "q1"}},
                {"type": "text", "data": {"text": PREFIX + "answer:q1"}},
            ],
        )
    ]
    assert [c[0] for c in h.agent.calls] == ["q1"]  # reused, not regenerated
    (row,) = h.new_version_rows()
    assert (row["state"], row["mode"], row["qq_message_id"], row["reply_to"]) == (
        "sent",
        "live_send",
        "nv1",
        "q1",
    )
    assert row["sent_at"] is not None
    (run,) = h.nv_runs()
    assert run["message_id"] == "q1" and run["state"] == "succeeded"
    assert json.loads(run["detail"])["new_version"]["source"] == "reuse"
    assert h.cmd.counts["sent"] == 1 and h.cmd.counts["source_reuse"] == 1
    assert h.store.summary()["sent_to_qq"] == 1
    # The answer outbound of the reply run is untouched (record only).
    states = {r["kind"]: r["state"] for r in h.store.con.execute("SELECT kind,state FROM outbound")}
    assert states == {"answer": "recorded_not_sent", "new_version": "sent"}


def test_command_right_after_question_waits_for_the_reply_run():
    async def main():
        h = H()
        h.feed(ev("q1", t=BASE))
        h.feed(command("c1", "q1", t=BASE + 1))  # the reply run is still running
        await h.settle()
        return h

    h = virtual(main)
    assert len(h.channel.calls) == 1 and [c[0] for c in h.agent.calls] == ["q1"]
    assert h.cmd.counts["source_reuse"] == 1


def test_non_owner_command_is_ignored_without_notice():
    async def main():
        h = H()
        h.feed(ev("q1", t=BASE))
        await h.settle()
        assert h.feed(command("c1", "q1", user="U2", t=BASE + 5)) == "ignored_not_owner"
        await h.settle()
        return h

    h = virtual(main)
    assert h.channel.calls == [] and h.new_version_rows() == [] and h.nv_runs() == []
    assert h.cmd.counts["ignored_not_owner"] == 1


def test_command_mentioning_the_bot_or_without_quote_does_not_trigger():
    async def main():
        h = H()
        h.feed(ev("q1", t=BASE))
        await h.settle()
        at_bot = ev("c1", user="OWNER", text="#新版", at=True, reply="q1", t=BASE + 5)
        assert h.feed(at_bot) is None  # an @bot request (AstrBot answers it too), not a command
        assert h.feed(ev("c2", user="OWNER", text="#新版", at=False, t=BASE + 6)) is None
        await h.settle()
        return h

    h = virtual(main)
    assert h.channel.calls == [] and h.new_version_rows() == []


def test_repeated_command_sends_only_once():
    async def main():
        h = H()
        h.feed(ev("q1", t=BASE))
        await h.settle()
        h.feed(command("c1", "q1", t=BASE + 10))
        h.feed(command("c2", "q1", t=BASE + 11))  # concurrent duplicate
        await h.settle()
        h.feed(command("c3", "q1", t=BASE + 20))  # later duplicate
        await h.settle()
        return h

    h = virtual(main)
    assert len(h.channel.calls) == 1 and len(h.new_version_rows()) == 1
    assert h.cmd.counts["duplicate"] == 2


class FallbackAgent(FakeAgent):
    """Every answer attempt failed: the live agent's fixed fallback reply (D-21)."""

    async def answer(self, trace, mocks, message, seq):
        await asyncio.sleep(1.0)
        self.calls.append((message.message_id, list(trace.context_ids), trace.run_id))
        trace.degraded.append("fallback_reply")
        trace.mark(self.host.now(), "fallback_reply", "fallback_reply")
        return "兜底回复"


def test_fallback_reply_is_recorded_but_never_reused_or_sent_as_new_version():
    class FallbackH(H):
        def _agent(self, host):
            self.agent = FallbackAgent(host)
            return self.agent

    async def main():
        h = FallbackH()
        h.feed(ev("q1", t=BASE))
        await h.settle()
        assert h.feed(command("c1", "q1", t=BASE + 60)) == "scheduled"
        await h.settle()
        return h

    h = virtual(main)
    reply = h.store.con.execute(
        "SELECT r.state, r.error_kind, o.state, o.text FROM runs r JOIN outbound o"
        " ON o.run_id=r.run_id WHERE r.purpose='reply'"
    ).fetchone()
    assert tuple(reply) == ("succeeded", "fallback_reply", "recorded_not_sent", "兜底回复")
    # The command regenerated instead of reusing the fallback, and sent nothing.
    assert [c[0] for c in h.agent.calls] == ["q1", "q1"]
    assert h.channel.calls == [] and h.new_version_rows() == []
    (run,) = h.nv_runs()
    assert run["state"] == "failed" and run["error_kind"] == "fallback_reply"
    assert h.cmd.counts["source_regenerate"] == 0


def test_after_30_minutes_regenerates_with_context_cut_at_the_question():
    async def main():
        h = H(billed=True)
        h.feed(ev("m0", text="我在看一部电影", at=False, t=BASE))
        h.feed(ev("q1", text="这部怎么样", t=BASE + 5))
        await h.settle()
        h.feed(ev("m_after", text="后来又说了别的", at=False, t=BASE + 20))
        await asyncio.sleep(1900)
        h.feed(command("c1", "q1", t=int(wall())))
        await h.settle()
        return h

    h = virtual(main)
    assert [c[0] for c in h.agent.calls] == ["q1", "q1"]
    _, context, run_id = h.agent.calls[1]
    assert "m0" in context and "m_after" not in context and "c1" not in context
    (run,) = h.nv_runs()
    assert run["run_id"] == run_id and run["state"] == "succeeded"
    assert h.store.context_snapshot(run_id) is not None
    billed = h.store.con.execute(
        "SELECT count(*) FROM usage_ledger WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    assert billed == 1
    assert h.cmd.counts["source_regenerate"] == 1 and h.cmd.counts["sent"] == 1
    assert h.new_version_rows()[0]["run_id"] == run_id


def test_ordinary_quoted_message_is_answered_once_at_full_cost_cap():
    async def main():
        h = H(daily=3.0)  # >= 100 %: non-owners get nothing, the owner's command still runs
        h.feed(ev("q1", t=BASE))
        h.feed(ev("m1", text="有人知道这个吗", at=False, t=BASE + 2))
        await h.settle()
        h.feed(command("c1", "q1", t=BASE + 30))
        h.feed(command("c2", "m1", t=BASE + 31))
        await h.settle()
        return h

    h = virtual(main)
    runs = {
        r["purpose"] + ":" + r["message_id"]: r["state"]
        for r in h.store.con.execute("SELECT * FROM runs")
    }
    assert runs["reply:q1"] == "rejected"
    assert runs["new_version:q1"] == "succeeded" and runs["new_version:m1"] == "succeeded"
    assert sorted(r["reply_to"] for r in h.new_version_rows()) == ["m1", "q1"]
    assert {r["state"] for r in h.new_version_rows()} == {"sent"}


def test_switch_off_records_only():
    async def main():
        h = H(send=False)
        h.feed(ev("q1", t=BASE))
        await h.settle()
        assert h.feed(command("c1", "q1", t=BASE + 5)) == "scheduled"
        await h.settle()
        return h

    h = virtual(main)
    assert h.new_version_rows() == [] and h.cmd.counts["send_disabled"] == 1
    assert not h.cmd.send_enabled and h.store.summary()["sent_to_qq"] == 0


def test_uncertain_send_is_not_retried():
    async def main():
        h = H(channel=FakeChannel([SendResult("uncertain", None, "timeout")]))
        h.feed(ev("q1", t=BASE))
        await h.settle()
        h.feed(command("c1", "q1", t=BASE + 5))
        await h.settle()
        h.feed(command("c2", "q1", t=BASE + 50))
        await h.settle()
        return h

    h = virtual(main)
    assert len(h.channel.calls) == 1
    (row,) = h.new_version_rows()
    assert (row["state"], row["error_kind"], row["qq_message_id"]) == ("uncertain", "timeout", None)
    assert h.cmd.counts["uncertain"] == 1 and h.cmd.counts["duplicate"] == 1


def test_recheck_refuses_recalled_question_and_account_numbers():
    async def main():
        h = H(answers={"q2": "他的QQ号是 98765432"})
        h.feed(ev("q1", t=BASE))
        h.feed(ev("q2", user="U2", t=BASE + 1))
        await h.settle()
        h.intake.intake(
            {
                "post_type": "notice",
                "notice_type": "group_recall",
                "self_id": "BOT",
                "group_id": "G1",
                "message_id": "q1",
                "user_id": "U1",
            }
        )
        h.feed(command("c1", "q1", t=BASE + 5))
        h.feed(command("c2", "q2", t=BASE + 6))
        await h.settle()
        return h

    h = virtual(main)
    assert h.channel.calls == []
    assert h.cmd.counts["question_recalled"] == 1
    (row,) = h.new_version_rows()
    assert (row["reply_to"], row["state"], row["error_kind"]) == (
        "q2",
        "failed",
        "contains_account_number",
    )


def test_sent_message_maps_back_to_the_question_for_followups():
    async def main():
        h = H()
        h.feed(ev("q1", text="推荐一部电影", t=BASE))
        await h.settle()
        h.feed(command("c1", "q1", t=BASE + 30))
        await h.settle()
        h.feed(ev("f1", user="U2", text="为什么推荐这部", reply="nv1", t=BASE + 60))
        await h.settle()
        return h

    h = virtual(main)
    assert h.store.answer_map("BOT", "G1") == {"nv1": "q1"}
    stored = h.store.lookup("BOT", "G1", "nv1")
    assert stored is not None and stored["user_id"] == "BOT"
    follow = [c for c in h.agent.calls if c[0] == "f1"][0]
    assert "q1" in follow[1] and "nv1" in follow[1]
    reasons = h.store.context_snapshot(follow[2])["reasons"]
    assert reasons["q1"].startswith("answers/") and reasons["nv1"].startswith("reply_to/")


def test_answers_map_lets_strategy_c_find_the_question_without_a_quote_segment():
    def event(mid, user, segs, t):
        return {"post_type": "message", "message_id": mid, "user_id": user, "self_id": "BOT",
                "group_id": "G1", "time": t, "message": segs}  # fmt: skip

    text = [{"type": "text", "data": {"text": "x"}}]
    events = [
        event("q1", "U1", text, 1),
        event("nv1", "BOT", text, 2),  # no reply segment: only the answers map links it
        event("f1", "U2", [{"type": "reply", "data": {"id": "nv1"}}] + text, 3),
    ]
    picks = {p.message_id: p.relation for p in explain_c(events, "f1", answers={"nv1": "q1"})}
    assert picks == {"nv1": "reply_to", "q1": "answers"}


def test_unquoted_old_bot_reply_is_paired_with_its_request():
    store = Store()
    q = Message("BOT", "G1", "q1", "U1", ({"type": "at", "data": {"qq": "BOT"}},))
    store.ingest_request(q, 100, 100.0)
    q2 = Message("BOT", "G1", "q2", "U2", ({"type": "at", "data": {"qq": "BOT"}},))
    store.ingest_request(q2, 110, 110.0)
    reply = Message("BOT", "G1", "b1", "BOT", ({"type": "text", "data": {"text": "a"}},))
    seq, _ = store.ingest(reply, 130, 200.0, source="backfill")
    assert store.link_unquoted_bot_reply(seq, reply, 130) == store.lookup("BOT", "G1", "q1")["seq"]
    reply2 = Message("BOT", "G1", "b2", "BOT", ({"type": "text", "data": {"text": "b"}},))
    seq2, _ = store.ingest(reply2, 140, 201.0, source="backfill")
    store.link_unquoted_bot_reply(seq2, reply2, 140)
    assert store.answer_map("BOT", "G1") == {"b1": "q1", "b2": "q2"}
    late = Message("BOT", "G1", "b3", "BOT", ({"type": "text", "data": {"text": "c"}},))
    seq3, _ = store.ingest(late, 1000, 202.0, source="backfill")
    assert store.link_unquoted_bot_reply(seq3, late, 1000) is None  # outside the window


# -- the send capability ---------------------------------------------------------------


class Ws:
    def __init__(self, respond=True):
        self.frames: list[dict] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.respond = respond

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.incoming.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def send(self, frame):
        request = json.loads(frame)
        self.frames.append(request)
        if self.respond:
            data = {"message_id": 4242}
            reply = {"status": "ok", "retcode": 0, "data": data, "echo": request["echo"]}
            self.incoming.put_nowait(json.dumps(reply))

    async def close(self):
        self.incoming.put_nowait(None)


def test_send_channel_is_one_shot_and_only_when_enabled():
    plain = OneBotGateway("ws://127.0.0.1:1", "t", allow_actions=True)
    with pytest.raises(ActionRefused):
        plain.owner_send_channel()
    gw = OneBotGateway("ws://127.0.0.1:1", "t", allow_actions=True, owner_send=True)
    assert isinstance(gw.owner_send_channel(), OwnerSendChannel)
    with pytest.raises(ActionRefused):
        gw.owner_send_channel()  # nobody else can take a second one
    receive_only = OneBotGateway("ws://127.0.0.1:1", "t", allow_actions=False, owner_send=True)
    with pytest.raises(ActionRefused):
        receive_only.owner_send_channel()
    with pytest.raises(ActionRefused):
        OwnerSendChannel(gw, object())  # a forged grant
    with pytest.raises(ActionRefused):
        asyncio.run(gw.call("send_group_msg", {"group_id": 1, "message": []}))

    async def frame():
        await ReadOnlyActions(Ws()).send_action("send_group_msg", {}, "e1")

    with pytest.raises(ActionRefused):
        asyncio.run(frame())
    with pytest.raises(ActionRefused):
        asyncio.run(ReadOnlyActions(Ws()).send_action("send_group_msg", {}, "e1", grant=object()))


def test_owner_sender_validates_before_using_the_channel():
    async def main():
        store = Store()
        q = Message("BOT", "G1", "123", "U1", ())
        seq, _ = store.ingest(q, BASE, BASE)
        store.create_run("r1", seq, q, state="succeeded", cost_mode="normal", now=BASE)
        channel = FakeChannel()
        sender = OwnerCommandSender(channel, store)
        no_row = await sender.send("r1", "G1", "G1", "123", PREFIX + "x")
        store.outbound_enqueue("r1", "new_version", q, PREFIX + "x", BASE, mode="live_send")
        results = [
            await sender.send("r1", "G1", "G2", "123", PREFIX + "x"),  # other group
            await sender.send("r1", "G1", "G1", "123", "x"),  # no prefix
            await sender.send("r1", "G1", "G1", "123", PREFIX + "y"),  # not the recorded text
            await sender.send("r1", "G1", "G1", "123", PREFIX + "x" * 4000),
            await sender.send("r1", "G1", "G1", "123", PREFIX + "x"),  # ok
            await sender.send("r1", "G1", "G1", "123", PREFIX + "x"),  # claimed already
        ]
        return no_row, results, channel

    no_row, results, channel = asyncio.run(main())
    assert no_row.error_kind == "no_outbound_row"
    assert [r.error_kind for r in results] == [
        "group_mismatch",
        "bad_prefix",
        "no_outbound_row",
        "bad_length",
        None,
        "already_sent",
    ]
    assert channel.calls == [
        (
            "G1",
            [
                {"type": "reply", "data": {"id": 123}},
                {"type": "text", "data": {"text": PREFIX + "x"}},
            ],
        )
    ]


def _gateway_send(respond: bool):
    async def main():
        ws = Ws(respond)

        async def connect(*a, **k):
            return ws

        gw = OneBotGateway("ws://x", "t", allow_actions=True, owner_send=True, connect=connect)
        channel = gw.owner_send_channel()
        stop = asyncio.Event()
        runner = asyncio.create_task(gw.run(stop))
        await gw.wait_connected(2)
        result = await channel.send_group_msg(
            "10001", [{"type": "text", "data": {"text": PREFIX + "x"}}], timeout=0.3
        )
        stop.set()
        await runner
        return ws.frames, result

    return asyncio.run(main())


def test_gateway_send_frame_and_timeout_is_uncertain():
    frames, result = _gateway_send(True)
    assert [f["action"] for f in frames] == ["send_group_msg"]
    assert frames[0]["params"]["group_id"] == 10001
    assert (result.status, result.message_id) == ("sent", "4242")
    frames, result = _gateway_send(False)
    assert len(frames) == 1 and (result.status, result.error_kind) == ("uncertain", "timeout")
