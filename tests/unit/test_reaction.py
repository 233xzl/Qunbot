"""D-20 "processing" reaction: one-shot narrow gateway channel, one reaction per admitted
request (persistent), never for other messages, failures recorded only."""

import asyncio
import json

import pytest
from test_admission import AT_BOT, event

from group_agent.admission import AccessPolicy, AccessSnapshot, RequestQueue
from group_agent.gateway import (
    ActionRefused,
    OneBotGateway,
    ReactionChannel,
    ReactionResult,
    ReadOnlyActions,
)
from group_agent.intake import EventIntake
from group_agent.reaction import ProcessingReaction
from group_agent.store import Store


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeChannel:
    def __init__(self, result=None, error=None):
        self.calls: list[str] = []
        self.result = result or ReactionResult("ok")
        self.error = error

    async def set_msg_emoji_like(self, message_id, *, timeout=5.0):
        self.calls.append(message_id)
        if self.error is not None:
            raise self.error
        return self.result


def setup(channel, store=None, *, sink=None, clock=None, should_react=None):
    store = store or Store()
    clock = clock or Clock(1000.0)
    access = AccessPolicy(AccessSnapshot("BOT"))
    holder = ProcessingReaction(
        channel, store, clock=clock, max_age_s=60, should_react=should_react
    )
    intake = EventIntake(
        store,
        access,
        sink=sink or RequestQueue(8),
        clock=clock,
        on_admitted=holder.admitted,
    )
    return store, holder, intake


async def drain(holder):
    await holder.aclose()


def test_admitted_request_gets_one_reaction_and_others_none():
    async def main():
        channel = FakeChannel()
        store, holder, intake = setup(channel)
        statuses = [
            intake.intake(event("m1")).status,  # admitted
            intake.intake(event("m1")).status,  # redelivered
            intake.intake(event("m2", segments=[{"type": "text", "data": {"text": "hi"}}])).status,
            # The owner's #新版 command shape: a quote without @bot is no request.
            intake.intake(
                event(
                    "m3",
                    user="OWNER",
                    segments=[
                        {"type": "reply", "data": {"id": "m1"}},
                        {"type": "text", "data": {"text": "#新版"}},
                    ],
                )
            ).status,
            intake.intake(event("m4", post_type="message_sent")).status,
        ]
        await drain(holder)
        return statuses, channel, store, holder

    statuses, channel, store, holder = asyncio.run(main())
    assert statuses == ["trigger", "duplicate", "stored", "stored", "stored"]
    assert channel.calls == ["m1"]
    assert store.reaction_state("BOT", "G1", "m1") == "ok"
    assert store.reaction_state("BOT", "G1", "m2") is None
    assert holder.counts == {"ok": 1}


def test_restart_and_replays_never_react_twice():
    async def main():
        store = Store()
        first = FakeChannel()
        _, holder, intake = setup(first, store)
        intake.intake(event("m1"))
        await drain(holder)
        # New process on the same store: the redelivered event and a direct call both refuse.
        second = FakeChannel()
        _, holder2, intake2 = setup(second, store)
        assert intake2.intake(event("m1")).status == "duplicate"
        from group_agent.admission import ReplyRequest
        from group_agent.contracts import Message

        forged = ReplyRequest(Message("BOT", "G1", "m1", "U1", (AT_BOT,)), 1, None, 1000.0, False)
        holder2.admitted(forged)
        holder2.admitted("m1")  # not a request object
        await drain(holder2)
        return first, second, holder2

    first, second, holder2 = asyncio.run(main())
    assert first.calls == ["m1"] and second.calls == []
    assert holder2.counts == {"refused_claim": 1, "refused_not_request": 1}


def test_unknown_or_stale_requests_are_refused():
    async def main():
        store = Store()
        clock = Clock(1000.0)
        channel = FakeChannel()
        _, holder, _ = setup(channel, store, clock=clock)
        from group_agent.admission import ReplyRequest
        from group_agent.contracts import Message

        # Never admitted here (no queued request row): refused.
        holder.admitted(ReplyRequest(Message("BOT", "G1", "x", "U1", ()), 1, None, 1000.0, False))
        # Admitted long ago (received 2 min before the claim): refused.
        old = Clock(880.0)
        _, _, intake = setup(FakeChannel(), store, clock=old)
        intake.on_admitted = None
        result = intake.intake(event("m9"))
        holder.admitted(result.request)
        await drain(holder)
        return channel, holder

    channel, holder = asyncio.run(main())
    assert channel.calls == [] and holder.counts == {"refused_claim": 2}


def test_switch_policy_and_failure_never_affect_admission():
    async def main():
        failing = FakeChannel(ReactionResult("failed", "failed", 100))
        logs: list[str] = []
        store, holder, intake = setup(failing)
        holder.log = logs.append
        first = intake.intake(event("m1"))
        broken = FakeChannel(error=RuntimeError("boom"))
        _, holder2, intake2 = setup(broken, store)
        second = intake2.intake(event("m2"))
        # Cost gate says no: skipped without a claim.
        none = FakeChannel()
        _, holder3, intake3 = setup(none, store, should_react=lambda r: False)
        third = intake3.intake(event("m3"))
        for h in (holder, holder2, holder3):
            await drain(h)
        return store, logs, (first, second, third), none, intake

    store, logs, results, none, intake = asyncio.run(main())
    assert [r.status for r in results] == ["trigger"] * 3
    rows = store.con.execute("SELECT message_id, state, reaction FROM reply_requests").fetchall()
    assert sorted(tuple(r) for r in rows) == [
        ("m1", "queued", "failed_failed_100"),
        ("m2", "queued", "failed_RuntimeError"),
        ("m3", "queued", None),
    ]
    assert logs == ["processing reaction failed_failed_100"] and none.calls == []
    assert intake.counts.get("admitted_hook_errors") is None


def test_dropped_requests_get_no_reaction():
    async def main():
        channel = FakeChannel()
        store, holder, intake = setup(channel, sink=RequestQueue(1))
        intake.intake(event("m1"))
        intake.intake(event("m2"))  # queue full: dropped
        await drain(holder)
        return channel

    assert asyncio.run(main()).calls == ["m1"]


# -- gateway channel ------------------------------------------------------------------------


class Ws:
    def __init__(self, retcode=0):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.frames: list[dict] = []
        self.retcode = retcode

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
        status = "ok" if self.retcode == 0 else "failed"
        reply = {"status": status, "retcode": self.retcode, "data": None, "echo": request["echo"]}
        self.incoming.put_nowait(json.dumps(reply))

    async def close(self):
        self.incoming.put_nowait(None)


def test_reaction_channel_is_one_shot_and_only_when_enabled():
    plain = OneBotGateway("ws://127.0.0.1:1", "t", allow_actions=True)
    with pytest.raises(ActionRefused):
        plain.reaction_channel("424")
    gw = OneBotGateway("ws://127.0.0.1:1", "t", allow_actions=True, reaction=True)
    assert isinstance(gw.reaction_channel("424"), ReactionChannel)
    with pytest.raises(ActionRefused):
        gw.reaction_channel("424")  # nobody else can take a second one
    receive_only = OneBotGateway("ws://127.0.0.1:1", "t", allow_actions=False, reaction=True)
    with pytest.raises(ActionRefused):
        receive_only.reaction_channel("424")
    with pytest.raises(ActionRefused):
        ReactionChannel(gw, object(), "424")  # forged grant
    fresh = OneBotGateway("ws://127.0.0.1:1", "t", allow_actions=True, reaction=True)
    with pytest.raises(ActionRefused):
        fresh.reaction_channel("424; drop")  # emoji id must be a face number
    with pytest.raises(ActionRefused):
        asyncio.run(gw.call("set_msg_emoji_like", {"message_id": 1, "emoji_id": "424"}))
    from group_agent import gateway

    for grant in (None, object(), gateway._OWNER_GRANT):  # the owner grant does not fit
        with pytest.raises(ActionRefused):
            asyncio.run(
                ReadOnlyActions(Ws()).send_action("set_msg_emoji_like", {}, "e", grant=grant)
            )
    with pytest.raises(ActionRefused):  # nor the reaction grant another action
        asyncio.run(
            ReadOnlyActions(Ws()).send_action(
                "send_group_msg", {}, "e", grant=gateway._REACTION_GRANT
            )
        )


def _react(retcode=0, n=1, clock=None):
    async def main():
        ws = Ws(retcode)

        async def connect(*a, **k):
            return ws

        gw = OneBotGateway(
            "ws://x",
            "t",
            allow_actions=True,
            reaction=True,
            connect=connect,
            clock=clock or Clock(),
        )
        channel = gw.reaction_channel("424")
        stop = asyncio.Event()
        runner = asyncio.create_task(gw.run(stop))
        await gw.wait_connected(2)
        results = [await channel.set_msg_emoji_like("10001", timeout=1) for _ in range(n)]
        stop.set()
        await runner
        return ws.frames, results

    return asyncio.run(main())


def test_reaction_frame_failure_and_rate_limit():
    frames, results = _react()
    assert frames == [
        {
            "action": "set_msg_emoji_like",
            "params": {"message_id": 10001, "emoji_id": "424", "set": True},
            "echo": frames[0]["echo"],
        }
    ]
    assert results == [ReactionResult("ok")]
    _, results = _react(retcode=1200)
    assert results == [ReactionResult("failed", "failed", 1200)]
    # Burst of 2 per second: the third immediate call is refused without a frame.
    frames, results = _react(n=3)
    assert [r.status for r in results] == ["ok", "ok", "failed"]
    assert results[2].error_kind == "rate_limited" and len(frames) == 2
