import asyncio
from collections import Counter
import json

from group_agent.backfill import ReplyBackfill, message_from_get_msg
from group_agent.config import Settings
from group_agent.contracts import Message
from group_agent.gateway import ActionTimeout
from group_agent.pipeline import Pipeline
from group_agent.shadow import ShadowIngestor, run_shadow
from group_agent.store import Store


def reply_message(message_id="m2", reply_to="m1", group_id="G1"):
    return Message(
        "BOT",
        group_id,
        message_id,
        "U2",
        (
            {"type": "reply", "data": {"id": reply_to}},
            {"type": "text", "data": {"text": "what is this"}},
        ),
    )


def get_msg_data(message_id="m1", group_id="G1", user_id="U1", reply_to=None, **extra):
    segments = [{"type": "text", "data": {"text": f"original {message_id}"}}]
    if reply_to:
        segments.insert(0, {"type": "reply", "data": {"id": reply_to}})
    data = {
        "time": 1700000000,
        "message_type": "group",
        "message_id": message_id,
        "real_id": message_id,
        "group_id": group_id,
        "user_id": user_id,
        "sender": {"user_id": user_id, "nickname": "n"},
        "message": segments,
    }
    data.update(extra)
    return data


class FakeApi:
    def __init__(self, messages=None, error=None):
        self.messages, self.error, self.calls = messages or {}, error, []

    async def get_msg(self, message_id):
        self.calls.append(message_id)
        if self.error is not None:
            raise self.error
        return self.messages[message_id]


def make(api, **kwargs):
    store = Store()
    return store, ReplyBackfill(store, api, "BOT", clock=lambda: 100.0, **kwargs)


async def test_missing_reference_is_fetched_and_stored_as_backfill():
    store, backfill = make(FakeApi({"m1": get_msg_data()}))
    current = reply_message()
    store.ingest(current)
    assert await backfill.resolve(current) == ["stored"]
    row = store.lookup("BOT", "G1", "m1")
    assert row["source"] == "backfill" and row["platform_time"] == 1700000000
    assert store.lookup("BOT", "G1", "m2")["source"] == "live"
    assert store.summary()["runs"] == 0 and store.summary()["backfilled"] == 1
    history = store.history(Message("BOT", "G1", "m9", "U9", ()), 10**9)
    assert [m.text for m in history if m.message_id == "m1"] == ["original m1"]


async def test_backfilled_message_never_triggers_a_run():
    # Even if the same message later arrives as a pushed event, it is a duplicate.
    store, backfill = make(FakeApi({"m1": get_msg_data()}))
    await backfill.resolve(reply_message())
    pipeline = Pipeline(Settings(), store)
    event = {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": "G1",
        "user_id": "U1",
        "message_id": "m1",
        "message": [{"type": "at", "data": {"qq": "BOT"}}],
    }
    assert pipeline.admit(event).status == "duplicate"


async def test_known_reference_is_not_looked_up():
    api = FakeApi()
    store, backfill = make(api)
    store.ingest(Message("BOT", "G1", "m1", "U1", ()))
    assert await backfill.resolve(reply_message()) == ["known"]
    assert api.calls == []


async def test_other_group_or_missing_group_is_rejected():
    for data, reason in [
        (get_msg_data(group_id="G2"), "rejected_other_group"),
        ({k: v for k, v in get_msg_data().items() if k != "group_id"}, "rejected_other_group"),
        (get_msg_data(message_id="m7"), "rejected_id_mismatch"),
        (get_msg_data(message="CQ string"), "rejected_invalid"),
        (get_msg_data(message_type="private"), "rejected_not_group"),
    ]:
        store, backfill = make(FakeApi({"m1": data}))
        assert await backfill.resolve(reply_message()) == [reason]
        assert store.summary()["messages"] == 0


async def test_failures_are_remembered_briefly():
    now = [100.0]
    api = FakeApi(error=ActionTimeout("timeout"))
    store = Store()
    backfill = ReplyBackfill(store, api, "BOT", clock=lambda: now[0], failure_ttl=600)
    assert await backfill.resolve(reply_message()) == ["failed_timeout"]
    assert await backfill.resolve(reply_message()) == ["recent_failure"]
    now[0] += 601
    assert await backfill.resolve(reply_message()) == ["failed_timeout"]
    assert len(api.calls) == 2


async def test_depth_and_lookup_limits():
    chain = {
        f"m{i}": get_msg_data(f"m{i}", reply_to=f"m{i - 1}" if i > 1 else None) for i in range(1, 6)
    }
    store, backfill = make(FakeApi(chain), max_depth=3)
    assert await backfill.resolve(reply_message("m9", "m5")) == ["stored"] * 3
    assert [store.lookup("BOT", "G1", f"m{i}") is not None for i in range(1, 6)] == [
        False,
        False,
        True,
        True,
        True,
    ]
    store, backfill = make(FakeApi(chain), max_depth=10, max_lookups=2)
    assert await backfill.resolve(reply_message("m9", "m5")) == ["stored", "stored", "limit"]


async def test_blocked_sender_and_recalled_reference_are_not_stored():
    store, backfill = make(
        FakeApi({"m1": get_msg_data(user_id="U9")}), blocked_users=frozenset({"U9"})
    )
    assert await backfill.resolve(reply_message()) == ["blocked"]
    assert store.summary()["messages"] == 0
    store, backfill = make(FakeApi({"m1": get_msg_data()}))
    store.revoke("BOT", "G1", "m1", "U1")  # tombstone: recall arrived first
    assert await backfill.resolve(reply_message()) == ["known"]


def test_message_from_get_msg_uses_sender_when_user_id_missing():
    data = get_msg_data()
    del data["user_id"]
    message, platform_time = message_from_get_msg(data, "BOT", "G1", "m1")
    assert message.user_id == "U1" and platform_time == 1700000000


class FakeWS:
    def __init__(self, frames, responses):
        self.inbound: asyncio.Queue = asyncio.Queue()
        for frame in frames:
            self.inbound.put_nowait(json.dumps(frame))
        self.responses, self.sent = responses, []

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        while True:
            frame = await self.inbound.get()
            if frame is None:
                return
            yield frame

    async def send(self, frame):
        request = json.loads(frame)
        self.sent.append(request)
        data = self.responses[str(request["params"]["message_id"])]
        response = {"status": "ok", "retcode": 0, "data": data, "echo": request["echo"]}
        self.inbound.put_nowait(json.dumps(response))

    async def close(self):
        self.inbound.put_nowait(None)


async def test_shadow_backfill_end_to_end_only_sends_get_msg():
    store = Store()
    ingestor = ShadowIngestor(store, "BOT", clock=lambda: 1.0)
    event = {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": "G1",
        "user_id": "U2",
        "message_id": "m2",
        "message": [
            {"type": "reply", "data": {"id": "m1"}},
            {"type": "text", "data": {"text": "?"}},
        ],
    }
    ws = FakeWS([event], {"m1": get_msg_data()})
    stop = asyncio.Event()
    counts = Counter()

    async def connect(url, **_):
        return ws

    task = asyncio.create_task(
        run_shadow(
            "ws://127.0.0.1:3001",
            "tok",
            ingestor,
            stop,
            connect=connect,
            backfill_replies=True,
            backfill_counts=counts,
        )
    )
    for _ in range(200):
        if counts["stored"]:
            break
        await asyncio.sleep(0.005)
    stop.set()
    stats = await asyncio.wait_for(task, 2)
    assert counts["stored"] == 1 and ingestor.counts["stored"] == 1
    assert [r["action"] for r in ws.sent] == ["get_msg"]
    assert stats["action_calls"] == 1
    assert store.lookup("BOT", "G1", "m1")["source"] == "backfill"
