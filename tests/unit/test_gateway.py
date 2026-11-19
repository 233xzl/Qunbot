import asyncio
import json

import pytest

from group_agent.gateway import (
    READ_ONLY_ACTIONS,
    ActionFailed,
    ActionRefused,
    ActionTimeout,
    GatewayDisconnected,
    GroupMessageEvent,
    IgnoredEvent,
    ImageInfo,
    MetaEvent,
    NoticeEvent,
    OneBotApi,
    OneBotGateway,
    ReadOnlyActions,
    ReceiveOnly,
    RecallEvent,
    TokenBucket,
    parse_event,
)


class FakeWS:
    """Scripted forward WebSocket. `responder(request)` may return a response dict."""

    def __init__(self, responder=None):
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False
        self.responder = responder

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
        if self.responder is not None:
            response = self.responder(request)
            if response is not None:
                self.push(response)

    async def close(self):
        self.closed = True
        self.inbound.put_nowait(None)

    def push(self, obj):
        self.inbound.put_nowait(json.dumps(obj))

    def drop(self):
        self.inbound.put_nowait(None)


def ok(request, data=None):
    return {"status": "ok", "retcode": 0, "data": data, "echo": request["echo"]}


async def start(connections, on_event=None, **kwargs):
    stop = asyncio.Event()
    queue = list(connections)

    async def connect(url, **_):
        if not queue:
            raise OSError("no more fake connections")
        return queue.pop(0)

    kwargs.setdefault("allow_actions", True)
    gateway = OneBotGateway(
        "ws://127.0.0.1:3001", "tok", on_event, connect=connect, reconnect_initial=0.01, **kwargs
    )
    task = asyncio.create_task(gateway.run(stop))
    assert await gateway.wait_connected(1)
    return gateway, stop, task


async def finish(stop, task):
    stop.set()
    return await asyncio.wait_for(task, 2)


FORBIDDEN = [
    "send_group_msg",
    "send_msg",
    "send_private_msg",
    "send_group_forward_msg",
    "delete_msg",
    "set_msg_emoji_like",
    "set_group_ban",
    "set_group_card",
    "set_group_leave",
    "",
    None,
    "GET_MSG",
]


def test_allowlist_contains_only_read_only_actions():
    assert READ_ONLY_ACTIONS == {
        "get_msg",
        "get_image",
        "get_forward_msg",
        "get_status",
        "get_login_info",
        "get_group_info",
        "get_group_list",
        "get_group_member_info",
        "get_group_member_list",
    }
    assert not any(a.startswith(("send", "set", "delete")) for a in READ_ONLY_ACTIONS)


@pytest.mark.parametrize("action", FORBIDDEN)
async def test_non_allowlisted_actions_are_refused_without_writing(action):
    ws = FakeWS(responder=ok)
    gateway, stop, task = await start([ws])
    with pytest.raises(ActionRefused) as error:
        await gateway.call(action, {"group_id": 1, "message": "hi"})
    assert error.value.kind == "action_not_allowed"
    # The connection wrapper refuses too, even if the gateway check were bypassed.
    with pytest.raises(ActionRefused):
        await ReadOnlyActions(ws).send_action(action, {}, "e1")
    await finish(stop, task)
    assert ws.sent == []


async def test_actions_disabled_by_default_and_connection_is_receive_only():
    ws = FakeWS(responder=ok)
    gateway, stop, task = await start([ws], allow_actions=False)
    assert type(gateway._connection) is ReceiveOnly
    assert not hasattr(gateway._connection, "send_action")
    with pytest.raises(ActionRefused) as error:
        await gateway.call("get_status")
    assert error.value.kind == "actions_disabled"
    await finish(stop, task)
    assert ws.sent == []


async def test_narrower_action_set_refuses_other_read_actions():
    ws = FakeWS(responder=ok)
    gateway, stop, task = await start([ws], actions=frozenset({"get_image"}))
    with pytest.raises(ActionRefused) as error:
        await gateway.call("get_group_member_list", {"group_id": 1})
    assert error.value.kind == "action_not_enabled"
    assert await gateway.call("get_image", {"file": "f"}) is None
    await finish(stop, task)
    assert [r["action"] for r in ws.sent] == ["get_image"]
    with pytest.raises(ValueError):
        OneBotGateway("ws://127.0.0.1:3001", "tok", actions=frozenset({"send_group_msg"}))


async def test_echo_matches_out_of_order_responses():
    ws = FakeWS()
    gateway, stop, task = await start([ws])
    first = asyncio.create_task(gateway.call("get_msg", {"message_id": 1}))
    second = asyncio.create_task(gateway.call("get_msg", {"message_id": 2}))
    while len(ws.sent) < 2:
        await asyncio.sleep(0)
    echoes = [r["echo"] for r in ws.sent]
    assert len(set(echoes)) == 2
    ws.push({"status": "ok", "retcode": 0, "data": {"n": 2}, "echo": echoes[1]})
    ws.push({"status": "ok", "retcode": 0, "data": {"n": 1}, "echo": echoes[0]})
    assert await first == {"n": 1} and await second == {"n": 2}
    assert ws.sent[0] == {"action": "get_msg", "params": {"message_id": 1}, "echo": echoes[0]}
    stats = await finish(stop, task)
    assert stats["responses"] == 2 and stats["action_ok"] == 2


async def test_timeout_then_late_and_unmatched_responses_are_dropped():
    ws = FakeWS()
    gateway, stop, task = await start([ws], action_timeout=0.05)
    with pytest.raises(ActionTimeout):
        await gateway.call("get_status")
    echo = ws.sent[0]["echo"]
    ws.push({"status": "ok", "retcode": 0, "data": {}, "echo": echo})
    ws.push({"status": "ok", "retcode": 0, "data": {}, "echo": "someone-else"})
    ws.push({"status": "ok", "retcode": 0, "data": {}})
    await asyncio.sleep(0.02)
    stats = await finish(stop, task)
    assert stats["action_timeouts"] == 1
    assert stats["late_responses"] == 1 and stats["unmatched_responses"] == 2
    assert gateway._pending == {}


async def test_disconnect_fails_pending_calls_and_reconnects():
    first, second = FakeWS(), FakeWS(responder=lambda r: ok(r, {"online": True}))
    gateway, stop, task = await start([first, second], action_timeout=2)
    pending = asyncio.create_task(gateway.call("get_status"))
    while not first.sent:
        await asyncio.sleep(0)
    first.drop()
    with pytest.raises(GatewayDisconnected) as error:
        await pending
    assert error.value.kind == "disconnected"
    assert await gateway.wait_connected(1)
    assert await gateway.call("get_status") == {"online": True}
    stats = await finish(stop, task)
    assert stats["connects"] == 2 and stats["disconnects"] == 2


async def test_call_without_connection_fails_fast():
    gateway = OneBotGateway("ws://127.0.0.1:3001", "tok", allow_actions=True)
    with pytest.raises(GatewayDisconnected) as error:
        await gateway.call("get_status")
    assert error.value.kind == "not_connected"


async def test_failed_status_keeps_retcode_only():
    ws = FakeWS(
        responder=lambda r: {
            "status": "failed",
            "retcode": 1200,
            "message": "message not found 12345",
            "wording": "secret wording",
            "echo": r["echo"],
        }
    )
    gateway, stop, task = await start([ws])
    with pytest.raises(ActionFailed) as error:
        await gateway.call("get_msg", {"message_id": 5})
    assert error.value.retcode == 1200 and error.value.kind == "failed"
    assert "secret" not in str(error.value) and "12345" not in str(error.value)
    await finish(stop, task)


def test_token_bucket_refills_at_rate():
    now = [0.0]
    bucket = TokenBucket(rate=2, capacity=2, clock=lambda: now[0])
    assert bucket.try_acquire() and bucket.try_acquire()
    assert not bucket.try_acquire()
    now[0] = 0.5
    assert bucket.try_acquire() and not bucket.try_acquire()
    now[0] = 100
    assert bucket.try_acquire() and bucket.try_acquire() and not bucket.try_acquire()


async def test_rate_limit_spaces_calls():
    ws = FakeWS(responder=ok)
    gateway, stop, task = await start([ws], rate_per_second=20, burst=1)
    loop = asyncio.get_running_loop()
    started = loop.time()
    for _ in range(4):
        await gateway.call("get_status")
    assert loop.time() - started >= 0.12  # 3 waits of 50 ms after the first token
    await finish(stop, task)


async def test_concurrency_cap_holds_back_extra_calls():
    ws = FakeWS()
    gateway, stop, task = await start([ws], max_concurrency=1)
    first = asyncio.create_task(gateway.call("get_status"))
    second = asyncio.create_task(gateway.call("get_status"))
    await asyncio.sleep(0.02)
    assert len(ws.sent) == 1  # the second call waits for the slot
    ws.push(ok(ws.sent[0], 1))
    assert await first == 1
    while len(ws.sent) < 2:
        await asyncio.sleep(0)
    ws.push(ok(ws.sent[1], 2))
    assert await second == 2
    await finish(stop, task)


async def test_async_event_handler_can_call_actions_in_order():
    seen = []
    ws = FakeWS(responder=lambda r: ok(r, {"for": r["params"]["message_id"]}))

    async def handler(event):
        data = await gateway.call("get_msg", {"message_id": event["n"]})
        seen.append((event["n"], data["for"]))

    gateway, stop, task = await start([ws], on_event=handler)
    for n in (1, 2, 3):
        ws.push({"post_type": "message", "n": n})
    while len(seen) < 3:
        await asyncio.sleep(0.005)
    await finish(stop, task)
    assert seen == [(1, 1), (2, 2), (3, 3)]


async def test_handler_errors_are_counted_not_fatal():
    calls = []

    def handler(event):
        calls.append(event)
        raise RuntimeError("boom")

    ws = FakeWS()
    gateway, stop, task = await start([ws], on_event=handler)
    ws.push({"post_type": "meta_event"})
    ws.push("not json")  # json string of a string: valid JSON, still an event frame
    ws.inbound.put_nowait("{bad")
    await asyncio.sleep(0.02)
    stats = await finish(stop, task)
    assert stats["handler_errors"] == 2 and stats["bad_frames"] == 1 and len(calls) == 2


async def test_api_wrappers_send_numeric_ids_and_validate_shapes():
    def responder(request):
        action = request["action"]
        if action == "get_image":
            return ok(request, {"file": "C:/cache/a.jpg", "url": "https://x/a", "file_size": "12"})
        if action == "get_group_member_list":
            return ok(request, {"not": "a list"})
        return ok(request, {"ok": True})

    ws = FakeWS(responder)
    gateway, stop, task = await start([ws])
    api = OneBotApi(gateway)
    await api.get_msg("123")
    await api.get_group_info("456")
    await api.get_group_member_info("456", "789")
    image = await api.get_image("abc.image")
    with pytest.raises(ActionFailed):
        await api.get_group_member_list("456")
    await finish(stop, task)
    assert image == ImageInfo("C:/cache/a.jpg", "https://x/a", 12, None)
    params = [r["params"] for r in ws.sent]
    assert params[0] == {"message_id": 123}
    assert params[1] == {"group_id": 456, "no_cache": False}
    assert params[2] == {"group_id": 456, "user_id": 789, "no_cache": False}
    assert params[3] == {"file": "abc.image"}
    assert {r["action"] for r in ws.sent} <= READ_ONLY_ACTIONS


def group_event(**extra):
    event = {
        "post_type": "message",
        "message_type": "group",
        "self_id": 1,
        "group_id": 2,
        "user_id": 3,
        "message_id": 4,
        "time": 1700000000,
        "message": [{"type": "text", "data": {"text": "hi"}}],
    }
    event.update(extra)
    return event


@pytest.mark.parametrize(
    "data,kind",
    [
        ({"messages": []}, None),
        ({"messages": [{"message": []}]}, None),
        ({"messages": "wrong"}, "bad_response"),
        ({"messages": [None]}, "bad_response"),
        ({"messages": [{"text": "x" * (1024 * 1024)}]}, "forward_oversize"),
    ],
)
async def test_forward_api_preserves_resource_id_and_bounds_response(data, kind):
    calls = []

    class Gateway:
        async def call(self, action, params, **kw):
            calls.append((action, params))
            return data

    api = OneBotApi(Gateway())
    if kind:
        with pytest.raises(ActionFailed) as error:
            await api.get_forward_msg("resource-opaque")
        assert error.value.kind == kind
    else:
        assert await api.get_forward_msg("resource-opaque") == data["messages"]
    assert calls == [("get_forward_msg", {"id": "resource-opaque"})]
    for value in ("", None, 123, "x" * 2049):
        with pytest.raises(ActionRefused):
            await api.get_forward_msg(value)


def test_parse_event_kinds():
    parsed = parse_event(group_event(), "1")
    assert isinstance(parsed, GroupMessageEvent)
    assert parsed.message.group_id == "2" and parsed.message.text == "hi"
    assert parsed.platform_time == 1700000000 and not parsed.self_sent
    assert parse_event(group_event(post_type="message_sent"), "1").self_sent
    assert parse_event(group_event(message="[CQ:at,qq=1]"), "1") == IgnoredEvent("invalid")
    assert parse_event(group_event(message_type="private"), "1") == IgnoredEvent("private")
    assert parse_event(group_event(self_id=9), "1") == IgnoredEvent("other_bot")
    assert parse_event({"retcode": 0, "echo": "e"}, "1") == IgnoredEvent("api_response")
    assert parse_event(
        {
            "post_type": "meta_event",
            "meta_event_type": "heartbeat",
            "interval": 30000,
            "status": {"online": True, "good": True},
        },
        "1",
    ) == MetaEvent("heartbeat", None, 30000, True)
    assert parse_event(
        {"post_type": "meta_event", "meta_event_type": "lifecycle", "sub_type": "connect"}, "1"
    ) == MetaEvent("lifecycle", "connect", None, None)
    recall = {
        "post_type": "notice",
        "notice_type": "group_recall",
        "self_id": 1,
        "group_id": 2,
        "user_id": 3,
        "operator_id": 5,
        "message_id": 4,
        "time": 7,
    }
    assert parse_event(recall, "1") == RecallEvent("2", "4", "3", "5", 7)
    assert parse_event({**recall, "message_id": True}, "1") == IgnoredEvent("invalid")
    assert parse_event(
        {"post_type": "notice", "notice_type": "group_increase", "self_id": 1, "group_id": 2},
        "1",
    ) == NoticeEvent("group_increase", None, "2")
    assert parse_event({"post_type": "request", "self_id": 1}, "1") == IgnoredEvent("request")
