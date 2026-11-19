import asyncio
import json

import pytest

from group_agent.config import ConfigError, load_snowluma_credentials
from group_agent.shadow import ShadowIngestor, run_shadow
from group_agent.store import Store


def group_event(message_id="m1", user_id="U1", group_id="G1", text="hi", **extra):
    event = {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": group_id,
        "user_id": user_id,
        "message_id": message_id,
        "time": 1700000000,
        "message": [{"type": "text", "data": {"text": text}}],
    }
    event.update(extra)
    return event


def recall_event(message_id="m1", user_id="U1", group_id="G1"):
    return {
        "post_type": "notice",
        "notice_type": "group_recall",
        "self_id": "BOT",
        "group_id": group_id,
        "user_id": user_id,
        "operator_id": user_id,
        "message_id": message_id,
        "time": 1700000001,
    }


def make_ingestor(**kwargs):
    store = Store()
    return store, ShadowIngestor(store, "BOT", clock=lambda: 42.0, **kwargs)


def test_stores_group_messages_with_times_and_dedups():
    store, ingestor = make_ingestor()
    assert ingestor.handle(group_event()) == "stored"
    assert ingestor.handle(group_event()) == "duplicate"
    row = store.con.execute("SELECT platform_time, received_at FROM messages").fetchone()
    assert tuple(row) == (1700000000, 42.0)


@pytest.mark.parametrize(
    "event,status",
    [
        ({"post_type": "meta_event", "meta_event_type": "heartbeat"}, "meta_heartbeat"),
        (group_event(message_type="private"), "ignored_private"),
        (group_event(self_id="OTHER"), "ignored_other_bot"),
        (
            {"post_type": "notice", "notice_type": "group_increase", "self_id": "BOT"},
            "notice_group_increase",
        ),
        ({"post_type": "request", "self_id": "BOT"}, "ignored_request"),
        ({"retcode": 0, "echo": "x"}, "ignored_api_response"),
        (group_event(message="CQ string"), "ignored_invalid"),
        ("not a dict", "ignored_invalid"),
    ],
)
def test_non_group_events_are_not_stored(event, status):
    store, ingestor = make_ingestor()
    assert ingestor.handle(event) == status
    assert store.summary()["messages"] == 0


def test_blocked_users_and_groups_are_not_collected():
    store, ingestor = make_ingestor(
        blocked_users=frozenset({"U9"}), blocked_groups=frozenset({"G9"})
    )
    assert ingestor.handle(group_event(user_id="U9")) == "ignored_blocked"
    assert ingestor.handle(group_event(group_id="G9")) == "ignored_blocked"
    assert ingestor.handle(recall_event(group_id="G9")) == "ignored_blocked"
    assert store.summary()["messages"] == 0


def test_recall_notice_removes_content():
    store, ingestor = make_ingestor()
    ingestor.handle(group_event(text="to be recalled"))
    assert ingestor.handle(recall_event()) == "recall_revoked"
    assert ingestor.handle(recall_event(message_id="m-later")) == "recall_tombstone"
    assert ingestor.handle(group_event(message_id="m-later")) == "duplicate"
    texts = [r[0] for r in store.con.execute("SELECT segments FROM messages")]
    assert texts == ["[]", "[]"]
    assert store.summary()["revoked"] == 2


class FakeConnection:
    """Yields scripted frames; any attempt to send fails the test."""

    def __init__(self, frames, stop=None):
        self.frames, self.stop, self.closed = list(frames), stop, False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for frame in self.frames:
            yield frame
        if self.stop is not None:
            self.stop.set()

    async def send(self, *_args, **_kwargs):
        raise AssertionError("shadow must never send")

    async def close(self):
        self.closed = True


async def test_run_shadow_receives_without_sending_and_reconnects():
    store, ingestor = make_ingestor()
    stop = asyncio.Event()
    calls = []
    connections = [
        FakeConnection([json.dumps(group_event("m1")), "{bad json"]),
        FakeConnection([json.dumps(group_event("m2")), json.dumps(recall_event("m1"))], stop),
    ]

    async def connect(url, **kwargs):
        calls.append((url, kwargs))
        return connections[len(calls) - 1]

    stats = await run_shadow(
        "ws://127.0.0.1:3001", "tok", ingestor, stop, connect=connect, reconnect_initial=0.01
    )
    assert stats["connects"] == 2 and stats["frames"] == 4 and stats["bad_frames"] == 1
    assert ingestor.counts["stored"] == 2 and ingestor.counts["recall_revoked"] == 1
    assert all(c.closed for c in connections)
    assert calls[0][1]["additional_headers"] == {"Authorization": "Bearer tok"}


async def test_run_shadow_backs_off_on_connect_failure():
    _, ingestor = make_ingestor()
    stop = asyncio.Event()
    attempts = 0

    async def connect(url, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            stop.set()
        raise OSError("refused")

    stats = await run_shadow(
        "ws://127.0.0.1:3001", "tok", ingestor, stop, connect=connect, reconnect_initial=0.01
    )
    assert stats["connect_failures"] == 3 and stats["connects"] == 0


class ActionWS(FakeConnection):
    """Answers get_image; anything else fails the test."""

    def __init__(self, frames):
        super().__init__([])
        self.inbound = asyncio.Queue()
        for frame in frames:
            self.inbound.put_nowait(json.dumps(frame))
        self.sent = []

    async def _iterate(self):
        while True:
            frame = await self.inbound.get()
            if frame is None:
                return
            yield frame

    async def send(self, frame):
        request = json.loads(frame)
        self.sent.append(request)
        assert request["action"] == "get_image"
        data = {
            "file": "https://multimedia.nt.qq.com.cn/x",
            "url": "https://multimedia.nt.qq.com.cn/x",
        }
        response = {"status": "ok", "retcode": 0, "data": data, "echo": request["echo"]}
        self.inbound.put_nowait(json.dumps(response))

    async def close(self):
        self.closed = True
        self.inbound.put_nowait(None)


async def test_shadow_collect_saves_images_and_records_requests_without_runs(tmp_path):
    from group_agent.admission import RecordOnly
    from group_agent.media import ImageSourcePolicy, MediaCollector

    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR\x00\x00\x00\x02\x00\x00\x00\x02rest"
    store = Store()
    sink = RecordOnly()
    ingestor = ShadowIngestor(store, "BOT", clock=lambda: 42.0, admit=True, sink=sink)
    image = group_event("m1", message=[{"type": "image", "data": {"file": "k1", "url": ""}}])
    ask = group_event("m2", message=[{"type": "at", "data": {"qq": "BOT"}}])
    ws = ActionWS([image, ask, ask])
    stop = asyncio.Event()

    async def resolver(host, port):
        return ["111.30.1.1"]

    async def download(url, max_bytes, timeout):
        return 200, png, None

    def factory(api):
        return MediaCollector(
            store,
            api,
            tmp_path / "media",
            policy=ImageSourcePolicy(resolver=resolver),
            download=download,
        )

    async def connect(url, **_):
        return ws

    task = asyncio.create_task(
        run_shadow(
            "ws://127.0.0.1:3001", "tok", ingestor, stop, connect=connect, media_factory=factory
        )
    )
    for _ in range(400):
        if store.summary()["media_files"] and ingestor.counts["duplicate"]:
            break
        await asyncio.sleep(0.005)
    stop.set()
    stats = await asyncio.wait_for(task, 3)
    assert [r["action"] for r in ws.sent] == ["get_image"] and stats["action_calls"] == 1
    assert ingestor.counts["trigger"] == 1 and ingestor.counts["duplicate"] == 1
    assert sink.counts["recorded"] == 1
    summary = store.summary()
    assert summary["reply_requests"] == 1 and summary["runs"] == 0
    assert summary["media_slots"] == {"available": 1} and summary["media_files"] == 1


@pytest.mark.parametrize(
    "url",
    [
        "ws://192.168.1.5:3001",
        "wss://127.0.0.1:3001",
        "ws://user:pw@127.0.0.1:3001",
        "ws://127.0.0.1:3001/?access_token=x",
        "http://127.0.0.1:3001",
    ],
)
def test_snowluma_credentials_require_plain_loopback_ws(tmp_path, url):
    path = tmp_path / "s.env"
    path.write_text(f"SNOWLUMA_WS_URL={url}\nSNOWLUMA_WS_TOKEN=secret-token", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_snowluma_credentials(path)
    assert "secret-token" not in str(error.value)


def test_snowluma_credentials_hidden_from_repr(tmp_path):
    path = tmp_path / "s.env"
    path.write_text(
        "SNOWLUMA_WS_URL=ws://127.0.0.1:3001\nSNOWLUMA_WS_TOKEN=secret-token", encoding="utf-8"
    )
    credentials = load_snowluma_credentials(path)
    assert credentials.token == "secret-token"
    assert "secret-token" not in repr(credentials)
