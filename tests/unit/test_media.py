import asyncio
import hashlib
import struct

import pytest

from group_agent.admission import AccessPolicy, AccessSnapshot
from group_agent.contracts import Message
from group_agent.gateway import ActionTimeout, ImageInfo
from group_agent.intake import EventIntake
from group_agent.media import ImageSourcePolicy, MediaCollector, MediaUnavailable, sniff_image
from group_agent.store import Store

QQ_URL = "https://multimedia.nt.qq.com.cn/download?fileid=abc"
GCHAT_URL = "http://gchat.qpic.cn/gchatpic_new/0/0-0-ABC/0"


def png(width=4, height=3, tail=b"x"):
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + tail


async def public_resolver(host, port):
    return ["203.0.113.10"] if host.endswith("example.org") else ["111.30.1.1"]


async def private_resolver(host, port):
    return ["10.0.0.5"]


class FakeApi:
    def __init__(self, infos=None, error=None):
        self.infos, self.error, self.calls = infos or {}, error, []

    async def get_image(self, file):
        self.calls.append(file)
        if self.error is not None:
            raise self.error
        return self.infos[file]


class FakeDownload:
    def __init__(self, bodies=None, status=200, redirects=None, gate=None):
        self.bodies, self.status = bodies or {}, status
        self.redirects, self.gate = redirects or {}, gate
        self.calls, self.active, self.peak = [], 0, 0

    async def __call__(self, url, max_bytes, timeout):
        self.calls.append(url)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
            if url in self.redirects:
                return 302, b"", self.redirects[url]
            body = self.bodies.get(url, png())
            if len(body) > max_bytes:
                raise MediaUnavailable("oversize")
            return self.status, body, None
        finally:
            self.active -= 1


def image_message(mid="m1", files=("f1",), url=QQ_URL, **data):
    segments = tuple({"type": "image", "data": {"file": f, "url": url, **data}} for f in files) + (
        {"type": "text", "data": {"text": "look"}},
    )
    return Message("BOT", "G1", mid, "U1", segments)


def make(tmp_path, api=None, download=None, **kwargs):
    store = Store()
    collector = MediaCollector(
        store,
        api,
        tmp_path / "media",
        policy=ImageSourcePolicy(resolver=public_resolver),
        download=download or FakeDownload(),
        rate_per_second=1000,
        burst=100,
        clock=lambda: 5.0,
        **kwargs,
    )
    return store, collector


def stored(store, message):
    seq, _ = store.ingest(message, 100, 1.0)
    return seq


async def test_get_image_url_is_downloaded_hashed_and_linked(tmp_path):
    api = FakeApi({"f1": ImageInfo(None, QQ_URL, 20, "a.png")})
    store, collector = make(tmp_path, api)
    message = image_message()
    seq = stored(store, message)
    assert collector.schedule(seq, message) == 1
    await collector.drain()
    [slot] = store.message_media(seq)
    digest = hashlib.sha256(png()).hexdigest()
    assert slot["status"] == "available" and slot["sha256"] == digest
    assert (slot["mime"], slot["width"], slot["height"]) == ("image/png", 4, 3)
    assert (tmp_path / "media" / f"{digest}.png").read_bytes() == png()
    assert api.calls == ["f1"] and collector.counts["files_written"] == 1


async def test_same_content_is_stored_once(tmp_path):
    download = FakeDownload()
    store, collector = make(tmp_path, FakeApi(error=ActionTimeout("timeout")), download)
    for mid, key in (("m1", "f1"), ("m2", "f2")):
        message = image_message(mid, (key,))
        collector.schedule(stored(store, message), message)
        await collector.drain()
    assert len(list((tmp_path / "media").iterdir())) == 1
    assert collector.counts["dedup_hash"] == 1
    # get_image failed; the segment's own URL was used as the fallback.
    assert download.calls == [QQ_URL, QQ_URL]
    # A repeated file key reuses the stored file without another fetch.
    message = image_message("m3", ("f1",))
    collector.schedule(stored(store, message), message)
    await collector.drain()
    assert collector.counts["dedup_key"] == 1 and len(download.calls) == 2


@pytest.mark.parametrize(
    "info,body,reason",
    [
        (ImageInfo(None, QQ_URL, 11 * 1024 * 1024, None), png(), "oversize"),
        (ImageInfo(None, QQ_URL, None, None), b"x" * 2048, "oversize"),
        (ImageInfo(None, QQ_URL, None, None), b"<html>not an image</html>", "not_image"),
        (ImageInfo(None, QQ_URL, None, None), b"", "empty"),
    ],
)
async def test_failures_mark_slot_unavailable(tmp_path, info, body, reason):
    download = FakeDownload({QQ_URL: body})
    max_bytes = 1024 if reason == "oversize" and info.file_size is None else 10 * 1024 * 1024
    store, collector = make(tmp_path, FakeApi({"f1": info}), download, max_bytes=max_bytes)
    message = image_message(url="")
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    [slot] = store.message_media(seq)
    assert (slot["status"], slot["reason"]) == ("unavailable", reason)
    assert not (tmp_path / "media").exists() or not list((tmp_path / "media").iterdir())
    if info.file_size and info.file_size > max_bytes:
        assert download.calls == []  # declared size over the cap is never fetched


async def test_declared_segment_size_over_cap_keeps_metadata_only(tmp_path):
    download = FakeDownload()
    store, collector = make(tmp_path, FakeApi(), download)
    message = image_message(file_size=str(20 * 1024 * 1024))
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    [slot] = store.message_media(seq)
    assert slot["reason"] == "oversize" and slot["declared_bytes"] == 20 * 1024 * 1024
    assert download.calls == []


async def test_local_file_from_get_image_is_read(tmp_path):
    local = tmp_path / "cache" / "img.bin"
    local.parent.mkdir()
    local.write_bytes(png(9, 9))
    api = FakeApi({"f1": ImageInfo(str(local), None, None, None)})
    download = FakeDownload()
    store, collector = make(tmp_path, api, download)
    message = image_message(url="")
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    [slot] = store.message_media(seq)
    assert slot["status"] == "available" and slot["width"] == 9
    assert download.calls == [] and collector.counts["local_reads"] == 1


async def test_relative_local_name_is_not_a_path(tmp_path):
    api = FakeApi({"f1": ImageInfo("img.png", None, None, None)})
    store, collector = make(tmp_path, api)
    message = image_message(url="")
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    assert store.message_media(seq)[0]["reason"] == "no_source"


@pytest.mark.parametrize(
    "url,reason",
    [
        ("https://evil.example.org/a.png", "host_not_allowed"),
        ("https://qpic.cn.example.org/a.png", "host_not_allowed"),
        ("http://127.0.0.1/a.png", "unsafe_url"),
        ("http://[::1]/a.png", "unsafe_url"),
        ("ftp://gchat.qpic.cn/a.png", "unsafe_url"),
        ("https://user:pw@gchat.qpic.cn/a.png", "unsafe_url"),
        ("https://gchat.qpic.cn:8443/a.png", "unsafe_url"),
        ("file:///C:/secret.png", "unsafe_url"),
    ],
)
async def test_source_policy_refuses_non_qq_and_local_targets(url, reason):
    policy = ImageSourcePolicy(resolver=public_resolver)
    with pytest.raises(MediaUnavailable) as error:
        await policy.check_url(url)
    assert error.value.reason == reason


async def test_qq_host_resolving_to_private_address_is_refused():
    with pytest.raises(MediaUnavailable) as error:
        await ImageSourcePolicy(resolver=private_resolver).check_url(GCHAT_URL)
    assert error.value.reason == "unsafe_url"
    await ImageSourcePolicy(resolver=public_resolver).check_url(GCHAT_URL)


async def test_redirect_target_is_rechecked(tmp_path):
    download = FakeDownload(redirects={QQ_URL: "http://127.0.0.1:8080/admin"})
    store, collector = make(
        tmp_path, FakeApi({"f1": ImageInfo(None, QQ_URL, None, None)}), download
    )
    message = image_message(url="")
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    assert store.message_media(seq)[0]["reason"] == "unsafe_url"
    assert download.calls == [QQ_URL]


async def test_concurrency_and_pending_bounds(tmp_path):
    gate = asyncio.Event()
    download = FakeDownload(gate=gate)
    store, collector = make(tmp_path, None, download, max_concurrency=2, max_pending=3)
    message = image_message(files=("a", "b", "c", "d", "e"))
    seq = stored(store, message)
    assert collector.schedule(seq, message) == 5  # schedule returns at once
    for _ in range(20):
        await asyncio.sleep(0)
    assert download.peak == 2 and collector.pending == 3
    gate.set()
    await collector.drain()
    statuses = [s["status"] for s in store.message_media(seq)]
    assert statuses.count("available") == 3
    assert collector.counts["unavailable_queue_full"] == 2


async def test_recall_during_download_leaves_no_file(tmp_path):
    gate = asyncio.Event()
    store, collector = make(tmp_path, None, FakeDownload(gate=gate))
    intake = EventIntake(store, AccessPolicy(AccessSnapshot("BOT")), media=collector)
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await asyncio.sleep(0)
    assert store.revoke("BOT", "G1", "m1", "U1") == "revoked"
    gate.set()
    await collector.drain()
    assert collector.counts["slot_gone"] == 1
    assert store.message_media(seq) == [] and store.summary()["media_files"] == 0
    assert not list((tmp_path / "media").iterdir())
    assert intake.media is collector


async def test_recall_removes_file_only_without_other_references(tmp_path):
    store, collector = make(tmp_path)
    intake = EventIntake(store, AccessPolicy(AccessSnapshot("BOT")), media=collector)
    for mid in ("m1", "m2"):
        message = image_message(mid, (mid,))
        collector.schedule(stored(store, message), message)
    await collector.drain()
    recall = {
        "post_type": "notice",
        "notice_type": "group_recall",
        "self_id": "BOT",
        "group_id": "G1",
        "user_id": "U1",
        "message_id": "m1",
    }
    assert intake.handle(recall) == "recall_revoked"
    assert len(list((tmp_path / "media").iterdir())) == 1  # m2 still uses the same file
    assert intake.handle({**recall, "message_id": "m2"}) == "recall_revoked"
    assert not list((tmp_path / "media").iterdir())
    assert store.summary()["media_files"] == 0


async def test_aclose_marks_unfinished_slots(tmp_path):
    store, collector = make(tmp_path, None, FakeDownload(gate=asyncio.Event()))
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await asyncio.sleep(0)
    await collector.aclose(drain_timeout=0.01)
    assert store.message_media(seq)[0]["reason"] == "interrupted"


def test_sniff_known_types():
    assert sniff_image(png(2, 5)) == ("image/png", 2, 5)
    assert sniff_image(b"GIF89a" + struct.pack("<HH", 7, 8)) == ("image/gif", 7, 8)
    assert sniff_image(b"plain text") is None


async def prepare(collector, seqs, seconds=1):
    await collector.prepare(lambda: seqs, asyncio.get_running_loop().time() + seconds)


@pytest.mark.parametrize("failure", [TimeoutError(), ConnectionError(), 429, 503])
async def test_selected_image_recovers_once(tmp_path, failure):
    calls = []

    async def download(url, *_):
        calls.append(url)
        if len(calls) == 1:
            if isinstance(failure, Exception):
                raise failure
            return failure, b"", None
        return 200, png(), None

    store, collector = make(tmp_path, download=download)
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    assert store.message_media(seq)[0]["status"] == "unavailable"
    await prepare(collector, [seq])
    assert store.message_media(seq)[0]["status"] == "available"
    assert len(calls) == 2
    await prepare(collector, [seq])
    assert len(calls) == 2


@pytest.mark.parametrize("status", [403, 404])
@pytest.mark.parametrize("refresh", [True, False])
async def test_expired_url_requires_new_address(tmp_path, status, refresh):
    fresh = GCHAT_URL if refresh else QQ_URL

    class Api(FakeApi):
        async def get_image(self, file):
            self.calls.append(file)
            return ImageInfo(None, QQ_URL if len(self.calls) == 1 else fresh, None, None)

    calls = []

    async def download(url, *_):
        calls.append(url)
        return (status, b"", None) if url == QQ_URL else (200, png(), None)

    store, collector = make(tmp_path, Api(), download)
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    await prepare(collector, [seq])
    assert calls == ([QQ_URL, GCHAT_URL] if refresh else [QQ_URL])
    assert (store.message_media(seq)[0]["status"] == "available") == refresh


@pytest.mark.parametrize("status,body", [(401, b""), (400, b""), (200, b"html"), (200, b"")])
async def test_permanent_image_failure_has_no_recovery(tmp_path, status, body):
    download = FakeDownload({QQ_URL: body}, status=status)
    store, collector = make(tmp_path, download=download)
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    await prepare(collector, [seq])
    assert len(download.calls) == 1


async def test_failed_recovery_cools_down_and_does_not_retry_on_ingestion(tmp_path):
    download = FakeDownload(status=503)
    store, collector = make(tmp_path, download=download)
    now = [5.0]
    collector.clock = lambda: now[0]
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    await prepare(collector, [seq])
    await prepare(collector, [seq])
    other = image_message("m2")
    collector.schedule(stored(store, other), other)
    await collector.drain()
    assert len(download.calls) == 2
    now[0] += 61
    await prepare(collector, [seq])
    assert len(download.calls) == 3


async def test_shared_download_survives_waiter_cancel_and_one_root_recall(tmp_path):
    gate = asyncio.Event()
    download = FakeDownload(gate=gate)
    store, collector = make(tmp_path, download=download)
    seqs = []
    for mid in ("m1", "m2"):
        message = image_message(mid)
        seq = stored(store, message)
        seqs.append(seq)
        collector.schedule(seq, message)
    waiter = asyncio.create_task(prepare(collector, seqs))
    while not download.calls:
        await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    store.revoke("BOT", "G1", "m1", "U1")
    gate.set()
    await collector.drain()
    slot = store.message_media(seqs[1])[0]
    assert slot["status"] == "available" and (collector.root / slot["path"]).is_file()
    assert len(download.calls) == 1


async def test_missing_cache_file_is_reacquired(tmp_path):
    download = FakeDownload()
    store, collector = make(tmp_path, download=download)
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    (collector.root / store.message_media(seq)[0]["path"]).unlink()
    await prepare(collector, [seq])
    assert (collector.root / store.message_media(seq)[0]["path"]).is_file()
    assert len(download.calls) == 2


async def test_recovery_deadline_caps_hanging_download(tmp_path):
    calls = []

    async def download(url, *_):
        calls.append(url)
        if len(calls) == 1:
            return 503, b"", None
        await asyncio.Event().wait()

    store, collector = make(tmp_path, download=download)
    message = image_message()
    seq = stored(store, message)
    collector.schedule(seq, message)
    await collector.drain()
    await prepare(collector, [seq], seconds=0.02)
    await collector.drain(0.1)
    assert len(calls) == 2 and collector.pending == 0
    assert store.message_media(seq)[0]["reason"] == "download_timeout"
