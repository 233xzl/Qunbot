"""Batch builder, media fetch and legacy linking; all inputs are invented, HTTP is mocked."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import sys
import urllib.error
import zlib
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/replay" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


batch = load("replay_batch_under_test", "build_batch.py")
media = batch.media
legacy = batch.legacy
base = batch.base


def png(width=3, height=2, salt=b"") -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"tEXt", b"c\x00" + salt)
        + chunk(b"IEND", b"")
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def call(self, action, payload):
        self.calls.append((action, payload))
        value = self.responses[payload["file"]]
        if isinstance(value, Exception):
            raise value
        return value


class NoWait:
    def wait(self):
        pass


def fetcher(tmp_path, client, contents, max_bytes=1000):
    downloads = []

    def download(url, cap):
        downloads.append(url)
        value = contents[url]
        if isinstance(value, Exception):
            raise value
        return value

    instance = media.MediaFetcher(
        client,
        tmp_path / "media",
        max_bytes=max_bytes,
        download=download,
        download_limiter=NoWait(),
    )
    return instance, downloads


def test_media_fetch_success_writes_content_addressed_file(tmp_path):
    content = png(5, 7)
    client = FakeClient({"key-a": {"url": "https://example.invalid/a", "file_size": "99"}})
    instance, _ = fetcher(tmp_path, client, {"https://example.invalid/a": content})
    ref = instance.fetch("media_a", "key-a")
    digest = hashlib.sha256(content).hexdigest()
    assert ref == {
        "status": "available",
        "reason": "fetched_via_onebot_get_image",
        "sha256": digest,
        "mime": "image/png",
        "bytes": len(content),
        "width": 5,
        "height": 7,
        "path": f"media/{digest}.png",
    }
    assert (tmp_path / ref["path"]).read_bytes() == content
    assert client.calls == [("get_image", {"file": "key-a"})]
    instance.save_index()
    stored = (tmp_path / "media/index.json").read_text(encoding="utf-8")
    assert "key-a" not in stored and "example.invalid" not in stored


def test_media_failures_are_unavailable_and_transient_ones_not_cached(tmp_path):
    client = FakeClient(
        {
            "broken": media.OneBotError("get_image returned failure"),
            "nourl": {"file_size": "1"},
            "http": {"url": "https://example.invalid/h"},
            "text": {"url": "https://example.invalid/t"},
        }
    )
    http_error = urllib.error.HTTPError("https://example.invalid/h", 400, "bad", {}, None)
    instance, _ = fetcher(
        tmp_path,
        client,
        {"https://example.invalid/h": http_error, "https://example.invalid/t": b"not an image"},
    )
    assert instance.fetch("m1", "broken")["reason"] == "get_image_failed"
    assert instance.fetch("m2", "nourl")["reason"] == "no_download_url"
    assert instance.fetch("m3", "http")["reason"] == "download_http_error"
    assert instance.fetch("m4", "text")["reason"] == "not_image"
    assert instance.fetch("m5", None)["reason"] == "missing_file_key"
    unavailable = instance.fetch("m1", "broken")
    assert unavailable["status"] == "unavailable" and unavailable["sha256"] is None
    # get_image failure retried (2 calls); permanent not_image is cached.
    assert [c[1]["file"] for c in client.calls].count("broken") == 2
    instance.fetch("m4", "text")
    assert [c[1]["file"] for c in client.calls].count("text") == 1


def test_media_oversize_by_declared_size_and_by_content(tmp_path):
    client = FakeClient(
        {
            "big-declared": {"url": "https://example.invalid/b", "file_size": "5000"},
            "big-body": {"url": "https://example.invalid/c", "file_size": "10"},
        }
    )
    instance, downloads = fetcher(
        tmp_path,
        client,
        {"https://example.invalid/c": media.OversizeError("over")},
        max_bytes=1000,
    )
    assert instance.fetch("m1", "big-declared")["reason"] == "oversize"
    assert downloads == []
    assert instance.fetch("m2", "big-body")["reason"] == "oversize"
    assert not (tmp_path / "media").exists() or not list((tmp_path / "media").glob("*.png"))


def test_default_download_caps_body_and_declared_length():
    class Response(io.BytesIO):
        def __init__(self, data, length=None):
            super().__init__(data)
            self.headers = {"Content-Length": str(length)} if length is not None else {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(body, length=None):
        return lambda request, timeout: Response(body, length)

    url = "https://example.invalid/x"
    assert media.default_download(url, 10, opener=opener(b"12345")) == b"12345"
    with pytest.raises(media.OversizeError):
        media.default_download(url, 10, opener=opener(b"1", length=11))
    with pytest.raises(media.OversizeError):
        media.default_download(url, 10, opener=opener(b"x" * 11))
    with pytest.raises(ValueError):
        media.default_download("file:///etc/passwd", 10, opener=opener(b""))


def test_media_dedup_by_sha256_and_cache_skips_refetch(tmp_path):
    content = png(1, 1)
    client = FakeClient(
        {
            "k1": {"url": "https://example.invalid/1"},
            "k2": {"url": "https://example.invalid/2"},
        }
    )
    instance, downloads = fetcher(
        tmp_path,
        client,
        {"https://example.invalid/1": content, "https://example.invalid/2": content},
    )
    first = instance.fetch("media_1", "k1")
    second = instance.fetch("media_2", "k2")
    assert first["sha256"] == second["sha256"]
    assert len(list((tmp_path / "media").glob("*.png"))) == 1
    assert instance.stats["dedup_hits"] == 1
    instance.save_index()
    rerun, rerun_downloads = fetcher(tmp_path, FakeClient({}), {})
    assert rerun.fetch("media_1", "k1") == first
    assert rerun_downloads == [] and rerun.stats["cache_hits"] == 1
    assert len(downloads) == 2


def test_onebot_client_refuses_non_read_actions_and_rate_limits():
    sleeps = []
    clock = iter([0.0, 0.05, 0.3])
    limiter = media.RateLimiter(5, clock=lambda: next(clock), sleep=sleeps.append)
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert sleeps == pytest.approx([0.15, 0.1])

    def opener(request, timeout):
        raise AssertionError("must not be called")

    client = media.OneBotReadClient("http://127.0.0.1:1", "t", opener=opener, limiter=NoWait())
    for action in ("send_group_msg", "set_group_ban", "delete_msg"):
        with pytest.raises(ValueError):
            client.call(action, {})


def test_onebot_client_sends_bearer_and_parses_failure():
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(request, timeout):
        seen.append(request)
        body = {"status": "ok", "retcode": 0, "data": {"url": "u"}}
        if json.loads(request.data)["file"] == "bad":
            body = {"status": "failed", "retcode": 1404}
        return Response(json.dumps(body).encode())

    client = media.OneBotReadClient("http://127.0.0.1:1/", "tok", opener=opener, limiter=NoWait())
    assert client.call("get_image", {"file": "ok"}) == {"url": "u"}
    assert seen[0].get_header("Authorization") == "Bearer tok"
    assert seen[0].full_url == "http://127.0.0.1:1/get_image"
    with pytest.raises(media.OneBotError) as caught:
        client.call("get_image", {"file": "bad"})
    assert "tok" not in str(caught.value)


def test_sniff_image_dimensions():
    assert media.sniff_image(png(9, 4)) == ("image/png", 9, 4)
    assert media.sniff_image(b"GIF89a" + struct.pack("<HH", 6, 8) + b"\0" * 4) == (
        "image/gif",
        6,
        8,
    )
    jpeg = b"\xff\xd8" + b"\xff\xe0\x00\x04ab" + b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", 3, 5)
    assert media.sniff_image(jpeg + b"\0" * 10) == ("image/jpeg", 5, 3)
    assert media.sniff_image(b"plain text") is None


# ---------------------------------------------------------------------------
# Legacy linking


def row(mid, time, *, user="person", message=None, group="g"):
    event = {
        "time": time,
        "post_type": "message_sent" if user == "bot" else "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": mid,
        "group_id": group,
        "self_id": "bot",
        "user_id": user,
        "sender": {"user_id": user, "nickname": "虚构"},
        "message": message if message is not None else [{"type": "text", "data": {"text": "x"}}],
    }
    return base.SourceRow(event, float(time), 0, f"row-{group}-{mid}")


def at():
    return {"type": "at", "data": {"qq": "bot"}}


def reply(mid):
    return {"type": "reply", "data": {"id": mid}}


def test_bot_reply_prefers_reply_segment_then_first_unquoted_message():
    a1 = row("a1", 100, message=[at()])
    a2 = row("a2", 110, message=[at()])
    a3 = row("a3", 400, message=[at()])
    a4 = row("a4", 1000, message=[at()])
    plain_for_a2 = row("b0", 115, user="bot")
    quoted_a1 = row("b1", 130, user="bot", message=[reply("a1")])
    quotes_other = row("b2", 405, user="bot", message=[reply("zzz")])
    plain_for_a3 = row("b3", 410, user="bot")
    late = row("b4", 1400, user="bot", message=[reply("a4")])
    other_group = row("b5", 1001, user="bot", message=[reply("a4")], group="other")
    rows = [a1, a2, plain_for_a2, quoted_a1, a3, quotes_other, plain_for_a3, a4, late, other_group]
    links = legacy.link_bot_replies(rows, [a1, a2, a3, a4])
    assert links["row-g-a1"]["rule"] == "reply_segment_to_anchor"
    assert links["row-g-a1"]["row"] is quoted_a1
    assert links["row-g-a2"]["rule"] == "first_unquoted_bot_message_after_anchor"
    assert links["row-g-a2"]["row"] is plain_for_a2
    assert links["row-g-a2"]["confidence"] == "medium"
    assert links["row-g-a3"]["row"] is plain_for_a3
    assert "row-g-a4" not in links


def diag_request(request_id, start, end, *, phases=("local_retrieval",), finished=True, **extra):
    rows = [
        {"time": start + 0.1 * i, "request": request_id, "phase": p} for i, p in enumerate(phases)
    ]
    if finished:
        rows.append(
            {
                "time": end,
                "request": request_id,
                "phase": "finished",
                "ms": round((end - start) * 1000),
                "search": extra.get("search", 0),
                "fetch": extra.get("fetch", 0),
            }
        )
    return legacy.DiagRequest(request_id, rows)


def test_diagnostics_matching_corrects_clock_offset_and_marks_ambiguity():
    # Diagnostics clock runs 22 s behind message time.
    windows = [
        legacy.AnchorWindow(f"a{i}", 1000.0 + 500 * i, 1000.0 + 500 * i + 20) for i in range(4)
    ]
    requests = [
        diag_request(f"r{i}", 1000 + 500 * i - 22 + 0.5, 1000 + 500 * i + 20 - 22 - 1.2)
        for i in range(4)
    ]
    # A competing request from another group finishing at nearly the same time.
    requests.append(diag_request("other", 1500 - 22 + 0.5, 1500 + 20 - 22 - 0.8))
    requests.sort(key=lambda r: r.start)
    offsets = legacy.estimate_offsets(windows, requests)
    assert offsets["a0"]["offset"] == pytest.approx(-23.2, abs=0.5)
    links = legacy.link_diagnostics(windows, requests, offsets)
    assert {k: v["request"].request_id for k, v in links.items()} == {
        "a0": "r0",
        "a1": "r1",
        "a2": "r2",
        "a3": "r3",
    }
    assert links["a0"]["confidence"] == "high"
    assert links["a1"]["confidence"] == "low"
    assert links["a1"]["candidate_count"] == 2
    # Without offset correction nothing matches the skewed clock.
    zero = {w.key: {"offset": 0.0, "seed_count": 0} for w in windows}
    assert legacy.link_diagnostics(windows, requests, zero) == {}


def test_request_summary_failure_flags_and_provider_scope():
    request = diag_request(
        "r",
        0,
        70,
        phases=("vision_fallback", "vision_failed", "local_retrieval", "web", "web_blocked"),
        search=2,
    )
    request.rows[1]["reason"] = "TimeoutError"
    request.rows[2]["ids"] = ["C0001", "has space"]
    summary = legacy.summarize_request(request)
    assert summary["latency_ms"] == 70000 and summary["search_count"] == 2
    assert summary["failure_flags"] == ["vision_failed", "web_blocked", "slow_over_60s"]
    assert summary["failure_reasons"] == ["TimeoutError"]
    assert summary["retrieved_knowledge_ids"] == ["C0001"]
    assert legacy.image_extraction(summary)["status"] == "failed"
    assert legacy.umo_in_group("bot:GroupMessage:111_222", "222")
    assert not legacy.umo_in_group("bot:GroupMessage:111_2223", "222")
    calls = [
        legacy.ProviderCall(1, 5, True, "completed", "m", 10, 2, 3),
        legacy.ProviderCall(2, 6, False, "completed", "m", 99, 0, 9),
        legacy.ProviderCall(50, 60, True, "completed", "m", 1, 1, 1),
    ]
    provider = legacy.provider_calls_between(calls, 0, 10, "diagnostics_request_span")
    assert provider["call_count"] == 1 and provider["other_group_overlap_count"] == 1
    assert provider["token_input_other"] == 10
    assert (
        legacy.adjust_diag_confidence("medium", {"call_count": 1, "other_group_overlap_count": 0})
        == "high"
    )
    assert (
        legacy.adjust_diag_confidence("high", {"call_count": 0, "other_group_overlap_count": 2})
        == "medium"
    )
    link = legacy.combine_link(None, None, provider, True)
    assert link["status"] == "partial"
    assert "no_bot_message_in_reply_window" in link["missing_reasons"]


# ---------------------------------------------------------------------------
# Quota selection


class Anchor:
    def __init__(self, name, timestamp):
        self.source_row = name
        self.timestamp = float(timestamp)
        self.order = (self.timestamp, 0, name)


def test_quota_selection_keeps_existing_and_reports_shortfall():
    anchors = [Anchor(f"a{i}", i * 1000) for i in range(12)]
    tags = {a.source_row: {"普通问答"} for a in anchors}
    for name in ("a0", "a1", "a2", "a3"):
        tags[name] = {"梗"}
    tags["a4"] = {"教学"}
    tags["a5"] = {"梗", "教学"}
    tags["a6"] = {"引用机器人回答", "引用追问"}
    tags["a7"] = {"梗"}
    quotas = {"本地梗解释": 3, "主人教学": 3, "引用机器人回答的追问": 1}
    keep = {"a5", "a11"}
    selected, assignment, stats = batch.select_with_quotas(
        anchors, tags, keep, quotas, target=8, signature=lambda a: a.source_row
    )
    names = {a.source_row for a in selected}
    assert keep <= names
    # The kept anchor goes to the scarcer category it is eligible for.
    assert assignment["a5"] == "主人教学"
    assert assignment["a11"] == batch.OVERFLOW
    assert stats["主人教学"] == {
        "quota": 3,
        "selected": 2,
        "shortfall": 1,
        "eligible_candidates": 2,
    }
    assert stats["本地梗解释"]["selected"] == 3
    assert stats["引用机器人回答的追问"]["selected"] == 1
    assert len(selected) == 8
    assert [a.order for a in selected] == sorted(a.order for a in selected)
    with pytest.raises(ValueError):
        batch.select_with_quotas(anchors, tags, {"missing"}, quotas, 8, lambda a: "")


def test_quota_selection_skips_duplicate_signatures():
    anchors = [Anchor(f"a{i}", i * 1000) for i in range(3)]
    tags = {a.source_row: {"梗"} for a in anchors}
    selected, _, stats = batch.select_with_quotas(
        anchors, tags, set(), {"本地梗解释": 3}, target=3, signature=lambda a: "same"
    )
    assert len(selected) == 1 and stats["本地梗解释"]["shortfall"] == 2


def test_quota_rules_map_candidate_tags():
    assert "旧链路失败或超时" in batch.quota_categories({"旧链路图片失败"})
    assert "先发图后@" in batch.quota_categories({"引用图片", "引用追问"})
    assert "指代不明" not in batch.quota_categories({"指代词", "先发图"})
    assert "本地梗解释" in batch.quota_categories({"旧链路仅本地检索"})
    assert "本地梗解释" not in batch.quota_categories({"旧链路仅本地检索", "短跟进"})
    assert batch.legacy_tags(None, False) == {"旧链路未回复"}


# ---------------------------------------------------------------------------
# Case enrichment and schema


@pytest.fixture
def pseudo():
    return base.Pseudonymizer(b"z" * 32)


def build_case(pseudo, tmp_path):
    image = {"type": "image", "data": {"file": "invented-key", "sub_type": 0}}
    earlier = row("img", 900, message=[image])
    anchor = row("q", 1000, message=[reply("img"), at()])
    answer = row("ans", 1020, user="bot", message=[reply("q")])
    rows = [earlier, anchor, answer]
    pseudo.learn(rows)
    case = base.make_case(rows, anchor, pseudo, ["x"])
    batch.finalize_case(case, ["配额:先发图后@", "引用图片"])
    bot = legacy.link_bot_replies(rows, [anchor])
    request = diag_request("req", 999.5, 1018.8, phases=("vision", "local_retrieval"))
    window = legacy.AnchorWindow(anchor.source_row, 1000.0, 1020.0)
    diag = legacy.link_diagnostics(
        [window], [request], {anchor.source_row: {"offset": 0.0, "seed_count": 1}}
    )
    calls = [legacy.ProviderCall(1001, 1015, True, "completed", "model-x", 5, 1, 2)]
    batch.fill_legacy(
        case,
        anchor,
        pseudo,
        bot.get(anchor.source_row),
        diag.get(anchor.source_row),
        {"offset": 0.0, "spread": 0.0},
        calls,
        (0.0, 5000.0),
    )
    client = FakeClient({"invented-key": {"url": "https://example.invalid/i"}})
    instance, _ = fetcher(tmp_path, client, {"https://example.invalid/i": png(2, 2)})
    keys = batch.raw_image_keys(rows, pseudo)
    refs = {mid: instance.fetch(mid, key) for mid, key in keys.items()}
    batch.apply_media(case, refs)
    return case


def test_enriched_case_validates_and_exports_no_locators(pseudo, tmp_path):
    case = build_case(pseudo, tmp_path)
    schema = json.loads((ROOT / "tests/replay/schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    validator.validate(case)
    result = case["legacy_result"]
    assert result["link"]["status"] == "linked"
    assert result["bot_reply"]["rule"] == "reply_segment_to_anchor"
    assert result["diagnostics"]["confidence"] == "high"
    assert result["latency_ms"] == 19300
    assert result["provider_calls"]["call_count"] == 1
    assert result["diagnostics_status"] == "joined"
    image = next(
        seg["data"] for e in case["events"] for seg in e["message"] if seg["type"] == "image"
    )
    assert image["availability"] == "available" and image["media_ref"]["width"] == 2
    serialized = json.dumps(case)
    for forbidden in ("invented-key", "example.invalid", '"req"'):
        assert forbidden not in serialized
    broken = json.loads(serialized)
    for event in broken["events"]:
        for seg in event["message"]:
            if seg["type"] == "image":
                seg["data"]["content_sha256"] = None
    assert not validator.is_valid(broken)


def test_reannotation_only_for_clarify_with_now_available_images(pseudo, tmp_path):
    case = build_case(pseudo, tmp_path)
    image_message = pseudo.message_id("img", "g")
    clarify = {"expected_route": "clarify", "required_context_ids": [image_message]}
    assert batch.reannotation_status(case, clarify) == "images_now_available"
    assert batch.reannotation_status(case, {"expected_route": "web"}) is None
    for event in case["events"]:
        for seg in event["message"]:
            if seg["type"] == "image":
                seg["data"]["availability"] = "unavailable"
    assert batch.reannotation_status(case, clarify) is None
