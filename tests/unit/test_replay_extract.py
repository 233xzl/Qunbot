"""All test payloads are invented; tests never open a production database."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "replay_extract_under_test", ROOT / "scripts/replay/extract_snowluma.py"
)
assert SPEC and SPEC.loader
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def event(
    mid="q",
    *,
    group="fake-group",
    user="fake-person",
    bot="fake-bot",
    time=10000,
    message=None,
    post_type="message",
):
    return {
        "time": time,
        "post_type": post_type,
        "message_type": "group",
        "sub_type": "normal",
        "message_id": mid,
        "group_id": group,
        "group_name": "虚构测试群名",
        "self_id": bot,
        "user_id": user,
        "sender": {
            "user_id": user,
            "nickname": "虚构昵称甲",
            "card": "虚构名片甲",
            "role": "member",
        },
        "message": message
        if message is not None
        else [{"type": "text", "data": {"text": "虚构文本"}}],
    }


def row(mid="q", *, sequence=0, **kwargs):
    payload = event(mid, **kwargs)
    return replay.SourceRow(
        payload, payload["time"], sequence, f"row-{payload['group_id']}-{mid}-{sequence}"
    )


def at(bot="fake-bot"):
    return {"type": "at", "data": {"qq": bot}}


def reply(mid):
    return {"type": "reply", "data": {"id": mid}}


def text(value):
    return {"type": "text", "data": {"text": value}}


@pytest.fixture
def pseudo():
    return replay.Pseudonymizer(b"a" * 32)


def test_hmac_is_stable_and_domain_separated(pseudo):
    again = replay.Pseudonymizer(b"a" * 32)
    assert pseudo.alias("account", 123) == again.alias("account", "123")
    assert pseudo.alias("nickname", "虚构昵称") == again.alias("nickname", "虚构昵称")
    assert pseudo.alias("account", 123) != pseudo.alias("group", 123)
    assert pseudo.message_id(123, "group-a") != pseudo.message_id(123, "group-b")
    assert pseudo.alias("account", 123) != replay.Pseudonymizer(b"b" * 32).alias("account", 123)
    assert len(pseudo.alias("account", 123)) == 26


def test_key_reused_only_under_private_directory(tmp_path, monkeypatch):
    private = tmp_path / "private"
    monkeypatch.setattr(replay, "PRIVATE_ROOT", private)
    key_path = private / "synthetic-test.key"
    first = replay.load_key(key_path)
    assert len(first) == 32
    assert replay.load_key(key_path) == first
    with pytest.raises(ValueError):
        replay.load_key(tmp_path / "public.key")
    assert not (tmp_path / "public.key").exists()


def test_known_accounts_names_and_group_names_replaced_in_body(pseudo):
    invented = row(user="123456789", group="987654321", bot="111222333")
    pseudo.learn([invented])
    body = "虚构昵称甲（虚构名片甲）的QQ123456789，在虚构测试群名987654321，@111222333"
    cleaned = pseudo.text(body)
    for sensitive in (
        "123456789",
        "987654321",
        "111222333",
        "虚构昵称甲",
        "虚构名片甲",
        "虚构测试群名",
    ):
        assert sensitive not in cleaned
    assert pseudo.alias("account", "123456789") in cleaned
    assert pseudo.alias("nickname", "虚构昵称甲") in cleaned
    assert pseudo.alias("group", "987654321") in cleaned
    assert pseudo.alias("group_name", "虚构测试群名") in cleaned


def test_text_transport_secrets_and_filenames_removed(pseudo):
    raw = (
        "https://example.invalid/image?token=secret-value "
        "token=synthetic-token Bearer invented-secret-value sk-abcdefghijklmnop "
        "private-name.png fake@example.invalid C:\\private\\invented.txt"
    )
    cleaned = pseudo.text(raw)
    for forbidden in (
        "https://",
        "secret-value",
        "synthetic-token",
        "abcdefghijklmnop",
        "private-name.png",
        "fake@example.invalid",
        "invented.txt",
    ):
        assert forbidden not in cleaned


def test_only_real_top_level_at_triggers():
    assert replay.top_level_at_bot(event(message=[at()]))
    nested = {"type": "reply", "data": {"id": "prior", "message": [at()]}}
    forwarded = {"type": "node", "data": {"content": [at()]}}
    assert not replay.top_level_at_bot(event(message=[nested, forwarded]))
    fake = event(message=[text("[CQ:at,qq=fake-bot] @fake-bot")])
    fake["raw_message"] = "[CQ:at,qq=fake-bot]"
    assert not replay.top_level_at_bot(fake)
    assert not replay.top_level_at_bot(event(message=[at("all")]))
    assert not replay.top_level_at_bot(event(message=[at("other-bot")]))
    assert not replay.top_level_at_bot(event(user="fake-bot", message=[at()]))
    assert not replay.top_level_at_bot(event(message=[at()], post_type="message_sent"))
    private = event(message=[at()])
    private["message_type"] = "private"
    assert not replay.top_level_at_bot(private)
    malformed = event(message="[CQ:at,qq=fake-bot]")
    assert not replay.top_level_at_bot(malformed)


def test_nested_at_and_reply_relationship_remain_nested_after_sanitization(pseudo):
    original = event(
        message=[{"type": "reply", "data": {"id": "prior", "message": [at()]}}, text("仅为引用")]
    )
    frozen = copy.deepcopy(original)
    cleaned = pseudo.event(original)
    assert cleaned["message"][0]["data"]["id"] == pseudo.message_id("prior", "fake-group")
    assert cleaned["message"][0]["data"]["message"][0]["type"] == "at"
    assert not replay.top_level_at_bot(cleaned)
    assert original == frozen
    assert replay.top_level_at_bot(pseudo.event(event(message=[at()])))


def test_media_opaque_payloads_never_export_locators_or_filenames(pseudo):
    original = event(
        message=[
            {
                "type": "image",
                "data": {
                    "file": "private-face.png",
                    "url": "https://example.invalid/x",
                    "key": "secret-token",
                    "width": 64,
                    "summary": "private nickname",
                },
            },
            {"type": "file", "data": {"name": "private-document.pdf", "file_id": "real-file-id"}},
            {"type": "json", "data": {"data": '{"token":"secret-token"}'}},
        ]
    )
    cleaned = pseudo.event(original)
    serialized = json.dumps(cleaned)
    for forbidden in (
        "private-face",
        "https://",
        "secret-token",
        "private nickname",
        "private-document",
        "real-file-id",
    ):
        assert forbidden not in serialized
    image = cleaned["message"][0]["data"]
    assert image["availability"] == "unavailable"
    assert image["content_sha256"] is None
    assert image["local_metadata"] == {"width": 64}
    assert cleaned["message"][2]["type"] == "redacted"


def test_context_uses_more_than_80_when_30_minutes_are_busy():
    history = [row(str(i), time=10000 + i) for i in range(100)]
    anchor = row("anchor", time=10100)
    assert len(replay.select_context([*history, anchor], anchor)) == 100


def test_context_uses_80_when_30_minutes_are_sparse():
    history = [row(str(i), time=i * 3600) for i in range(100)]
    anchor = row("anchor", time=100 * 3600)
    assert replay.select_context([*history, anchor], anchor) == history[-80:]


def test_context_boundary_ties_scope_and_missing_history():
    anchor = row("anchor", time=10000, sequence=10)
    earlier = row("earlier", time=10000, sequence=9)
    future = row("future", time=10000, sequence=11)
    other_group = row("other", time=9999, group="foreign-group")
    threshold = row("threshold", time=8200)
    rows = sorted([earlier, anchor, future, other_group, threshold], key=lambda r: r.order)
    assert replay.select_context(rows, anchor, minimum_messages=0) == [threshold, earlier]
    assert replay.select_context([anchor], anchor) == []


def test_old_reply_association_never_crosses_group_and_timing_is_uncertain():
    anchor = row("anchor", message=[at()])
    old = row(
        "old", time=9999, user="fake-bot", post_type="message_sent", message=[reply("anchor")]
    )
    foreign = row(
        "foreign",
        time=10001,
        group="foreign-group",
        user="fake-bot",
        post_type="message_sent",
        message=[reply("anchor")],
    )
    uncertain = row("uncertain", time=10002, user="fake-bot", post_type="message_sent")
    linked = row(
        "linked", time=10003, user="fake-bot", post_type="message_sent", message=[reply("anchor")]
    )
    chain = row(
        "chain", time=10004, user="fake-bot", post_type="message_sent", message=[reply("linked")]
    )
    late = row(
        "late", time=12000, user="fake-bot", post_type="message_sent", message=[reply("anchor")]
    )
    rows = [old, anchor, foreign, uncertain, linked, chain, late]
    future, results = replay.collect_followup(rows, anchor)
    assert future == [uncertain, linked, chain]
    assert results == [
        (uncertain, "uncertain"),
        (linked, "explicit_reply"),
        (chain, "explicit_reply_chain"),
    ]


def test_duplicate_ids_are_not_resolved():
    one = row("duplicate", sequence=1)
    two = row("duplicate", sequence=2)
    assert replay.build_index([one, two]) == {}
    foreign = row("duplicate", group="foreign-group")
    assert len(replay.build_index([one, foreign])) == 2


def test_reference_ancestors_recovered_beyond_80_without_cross_group(pseudo):
    root = row("root", time=100)
    foreign = row("missing", time=101, group="foreign-group")
    history = [row(str(i), time=1000 + i * 60) for i in range(100)]
    anchor = row("anchor", time=10000, message=[reply("root"), reply("missing"), at()])
    rows = [root, foreign, *history, anchor]
    pseudo.learn(rows)
    case = replay.make_case(rows, anchor, pseudo, ["引用追问"])
    ids = {e["message_id"] for e in case["events"]}
    assert pseudo.message_id("root", "fake-group") in ids
    assert all(e["group_id"] == pseudo.alias("group", "fake-group") for e in case["events"])
    assert case["provenance"]["window"]["base_context_count"] == 80
    assert case["provenance"]["window"]["reference_ancestor_count"] == 1
    assert [r["status"] for r in case["provenance"]["reply_relations"]] == [
        "resolved",
        "unresolved",
    ]


def test_case_never_calls_temporal_candidate_measured_reply(pseudo):
    anchor = row("anchor", message=[at()])
    candidate = row("later", time=10001, user="fake-bot", post_type="message_sent")
    case = replay.make_case([anchor, candidate], anchor, pseudo, ["普通问答"])
    assert case["legacy_result"]["status"] == "uncertain"
    assert case["legacy_result"]["responses"][0]["observed_delay_seconds"] is None
    assert case["legacy_result"]["latency_ms"] is None
    assert case["legacy_result"]["search_count"] is None
    assert case["annotations"] == replay.empty_annotations()
    assert not case["provenance"]["window"]["followup_complete"]


def test_selection_deduplicates_text_and_tags_are_candidates_only():
    history = [row("img", time=9999, message=[{"type": "image", "data": {"file": "invented"}}])]
    first = row("first", message=[at(), text("这个是什么梗，记住它")])
    duplicate = row("duplicate", time=10001, message=[at(), text("这个是什么梗，记住它")])
    second = row("second", time=11000, message=[at(), text("请搜索网页")])
    rows = [*history, first, duplicate, second]
    selected, tags, stats = replay.choose_anchors(rows, count=3)
    assert len(selected) == 2
    assert {"先发图", "指代词", "梗", "教学"} <= set(tags[first.source_row])
    assert "网页" in tags[second.source_row]
    assert stats["candidate_pool_count"] == 3


def test_sqlite_reader_is_read_only_and_double_checks_scope(tmp_path, monkeypatch):
    db = tmp_path / "synthetic.db"
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE messages (message_hash INTEGER PRIMARY KEY, timestamp INTEGER, "
            "sequence INTEGER, sequence_authoritative INTEGER, data TEXT, is_group INTEGER, session_id TEXT)"
        )
        for i, (session, payload) in enumerate(
            [
                ("fake-group", event("valid", message=[at()])),
                ("foreign-group", event("not-selected", group="foreign-group")),
                ("fake-group", event("mismatch", group="foreign-group")),
                ("fake-group", event("not-array", message="[CQ:at,qq=fake-bot]")),
            ]
        ):
            con.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?)",
                (i, 10000000, i, 1, json.dumps(payload), 1, session),
            )
    before = hashlib.sha256(db.read_bytes()).digest()
    original_connect = sqlite3.connect
    captured = []

    def spy_connect(database, **kwargs):
        captured.append((database, kwargs))
        return original_connect(database, **kwargs)

    monkeypatch.setattr(replay.sqlite3, "connect", spy_connect)
    rows, stats = replay.read_rows(db, "fake-group")
    assert len(rows) == 1
    assert stats["payload_scope_mismatch_skipped"] == 1
    assert stats["non_array_or_non_message_skipped"] == 1
    assert captured[0][0].endswith("?mode=ro")
    assert captured[0][1]["uri"] is True
    assert hashlib.sha256(db.read_bytes()).digest() == before


def test_real_output_cannot_escape_raw_root(tmp_path):
    with pytest.raises(ValueError):
        replay.require_under(tmp_path / "public.json", replay.RAW_ROOT)
    with pytest.raises(ValueError):
        replay.require_under(replay.RAW_ROOT, replay.RAW_ROOT)
    assert (
        replay.require_under(replay.RAW_ROOT / "test-batch", replay.RAW_ROOT).parent
        == replay.RAW_ROOT
    )


def test_review_marks_chat_untrusted_and_does_not_render_active_media(pseudo):
    anchor = row(
        "anchor", message=[at(), text("<script>fake</script> ![x](https://example.invalid/p)")]
    )
    case = replay.make_case([anchor], anchor, pseudo, ["普通问答"])
    review = replay.markdown_review(case)
    assert "不可信材料" in review
    assert "<script>" not in review
    assert "https://" not in review
    assert "![x](" not in review


def test_schema_itself_and_handwritten_cases_validate():
    schema = json.loads((ROOT / "tests/replay/schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    cases = sorted((ROOT / "tests/replay/cases").glob("synthetic-*.json"))
    assert len(cases) >= 30
    for path in cases:
        case = json.loads(path.read_text(encoding="utf-8"))
        validator.validate(case)
        assert case["source"] == "synthetic"


def test_schema_validates_generated_case_and_rejects_media_url(pseudo):
    schema = json.loads((ROOT / "tests/replay/schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    anchor = row("anchor", message=[at(), {"type": "image", "data": {"file": "invented.png"}}])
    case = replay.make_case([anchor], anchor, pseudo, ["随附图片"])
    validator.validate(case)
    case["events"][0]["message"][1]["data"]["url"] = "https://example.invalid/private"
    assert not validator.is_valid(case)


def test_schema_rejects_real_sample_marked_safe_to_commit(pseudo):
    schema = json.loads((ROOT / "tests/replay/schema.json").read_text(encoding="utf-8"))
    anchor = row("anchor", message=[at()])
    case = replay.make_case([anchor], anchor, pseudo, ["普通问答"])
    case["provenance"]["storage_policy"] = "synthetic_versionable"
    assert not jsonschema.Draft202012Validator(schema).is_valid(case)
