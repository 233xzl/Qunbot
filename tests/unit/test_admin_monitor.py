from datetime import datetime

import pytest

from group_agent.admin import monitor
from group_agent.contracts import Message
from group_agent.store import Store
from group_agent.usage import CN_TZ


NOW = datetime(2026, 11, 19, 1, 30, tzinfo=CN_TZ).timestamp()


@pytest.fixture
def store():
    db = Store()
    try:
        yield db
    finally:
        db.close()


def charge(store, gid, service, when, cost):
    store.con.execute(
        "INSERT INTO usage_ledger(run_id,group_id,day,created_at,provider,model,service,cost_cny) "
        "VALUES ('r',?,'d',?,'provider','model',?,?)",
        (gid, when, service, cost),
    )


def test_usage_boundaries_and_grouping(store):
    for gid in ("A", "B"):
        store.con.execute("INSERT INTO group_names VALUES ('BOT',?,'同名群',0)", (gid,))
    for args in [
        ("A", "generate", NOW - 86401, 1),
        ("A", "judge", NOW - 86400, 2),
        ("A", "search", NOW - 7200, 4),
        ("B", "fetch", NOW - 5400, 8),
        (None, None, NOW - 1, 16),
    ]:
        charge(store, *args)
    for period, cost, buckets in [
        ("today", 24, 2),
        ("24h", 30, 24),
        ("yesterday", 7, 24),
        ("7d", 31, 7),
        ("14d", 31, 14),
        ("30d", 31, 30),
    ]:
        for dimension in ["total", "model", "group"]:
            u = monitor.usage(store.con, "BOT", period, dimension, None, NOW)
            assert len(u["days"]) == buckets
            assert sum(r["cost"] for r in u["days"]) == u["cost"] == cost
            assert sum(sum(s["values"]) for s in u["series"]) == cost
    u = monitor.usage(store.con, "BOT", "7d", "group", None, NOW)
    assert sorted((s["label"], s["cost"]) for s in u["series"]) == [
        ("同名群", 7),
        ("同名群", 8),
        ("未归属", 16),
    ]
    u = monitor.usage(store.con, "BOT", "7d", "model", None, NOW)
    assert {s["label"].split(" · ")[1] for s in u["series"]} == {"搜索", "获取", "对话", "其他"}


def test_shared_request_projection_and_old_names(store):
    segs = (
        {"type": "reply", "data": {"id": "old"}},
        {"type": "at", "data": {"qq": "BOT"}},
        {"type": "text", "data": {"text": "  帮我查一下天气"}},
    )
    m = Message("BOT", "A", "msg", "U", segs)
    seq, *_ = store.ingest_request(m, member_name="春风")
    store.create_run("r", seq, m, state="succeeded", cost_mode="normal", now=NOW - 1)
    store.record_outbound("r", m, "晴天")
    store.outbound_update("r", NOW, state="recorded_not_sent")
    for amount in [0.001, 0.002]:
        charge(store, "A", "search", NOW - 1, amount)
    store.con.execute("INSERT INTO group_names VALUES ('BOT','A','读书群',0)")
    (row,) = monitor.requests(store.con, bounds=monitor.window("today", NOW))
    (detail,) = monitor.requests(store.con, run_id="r")
    assert row["preview"] == detail["message"] == "帮我查一下天气"
    assert row["cost"] == detail["cost"] == pytest.approx(0.003)
    assert (detail["group_name"], detail["member_name"], detail["reply_text"]) == (
        "读书群",
        "春风",
        "晴天",
    )
    assert detail["delivery"] == "recorded_not_sent"
    store.revoke("BOT", "A", "msg", "U", NOW)
    assert monitor.requests(store.con)[0]["preview"] == "消息已撤回"
    store.con.executescript("DROP TABLE group_names; DROP TABLE request_members;")
    (d,) = monitor.requests(store.con, run_id="r")
    assert d["group_name"] == "未知群聊" and d["member_name"] is None


def test_empty_and_image():
    assert monitor.message_text('[{"type":"image","data":{"url":"private"}}]', 0, "BOT") == "[图片]"
    assert monitor.message_text(None, 0, "BOT") == "消息已清理"
    assert not monitor.usage(None, "BOT", "24h", "model", None, NOW)["series"]
