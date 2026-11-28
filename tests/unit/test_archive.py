"""Text archive and history tools (`archive.py`), with fake models only."""

from dataclasses import replace
import json
import sqlite3

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from group_agent import archive
from group_agent.config import ArchiveSpec, ConfigError, parse_settings
from group_agent.config_service import validate_keys
from group_agent.live_agent import LiveAgent
from group_agent.policy import RuntimeLimits

from test_live_replay import USAGE, live_config
from test_orchestrator import Harness, ev, recall, wall

QUICK = RuntimeLimits(group_min_interval_seconds=0, group_per_minute=1000)


def archived(h, group="G1"):
    return [
        dict(r)
        for r in h.store.con.execute(
            "SELECT * FROM archive WHERE group_id=? ORDER BY time, rowid", (group,)
        )
    ]


def on(h, **spec):
    h.orch._base_settings = replace(h.settings, archive=ArchiveSpec(enabled=True, **spec))


def chat(mid, user="U1", group="G1", text="闲聊", t=None, reply=None):
    return ev(mid, user=user, group=group, text=text, at=False, t=t, reply=reply)


# -- writing ----------------------------------------------------------------------------


async def test_archive_is_off_by_default():
    h = Harness(limits=QUICK)
    h.feed(chat("m1", t=int(wall())))
    assert archived(h) == []


async def test_chat_is_archived_as_text_and_recall_deletes_it():
    h = Harness(limits=QUICK)
    on(h)
    t = int(wall())
    event = chat("m1", text="原神 4.0 什么时候", t=t)
    event["message"] += [
        {"type": "image", "data": {"file": "x"}},
        {"type": "at", "data": {"qq": "U2"}},
    ]
    h.feed(event)
    h.feed(chat("m2", user="U2", text="下周", t=t + 1, reply="m1"))
    rows = archived(h)
    assert [r["text"] for r in rows] == ["原神 4.0 什么时候[图片]@{U2}", "下周"]
    assert rows[1]["reply_to"] == "m1" and rows[0]["source"] == "live"
    h.feed(recall("m1"))
    assert [r["message_id"] for r in archived(h)] == ["m2"]


async def test_blocked_members_and_groups_without_collection_are_not_archived():
    h = Harness(limits=QUICK)
    on(h)
    t = int(wall())
    h.orch.apply_access(blocked_users={"U9"}, collection_off_groups={"G2"})
    h.feed(chat("m1", user="U9", t=t))
    h.feed(chat("m2", group="G2", t=t))
    h.feed(chat("m3", t=t))
    assert [r["message_id"] for r in archived(h)] == ["m3"]
    assert archived(h, "G2") == []


async def test_group_override_switches_the_archive_per_group():
    h = Harness(limits=QUICK)
    h.orch._base_settings = replace(
        h.settings, group_profiles={"G2": {"enabled": True, "archive": {"enabled": True}}}
    )
    t = int(wall())
    h.feed(chat("m1", t=t))
    h.feed(chat("m2", group="G2", t=t))
    assert archived(h) == [] and len(archived(h, "G2")) == 1


# -- tools ----------------------------------------------------------------------------


def seeded(tmp_path=None):
    h = Harness(limits=QUICK)
    on(h)
    base = 1_790_000_000
    rows = [
        ("a1", "U1", base, "我们约定周五晚上开黑打原神"),
        ("a2", "U2", base + 60, "好的周五见，我的 QQ 是 123456789"),
        ("a3", "BOT", base + 120, "记住了，周五晚上原神"),
        ("a4", "U3", base + 180, "原神新角色好强"),
        ("a5", "U1", base + 240, "别的群也在聊原神"),
    ]
    con = h.store.con
    with con:
        for mid, user, t, text in rows:
            con.execute(
                "INSERT INTO archive VALUES ('BOT', ?, ?, ?, ?, NULL, ?, 'import')",
                ("G2" if mid == "a5" else "G1", mid, user, t, text),
            )
    return h, base


def tools(h, before, labels=None, blocked=frozenset(), **kw):
    return archive.HistoryTools(
        h.store.con,
        "BOT",
        "G1",
        "U1",
        before,
        labels or {"U1": "群友A"},
        lambda: blocked,
        searches=kw.get("searches", 3),
        reads=kw.get("reads", 5),
    )


async def test_grep_is_bound_to_the_group_and_the_request_time():
    h, base = seeded()
    t = tools(h, base + 150)
    out = t.grep(["原神"])
    assert "共命中 2 条" in out  # a4 is after the request, a5 in another group
    assert "开黑" in out and "新角色" not in out and "别的群" not in out
    assert "机器人（你）（此前的回答，不是证据）" in out
    assert out.index("记住了") < out.index("开黑")  # newest first
    assert "本群归档最早可查到" in out
    assert t.calls[-1] == {
        "name": "history_search",
        "args": {"keywords": 1},
        "status": "ok",
        "hits": 2,
        "shown": 2,
    }


async def test_grep_keywords_speaker_numbers_and_blocked():
    h, base = seeded()
    t = tools(h, base + 1000, blocked=frozenset({"U3"}))
    assert "共命中 1 条" in t.grep(["周五", "开黑"], match="all")
    assert "共命中 3 条" in t.grep(["开黑", "周五见", "记住"], match="any")
    mine = t.grep(["周五"], speaker="提问人")
    assert "共命中 1 条" in mine and "群友A" in mine
    out = tools(h, base + 1000).grep(["QQ"])
    assert "某号码" in out and "123456789" not in out
    assert "共 0 条" in tools(h, base + 1000, blocked=frozenset({"U3"})).grep(["新角色"])


async def test_grep_rejects_bad_arguments_and_stops_at_its_budget():
    h, base = seeded()
    t = tools(h, base + 1000, searches=1)
    assert "至少 2 个字" in t.grep(["原"])
    assert "speaker 只能是" in t.grep(["原神"], speaker="张三")
    assert "日期格式" in t.grep(["原神"], since="昨天")
    assert "共命中" in t.grep(["原神"])
    assert "已用完" in t.grep(["原神"])
    assert [c["status"] for c in t.calls] == ["bad_args", "bad_speaker", "bad_args", "ok", "budget"]


async def test_read_shows_neighbours_before_the_request_only():
    h, base = seeded()
    t = tools(h, base + 150)
    t.grep(["周五见"])
    out = t.read("h1", before=5, after=5)
    assert "→ [h1]" in out and "开黑" in out and "记住了" in out
    assert "新角色" not in out
    assert "编号" in t.read("h99")


async def test_dates_limit_the_search():
    h, base = seeded()
    t = tools(h, base + 1000)
    day = __import__("datetime").datetime.fromtimestamp(base, archive._tz()).strftime("%Y-%m-%d")
    assert "共命中 3 条" in t.grep(["原神"], since=day, until=day)
    assert "共 0 条" in t.grep(["原神"], until="2020-01-01")


async def test_prune_and_clear():
    h, base = seeded()
    con = h.store.con
    assert archive.prune(con, base + 86400 * 2, lambda g: 1) == {"groups": 2, "messages": 5}
    h, base = seeded()
    assert archive.clear(h.store.con, "BOT", "G1", user_id="U1") == 1
    assert archive.clear(h.store.con, "BOT", "G1", since=base + 100) == 2


# -- the answer model gets the tools -------------------------------------------------------


async def test_answer_model_searches_the_group_history(tmp_path):
    h = Harness(limits=QUICK)
    on(h)
    t = int(wall())
    with h.store.con:
        h.store.con.execute(
            "INSERT INTO archive VALUES ('BOT','G1','old','U2',?,NULL,'上次说的暗号是蓝色海豚','import')",
            (t - 86400,),
        )
    seen = {}

    def gemini():
        async def fn(messages, info):
            seen.setdefault("tools", [d.name for d in info.function_tools])
            returns = [
                p
                for m in messages
                for p in getattr(m, "parts", [])
                if isinstance(p, ToolReturnPart)
            ]
            if not returns:
                return ModelResponse(
                    parts=[ToolCallPart("grep_group_history", {"keywords": ["暗号"]})], usage=USAGE
                )
            seen["result"] = returns[-1].content
            return ModelResponse(parts=[TextPart("暗号是蓝色海豚")], usage=USAGE)

        return FunctionModel(fn)

    def deepseek():
        async def fn(messages, info):
            return ModelResponse(parts=[TextPart("备用")], usage=USAGE)

        return FunctionModel(fn)

    config = live_config(tmp_path, gemini, deepseek, decision_mode="lightweight", answer_tools=True)
    h.orch.agent = LiveAgent(h.orch, config)
    h.feed(ev("q1", text="上次说的暗号是什么", t=t))
    await h.settle()
    assert {"grep_group_history", "read_group_history"} <= set(seen["tools"])
    assert "蓝色海豚" in seen["result"] and "共命中 1 条" in seen["result"]
    detail = json.loads(h.runs()["q1"]["detail"])
    assert "history_search:ok" in detail["live"]["tools"]
    assert h.outbound()[0]["text"] == "暗号是蓝色海豚"


async def test_no_history_tools_while_the_archive_is_off(tmp_path):
    h = Harness(limits=QUICK)
    seen = {}

    def gemini():
        async def fn(messages, info):
            seen["tools"] = [d.name for d in info.function_tools]
            return ModelResponse(parts=[TextPart("好")], usage=USAGE)

        return FunctionModel(fn)

    config = live_config(tmp_path, gemini, gemini, decision_mode="lightweight", answer_tools=True)
    h.orch.agent = LiveAgent(h.orch, config)
    h.feed(ev("q1", t=int(wall())))
    await h.settle()
    assert "grep_group_history" not in seen.get("tools", [])


# -- import -------------------------------------------------------------------------------


def snowluma(tmp_path, events):
    path = tmp_path / "messages.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE messages (message_hash TEXT, is_group INTEGER, session_id TEXT,"
        " timestamp INTEGER, data TEXT)"
    )
    for i, e in enumerate(events):
        con.execute(
            "INSERT INTO messages VALUES (?,1,?,?,?)",
            (str(i), e["group_id"], e["time"] * 1000, json.dumps(e, ensure_ascii=False)),
        )
    con.commit()
    con.close()
    return path


async def test_import_dedupes_and_skips_recalled_old_and_refused(tmp_path):
    h = Harness(limits=QUICK)
    now = 1_790_000_000
    events = [
        chat("s1", text="早期消息", t=now - 200 * 86400),
        chat("s2", text="正常导入", t=now - 10),
        chat("s3", text="被撤回的", t=now - 9),
        chat("s4", user="U9", text="屏蔽成员", t=now - 8),
        chat("s5", group="G2", text="未开归档的群", t=now - 7),
        chat("s6", text="[图片]", t=now - 6),
    ]
    with h.store.con:
        h.store.con.execute(
            "INSERT INTO messages(bot_id,group_id,message_id,user_id,segments,revoked)"
            " VALUES ('BOT','G1','s3','U1','[]',1)"
        )
    path = snowluma(tmp_path, events)

    def run(dry=False):
        return archive.import_snowluma(
            h.store.con,
            path,
            "BOT",
            lambda g, u: g == "G1" and u != "U9",
            lambda g: 180,
            now=now,
            dry_run=dry,
        )

    dry = run(True)
    assert dry["inserted"] == 2 and archived(h) == []
    report = run()
    assert report["inserted"] == 2
    assert report["skipped"] == {"retention": 1, "recalled": 1, "policy": 2}
    assert report["groups"]["G1"]["inserted"] == 2
    assert [r["message_id"] for r in archived(h)] == ["s2", "s6"]
    again = run()
    assert again["inserted"] == 0 and again["groups"]["G1"]["duplicate"] == 2


# -- configuration --------------------------------------------------------------------------


async def test_archive_settings_validate(tmp_path):
    s = parse_settings({"archive": {"enabled": True, "retention_days": 30}}, tmp_path / "s")
    assert s.archive == ArchiveSpec(enabled=True, retention_days=30)
    for bad in ({"enabled": "yes"}, {"retention_days": 0}, {"searches": 11}, {"x": 1}):
        with pytest.raises(ConfigError):
            parse_settings({"archive": bad}, tmp_path / "s")
    validate_keys({"group_profiles": {"G1": {"archive": {"enabled": True, "retention_days": 7}}}})
    with pytest.raises(ConfigError):
        validate_keys({"group_profiles": {"G1": {"archive": {"searches": 1}}}})
    profiled = replace(s, group_profiles={"G1": {"enabled": True, "archive": {"enabled": False}}})
    assert profiled.archive_for_group("G1").enabled is False
    assert profiled.archive_for_group("G2").enabled is True


# -- command line ---------------------------------------------------------------------------


async def test_archive_command_imports_reports_and_clears(tmp_path, monkeypatch, capsys):
    from group_agent import archive_cli
    from group_agent.config import Settings
    from group_agent.store import Store

    db = tmp_path / "agent.sqlite3"
    Store(db).close()
    now = int(__import__("time").time())
    path = snowluma(
        tmp_path, [chat("s1", text="导入的消息", t=now - 60), chat("s2", user="U2", t=now - 30)]
    )
    settings = Settings(
        bot_id="BOT", serve_db=db, snowluma_db=path, archive=ArchiveSpec(enabled=True)
    )
    monkeypatch.setattr(archive_cli, "_settings", lambda instance: settings)
    assert archive_cli.main(["import", "--instance", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inserted"] == 2 and list(report["groups"]) == ["…G1"]
    assert archive_cli.main(["stats", "--instance", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2
    assert (
        archive_cli.main(["clear", "--instance", str(tmp_path), "--group", "G1", "--user", "U2"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["cleared"] == 1
    off = replace(settings, archive=ArchiveSpec())
    monkeypatch.setattr(archive_cli, "_settings", lambda instance: off)
    assert archive_cli.main(["import", "--instance", str(tmp_path)]) == 1


async def test_retired_owner_command_send_still_loads(tmp_path):
    from group_agent.config_service import build_bundle

    path = tmp_path / "settings.toml"
    data = {"schema_version": 1, "serve": {"owner_command_send": True}}
    bundle = build_bundle(data, path)
    assert "owner_command_send" not in bundle.data.get("serve", {})
    assert not hasattr(bundle.settings, "owner_command_send")
    with pytest.raises(ConfigError):
        build_bundle({"schema_version": 1, "serve": {"owner_command_sendx": True}}, path)