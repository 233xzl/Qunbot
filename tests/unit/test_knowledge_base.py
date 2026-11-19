"""T4.1-T4.3: knowledge migration and check, search on the new base, owner teaching.

Everything runs on fabricated legacy stores and temporary files; ids, names and texts are
invented. No network, credentials or QQ sending.
"""

import asyncio
import hashlib
import io
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest
from test_orchestrator import Harness, ev, virtual

from group_agent.knowledge_base import KnowledgeBase, content_hash
from group_agent.knowledge_migration import migrate, verify
from group_agent.knowledge_search import KnowledgeSnapshot
from group_agent.teaching import UNAUTHORIZED_TEXT, parse_teaching, reply_for


def png_bytes(color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


def legacy_store(path: Path, media: Path) -> dict:
    """A small store in the legacy plugin's schema: 3 CHIME entries (one disabled), one
    owner entry with alias and image, one owner correction superseding a CHIME entry."""
    media.mkdir(parents=True, exist_ok=True)
    raw = png_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    (media / f"{sha}.png").write_bytes(raw)
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE entries (id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,"
        " revision INTEGER NOT NULL, active INTEGER NOT NULL, doc_id TEXT NOT NULL,"
        " data TEXT NOT NULL);"
        "CREATE TABLE revisions (id TEXT NOT NULL, revision INTEGER NOT NULL, data TEXT NOT NULL,"
        " created REAL NOT NULL, PRIMARY KEY(id, revision));"
        "CREATE TABLE imports (sha256 TEXT PRIMARY KEY, count INTEGER NOT NULL,"
        " doc_id TEXT NOT NULL, imported REAL NOT NULL);"
    )

    def base(kid, name, expl, active=True, revision=1):
        return {
            "id": kid,
            "kind": "base",
            "name": name,
            "aliases": [],
            "category": "其他",
            "explanation": expl,
            "source": "虚构出处",
            "examples": ["虚构例句一二三四"],
            "images": [],
            "supersedes": [],
            "revision": revision,
            "active": active,
            "fingerprint": "f" + kid,
            "snapshot": "s",
            "ordinal": 0,
            "original": {"meme": name},
            "doc_id": "d1",
        }

    rows = [
        base("C0001", "虚构梗甲", "甲的虚构解释"),
        base("C0002", "虚构梗乙", "乙的虚构解释"),
        base("C0003", "虚构梗丙", "丙的虚构解释", active=False, revision=2),
        {
            "id": "K001",
            "kind": "owner",
            "name": "虚构教学丁",
            "aliases": ["丁别名"],
            "category": "其他",
            "explanation": "丁的虚构解释",
            "technical_meaning": "",
            "source": "",
            "examples": [],
            "images": [f"images/{sha}.png"],
            "supersedes": [],
            "revision": 1,
            "active": True,
            "doc_id": "d2",
        },
        {
            "id": "K002",
            "kind": "owner",
            "name": "虚构梗乙",
            "aliases": [],
            "category": "其他",
            "explanation": "乙的主人更正",
            "technical_meaning": "",
            "source": "",
            "examples": [],
            "images": [],
            "supersedes": ["C0002"],
            "revision": 1,
            "active": True,
            "doc_id": "d3",
        },
    ]
    for d in rows:
        con.execute(
            "INSERT INTO entries VALUES (?,?,?,?,?,?,?)",
            (d["id"], d["kind"], d["name"], d["revision"], int(d["active"]), d["doc_id"],
             json.dumps(d, ensure_ascii=False)),
        )  # fmt: skip
        for rev in range(1, d["revision"] + 1):
            old = {**d, "revision": rev, "active": d["active"] if rev == d["revision"] else True}
            con.execute(
                "INSERT INTO revisions VALUES (?,?,?,?)",
                (d["id"], rev, json.dumps(old, ensure_ascii=False), 1000.0 + rev),
            )
    con.commit()
    con.close()
    return {"sha": sha, "rows": rows}


@pytest.fixture
def legacy(tmp_path):
    info = legacy_store(tmp_path / "legacy.sqlite3", tmp_path / "legacy-images")
    return tmp_path / "legacy.sqlite3", tmp_path / "legacy-images", info


# -- T4.1 migration ----------------------------------------------------------------------


def test_migration_copies_everything_and_checks_per_entry(tmp_path, legacy):
    snap, media, info = legacy
    kb = KnowledgeBase(tmp_path / "kb" / "knowledge.sqlite3")
    result = migrate(snap, kb, media)
    assert (result["entries_inserted"], result["revisions_inserted"]) == (5, 6)
    assert result["media_copied"] == 1 and not result["media_missing"]
    report = verify(snap, kb, media)
    assert report["ok"], report["mismatches"]
    assert report["legacy"] == {
        "entries": 5,
        "by_kind": {"base": 3, "owner": 2},
        "active": 4,
        "revisions": 6,
        "aliases": 1,
        "images": 1,
    }
    assert report["checked"]["images_ok"] == 1
    image = tmp_path / "kb" / "images" / f"{info['sha']}.png"
    assert hashlib.sha256(image.read_bytes()).hexdigest() == info["sha"]
    # Data is carried verbatim: the new hash equals the hash of the legacy JSON.
    row = kb.con.execute("SELECT content_sha256, scope FROM entries WHERE id='K001'").fetchone()
    assert tuple(row) == (content_hash(info["rows"][3]), "public")
    # Report text holds no knowledge content.
    assert "虚构" not in json.dumps(report, ensure_ascii=False)


def test_migration_is_idempotent_and_follows_legacy_updates(tmp_path, legacy):
    snap, media, _ = legacy
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    migrate(snap, kb, media)
    again = migrate(snap, kb, media)
    assert (again["entries_inserted"], again["revisions_inserted"]) == (0, 0)
    assert (again["entries_unchanged"], again["revisions_present"], again["media_present"]) == (
        5,
        6,
        1,
    )
    assert kb.counts()["entries"] == 5 and kb.counts()["revisions"] == 6
    # The legacy side moved on before the cutover: a new revision is picked up.
    con = sqlite3.connect(snap)
    d = json.loads(con.execute("SELECT data FROM entries WHERE id='C0001'").fetchone()[0])
    d.update(revision=2, explanation="甲的新解释")
    con.execute(
        "UPDATE entries SET revision=2, data=? WHERE id='C0001'",
        (json.dumps(d, ensure_ascii=False),),
    )
    con.execute("INSERT INTO revisions VALUES ('C0001', 2, ?, 2000.0)", (json.dumps(d),))
    con.commit()
    con.close()
    third = migrate(snap, kb, media)
    assert (third["entries_updated"], third["revisions_inserted"]) == (1, 1)
    assert kb.revision("C0001") == 2 and verify(snap, kb, media)["ok"]


def test_verify_reports_mismatches_by_id_and_teaching_divergence(tmp_path, legacy):
    snap, media, _ = legacy
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    migrate(snap, kb, media)
    with kb.con:
        kb.con.execute("UPDATE entries SET active=0 WHERE id='C0001'")
        kb.con.execute("DELETE FROM revisions WHERE id='C0003' AND revision=1")
    report = verify(snap, kb, media)
    assert not report["ok"]
    assert {"id": "C0001", "field": "active"} in report["mismatches"]
    assert {"id": "C0003", "field": "revision 1 missing"} in report["mismatches"]
    # A teaching change after the migration is not a mismatch; a re-run keeps it.
    kb2 = KnowledgeBase(tmp_path / "kb2.sqlite3")
    migrate(snap, kb2, media)
    fix = parse_teaching("更正 K001：丁的新解释", "g_1")
    out = kb2.apply(fix, actor="o", owner_id="o", source_message="m1", group_id="g_1", now=1)
    assert out["outcome"] == "committed" and out["revision"] == 2
    con = sqlite3.connect(snap)
    con.execute("UPDATE entries SET active=0 WHERE id='K001'")
    con.commit()
    con.close()
    assert migrate(snap, kb2, media)["entries_diverged"] == ["K001"]
    report = verify(snap, kb2, media)
    assert report["ok"] and report["checked"]["changed_by_teaching_since"] == 1
    assert kb2.revision("K001") == 2


# -- T4.2 search on the new base ------------------------------------------------------------


def test_search_reads_the_new_base_with_scope_and_live_rules(tmp_path, legacy):
    snap, media, _ = legacy
    path = tmp_path / "kb.sqlite3"
    kb = KnowledgeBase(path)
    migrate(snap, kb, media)
    hits = kb.search("虚构梗甲是什么意思")
    assert [h.entry.id for h in hits][:1] == ["C0001"] and hits[0].match == "exact"
    assert [h.entry.id for h in kb.search("丁别名")] == ["K001"]  # alias
    assert kb.search("虚构梗丙") == []  # disabled
    assert [h.entry.id for h in kb.search("虚构梗乙")] == ["K002"]  # C0002 superseded
    assert kb.revision("C0002") is None and kb.entry("C0002").id == "K002"
    add = parse_teaching("记住 本群：「虚构群梗」本群的虚构说法", "g_1")
    kb.apply(add, actor="o", owner_id="o", source_message="m1", group_id="g_1", now=1)
    assert [h.entry.id for h in kb.search("虚构群梗", group_id="g_1")] == ["K003"]
    assert kb.search("虚构群梗", group_id="g_2") == [] and kb.search("虚构群梗") == []
    # Read-only view of the same file (replay tools) sees the same live entries.
    snapshot = KnowledgeSnapshot(path)
    assert {e.id for e in snapshot.entries} == {"C0001", "K001", "K002", "K003"}
    assert snapshot.get("K003").scope == "group:g_1"


# -- T4.3 teaching -----------------------------------------------------------------------


def apply(kb, text, actor="o", mid="m", group="g_1", quotes_bot=False, **kw):
    command = parse_teaching(text, group, quotes_bot)
    assert command is not None, text
    entry = kb.apply(command, actor=actor, owner_id="o", source_message=mid, group_id=group,
                     now=1.0, **kw)  # fmt: skip
    return entry, reply_for(entry, kb.describe)


def test_teaching_add_scope_idempotency_and_replies(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    e, text = apply(kb, "记住 公共：「虚构甲」一种虚构说法", mid="m1")
    assert e["applied"] and text == "已记住 K001：虚构甲，所有群可用。"
    e, text = apply(kb, "教学 本群：「虚构乙」本群的虚构说法", mid="m2")
    assert text == "已记住 K002：虚构乙，本群可用。"
    assert kb.con.execute("SELECT scope FROM entries WHERE id='K002'").fetchone()[0] == "group:g_1"
    e, text = apply(kb, "记住：「虚构丙」没写范围", mid="m3")
    assert e["outcome"] == "missing_scope" and "公共" in text and "本群" in text
    assert kb.counts()["entries"] == 2
    e, _ = apply(kb, "记住 公共：「虚构甲」一种虚构说法", mid="m1")  # same event again
    assert e["outcome"] == "duplicate" and not e["applied"]
    e, _ = apply(kb, "记住 公共：「虚构甲」一种虚构说法", mid="m4")  # re-sent, same content
    assert e["outcome"] == "duplicate"
    e, text = apply(kb, "记住 公共：「虚构甲」另一个意思", mid="m5")
    assert e["outcome"] == "conflict" and "更正 K001" in text
    e, text = apply(kb, "记一下 公共：这是一个没有名字的说法", mid="m6")
    assert e["outcome"] == "missing_name" and "梗名" in text
    assert kb.counts()["revisions"] == 2
    author = kb.con.execute("SELECT author, source_message FROM entries WHERE id='K001'").fetchone()
    assert tuple(author) == ("o", "m1")


def test_non_owner_is_refused_without_revealing_anything(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    apply(kb, "记住 公共：「虚构甲」一种虚构说法", mid="m1")
    replies = set()
    for i, text in enumerate(
        ["停用 K001", "停用 K999", "更正 K001：别的", "记住 公共：「虚构乙」x", "恢复「不存在」"]
    ):
        e, reply = apply(kb, text, actor="someone", mid=f"x{i}")
        assert e["outcome"] == "unauthorized" and e["id"] is None and not e["applied"]
        replies.add(reply)
    assert replies == {UNAUTHORIZED_TEXT}
    assert kb.revision("K001") == 1 and kb.counts()["entries"] == 1


def test_correct_with_revision_check_disable_restore_and_view(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    apply(kb, "记住 公共：「虚构甲」第一版", mid="m1")
    e, text = apply(kb, "更正 K001：第二版", mid="m2")
    assert e["revision"] == 2 and text == "已更正 K001：虚构甲，所有群可用。"
    e, text = apply(kb, "更正 K001 v1：基于旧版的改动", mid="m3")
    assert e["outcome"] == "conflict" and e["expected"] == 1 and "v2" in text
    e, _ = apply(kb, "把 K001 改成 第三版", mid="m4")
    assert e["revision"] == 3
    e, _ = apply(kb, "更正 「虚构甲」：第四版", mid="m5")
    assert e["revision"] == 4
    # Quoting an answer that used revision 2: conflict, nothing written.
    e, _ = apply(kb, "不对，更正为：第五版", mid="m6", quotes_bot=True, target=("K001", 2))
    assert e["outcome"] == "conflict" and kb.revision("K001") == 4
    e, text = apply(kb, "停用 K001", mid="m7")
    assert text == "已停用 K001：虚构甲，所有群生效。" and kb.revision("K001") is None
    assert kb.search("虚构甲") == []
    e, _ = apply(kb, "停用 K001", mid="m8")
    assert e["outcome"] == "duplicate"
    e, text = apply(kb, "看看 K001", actor="anyone", mid="m9")
    assert e["outcome"] == "viewed" and "第四版" in text and "状态：已停用" in text
    e, text = apply(kb, "恢复「虚构甲」", mid="m10")
    assert text == "已恢复 K001：虚构甲，所有群生效。" and kb.revision("K001") == 6
    e, text = apply(kb, "查看「虚构甲」", actor="anyone", mid="m11")
    assert "[K001 v6] 虚构甲" in text and "状态：已启用" in text
    history = kb.con.execute("SELECT revision, op FROM revisions WHERE id='K001'").fetchall()
    assert [tuple(r) for r in history] == [
        (1, "add"),
        (2, "correct"),
        (3, "correct"),
        (4, "correct"),
        (5, "disable"),
        (6, "restore"),
    ]


def test_group_entries_are_invisible_elsewhere_and_base_corrections_supersede(tmp_path, legacy):
    snap, media, _ = legacy
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    migrate(snap, kb, media)
    apply(kb, "记住 本群：「虚构群梗」本群说法", mid="m1", group="g_1")
    e, text = apply(kb, "看看 K003", actor="anyone", mid="m2", group="g_2")
    assert e["outcome"] == "not_found" and "本群说法" not in text
    e, _ = apply(kb, "停用 K003", mid="m3", group="g_2")
    assert e["outcome"] == "not_found" and kb.revision("K003") == 1
    # Correcting a CHIME entry creates an owner entry superseding exactly that entry.
    e, text = apply(kb, "更正 C0001：甲的主人解释", mid="m4")
    assert (e["id"], e["revision"], e["supersedes"]) == ("K004", 1, "C0001")
    assert kb.revision("C0001") is None and kb.revision("K004") == 1
    assert [h.entry.id for h in kb.search("虚构梗甲")] == ["K004"]
    # A second correction of the same CHIME id goes to the owner entry.
    e, _ = apply(kb, "更正 C0001：甲的主人解释二", mid="m5")
    assert (e["id"], e["revision"]) == ("K004", 2)


def test_teaching_images_saved_by_hash(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb" / "knowledge.sqlite3")
    image = tmp_path / "in.png"
    image.write_bytes(png_bytes((0, 0, 255)))
    e, _ = apply(kb, "记住 公共：「虚构图梗」带图", mid="m1", images=[image, None])
    assert e["applied"] and e["images"] == 1
    data = json.loads(kb.con.execute("SELECT data FROM entries WHERE id='K001'").fetchone()[0])
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    assert data["images"] == [f"images/{sha}.png"]
    assert (tmp_path / "kb" / "images" / f"{sha}.png").is_file()
    e, text = apply(kb, "记住 公共：「虚构图梗二」图没取回", mid="m2", images=[None])
    assert e["outcome"] == "image_unavailable" and "图片" in text
    assert kb.counts()["entries"] == 1


def test_parse_view_disable_restore_and_id_corrections():
    p = parse_teaching
    assert (p("看看 k12", "g").op, p("看看 k12", "g").target_id) == ("view", "K12")
    assert p("查看「虚构」", "g").term == "虚构"
    assert p("请停用 C0001", "g").op == "disable"
    assert p("重新启用 K001", "g").op == "restore"
    fix = p("更正 K001 v3：新解释", "g")
    assert (fix.op, fix.target_id, fix.expected, fix.content) == ("correct", "K001", 3, "新解释")
    assert p("把K002改为 另一个解释", "g").content == "另一个解释"
    assert p("更正 K001", "g").content == ""
    assert p("看看这个梗什么意思", "g") is None
    assert p("恢复出厂设置是什么梗", "g") is None


def test_live_agent_refresh_reads_current_entries(tmp_path, legacy):
    from types import SimpleNamespace

    from group_agent.live_agent import LiveAgent, RunState

    snap, media, _ = legacy
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    migrate(snap, kb, media)
    state = RunState(record={})
    state.kb_hits = kb.search("虚构梗甲") + kb.search("丁别名")
    state.decision = SimpleNamespace(knowledge_ids=["C0001", "K001"])
    fake = SimpleNamespace(config=SimpleNamespace(knowledge=kb), _state=lambda trace: state)
    apply(kb, "更正 C0001：甲的主人解释", mid="m1")  # -> K003 supersedes C0001
    apply(kb, "停用 K001", mid="m2")
    LiveAgent.refresh_knowledge(fake, SimpleNamespace(run_id="r"))
    assert [(h.entry.id, h.entry.revision) for h in state.kb_hits] == [("K003", 1)]
    assert state.decision.knowledge_ids == ["K003"]


# -- orchestrator: online teaching and the pre-publish knowledge check ------------------------


def harness_with_kb(tmp_path, **kw):
    h = Harness(**kw)
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    h.orch.knowledge = h.orch.ledger = kb
    return h, kb


def test_online_teaching_writes_once_and_refuses_others(tmp_path):
    async def main():
        h, kb = harness_with_kb(tmp_path)
        event = ev("t1", user="OWNER", text="记住 本群：「虚构开黑」晚上九点")
        h.feed(event)
        assert h.feed(dict(event)) == "duplicate"  # the same event delivered twice
        h.feed(ev("t2", user="U1", text="记住 公共：「虚构开黑」随时"))
        h.feed(ev("t3", user="OWNER", text="记住：「虚构早八」早上八点"))
        await h.settle()
        return h, kb

    h, kb = virtual(main)
    out = {o["reply_to"]: o["text"] for o in h.outbound()}
    assert out["t1"] == "已记住 K001：虚构开黑，本群可用。"
    assert out["t2"] == UNAUTHORIZED_TEXT
    assert "公共" in out["t3"] and "本群" in out["t3"]
    assert kb.counts()["entries"] == 1 and kb.counts()["revisions"] == 1
    detail = json.loads(h.runs()["t1"]["detail"])
    assert (
        detail["teaching"]["outcome"] == "committed" and detail["teaching"]["scope"] == "group:G1"
    )
    assert json.loads(h.runs()["t3"]["detail"])["teaching"]["outcome"] == "missing_scope"
    assert h.agent.answered == []


class KnowledgeAgent:
    """Answers with entry K001 as evidence; the first answer takes 5 s."""

    def __init__(self, host, kb):
        self.host, self.kb, self.answers, self.refreshed, self.runs = host, kb, [], 0, {}

    async def vision(self, trace, mocks):
        return True

    async def decide_route(self, trace, mocks):
        return "local_knowledge"

    async def run_tools(self, trace, m, budget):
        pass

    def refresh_knowledge(self, trace):
        self.refreshed += 1

    async def answer(self, trace, mocks, message, seq):
        entry = self.kb.entry("K001")
        trace.knowledge = {entry.id: entry.revision}
        self.answers.append(entry.revision)
        await asyncio.sleep(5 if len(self.answers) == 1 else 0.1)
        return f"answer from v{entry.revision}"


def test_knowledge_change_before_publish_regenerates(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    command = parse_teaching("记住 公共：「虚构梗」第一版", "G1")
    kb.apply(command, actor="OWNER", owner_id="OWNER", source_message="seed", group_id="G1", now=0)

    async def main():
        h = Harness()
        h.orch.knowledge = h.orch.ledger = kb
        h.orch.agent = agent = KnowledgeAgent(h.orch, kb)
        h.feed(ev("q1", user="U1", text="虚构梗是什么意思"))
        await asyncio.sleep(1)
        h.feed(ev("f1", user="OWNER", text="更正 K001：第二版"))
        await h.settle()
        return h, agent

    h, agent = virtual(main)
    assert agent.answers == [1, 2] and agent.refreshed == 1
    out = {o["reply_to"]: o for o in h.outbound()}
    assert out["q1"]["text"] == "answer from v2" and out["q1"]["state"] == "recorded_not_sent"
    assert out["f1"]["text"] == "已更正 K001：虚构梗，所有群可用。"
    assert h.runs()["q1"]["state"] == "succeeded"
