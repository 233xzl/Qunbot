"""scripts/shadow_report.py on fabricated serve and SnowLuma stores (temporary SQLite)."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

from group_agent.contracts import Message
from group_agent.store import Store

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("shadow_report", ROOT / "scripts/shadow_report.py")
report = importlib.util.module_from_spec(spec)
sys.modules["shadow_report"] = report  # dataclasses look the module up
spec.loader.exec_module(report)

T = 1_800_000_000
AT = {"type": "at", "data": {"qq": "BOT"}}


def text(value):
    return {"type": "text", "data": {"text": value}}


def serve_db(path: Path) -> None:
    store = Store(path)

    def request(mid, user, segs, t, run_id, answer, state="succeeded", route="chat", cost=0.0):
        m = Message("BOT", "G1", mid, user, tuple(segs))
        seq, _, _ = store.ingest_request(m, t, t + 0.5)
        store.create_run(run_id, seq, m, state="queued", cost_mode="normal", now=t + 1)
        store.update_run(run_id, t + 20, state=state, route=route, ended_at=t + 20)
        if answer is not None:
            store.outbound_enqueue(run_id, "answer", m, answer, t + 19)
            store.outbound_update(run_id, t + 19, state="recorded_not_sent", recorded_at=t + 19)
        if cost:
            store.record_usage(
                {
                    "run_id": run_id,
                    "day": "2027-01-15",
                    "created_at": t + 5,
                    "provider": "gemini",
                    "model": "m",
                    "task": "answer",
                    "mode": "text",
                    "status": "ok",
                    "input_tokens": 1,
                    "cached_tokens": 0,
                    "output_tokens": 1,
                    "thinking_tokens": None,
                    "requests": 1,
                    "cost_cny": cost,
                    "latency_s": 1.0,
                }
            )
        return seq, m

    store.ingest(Message("BOT", "G1", "ctx", "U9", (text("被引用的<b>原话</b>"),)), T - 30, T - 30)
    seq1, m1 = request(
        "q1", "U1", [{"type": "reply", "data": {"id": "ctx"}}, AT, text("<script>x</script>?")],
        T, "r1", "新回答一", cost=0.004,
    )  # fmt: skip
    store.save_context_snapshot(
        "r1", seq1, "C", ["ctx"], {"ctx": "reply_to/quote_chain"}, [], False, T + 2
    )
    request("q2", "U2", [AT, text("第二个问题")], T + 100, "r2", "新回答二", cost=0.002)
    request("q3", "U3", [AT, text("第三个问题")], T + 200, "r3", None, state="failed", route=None)
    request("q4", "U4", [AT, text("明天的问题")], T + 90_000, "r4", "新回答四")
    # The owner's #新版 send for q2 (a new_version run keyed by the command message).
    cmd = Message(
        "BOT", "G1", "c2", "OWNER", ({"type": "reply", "data": {"id": "q2"}}, text("#新版"))
    )
    cseq, _ = store.ingest(cmd, T + 150, T + 150)
    q2 = Message("BOT", "G1", "q2", "U2", ())
    store.create_run(
        "nv2", cseq, q2, state="succeeded", cost_mode="normal", now=T + 150, purpose="new_version"
    )
    store.outbound_enqueue("nv2", "new_version", q2, "〔新版〕新回答二", T + 151, mode="live_send")
    store.outbound_update("nv2", T + 152, state="sent", qq_message_id="b_nv", sent_at=T + 152)
    store.close()


def snowluma_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE messages (message_hash INTEGER PRIMARY KEY, is_group INTEGER NOT NULL,"
        " session_id INTEGER NOT NULL, sequence INTEGER NOT NULL, timestamp INTEGER, data TEXT)"
    )
    rows = []

    def add(mid, user, segs, t, seq):
        event = {"post_type": "message", "message_type": "group", "self_id": "BOT",
                 "group_id": "G1", "message_id": mid, "user_id": user, "time": t,
                 "message": segs}  # fmt: skip
        rows.append((len(rows) + 1, 1, 1, seq, t, json.dumps(event, ensure_ascii=False)))

    add("q1", "U1", [AT, text("q")], T, 1)
    add("q2", "U2", [AT, text("q")], T + 100, 2)
    add("old1", "BOT", [{"type": "reply", "data": {"id": "q1"}}, text("旧回答<一>")], T + 25, 3)
    add("old2", "BOT", [text("旧回答二，无引用")], T + 140, 4)
    add(
        "b_nv",
        "BOT",
        [{"type": "reply", "data": {"id": "q2"}}, text("〔新版〕新回答二")],
        T + 152,
        5,
    )
    add("q3", "U3", [AT, text("q")], T + 200, 6)
    add("other", "U5", [text("闲聊")], T + 210, 7)
    con.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def make(tmp_path):
    serve, snow = tmp_path / "serve.sqlite3", tmp_path / "messages.db"
    serve_db(serve)
    snowluma_db(snow)
    return serve, snow


def test_pairs_old_replies_and_writes_escaped_report(tmp_path):
    serve, snow = make(tmp_path)
    out = tmp_path / "report"
    args = ["--serve-db", str(serve), "--snowluma-db", str(snow), "--out", str(out)]
    assert report.main(args + ["--until", str(T + 1000)]) == 0
    items, _ = report.build(serve, snow, snowluma_root=tmp_path, since=None, until=T + 1000)
    by_run = {i["run_id"]: i for i in items}
    assert set(by_run) == {"r1", "r2", "r3"}  # reply runs only; the #新版 run is not a request
    assert by_run["r1"]["old"]["rule"] == "reply_segment_to_anchor"
    assert by_run["r1"]["old"]["text"] == "旧回答<一>" and by_run["r1"]["latency_old"] == 25
    assert by_run["r2"]["old"]["rule"] == "first_unquoted_bot_message_after_anchor"
    assert by_run["r3"]["old"] is None  # the 〔新版〕 message is never an old reply
    assert by_run["r1"]["quoted"] == ["群友：被引用的<b>原话</b>"]
    assert by_run["r1"]["context_count"] == 1 and by_run["r1"]["latency_new"] == 19
    assert by_run["r2"]["new_version"] == {"state": "sent", "error_kind": None}

    page = (out / "index.html").read_text("utf-8")
    assert "<script>x</script>" not in page and "&lt;script&gt;x&lt;/script&gt;" in page
    assert "&lt;b&gt;原话&lt;/b&gt;" in page and "旧回答&lt;一&gt;" in page
    assert page.count('class="item"') == 3 and "localStorage" in page

    summary = json.loads((out / "summary.json").read_text("utf-8"))
    assert summary["requests"] == 3 and summary["paired_with_old_reply"] == 2
    assert summary["pairing_rate"] == round(2 / 3, 4)
    assert summary["run_states"] == {"succeeded": 2, "failed": 1}
    assert summary["cost_cny"]["total"] == 0.006
    assert summary["latency_old_s"]["n"] == 2 and summary["latency_new_s"]["n"] == 3
    assert summary["new_version"]["bot_messages_seen"] == 1
    # Aggregates only: no message text and no ids in the summaries.
    flat = (out / "summary.json").read_text("utf-8") + (out / "summary.md").read_text("utf-8")
    for word in ("新回答", "旧回答", "问题", "q1", "r1", "U1", "G1", "BOT"):
        assert word not in flat, word


def test_since_until_and_regeneration_is_idempotent(tmp_path):
    serve, snow = make(tmp_path)
    out = tmp_path / "report"
    args = ["--serve-db", str(serve), "--snowluma-db", str(snow), "--out", str(out)]
    report.main(args)
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    report.main(args)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first
    assert json.loads(first["summary.json"])["requests"] == 4
    report.main(args + ["--since", str(T + 50), "--until", str(T + 150)])
    assert json.loads((out / "summary.json").read_text("utf-8"))["requests"] == 1
    # The serve store is untouched (read-only).
    assert Store(serve).summary()["runs"] == 5


def test_import_ratings_summary(tmp_path):
    serve, snow = make(tmp_path)
    ratings = tmp_path / "ratings.json"
    ratings.write_text(
        json.dumps(
            {
                "version": 1,
                "ratings": {
                    "r1": {"rating": "new_better", "note": "更准"},
                    "r2": {"rating": "same"},
                    "zz": {"rating": "old_better"},
                },
            }
        ),
        "utf-8",
    )
    out = tmp_path / "report"
    report.main(
        ["--serve-db", str(serve), "--snowluma-db", str(snow), "--out", str(out),
         "--import-ratings", str(ratings), "--until", str(T + 1000)]
    )  # fmt: skip
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    assert summary["ratings"] == {
        "rated": 2,
        "unrated": 1,
        "counts": {"新更好": 1, "差不多": 1},
        "with_note": 1,
    }
    assert "更准" not in (out / "summary.md").read_text("utf-8")
