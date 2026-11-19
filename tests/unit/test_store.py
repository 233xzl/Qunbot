import sqlite3

import pytest

from group_agent.contracts import Message
from group_agent.store import SCHEMA_VERSION, Store


def test_reopen_preserves_schema_and_duplicate_state(tmp_path):
    path = tmp_path / "test.db"
    message = Message("BOT", "G1", "m1", "U1", ())
    first = Store(path)
    seq, fresh = first.ingest(message)
    assert fresh
    assert first.con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert first.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    first.close()
    second = Store(path)
    assert second.ingest(message) == (seq, False)
    second.close()


def test_v7_database_gets_the_reaction_column(tmp_path):
    path = tmp_path / "v7.db"
    store = Store(path)
    store.ingest_request(Message("BOT", "G1", "m1", "U1", ()), None, 1.0)
    store.con.executescript(
        "ALTER TABLE reply_requests DROP COLUMN reaction; PRAGMA user_version=7;"
    )
    store.close()
    store = Store(path)
    assert store.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert store.reaction_state("BOT", "G1", "m1") is None
    store.set_request_state("BOT", "G1", "m1", "queued")
    assert store.claim_reaction("BOT", "G1", "m1", now=2.0, max_age_s=60)
    assert not store.claim_reaction("BOT", "G1", "m1", now=2.0, max_age_s=60)
    assert store.reaction_state("BOT", "G1", "m1") == "pending"
    store.close()


def test_v1_database_migrates_in_place(tmp_path):
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as con:
        con.executescript(
            "CREATE TABLE messages (seq INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL,"
            " group_id TEXT NOT NULL, message_id TEXT NOT NULL, user_id TEXT NOT NULL,"
            " segments TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,"
            " UNIQUE(bot_id, group_id, message_id));"
            "INSERT INTO messages(bot_id,group_id,message_id,user_id,segments)"
            " VALUES ('BOT','G1','m1','U1','[]');"
            "PRAGMA user_version=1;"
        )
    store = Store(path)
    assert store.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    row = store.con.execute("SELECT platform_time, received_at FROM messages").fetchone()
    assert tuple(row) == (None, None)
    store.close()


def test_v2_database_migrates_to_message_source(tmp_path):
    path = tmp_path / "v2.db"
    with sqlite3.connect(path) as con:
        con.executescript(
            "CREATE TABLE messages (seq INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL,"
            " group_id TEXT NOT NULL, message_id TEXT NOT NULL, user_id TEXT NOT NULL,"
            " segments TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,"
            " platform_time INTEGER, received_at REAL,"
            " UNIQUE(bot_id, group_id, message_id));"
            "INSERT INTO messages(bot_id,group_id,message_id,user_id,segments)"
            " VALUES ('BOT','G1','m1','U1','[]');"
            "PRAGMA user_version=2;"
        )
    store = Store(path)
    assert store.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert store.lookup("BOT", "G1", "m1")["source"] == "live"
    store.ingest(Message("BOT", "G1", "m0", "U0", ()), 1, 2.0, source="backfill")
    assert store.lookup("BOT", "G1", "m0")["source"] == "backfill"
    assert store.summary()["backfilled"] == 1
    with pytest.raises(ValueError):
        store.ingest(Message("BOT", "G1", "m9", "U9", ()), source="model")
    with pytest.raises(sqlite3.IntegrityError):
        with store.con:
            store.con.execute("UPDATE messages SET source='other'")
    store.close()


def test_v3_database_migrates_to_v4_tables(tmp_path):
    path = tmp_path / "v3.db"
    with sqlite3.connect(path) as con:
        con.executescript(
            "CREATE TABLE messages (seq INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT NOT NULL,"
            " group_id TEXT NOT NULL, message_id TEXT NOT NULL, user_id TEXT NOT NULL,"
            " segments TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,"
            " platform_time INTEGER, received_at REAL,"
            " source TEXT NOT NULL DEFAULT 'live' CHECK(source IN ('live', 'backfill')),"
            " UNIQUE(bot_id, group_id, message_id));"
            "INSERT INTO messages(bot_id,group_id,message_id,user_id,segments)"
            " VALUES ('BOT','G1','m1','U1','[]');"
            "PRAGMA user_version=3;"
        )
    store = Store(path)
    assert store.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    tables = {r[0] for r in store.con.execute("SELECT name FROM sqlite_master")}
    assert {"reply_requests", "related_marks", "media_assets", "message_media"} <= tables
    assert {"runs", "outbound", "context_snapshots", "usage_ledger"} <= tables
    assert store.lookup("BOT", "G1", "m1")["source"] == "live"
    store.close()


V5_DDL = """
CREATE TABLE messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, message_id TEXT NOT NULL,
    user_id TEXT NOT NULL, segments TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
    platform_time INTEGER, received_at REAL,
    source TEXT NOT NULL DEFAULT 'live' CHECK(source IN ('live', 'backfill')),
    UNIQUE(bot_id, group_id, message_id));
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, message_seq INTEGER NOT NULL UNIQUE REFERENCES messages(seq),
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'succeeded', 'failed',
        'cancelled', 'expired', 'rejected', 'recorded')),
    error_kind TEXT, bot_id TEXT, group_id TEXT, message_id TEXT, user_id TEXT,
    cost_mode TEXT, route TEXT, queued INTEGER NOT NULL DEFAULT 0,
    created_at REAL, started_at REAL, ended_at REAL, updated_at REAL, detail TEXT);
CREATE TABLE outbound (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, reply_to TEXT NOT NULL,
    requester TEXT NOT NULL, text TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending', 'recorded_not_sent', 'failed')),
    mode TEXT NOT NULL CHECK(mode IN ('offline', 'record_only')),
    kind TEXT NOT NULL DEFAULT 'answer' CHECK(kind IN ('answer', 'busy_notice')),
    idempotency_key TEXT UNIQUE,
    error_kind TEXT, created_at REAL, scheduled_at REAL, recorded_at REAL,
    qq_message_id TEXT);
CREATE TABLE context_snapshots (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    anchor_seq INTEGER NOT NULL, strategy TEXT NOT NULL,
    message_ids TEXT NOT NULL, reasons TEXT NOT NULL, gaps TEXT NOT NULL,
    degraded INTEGER NOT NULL DEFAULT 0, created_at REAL);
INSERT INTO messages(bot_id,group_id,message_id,user_id,segments) VALUES ('BOT','G1','q','U1','[]');
INSERT INTO runs(run_id,message_seq,state,message_id) VALUES ('r1',1,'succeeded','q');
INSERT INTO outbound(run_id,bot_id,group_id,reply_to,requester,text,state,mode,kind,
    idempotency_key,recorded_at) VALUES ('r1','BOT','G1','q','U1','answer text',
    'recorded_not_sent','record_only','answer','BOT:G1:q:answer',5.0);
INSERT INTO context_snapshots VALUES ('r1',1,'C','[]','{}','[]',0,1.0);
PRAGMA user_version=5;
"""


def test_v5_database_migrates_to_v6_keeping_runs_outbound_and_snapshots(tmp_path):
    path = tmp_path / "v5.db"
    with sqlite3.connect(path) as con:
        con.executescript(V5_DDL)
    store = Store(path)
    assert store.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert store.run("r1")["purpose"] == "reply"
    row = store.outbound("r1")
    assert (row["text"], row["state"], row["recorded_at"], row["sent_at"]) == (
        "answer text",
        "recorded_not_sent",
        5.0,
        None,
    )
    assert store.context_snapshot("r1")["strategy"] == "C"  # not cascaded away
    assert store.reply_run(1)["outbound_text"] == "answer text"
    with store.con:  # the new live-send states are accepted now
        store.con.execute("UPDATE outbound SET state='uncertain' WHERE run_id='r1'")
    with pytest.raises(sqlite3.IntegrityError):
        with store.con:
            store.con.execute("UPDATE outbound SET state='acked'")
    store.close()
    again = Store(path)  # reopening a v6 database changes nothing
    assert again.outbound("r1")["state"] == "uncertain"
    again.close()


def test_history_orders_by_platform_time_and_includes_late_backfill():
    store = Store()

    def text(mid, value):
        return Message("BOT", "G1", mid, "U1", ({"type": "text", "data": {"text": value}},))

    store.ingest(text("a", "a"), 100, 1.0)
    store.ingest(text("c", "c"), 300, 2.0)
    current = Message("BOT", "G1", "q", "U2", ({"type": "reply", "data": {"id": "b"}},))
    boundary, _ = store.ingest(current, 400, 3.0)
    # Fetched by reference after the question arrived: larger seq, earlier platform time.
    store.ingest(text("b", "b"), 200, 4.0, source="backfill")
    store.ingest(text("future", "f"), 500, 5.0, source="backfill")
    store.ingest(text("later", "l"), 450, 6.0)
    assert [m.message_id for m in store.history(current, boundary)] == ["a", "b", "c"]
    # Same platform time: seq decides.
    store.ingest(text("c2", "c2"), 300, 7.0, source="backfill")
    assert [m.message_id for m in store.history(current, boundary)] == ["a", "b", "c", "c2"]
    store.close()


def test_recall_removes_content_and_tombstone_blocks_late_insert():
    store = Store()
    text = ({"type": "text", "data": {"text": "secret words"}},)
    store.ingest(Message("BOT", "G1", "m1", "U1", text), 100, 1.0)
    assert store.revoke("BOT", "G1", "m1", "U1") == "revoked"
    assert store.revoke("BOT", "G1", "m1", "U1") == "duplicate"
    row = store.con.execute("SELECT segments, revoked FROM messages WHERE message_id='m1'")
    assert tuple(row.fetchone()) == ("[]", 1)
    # Recall arriving before its message leaves a placeholder; the late message stays out.
    assert store.revoke("BOT", "G1", "m2", "U2") == "tombstone"
    assert store.ingest(Message("BOT", "G1", "m2", "U2", text))[1] is False
    assert "secret" not in "".join(r[0] for r in store.con.execute("SELECT segments FROM messages"))
    current = Message("BOT", "G1", "m3", "U3", ())
    assert store.history(current, 10**9) == []
    store.close()


def test_future_schema_rejected_without_mutation(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="schema"):
        Store(path)
    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 99


def test_v10_group_usage_survives_run_cleanup(tmp_path):
    path = tmp_path / "usage.db"
    store = Store(path)
    message = Message("BOT", "A", "m1", "U", ())
    seq, _ = store.ingest(message)
    store.create_run("known", seq, message, state="running", cost_mode="normal", now=1)
    store.con.executescript("""
        DROP INDEX usage_group_day;
        ALTER TABLE usage_ledger DROP COLUMN group_id;
        INSERT INTO usage_ledger(run_id,day,created_at,provider,model,cost_cny)
        VALUES ('known','d',1,'p','m',1), ('missing','d',1,'p','m',2);
        PRAGMA user_version=10;
    """)
    store.close()
    store = Store(path)
    store.con.execute("DELETE FROM runs")
    assert [
        tuple(r)
        for r in store.con.execute("SELECT group_id,cost_cny FROM usage_ledger ORDER BY id")
    ] == [("A", 1), (None, 2)]
    assert store.usage_total("d") == 3
    store.close()
