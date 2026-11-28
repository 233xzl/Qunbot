"""Read-only self-sync of the bot's own messages (self_sync.py) and owner command
messages kept out of context (kind='owner_command').

The SnowLuma side is a temporary SQLite file with the minimal subset of SnowLuma's
`messages` table that the sync reads (PRAGMA table_info of the real store: is_group,
session_id, sequence, timestamp, data; index on (is_group, session_id, sequence))."""

import asyncio
import hashlib
import json
import sqlite3
import time

import pytest
from test_orchestrator import BASE, Harness, ev, virtual, wall

from group_agent.app import main as app_main
from group_agent.config import Retention, load_settings
from group_agent.contracts import Message
from group_agent.retention import related_set
from group_agent.self_sync import SelfSync, connect_ro
from group_agent.store import Store

GROUP = "20001"  # SnowLuma keys group sessions by the numeric group id
T = 1_800_000_000

SNOWLUMA_DDL = """
CREATE TABLE messages (
    message_hash INTEGER PRIMARY KEY, is_group INTEGER NOT NULL, session_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL, timestamp INTEGER NOT NULL, data TEXT
);
CREATE INDEX idx_messages_session_seq ON messages(is_group, session_id, sequence);
"""


class FakeSnowLuma:
    def __init__(self, path):
        self.path = path
        self.sequence = 0
        with sqlite3.connect(path) as con:
            con.executescript(SNOWLUMA_DDL)
        con.close()

    def add(self, mid, t, text="答复", user="BOT", reply=None, group=GROUP):
        self.sequence += 1
        segments = [{"type": "reply", "data": {"id": reply}}] if reply else []
        segments.append({"type": "text", "data": {"text": text}})
        event = {
            "time": int(t),
            "self_id": "BOT",
            "post_type": "message_sent" if user == "BOT" else "message",
            "message_type": "group",
            "message_id": mid,
            "group_id": int(group),
            "user_id": user,
            "message": segments,
        }
        con = sqlite3.connect(self.path)
        with con:
            con.execute(
                "INSERT INTO messages(is_group, session_id, sequence, timestamp, data)"
                " VALUES (1, ?, ?, ?, ?)",
                (int(group), self.sequence, int(t), json.dumps(event, ensure_ascii=False)),
            )
        con.close()

    def digest(self):
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def member(mid, user="U1", text="问一下", group=GROUP, reply=None):
    segments = [{"type": "reply", "data": {"id": reply}}] if reply else []
    segments += [{"type": "at", "data": {"qq": "BOT"}}, {"type": "text", "data": {"text": text}}]
    return Message("BOT", group, mid, user, tuple(segments))


def request(store, mid, t, user="U1", group=GROUP):
    m = member(mid, user=user, group=group)
    seq = store.ingest_request(m, int(t), float(t))[0]
    return m, seq


def sync_for(store, sl, **kw):
    kw.setdefault("clock", lambda: T)
    return SelfSync(store, "BOT", sl.path, **kw)


@pytest.fixture
def sl(tmp_path):
    return FakeSnowLuma(tmp_path / "messages.db")


# -- incremental read, dedup, read-only ------------------------------------------------


def test_incremental_sync_dedup_and_read_only(sl):
    store = Store()
    request(store, "q1", T - 100)
    sl.add("q1", T - 100, user="U1")  # member messages arrive by push, not by the sync
    sl.add("a1", T - 90, reply="q1")
    sl.add("x1", T - 80, group="20002")  # a group this store does not collect
    sync = sync_for(store, sl)
    assert asyncio.run(sync.poll()) == 1
    row = store.lookup("BOT", GROUP, "a1")
    assert row["source"] == "self_sync" and row["user_id"] == "BOT"
    assert store.lookup("BOT", "20002", "x1") is None
    assert store.answer_map("BOT", GROUP) == {"a1": "q1"}
    assert asyncio.run(sync.poll()) == 0  # nothing new after the cursor
    sl.add("a2", T - 10)
    assert asyncio.run(sync.poll()) == 1  # incremental
    assert sync.counts["synced"] == 2 and sync.counts["paired_quote"] == 1
    # A restarted sync re-reads the lookback window: the store deduplicates by message id.
    again = sync_for(store, sl)
    assert asyncio.run(again.poll()) == 0 and again.counts["duplicate"] == 2
    assert store.summary()["self_synced"] == 2
    with pytest.raises(sqlite3.OperationalError):
        connect_ro(sl.path).execute("CREATE TABLE t(x)")


def test_sync_does_not_write_the_snowluma_file(sl):
    store = Store()
    request(store, "q1", T - 100)
    sl.add("a1", T - 90, reply="q1")
    before = sl.digest()
    names = {p.name for p in sl.path.parent.iterdir()}
    asyncio.run(sync_for(store, sl).poll())
    assert sl.digest() == before
    assert {p.name for p in sl.path.parent.iterdir()} == names  # no -wal / -shm / journal


def test_message_stored_by_the_send_path_is_not_stored_twice(sl):
    store = Store()
    _, q_seq = request(store, "q1", T - 100)
    sent = Message("BOT", GROUP, "nv1", "BOT", ({"type": "text", "data": {"text": "〔新版〕甲"}},))
    store.ingest(sent, T - 98, T - 98)
    store.mark_related(q_seq, [store.lookup("BOT", GROUP, "nv1")["seq"]], "answered_by")
    sl.add("nv1", T - 98, text="〔新版〕甲")
    sl.add("nv2", T - 95, text="〔新版〕乙")  # a 〔新版〕 send the send path did not store
    sl.add("a1", T - 90)  # the old bot's unquoted answer
    sync = sync_for(store, sl)
    assert asyncio.run(sync.poll()) == 2
    assert sync.counts["duplicate"] == 1 and sync.counts["paired_new_version"] == 1
    # 〔新版〕 sends never claim a request: the old bot's answer still pairs with q1.
    assert store.answer_map("BOT", GROUP)["a1"] == "q1"
    assert "nv2" not in store.answer_map("BOT", GROUP)


# -- answered_by ---------------------------------------------------------------------------


def test_unquoted_answers_pair_by_the_shadow_report_rule_and_quotes_link_directly(sl):
    store = Store()
    request(store, "q0", T - 500)  # outside the 300 s window of every answer below
    request(store, "q1", T - 200, user="U1")
    request(store, "q2", T - 150, user="U2")
    request(store, "q9", T - 150, group="20002")  # another group
    sl.add("a1", T - 140)
    sl.add("a2", T - 130)
    sl.add("a3", T - 120)  # no request left to claim
    sync = sync_for(store, sl)
    assert asyncio.run(sync.poll()) == 3
    assert store.answer_map("BOT", GROUP) == {"a1": "q1", "a2": "q2"}
    assert sync.counts["paired_unquoted"] == 2 and sync.counts["unpaired"] == 1
    # A later quoted answer to q1 links directly and releases a1's unquoted claim.
    sl.add("a4", T - 110, reply="q1")
    asyncio.run(sync.poll())
    assert store.answer_map("BOT", GROUP) == {"a4": "q1", "a2": "q2"}
    assert sync.counts["paired_quote"] == 1


def test_sync_never_starts_a_run(tmp_path):
    path = tmp_path / "messages.db"
    sl = FakeSnowLuma(path)

    async def main():
        h = Harness()
        h.feed(ev("m0", user="U1", group=GROUP, at=False, t=BASE - 30))
        sl.add("b1", BASE - 20, text="@BOT 你好")  # bot's own text, never a request
        sl.add("b2", BASE - 10, reply="m0")
        sync = SelfSync(h.store, "BOT", path, clock=wall)
        stored = await sync.poll()
        await h.settle()
        return h, stored

    h, stored = virtual(main)
    assert stored == 2
    assert h.queue.queue.empty() and h.runs() == {}
    assert h.store.con.execute("SELECT count(*) FROM reply_requests").fetchone()[0] == 0


# -- unreadable store ----------------------------------------------------------------------


def test_unreadable_store_warns_once_and_the_loop_continues(tmp_path):
    store = Store()
    request(store, "q1", T - 100)
    path = tmp_path / "messages.db"
    logs = []
    sync = SelfSync(store, "BOT", path, interval=0.01, clock=lambda: T, log=logs.append)

    async def loop_for(seconds):
        stop = asyncio.Event()
        task = asyncio.create_task(sync.run(stop))
        await asyncio.sleep(seconds)
        stop.set()
        return await task

    assert asyncio.run(loop_for(0.2)) == 0  # missing file
    path.write_bytes(b"not a sqlite database" * 100)
    assert asyncio.run(loop_for(0.2)) == 0  # corrupt file
    assert sync.counts["read_errors"] >= 4
    assert len([x for x in logs if "not readable" in x]) == 1
    path.unlink()
    sl = FakeSnowLuma(path)
    sl.add("a1", T - 90, reply="q1")
    assert asyncio.run(sync.poll()) == 1
    assert logs[-1].endswith("readable again")
    assert all(str(path) not in x and "q1" not in x for x in logs)


# -- immediate sync before a run's context -------------------------------------------------


def test_catch_up_syncs_once_when_the_askers_previous_request_is_unanswered(sl):
    store = Store()
    request(store, "q1", T - 60, user="U1")
    q2, q2_seq = request(store, "q2", T, user="U1")
    q3, q3_seq = request(store, "q3", T, user="U2")
    sl.add("a1", T - 50)
    sync = sync_for(store, sl)
    assert asyncio.run(sync.catch_up(q3, q3_seq)) == "skipped"  # U2 has no earlier request
    assert asyncio.run(sync.catch_up(q2, q2_seq)) == "synced"
    assert store.answer_map("BOT", GROUP) == {"a1": "q1"}
    assert asyncio.run(sync.catch_up(q2, q2_seq)) == "skipped"  # answered now
    assert sync.counts["catch_up"] == 1


def test_catch_up_is_bounded(sl):
    store = Store()
    request(store, "q1", T - 60)
    q2, q2_seq = request(store, "q2", T)

    def slow_reader(*args):
        time.sleep(0.6)
        return [], {}

    sync = sync_for(store, sl, reader=slow_reader, catch_up_seconds=0.1)

    async def main():
        started = time.perf_counter()
        outcome = await sync.catch_up(q2, q2_seq)
        elapsed = time.perf_counter() - started
        await sync.aclose()
        return outcome, elapsed

    outcome, elapsed = asyncio.run(main())
    assert outcome == "timeout" and elapsed < 0.5


def test_strategy_c_continues_with_the_old_bots_synced_answer(tmp_path):
    """Shadow trial: the old bot's answer reaches the store only through the sync; the
    asker's short follow-up picks it (strategy C `continues`) via the immediate sync."""
    path = tmp_path / "messages.db"
    sl = FakeSnowLuma(path)

    async def main():
        h = Harness()
        h.orch.self_sync = SelfSync(h.store, "BOT", path, clock=wall)
        h.feed(ev("q1", user="U1", group=GROUP, text="这个报错怎么修", t=BASE - 60))
        await h.settle()
        sl.add("a1", BASE - 50, text="把缓存清掉再重启")  # unquoted, never pushed
        h.feed(ev("q2", user="U1", group=GROUP, text="为啥", t=BASE))
        await h.settle()
        return h

    h = virtual(main)
    snap = h.store.context_snapshot(h.runs()["q2"]["run_id"])
    assert snap["reasons"]["a1"].startswith("continues")
    assert "q1" in snap["message_ids"]
    assert h.orch.counts["self_sync_synced"] == 1
    assert h.store.lookup("BOT", GROUP, "a1")["source"] == "self_sync"


# -- owner command messages are not context ---------------------------------------------------


def test_owner_command_message_is_excluded_from_history_and_related_set():
    store = Store()
    q0 = Message("BOT", "G1", "q0", "U2", ({"type": "text", "data": {"text": "原帖"}},))
    c1 = Message("BOT", "G1", "c1", "OWNER", ({"type": "text", "data": {"text": "#暂停"}},))
    n1 = Message("BOT", "G1", "n1", "U3", ({"type": "text", "data": {"text": "#暂停"}},))
    store.ingest(q0, BASE, BASE)
    c1_seq, _ = store.ingest(c1, BASE + 5, BASE + 5, kind="owner_command")
    n1_seq, _ = store.ingest(n1, BASE + 6, BASE + 6)
    q3 = Message("BOT", "G1", "q3", "U1", ({"type": "text", "data": {"text": "那这个呢"}},))
    q3_seq, _ = store.ingest(q3, BASE + 20, BASE + 20)
    history = [m.message_id for m in store.history(q3, q3_seq)]
    assert "c1" not in history and "n1" in history
    # Retention: a command is not related through a context mark; a `#新版` command of an
    # earlier release stays related only as the anchor of its own run.
    store.mark_related(q3_seq, [c1_seq, n1_seq], "context")
    related = related_set(store, Retention())
    assert c1_seq not in related and n1_seq in related
    store.con.execute("DELETE FROM related_marks WHERE seq=?", (c1_seq,))
    store.mark_related(q3_seq, [c1_seq], "new_version_command")
    assert c1_seq in related_set(store, Retention())


# -- settings / CLI ----------------------------------------------------------------------------


def test_serve_settings_and_flags(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        '[identity]\nbot_id = "1"\nowner_id = "2"\n'
        "[serve]\nself_sync = false\nself_sync_interval_s = 2.5\n"
        'snowluma_db = "sl/messages.db"\n',
        encoding="utf-8",
    )
    s = load_settings(path)
    assert (s.self_sync, s.self_sync_interval_s) == (False, 2.5)
    assert s.snowluma_db == (tmp_path / "sl" / "messages.db").resolve()
    with pytest.raises(SystemExit):
        app_main(["--no-self-sync"])  # serve-only flag
    with pytest.raises(SystemExit):
        app_main(["--serve", "--allow-owner-send"])  # retired with the `#新版` command
