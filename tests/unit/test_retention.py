import json
import os

import pytest

from group_agent.app import main
from group_agent.config import Retention
from group_agent.contracts import Message
from group_agent.retention import related_set, run_cleanup
from group_agent.store import Store

NOW = 2_000_000_000
H, D = 3600, 86400
AT_BOT = {"type": "at", "data": {"qq": "BOT"}}


def msg(mid, *segments, group="G1", user="U1"):
    return Message("BOT", group, mid, user, tuple(segments))


def reply(mid):
    return {"type": "reply", "data": {"id": mid}}


def text(value="x"):
    return {"type": "text", "data": {"text": value}}


def ids(store):
    return sorted(r[0] for r in store.con.execute("SELECT message_id FROM messages"))


@pytest.fixture
def store():
    value = Store()
    yield value
    value.close()


def test_recent_buffer_and_related_set(store):
    retention = Retention(recent_hours=24, related_days=7)
    store.ingest(msg("fresh", text()), NOW - 2 * H)
    store.ingest(msg("old", text()), NOW - 30 * H)
    store.ingest(msg("material", text()), NOW - 3 * D)
    store.ingest(msg("mid", reply("material")), NOW - 3 * D + 60)
    store.ingest_request(msg("ask", AT_BOT, reply("mid")), NOW - 3 * D + 120)
    store.ingest(msg("ancient", text()), NOW - 8 * D)
    store.ingest_request(msg("ancient_ask", AT_BOT, reply("ancient")), NOW - 8 * D)
    store.ingest(msg("other_group_old", text(), group="G2"), NOW - 30 * H)
    report = run_cleanup(store, retention, now=NOW)
    assert ids(store) == ["ask", "fresh", "material", "mid"]
    assert report["expired_recent"] == 2 and report["expired_hard"] == 2
    assert report["kept_related"] == 3 and report["requests_expired"] == 1
    assert store.summary()["reply_requests"] == 1


def test_reference_across_groups_is_not_related(store):
    store.ingest(msg("shared", text(), group="G2"), NOW - 2 * D)
    store.ingest_request(msg("ask", AT_BOT, reply("shared")), NOW - 2 * D)
    assert len(related_set(store, Retention())) == 1  # only the request itself


def test_branch_limits(store):
    store.ingest(msg("c0", text()), NOW - 2 * D)
    for i in range(1, 6):
        store.ingest(msg(f"c{i}", reply(f"c{i - 1}")), NOW - 2 * D)
    store.ingest_request(msg("ask", AT_BOT, reply("c5")), NOW - 2 * D)
    deep = related_set(store, Retention(max_branch_depth=2))
    assert len(deep) == 3  # ask, c5, c4
    capped = related_set(store, Retention(max_branch_messages=4))
    assert len(capped) == 4


def test_mark_related_keeps_snapshot_until_hard_limit(store):
    seq, _ = store.ingest(msg("context", text()), NOW - 2 * D)
    anchor, _, _ = store.ingest_request(msg("ask", AT_BOT), NOW - 2 * D)
    assert store.mark_related(anchor, [seq, 999], "context_snapshot") == 1
    run_cleanup(store, Retention(), now=NOW)
    assert ids(store) == ["ask", "context"]
    run_cleanup(store, Retention(), now=NOW + 6 * D)
    assert ids(store) == []
    assert store.con.execute("SELECT count(*) FROM related_marks").fetchone()[0] == 0


def test_capacity_drops_unrelated_oldest_first(store):
    retention = Retention(recent_hours=24, max_messages_per_group=3, max_branch_messages=3)
    store.ingest(msg("material", text()), NOW - 5 * H)
    store.ingest_request(msg("ask", AT_BOT, reply("material")), NOW - 5 * H)
    for i in range(3):
        store.ingest(msg(f"chat{i}", text()), NOW - (3 - i) * H)
    report = run_cleanup(store, retention, now=NOW)
    assert report["over_capacity"] == 2
    assert ids(store) == ["ask", "chat2", "material"]


def test_tombstones_age_out_and_unknown_time_is_kept(store):
    store.revoke("BOT", "G1", "gone", "U1", received_at=NOW - 30 * H)
    store.ingest(msg("no_time", text()))
    report = run_cleanup(store, Retention(), now=NOW)
    assert ids(store) == ["no_time"] and report["kept_no_time"] == 1


def test_runs_of_deleted_messages_are_removed(store):
    seq, _, _ = store.ingest_request(msg("ask", AT_BOT), NOW - 8 * D)
    run_id = store.start_run(seq)
    store.record_outbound(run_id, msg("ask", AT_BOT), "answer")
    run_cleanup(store, Retention(), now=NOW)
    assert store.summary()["runs"] == 0 and store.summary()["outbound_records"] == 0


def test_dry_run_counts_without_changes(store, tmp_path):
    store.ingest(msg("old", text()), NOW - 30 * H)
    before = ids(store)
    report = run_cleanup(store, Retention(), now=NOW, media_root=tmp_path, dry_run=True)
    assert report["dry_run"] is True and report["expired_recent"] == 1
    assert ids(store) == before


def add_media(store, seq, sha, size=10, created=0.0):
    asset = {"sha256": sha, "mime": "image/png", "bytes": size, "width": 1, "height": 1}
    asset["path"] = f"{sha}.png"
    store.add_media_slots(seq, [(0, None)])
    assert store.attach_media(seq, 0, asset, created)
    return asset["path"]


def test_media_orphans_cap_and_stray_files(store, tmp_path):
    old_seq, _ = store.ingest(msg("old", text()), NOW - 30 * H)
    new_seq, _ = store.ingest(msg("new", text()), NOW - H)
    other_seq, _ = store.ingest(msg("other", text()), NOW - H)
    keep_seq, _ = store.ingest(msg("keep", text()), NOW - H)
    shared, alone = "a" * 64, "b" * 64
    big_old, big_new = "c" * 64, "d" * 64
    (tmp_path / add_media(store, old_seq, alone)).write_bytes(b"1")
    add_media(store, new_seq, shared, created=3.0)
    (tmp_path / f"{shared}.png").write_bytes(b"1")
    half = 512 * 1024
    (tmp_path / add_media(store, other_seq, big_old, half, 1.0)).write_bytes(b"1")
    (tmp_path / add_media(store, keep_seq, big_new, half, 2.0)).write_bytes(b"1")
    stray = tmp_path / ("e" * 64 + ".jpg")
    stray.write_bytes(b"1")
    os.utime(stray, (NOW - 3600, NOW - 3600))
    fresh_part = tmp_path / ("f" * 64 + ".jpg.0123abcd.part")
    fresh_part.write_bytes(b"1")
    os.utime(fresh_part, (NOW - 10, NOW - 10))
    unrelated = tmp_path / "notes.txt"
    unrelated.write_bytes(b"1")
    retention = Retention(media_max_mb=1)
    dry = run_cleanup(store, retention, now=NOW, media_root=tmp_path, dry_run=True)
    assert (dry["media_orphaned"], dry["media_evicted"], dry["stray_files"]) == (1, 1, 1)
    report = run_cleanup(store, retention, now=NOW, media_root=tmp_path)
    assert report["files_deleted"] == 3
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == sorted([f"{shared}.png", f"{big_new}.png", fresh_part.name, "notes.txt"])
    evicted = store.message_media(other_seq)[0]
    assert (evicted["status"], evicted["reason"], evicted["sha256"]) == (
        "unavailable",
        "evicted",
        None,
    )


def test_cleanup_cli_dry_run(tmp_path, capsys):
    db = tmp_path / "shadow.sqlite3"
    store = Store(db)
    store.ingest(msg("old", text()), 1)
    store.close()
    assert main(["--cleanup", "--db", str(db), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "cleanup" and report["expired_hard"] == 1
    assert report["dry_run"] is True
    store = Store(db)
    assert ids(store) == ["old"]
    store.close()
    assert main(["--cleanup", "--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["files_deleted"] == 0
    store = Store(db)
    assert ids(store) == []
    store.close()
