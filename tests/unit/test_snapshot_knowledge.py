import importlib.util
import json
from pathlib import Path
import sqlite3

SPEC = importlib.util.spec_from_file_location(
    "snapshot_knowledge", Path("scripts/snapshot_knowledge.py")
)
snapshot_knowledge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_knowledge)


def make_source(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        "CREATE TABLE entries (id TEXT, kind TEXT, active INTEGER, data TEXT);"
        "CREATE TABLE revisions (id TEXT, revision INTEGER, data TEXT);"
        "INSERT INTO entries VALUES ('K1','base',1,'{}'),('K2','owner',1,'{}');"
        "INSERT INTO revisions VALUES ('K1',1,'{}'),('K2',1,'{}');"
    )
    con.commit()
    con.close()


def test_snapshot_is_consistent_and_leaves_source_alone(tmp_path):
    source = tmp_path / "knowledge.sqlite3"
    make_source(source)
    media = tmp_path / "images"
    media.mkdir()
    (media / "a.png").write_bytes(b"fake")
    before = source.read_bytes()

    manifest = snapshot_knowledge.snapshot(source, tmp_path / "out", media)

    assert source.read_bytes() == before
    assert manifest["integrity_check"] == "ok"
    assert manifest["table_counts"] == {"entries": 2, "revisions": 2}
    assert manifest["status_breakdown"]["entries.kind"] == {"base": 1, "owner": 1}
    assert manifest["media"]["files"] == 1
    saved = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert saved["snapshot_sha256"] == manifest["snapshot_sha256"]
    copy = sqlite3.connect(tmp_path / "out" / manifest["snapshot"])
    assert copy.execute("SELECT count(*) FROM entries").fetchone()[0] == 2
    copy.close()


def test_row_digest_ignores_row_order(tmp_path):
    a, b = sqlite3.connect(tmp_path / "a.db"), sqlite3.connect(tmp_path / "b.db")
    for con, rows in ((a, [(1,), (2,)]), (b, [(2,), (1,)])):
        con.execute("CREATE TABLE t (x INTEGER)")
        con.executemany("INSERT INTO t VALUES (?)", rows)
    assert snapshot_knowledge.content_digest(a, "t") == snapshot_knowledge.content_digest(b, "t")
    a.close()
    b.close()
