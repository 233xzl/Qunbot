"""Legacy knowledge snapshot -> the framework's knowledge base, with a per-entry check
(T4.1, KB-01, AC-16).

Input is a consistent snapshot of the AstrBot plugin store (`scripts/snapshot_knowledge.py`,
read-only backup API) plus the plugin's image directory (read-only). Every legacy entry,
every revision and every referenced example image is carried over unchanged:

- entry `data` JSON verbatim (aliases, category, explanation, technical meaning, source,
  examples, images, supersedes, CHIME original fields), so `content_sha256` of the new row
  equals the hash of the legacy JSON; scope `public` (the legacy library is shared by all
  groups, REQUIREMENTS 3.1); origin `migrated`;
- revision rows with their original timestamps (op `migrated`);
- images copied to `<kb dir>/images/<sha256>.<ext>` (the legacy relative path), the hash
  checked against the file name and the copy.

Re-running is idempotent: identical rows are left alone; an entry whose legacy side moved
on (a newer legacy revision, still before the cutover) is updated, unless the new base has
changed it by teaching since - then it is reported as `diverged` and kept. The check report
holds counts and ids only, never knowledge text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from .knowledge_base import KnowledgeBase, canonical, content_hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy(snapshot: Path) -> tuple[dict, dict]:
    con = sqlite3.connect(f"{Path(snapshot).resolve().as_uri()}?mode=ro", uri=True)
    try:
        entries = {
            r[0]: {
                "kind": r[1],
                "name": r[2],
                "revision": int(r[3]),
                "active": bool(r[4]),
                "data": json.loads(r[5]),
            }
            for r in con.execute("SELECT id, kind, name, revision, active, data FROM entries")
        }
        revisions = {
            (r[0], int(r[1])): {"data": json.loads(r[2]), "created": float(r[3])}
            for r in con.execute("SELECT id, revision, data, created FROM revisions")
        }
    finally:
        con.close()
    return entries, revisions


def _images(entries: dict) -> list[str]:
    return sorted({str(p) for e in entries.values() for p in e["data"].get("images") or []})


def migrate(snapshot: Path, kb: KnowledgeBase, media_dir: Path | None, now: float | None = None):
    """Copy the snapshot into `kb`; returns counts (and ids of conflicts)."""
    now = time.time() if now is None else now
    entries, revisions = _legacy(snapshot)
    out = {
        "revisions_inserted": 0,
        "revisions_present": 0,
        "revision_conflicts": [],
        "entries_inserted": 0,
        "entries_updated": 0,
        "entries_unchanged": 0,
        "entries_diverged": [],
        "media_copied": 0,
        "media_present": 0,
        "media_missing": [],
    }
    con = kb.con
    with con:
        for (kid, rev), item in sorted(revisions.items()):
            data, digest = item["data"], content_hash(item["data"])
            have = con.execute(
                "SELECT content_sha256 FROM revisions WHERE id=? AND revision=?", (kid, rev)
            ).fetchone()
            if have is None:
                con.execute(
                    "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        kid,
                        rev,
                        "public",
                        int(bool(data.get("active", True))),
                        canonical(data),
                        digest,
                        item["created"],
                        "migrated",
                        None,
                        None,
                    ),
                )
                out["revisions_inserted"] += 1
            elif have[0] == digest:
                out["revisions_present"] += 1
            else:
                out["revision_conflicts"].append(kid)
        for kid, e in sorted(entries.items()):
            digest = content_hash(e["data"])
            created = [v["created"] for (i, _), v in revisions.items() if i == kid] or [now]
            row = con.execute("SELECT * FROM entries WHERE id=?", (kid,)).fetchone()
            values = (
                e["name"],
                "public",
                e["revision"],
                int(e["active"]),
                canonical(e["data"]),
                digest,
            )
            if row is None:
                con.execute(
                    "INSERT INTO entries (id, kind, name, scope, revision, active, data,"
                    " content_sha256, origin, author, source_message, created, updated)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (kid, e["kind"], *values, "migrated", None, None, min(created), max(created)),
                )
                out["entries_inserted"] += 1
                continue
            same = (row["content_sha256"], row["revision"], bool(row["active"])) == (
                digest,
                e["revision"],
                e["active"],
            )
            if same:
                out["entries_unchanged"] += 1
                continue
            taught = con.execute(
                "SELECT count(*) FROM revisions WHERE id=? AND op!='migrated'", (kid,)
            ).fetchone()[0]
            if row["origin"] == "migrated" and not taught and e["revision"] >= row["revision"]:
                con.execute(
                    "UPDATE entries SET name=?, scope=?, revision=?, active=?, data=?,"
                    " content_sha256=?, updated=? WHERE id=?",
                    (*values, max(created), kid),
                )
                out["entries_updated"] += 1
            else:
                out["entries_diverged"].append(kid)
    for relative in _images(entries):
        name = Path(relative).name
        source = media_dir / name if media_dir is not None else None
        if source is None or not source.is_file():
            out["media_missing"].append(name)
            continue
        digest = sha256_file(source)
        target = kb.root / relative
        if target.is_file() and sha256_file(target) == digest:
            out["media_present"] += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, tmp)
            tmp.replace(target)
            out["media_copied"] += 1
        with con:
            con.execute(
                "INSERT OR IGNORE INTO media VALUES (?,?,?,?)",
                (digest, relative, target.stat().st_size, now),
            )
    kb._writes += 1  # search index rebuilds on next use
    return out


def verify(snapshot: Path, kb: KnowledgeBase, media_dir: Path | None) -> dict:
    """Per-entry comparison of the snapshot with the base: counts and mismatching ids."""
    entries, revisions = _legacy(snapshot)
    mismatches: list[dict] = []
    changed_since = 0

    def bad(kid: str, what: str) -> None:
        mismatches.append({"id": kid, "field": what})

    for (kid, rev), item in sorted(revisions.items()):
        row = kb.con.execute(
            "SELECT content_sha256 FROM revisions WHERE id=? AND revision=?", (kid, rev)
        ).fetchone()
        if row is None:
            bad(kid, f"revision {rev} missing")
        elif row[0] != content_hash(item["data"]):
            bad(kid, f"revision {rev} hash")
    for kid, e in sorted(entries.items()):
        row = kb.con.execute("SELECT * FROM entries WHERE id=?", (kid,)).fetchone()
        if row is None:
            bad(kid, "entry missing")
            continue
        taught = kb.con.execute(
            "SELECT count(*) FROM revisions WHERE id=? AND op!='migrated'", (kid,)
        ).fetchone()[0]
        if taught:  # changed by owner teaching after the migration: compare history only
            changed_since += 1
            continue
        data = json.loads(row["data"])
        checks = {
            "kind": row["kind"] == e["kind"],
            "name": row["name"] == e["name"],
            "revision": row["revision"] == e["revision"],
            "active": bool(row["active"]) == e["active"],
            "scope": row["scope"] == "public",
            "content_sha256": row["content_sha256"] == content_hash(e["data"])
            and content_hash(data) == row["content_sha256"],
            "aliases": data.get("aliases") == e["data"].get("aliases"),
            "source": data.get("source") == e["data"].get("source"),
            "supersedes": data.get("supersedes") == e["data"].get("supersedes"),
            "images": data.get("images") == e["data"].get("images"),
        }
        for what, ok in checks.items():
            if not ok:
                bad(kid, what)
    media_ok = 0
    for relative in _images(entries):
        target = kb.root / relative if kb.root is not None else None
        name = Path(relative).name
        if target is None or not target.is_file():
            bad(name, "image missing")
            continue
        digest = sha256_file(target)
        expected = Path(name).stem
        source = media_dir / name if media_dir is not None else None
        if digest != expected or (
            source is not None and source.is_file() and sha256_file(source) != digest
        ):
            bad(name, "image hash")
        else:
            media_ok += 1
    kinds: dict[str, int] = {}
    for e in entries.values():
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    new_only = (
        kb.con.execute(
            "SELECT origin, count(*) FROM entries WHERE id NOT IN (%s) GROUP BY origin"
            % ",".join("?" * len(entries)),
            tuple(entries),
        ).fetchall()
        if entries
        else []
    )
    return {
        "snapshot": Path(snapshot).name,
        "snapshot_sha256": sha256_file(Path(snapshot)),
        "legacy": {
            "entries": len(entries),
            "by_kind": kinds,
            "active": sum(e["active"] for e in entries.values()),
            "revisions": len(revisions),
            "aliases": sum(len(e["data"].get("aliases") or []) for e in entries.values()),
            "images": len(_images(entries)),
        },
        "base": kb.counts(),
        "checked": {
            "entries": len(entries),
            "revisions": len(revisions),
            "images_ok": media_ok,
            "changed_by_teaching_since": changed_since,
            "new_entries_not_in_snapshot": dict(new_only),
        },
        "mismatches": mismatches,
        "ok": not mismatches,
    }
