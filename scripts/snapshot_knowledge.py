"""Take a consistent read-only snapshot of the legacy knowledge base (T0.4, KB-01).

Uses the SQLite online backup API from a read-only connection, so the running AstrBot
plugin is neither locked for long nor modified. Writes:

    <out>/knowledge-<UTC timestamp>.sqlite3
    <out>/knowledge-<UTC timestamp>.manifest.json   (counts, hashes; no knowledge text)

Usage:
    python scripts/snapshot_knowledge.py --source <knowledge.sqlite3> [--out data/fixtures]
        [--media-dir <dir of teaching images>] [--expect-entries N --expect-revisions N]
"""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table_counts(con: sqlite3.Connection) -> dict[str, int]:
    names = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            " ORDER BY name"
        )
    ]
    return {n: con.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0] for n in names}


def grouped(con: sqlite3.Connection, table: str, column: str) -> dict[str, int] | None:
    cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
    if column not in cols:
        return None
    return {
        str(k): v
        for k, v in con.execute(
            f'SELECT "{column}", count(*) FROM "{table}" GROUP BY "{column}" ORDER BY 1'
        )
    }


def content_digest(con: sqlite3.Connection, table: str) -> str:
    """Order-independent digest of all rows, for later comparison with a migrated copy."""
    rows = sorted(
        json.dumps(list(r), ensure_ascii=False, default=str)
        for r in con.execute(f'SELECT * FROM "{table}"')
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def snapshot(source: Path, out_dir: Path, media_dir: Path | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / f"knowledge-{stamp}.sqlite3"
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)  # one consistent read transaction
    finally:
        src.close()
    dst.execute("PRAGMA journal_mode=DELETE")
    integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
    counts = table_counts(dst)
    manifest = {
        "snapshot": target.name,
        "taken_at_utc": stamp,
        "source_name": source.name,
        "integrity_check": integrity,
        "table_counts": counts,
        "status_breakdown": {
            f"{t}.{c}": s
            for t in counts
            for c in ("status", "kind", "active")
            if (s := grouped(dst, t, c)) is not None
        },
        "row_digests": {t: content_digest(dst, t) for t in counts},
    }
    dst.close()
    manifest["snapshot_sha256"] = sha256_file(target)
    if media_dir is not None and media_dir.is_dir():
        files = sorted(p for p in media_dir.rglob("*") if p.is_file())
        manifest["media"] = {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "sha256_by_relative_path": {
                p.relative_to(media_dir).as_posix(): sha256_file(p) for p in files
            },
        }
    manifest_path = out_dir / f"knowledge-{stamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = manifest_path.as_posix()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/fixtures"))
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--expect-entries", type=int)
    parser.add_argument("--expect-revisions", type=int)
    args = parser.parse_args()
    manifest = snapshot(args.source, args.out, args.media_dir)
    counts = manifest["table_counts"]
    print(
        json.dumps(
            {
                "snapshot": manifest["snapshot"],
                "integrity_check": manifest["integrity_check"],
                "table_counts": counts,
                "status_breakdown": manifest["status_breakdown"],
                "media_files": manifest.get("media", {}).get("files"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    ok = manifest["integrity_check"] == "ok"
    if args.expect_entries is not None:
        ok &= counts.get("entries") == args.expect_entries
    if args.expect_revisions is not None:
        ok &= counts.get("revisions") == args.expect_revisions
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
