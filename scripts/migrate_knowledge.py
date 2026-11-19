"""Migrate the legacy knowledge base into the framework's own base and check it (T4.1).

Reads a snapshot taken by `scripts/snapshot_knowledge.py` (SQLite backup API, read-only on
the AstrBot side) and the legacy image directory (read-only), writes
`data/knowledge/knowledge.sqlite3` (+ `images/`), then compares entry by entry. Safe to
re-run (idempotent). The report has counts and ids only, no knowledge text.

Usage:
    python scripts/migrate_knowledge.py [--snapshot <file> | --source <legacy sqlite3>]
        [--media-dir <legacy images dir>] [--kb data/knowledge/knowledge.sqlite3]
        [--report data/knowledge/migration-report.json] [--verify-only]

Without --snapshot/--source the newest data/fixtures/knowledge-*.sqlite3 is used;
--source takes a fresh snapshot first. Exit status 1 when the check finds mismatches.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group_agent.knowledge_base import KnowledgeBase  # noqa: E402
from group_agent.knowledge_migration import migrate, verify  # noqa: E402


def take_snapshot(source: Path, out: Path, media_dir: Path | None) -> Path:
    spec = importlib.util.spec_from_file_location(
        "snapshot_knowledge", ROOT / "scripts/snapshot_knowledge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.snapshot(source, out, media_dir)
    if manifest["integrity_check"] != "ok":
        raise SystemExit("snapshot integrity check failed")
    return out / manifest["snapshot"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source", type=Path, help="legacy knowledge.sqlite3 (snapshot first)")
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--kb", type=Path, default=ROOT / "data/knowledge/knowledge.sqlite3")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "data/fixtures")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    media_dir = args.media_dir if args.media_dir and args.media_dir.is_dir() else None
    if args.source is not None:
        snapshot = take_snapshot(args.source, args.fixtures, media_dir)
    elif args.snapshot is not None:
        snapshot = args.snapshot
    else:
        found = sorted(args.fixtures.glob("knowledge-*.sqlite3"))
        if not found:
            parser.error("no snapshot: pass --snapshot or --source")
        snapshot = found[-1]
    kb = KnowledgeBase(args.kb)
    try:
        result = None if args.verify_only else migrate(snapshot, kb, media_dir)
        report = verify(snapshot, kb, media_dir)
        report["migration"] = result
        report["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if result is not None:
            with kb.con:
                kb.con.execute(
                    "INSERT INTO imports VALUES (?,?,?)",
                    (report["snapshot_sha256"], time.time(), json.dumps(report)),
                )
    finally:
        kb.close()
    path = args.report or args.kb.with_name("migration-report.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = {k: report[k] for k in ("snapshot", "legacy", "base", "checked", "ok")}
    shown["mismatches"] = len(report["mismatches"])
    if result is not None:
        shown["migration"] = {k: (len(v) if isinstance(v, list) else v) for k, v in result.items()}
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
