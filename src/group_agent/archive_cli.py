"""`qunbot archive`: import SnowLuma history into the text archive, show or clear it.

    qunbot archive stats  --instance data/production
    qunbot archive import --instance data/production [--snowluma-db PATH] [--dry-run]
    qunbot archive clear  --instance data/production --group GID [--user UID | --since DATE]

The import reads a backup copy of SnowLuma's message store (the live file is only read
through SQLite's backup API) and is safe while the service runs. Group and account ids are
printed only as their last four digits.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from .config import ConfigError


def _mask(value: str) -> str:
    return "…" + str(value)[-4:]


def _settings(instance: Path):
    from .config_service import load_bundle

    return load_bundle(instance.resolve() / "config-history").settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="qunbot archive", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("stats", "import", "clear"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--instance", type=Path, required=True)
        if name == "import":
            cmd.add_argument("--snowluma-db", type=Path)
            cmd.add_argument("--dry-run", action="store_true")
        if name == "clear":
            cmd.add_argument("--group", required=True)
            who = cmd.add_mutually_exclusive_group(required=True)
            who.add_argument("--user")
            who.add_argument("--since", help="YYYY-MM-DD: clear the group's archive from this day")
            who.add_argument("--all", action="store_true", help="clear the whole group")
    args = parser.parse_args(argv)
    try:
        settings = _settings(args.instance)
    except ConfigError as exc:
        parser.error(str(exc))
    from . import archive
    from .admission import AccessPolicy
    from .store import Store

    db = settings.serve_db
    if not db.is_file():
        parser.error("archive: the instance's message store does not exist yet")
    store = Store(db)
    try:
        con = store.con
        if args.action == "stats":
            rows = archive.stats(con, settings.bot_id)
            out = []
            for r in rows:
                spec = settings.archive_for_group(r["group_id"])
                out.append(
                    {
                        "group": _mask(r["group_id"]),
                        "messages": r["messages"],
                        "from": datetime.fromtimestamp(r["earliest"]).strftime("%Y-%m-%d"),
                        "to": datetime.fromtimestamp(r["latest"]).strftime("%Y-%m-%d"),
                        "enabled": spec.enabled,
                        "retention_days": spec.retention_days,
                    }
                )
            print(
                json.dumps(
                    {"archive": out, "total": sum(r["messages"] for r in rows)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.action == "clear":
            since = None
            if args.since:
                since = datetime.strptime(args.since, "%Y-%m-%d").timestamp()
            n = archive.clear(con, settings.bot_id, str(args.group), user_id=args.user, since=since)
            print(json.dumps({"cleared": n, "group": _mask(args.group)}, ensure_ascii=False))
            return 0
        snowluma = args.snowluma_db or settings.snowluma_db
        if snowluma is None or not Path(snowluma).is_file():
            parser.error("archive import: SnowLuma messages.db not found (use --snowluma-db)")
        snap = AccessPolicy.from_settings(settings).snapshot

        def allowed(group: str, user: str) -> bool:
            return (
                settings.archive_for_group(group).enabled
                and group not in snap.blocked_groups
                and group not in snap.collection_off_groups
                and user not in snap.blocked_users
            )

        groups_on = any(settings.archive_for_group(g).enabled for g in settings.group_profiles)
        if not (settings.archive.enabled or groups_on):
            print(
                "archive import: the archive is off (enable [archive] or a group's archive first)",
                file=sys.stderr,
            )
            return 1
        report = archive.import_snowluma(
            con,
            Path(snowluma),
            settings.bot_id,
            allowed,
            lambda g: settings.archive_for_group(g).retention_days,
            now=time.time(),
            dry_run=args.dry_run,
        )
        report["groups"] = {_mask(g): v for g, v in report["groups"].items()}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
