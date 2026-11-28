"""CLI: offline replay, shadow ingestion, online record-only mode, or retention cleanup.

Nothing here sends QQ messages. The shadow uses read-only actions only on explicit opt-in:
`get_msg` with --backfill-replies, `get_image` with --collect (which also records reply
admission decisions without running anything). `--serve` (serve.py) runs answers with the
read-only action allowlist and records outbound intents as `recorded_not_sent`.
"""

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from .admission import AccessPolicy, RecordOnly
from .config import ConfigError, load_snowluma_credentials
from .media import ImageSourcePolicy, MediaCollector
from .pipeline import Pipeline
from .retention import periodic_cleanup, run_cleanup
from .shadow import ShadowIngestor, run_shadow
from .store import Store

STORE_KEYS = ("messages", "revoked", "backfilled", "reply_requests", "media_slots", "media_files")


def media_dir_for(settings, args, db_path: Path) -> Path:
    """Media directory belongs to one database: orphan-file cleanup deletes files the
    database does not know, so a --db override never shares the configured directory."""
    if args.media_dir is not None:
        return args.media_dir
    if args.db is None and settings.collection.media_dir is not None:
        return settings.collection.media_dir
    return db_path.with_name(db_path.stem + ".media")


def cleanup_main(settings, args) -> int:
    db_path = args.db or settings.shadow_db
    if not db_path.is_file():
        print(json.dumps({"mode": "cleanup", "error": "database_not_found"}))
        return 1
    store = Store(db_path)
    try:
        report = run_cleanup(
            store,
            settings.retention,
            media_root=media_dir_for(settings, args, db_path),
            dry_run=args.dry_run,
        )
    finally:
        store.close()
    print(json.dumps({"mode": "cleanup", **report}, ensure_ascii=False, indent=2))
    return 0


def shadow_main(parser, settings, args) -> int:
    if settings.bot_id == "BOT":
        parser.error("shadow: identity.bot_id is the example placeholder; use local settings")
    try:
        credentials = load_snowluma_credentials(settings.credentials)
    except ConfigError as exc:
        parser.error(str(exc))
    db_path = args.db or settings.shadow_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    media_root = media_dir_for(settings, args, db_path)
    store = Store(db_path)
    sink = RecordOnly() if args.collect else None
    ingestor = ShadowIngestor(
        store,
        settings.bot_id,
        access=AccessPolicy.from_settings(settings),
        admit=args.collect,
        sink=sink,
    )
    started = time.time()

    def log(text: str) -> None:
        print(f"[shadow {time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)

    backfill_counts = Counter()
    collection = settings.collection
    media_holder: list[MediaCollector] = []

    def media_factory(api):
        media = MediaCollector(
            store,
            api,
            media_root,
            policy=ImageSourcePolicy(collection.image_host_suffixes),
            max_bytes=collection.max_image_mb * 1024 * 1024,
            rate_per_second=collection.download_per_second,
            max_concurrency=collection.max_concurrent_downloads,
            max_pending=collection.max_pending_downloads,
            log=log,
        )
        media_holder.append(media)
        return media

    cleanup_reports: list[dict] = []

    async def main():
        stop = asyncio.Event()
        if args.duration:
            asyncio.get_running_loop().call_later(args.duration, stop.set)
        tasks = []
        if not args.no_cleanup:
            tasks.append(
                periodic_cleanup(
                    stop,
                    settings.retention.cleanup_interval_minutes * 60,
                    lambda: run_cleanup(store, settings.retention, media_root=media_root),
                    on_report=cleanup_reports.append,
                    log=log,
                )
            )
        shadow = run_shadow(
            credentials.ws_url,
            credentials.token,
            ingestor,
            stop,
            log=log,
            backfill_replies=args.backfill_replies,
            backfill_counts=backfill_counts,
            media_factory=media_factory if args.collect else None,
        )
        results = await asyncio.gather(shadow, *tasks)
        return results[0]

    stats = Counter()
    try:
        stats = asyncio.run(main())
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        read_actions = []
        if args.backfill_replies:
            read_actions.append("get_msg")
        if args.collect:
            read_actions.append("get_image")
        summary = {
            "mode": "shadow_read_only",
            "read_actions": read_actions,
            "started_at": started,
            "ended_at": time.time(),
            "connection": dict(stats),
            "events": dict(ingestor.counts),
            "store": {k: v for k, v in store.summary().items() if k in STORE_KEYS},
            "sent_to_qq": 0,
            "runs_started": 0,
        }
        if args.backfill_replies:
            summary["backfill"] = dict(backfill_counts)
        if args.collect:
            summary["admission"] = dict(sink.counts)
            summary["media"] = dict(media_holder[0].counts) if media_holder else {}
        if cleanup_reports:
            first, last = cleanup_reports[0], cleanup_reports[-1]
            summary["cleanup"] = {
                "runs": len(cleanup_reports),
                "first": {k: v for k, v in first.items() if k != "now"},
                "last": {k: v for k, v in last.items() if k != "now"},
            }
        store.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def serve_main(parser, settings, args) -> int:
    """Online mode, record only (serve.py): runs answers, never sends."""
    from .runtime import run_controlled as run_serve

    if settings.bot_id == "BOT":
        parser.error("serve: identity.bot_id is the example placeholder; use local settings")
    if args.budget_cny is not None and args.budget_cny <= 0:
        parser.error("--budget-cny must be positive")
    inject = None
    if args.inject is not None:
        try:
            inject = json.loads(args.inject.read_text(encoding="utf-8"))
            if isinstance(inject, dict):
                inject = [inject]
            if not isinstance(inject, list) or not all(isinstance(e, dict) for e in inject):
                raise ValueError
        except (OSError, ValueError):
            parser.error("--inject: expected a JSON event object or array")
    if args.dry_models:
        # Fake models never reach QQ: their placeholder text must not be sent, and the
        # settings' send switches are overridden (explicit send flags are refused above).
        args.no_live_send = args.no_owner_send = args.no_reaction = True
        args.live_send = args.allow_owner_send = args.reaction = False
    db_path = args.db or settings.serve_db
    # The media directory belongs to this database (orphan cleanup deletes unknown files).
    media_root = media_dir_for(settings, args, db_path)

    def log(text: str) -> None:
        print(f"[serve {time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)

    from .config_service import ConfigService

    service = ConfigService(settings.config_store) if settings.config_store else None
    adopted = False

    def on_ready():
        nonlocal adopted
        if service:
            service.record_runtime(settings, "adopted")
        adopted = True

    try:
        summary = asyncio.run(
            run_serve(
                settings,
                db_path=db_path,
                media_root=media_root,
                duration=args.duration,
                dry_models=args.dry_models,
                budget_cny=args.budget_cny,
                web=not args.no_web,
                cleanup=not args.no_cleanup,
                inject=inject,
                knowledge_path=args.knowledge,
                log=log,
                owner_send=False
                if args.no_owner_send
                else (True if args.allow_owner_send else None),
                self_sync=False if args.no_self_sync else None,
                snowluma_db=args.snowluma_db,
                reaction=False if args.no_reaction else (True if args.reaction else None),
                live_send=False if args.no_live_send else (True if args.live_send else None),
                stop_file=args.stop_file,
                account_book=not args.no_account_book,
                on_ready=on_ready,
            )
        )
    except ConfigError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    finally:
        if service:
            service.record_runtime(settings, "stopped" if adopted else "failed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cli_settings(args):
    from .config_service import load_bundle

    overrides = {}
    path_fields = {
        "knowledge": "knowledge.db_path",
        "snowluma_db": "serve.snowluma_db",
        "media_dir": "collection.media_dir",
    }
    if args.db is not None:
        overrides["serve.db_path" if args.serve else "shadow.db_path"] = str(args.db.resolve())
        if args.media_dir is None:
            overrides["collection.media_dir"] = str(
                args.db.with_name(args.db.stem + ".media").resolve()
            )
    for flag, key in path_fields.items():
        value = getattr(args, flag)
        if value is not None:
            overrides[key] = str(value.resolve())
    for positive, negative, key in (
        ("live_send", "no_live_send", "live_send"),
        ("allow_owner_send", "no_owner_send", "owner_command_send"),
        ("reaction", "no_reaction", "processing_reaction"),
    ):
        if getattr(args, positive) or getattr(args, negative) or args.dry_models:
            overrides[f"serve.{key}"] = bool(getattr(args, positive)) and not args.dry_models
    if args.no_self_sync:
        overrides["serve.self_sync"] = False
    if args.no_web:
        overrides["agent.web_enabled"] = False
    if args.bot_id:
        overrides["identity.bot_id"] = args.bot_id
    return load_bundle(args.settings, overrides).settings


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "admin":
        from .admin.cli import main as admin_main

        return admin_main(argv[1:])
    if argv and argv[0] == "config":
        from .config_cli import main as config_main

        return config_main(argv[1:])
    if argv and argv[0] == "archive":
        from .archive_cli import main as archive_main

        return archive_main(argv[1:])
    from . import __version__

    parser = argparse.ArgumentParser(description="Qunbot QQ group chat backend")
    parser.add_argument("--version", action="version", version=f"Qunbot {__version__}")
    parser.add_argument("--settings", type=Path, default=Path("config/settings.example.toml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--replay", type=Path, help="JSON case containing events, or a raw event array"
    )
    mode.add_argument(
        "--shadow",
        action="store_true",
        help="receive group messages/recalls from local SnowLuma into the store; never sends",
    )
    mode.add_argument(
        "--cleanup",
        action="store_true",
        help="apply the retention rules once to the store (default: the shadow database)",
    )
    mode.add_argument(
        "--serve",
        action="store_true",
        help="online mode: intake, scheduling and answers, outbound recorded only (never sent)",
    )
    parser.add_argument("--bot-id", help="override replay bot identity")
    parser.add_argument("--duration", type=float, help="shadow/serve: stop after this many seconds")
    parser.add_argument(
        "--db", type=Path, help="shadow/serve/cleanup: SQLite path (default from settings)"
    )
    parser.add_argument(
        "--dry-models",
        action="store_true",
        help="serve: local fake models (no model/search credentials, zero cost)",
    )
    parser.add_argument(
        "--budget-cny", type=float, help="serve: hard spending limit for this process (CNY)"
    )
    parser.add_argument("--no-web", action="store_true", help="serve: no web search/fetch")
    parser.add_argument(
        "--knowledge",
        type=Path,
        help="serve: knowledge base path (default: settings [knowledge] db_path)",
    )
    parser.add_argument(
        "--allow-owner-send",
        action="store_true",
        help="serve: let the owner's #新版 command send its answer (D-19; the only send path)",
    )
    parser.add_argument(
        "--no-owner-send",
        action="store_true",
        help="serve: record-only even when settings turn owner_command_send on",
    )
    parser.add_argument(
        "--reaction",
        action="store_true",
        help="serve: put the 'processing' emoji on admitted requests (D-20; off in the shadow)",
    )
    parser.add_argument(
        "--no-reaction",
        action="store_true",
        help="serve: no 'processing' emoji even when settings turn processing_reaction on",
    )
    parser.add_argument(
        "--live-send",
        action="store_true",
        help="serve: send every answer to QQ (cutover; switches the #新版 command off)",
    )
    parser.add_argument(
        "--no-live-send",
        action="store_true",
        help="serve: record-only even when settings turn live_send on",
    )
    parser.add_argument(
        "--no-account-book",
        action="store_true",
        help="serve: do not append calls to the account-wide usage book (tests, experiments)",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="serve: stop gracefully once this file exists (used by Stop-GroupAgent.cmd)",
    )
    parser.add_argument(
        "--no-self-sync",
        action="store_true",
        help="serve: do not read the bot's own messages from SnowLuma's message store",
    )
    parser.add_argument(
        "--snowluma-db",
        type=Path,
        help="serve: SnowLuma messages.db for the read-only self-sync (default from bot id)",
    )
    parser.add_argument(
        "--inject",
        type=Path,
        help="serve: JSON event(s) fed through the normal intake after connecting (smoke test)",
    )
    parser.add_argument(
        "--backfill-replies",
        action="store_true",
        help="shadow: fetch referenced messages missing locally with read-only get_msg",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="shadow: save images (read-only get_image) and record reply admission decisions",
    )
    parser.add_argument(
        "--media-dir", type=Path, help="shadow/cleanup: media directory (default: <db>.media)"
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", help="shadow: skip the periodic retention cleanup"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="cleanup: only count what would be deleted"
    )
    args = parser.parse_args(argv)
    for flag in ("backfill_replies", "collect"):
        if getattr(args, flag) and not args.shadow:
            parser.error(f"--{flag.replace('_', '-')} requires --shadow")
    if args.no_cleanup and not (args.shadow or args.serve):
        parser.error("--no-cleanup requires --shadow or --serve")
    for flag in (
        "dry_models",
        "budget_cny",
        "no_web",
        "knowledge",
        "inject",
        "allow_owner_send",
        "no_owner_send",
        "no_self_sync",
        "snowluma_db",
        "reaction",
        "no_reaction",
        "live_send",
        "no_live_send",
        "no_account_book",
        "stop_file",
    ):
        if getattr(args, flag) not in (None, False) and not args.serve:
            parser.error(f"--{flag.replace('_', '-')} requires --serve")
    if args.live_send and args.no_live_send:
        parser.error("--live-send and --no-live-send exclude each other")
    for flag in ("live_send", "allow_owner_send", "reaction"):
        if args.dry_models and getattr(args, flag):
            parser.error(f"--{flag.replace('_', '-')} cannot be combined with --dry-models")
    if args.allow_owner_send and args.no_owner_send:
        parser.error("--allow-owner-send and --no-owner-send exclude each other")
    if args.reaction and args.no_reaction:
        parser.error("--reaction and --no-reaction exclude each other")
    if args.dry_run and not args.cleanup:
        parser.error("--dry-run requires --cleanup")
    if args.shadow or args.cleanup or args.serve:
        try:
            settings = cli_settings(args)
        except ConfigError as exc:
            parser.error(str(exc))
        if args.cleanup:
            return cleanup_main(settings, args)
        if args.serve:
            return serve_main(parser, settings, args)
        return shadow_main(parser, settings, args)
    try:
        settings = cli_settings(args)
        payload = json.loads(args.replay.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            events = payload["events"]
            env = payload.get("environment", {})
            if not isinstance(env, dict):
                raise ValueError
            for name in ("bot_id", "owner_id"):
                value = env.get(name, getattr(settings, name))
                if name == "owner_id" and value is None:
                    continue  # Real extraction has no verified owner mapping; do not infer it.
                if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
                    raise ValueError
            for name in ("blocked_users", "blocked_groups", "blacklist"):
                values = env.get(name, [])
                if not isinstance(values, list) or any(
                    isinstance(v, bool) or not isinstance(v, (str, int)) or not str(v)
                    for v in values
                ):
                    raise ValueError
            if env.get("config_overrides"):
                parser.error("replay config_overrides are not implemented; no run was executed")
            settings = replace(
                settings,
                bot_id=str(env.get("bot_id", settings.bot_id)),
                owner_id=None
                if "owner_id" in env and env["owner_id"] is None
                else str(env.get("owner_id", settings.owner_id)),
                blocked_users=settings.blocked_users
                | frozenset(map(str, env.get("blocked_users", [])))
                | frozenset(map(str, env.get("blacklist", []))),
                blocked_groups=settings.blocked_groups
                | frozenset(map(str, env.get("blocked_groups", []))),
            )
        else:
            events = payload
        if args.bot_id:
            settings = replace(settings, bot_id=args.bot_id)
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            raise ValueError
    except ConfigError as exc:
        parser.error(str(exc))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        parser.error("replay: expected readable JSON event array or case with events/environment")
    store = Store()

    async def replay():
        pipeline = Pipeline(settings, store)
        statuses = Counter()
        for event in events:
            statuses[await pipeline.process(event)] += 1
        return dict(statuses)

    try:
        statuses = asyncio.run(replay())
        print(
            json.dumps(
                {"mode": "offline_mock", "events": statuses, **store.summary()},
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
