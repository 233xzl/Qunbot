"""Read-only smoke probe of the OneBot gateway against local SnowLuma (T3.1 / T3.11).

Calls only get_status, get_login_info, and for one group get_group_info and
get_group_member_list (the last two through GroupInfoTool, as in a run). The group is
taken from the shadow store (most messages, blocked groups skipped); it is never printed.
Output is counts and field names only: no QQ numbers, nicknames or group names.

Usage:
    python scripts/probe_gateway.py [--settings config/settings.toml] [--shadow-db PATH]
"""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import time

from group_agent.config import ConfigError, load_settings, load_snowluma_credentials
from group_agent.gateway import GatewayError, OneBotApi, OneBotGateway
from group_agent.group_info import GroupInfoCache, GroupInfoTool


def pick_group(paths: list[Path], blocked: frozenset[str]) -> str | None:
    for path in paths:
        if not path.exists():
            continue
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            con.execute("PRAGMA temp_store=MEMORY")
            rows = con.execute(
                "SELECT group_id, count(*) AS n FROM messages GROUP BY group_id ORDER BY n DESC"
            ).fetchall()
        finally:
            con.close()
        for group_id, _ in rows:
            if group_id not in blocked:
                return group_id
    return None


async def probe(credentials, group_id: str | None, bot_id: str) -> dict:
    events: Counter[str] = Counter()

    def on_event(event):
        events[str(event.get("post_type", "unknown")) if isinstance(event, dict) else "other"] += 1

    stop = asyncio.Event()
    gateway = OneBotGateway(
        credentials.ws_url,
        credentials.token,
        on_event,
        allow_actions=True,
        action_timeout=15,
        reconnect_initial=1,
        reconnect_max=2,
    )
    task = asyncio.create_task(gateway.run(stop))
    report: dict = {"connected": await gateway.wait_connected(10)}
    api = OneBotApi(gateway)
    try:
        if not report["connected"]:
            return report
        for name in ("get_status", "get_login_info"):
            started = time.monotonic()
            try:
                data = await getattr(api, name)()
                report[name] = {
                    "ok": True,
                    "fields": sorted(data),
                    "ms": round((time.monotonic() - started) * 1000),
                }
                if name == "get_status":
                    report[name]["online"] = data.get("online")
                if name == "get_login_info":
                    report[name]["matches_settings_bot_id"] = str(data.get("user_id")) == bot_id
            except GatewayError as exc:
                report[name] = {"ok": False, "error": exc.kind, "retcode": exc.retcode}
        if group_id is None:
            report["group"] = "skipped: no group in shadow store"
            return report
        started = time.monotonic()
        try:
            info = await api.get_group_info(group_id)
            members = await api.get_group_member_list(group_id, timeout=30)
            fields = sorted({k for m in members for k in m})
            report["get_group_info"] = {"ok": True, "fields": sorted(info)}
            report["get_group_member_list"] = {
                "ok": True,
                "members": len(members),
                "fields": fields,
                "roles": dict(Counter(str(m.get("role")) for m in members)),
                "with_card": sum(1 for m in members if m.get("card")),
                "with_title": sum(1 for m in members if m.get("title")),
                "with_join_time": sum(1 for m in members if m.get("join_time")),
                "ms": round((time.monotonic() - started) * 1000),
            }
        except GatewayError as exc:
            report["group_queries"] = {"ok": False, "error": exc.kind, "retcode": exc.retcode}
            return report
        # The tool path as a run would use it: a fresh cache fetches once, then hits.
        calls_before = gateway.stats["action_calls"]
        cache = GroupInfoCache(api)
        tool = GroupInfoTool(cache, group_id)
        counts = await tool.call({"query": "role_counts"})
        summary = await tool.call({"query": "summary"})
        refused = await tool.call({"query": "summary", "group_id": "1"})
        text = json.dumps([counts, summary], ensure_ascii=False)
        leaked = sum(1 for m in members if m.get("user_id") and str(m["user_id"]) in text)
        report["group_info_tool"] = {
            "role_counts": counts["result"],
            "summary_status": summary["status"],
            "summary_member_count": summary["result"]["member_count"],
            "cross_group_status": refused["status"],
            "cache": dict(cache.counts),
            "platform_calls": gateway.stats["action_calls"] - calls_before,
            "user_ids_in_output": leaked,
        }
        return report
    finally:
        stop.set()
        stats = await asyncio.wait_for(task, 10)
        report["connection"] = {
            k: v
            for k, v in stats.items()
            if k
            in (
                "connects",
                "disconnects",
                "connect_failures",
                "frames",
                "action_calls",
                "action_ok",
                "action_failed",
                "action_timeouts",
                "unmatched_responses",
                "late_responses",
            )
        }
        report["events_received"] = dict(events)
        report["sent_to_qq"] = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path("config/settings.toml"))
    parser.add_argument("--shadow-db", type=Path, action="append")
    args = parser.parse_args()
    try:
        settings = load_settings(args.settings)
        credentials = load_snowluma_credentials(settings.credentials)
    except ConfigError as exc:
        print(json.dumps({"skipped": str(exc)}, ensure_ascii=False))
        return 2
    paths = args.shadow_db or [
        settings.shadow_db,
        *sorted(Path("data/shadow").glob("*.sqlite3"), key=lambda p: -p.stat().st_size),
    ]
    group_id = pick_group(paths, settings.blocked_groups)
    report = asyncio.run(probe(credentials, group_id, settings.bot_id))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("connected") else 1


if __name__ == "__main__":
    sys.exit(main())
