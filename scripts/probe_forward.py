"""Read one stored forward through SnowLuma; print only counts and field names.

Never sends messages or calls a model. Raw output, when requested, stays under data/.
"""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sqlite3

from group_agent.config import load_settings, load_snowluma_credentials
from group_agent.gateway import GatewayError, OneBotApi, OneBotGateway


async def probe(settings, resource_id):
    credentials = load_snowluma_credentials(settings.credentials)
    gateway = OneBotGateway(
        credentials.ws_url,
        credentials.token,
        lambda _: None,
        allow_actions=True,
        action_timeout=10,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(gateway.run(stop))
    try:
        if not await gateway.wait_connected(10):
            return {"ok": False, "error": "not_connected"}, []
        nodes = await OneBotApi(gateway).get_forward_msg(resource_id)
        segments = [s for n in nodes for s in n.get("message", []) if isinstance(s, dict)]
        report = {
            "ok": True,
            "nodes": len(nodes),
            "node_fields": sorted({k for n in nodes for k in n}),
            "segment_types": dict(Counter(s.get("type") for s in segments)),
            "segment_data_fields": {
                kind: sorted(
                    {k for s in segments if s.get("type") == kind for k in (s.get("data") or {})}
                )
                for kind in sorted({s.get("type") for s in segments})
            },
            "bytes": len(json.dumps(nodes, ensure_ascii=False).encode("utf-8")),
            "qq_sends": 0,
            "model_calls": 0,
        }
        return report, nodes
    except GatewayError as exc:
        return {"ok": False, "error": exc.kind, "retcode": exc.retcode}, []
    finally:
        stop.set()
        await asyncio.wait_for(task, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path("config/settings.toml"))
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and not args.output.resolve().is_relative_to(Path("data").resolve()):
        parser.error("raw output must be inside data/")
    settings = load_settings(args.settings)
    with sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True) as con:
        row = con.execute(
            "SELECT bot_id,group_id,revoked,segments FROM messages WHERE seq=?", (args.seq,)
        ).fetchone()
    if row is None or row[2] or row[0] != settings.bot_id or row[1] in settings.blocked_groups:
        parser.error("root message unavailable")
    resources = [
        s["data"].get("id") or s["data"].get("res_id") or s["data"].get("forward_id")
        for s in json.loads(row[3])
        if s.get("type") == "forward"
    ]
    if not resources or not resources[0]:
        parser.error("no forward resource in root message")
    report, nodes = asyncio.run(probe(settings, resources[0]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"report": report, "nodes": nodes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
