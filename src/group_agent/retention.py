"""Retention cleanup (T3.3, SEC-02, CTX-05, D-14; docs/CONTEXT-RETENTION.md).

Rules, applied per (bot, group) by message production time (platform time; local receive
time when missing; rows with neither are kept and counted):

1. older than `related_days` -> deleted, whatever references it (hard limit; a new
   reference never extends a message's life);
2. older than `recent_hours` and not in the related set -> deleted;
3. more than `max_messages_per_group` left -> the oldest unrelated messages go first,
   then the oldest related ones.

The related set is the "提问关联集合": every recorded reply request (an admitted
top-level @bot), its explicit reply ancestors in the same group (depth
`max_branch_depth`, at most `max_branch_messages` per request), and messages explicitly
marked for a request through `Store.mark_related` (the context snapshot / answer mapping
of the future scheduler and sender, T3.4-T3.7). Same-topic or time-adjacent messages are
not inferred into it. The owner's `#新版` command messages (`kind='owner_command'`) are
excluded from the reply ancestry and from context marks; the only mark that keeps one is
`new_version_command`, the anchor of its own `new_version` run.

Recall tombstones follow the same ages. Reply request rows (persistent dedup) are kept
for `related_days`. Runs and offline outbound records of deleted messages are deleted
with them. Afterwards media files that no message references are deleted; the media
cache is capped at `media_max_mb` (oldest files evicted, their slots marked
`unavailable/evicted`); unknown files left in the media directory (crash leftovers)
older than a grace period are removed.

`dry_run=True` computes the same counts and changes nothing. The same function serves the
production store and the shadow store (D-14).
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Callable
import json
from pathlib import Path
import time
from typing import Any

from .config import Retention
from .media import MEDIA_FILE, remove_media_files
from .store import Store

HOUR = 3600
DAY = 86400
REPORT_KEYS = (
    "messages_scanned",
    "expired_hard",
    "expired_recent",
    "over_capacity",
    "kept_related",
    "kept_no_time",
    "requests_expired",
    "media_orphaned",
    "media_evicted",
    "media_bytes_freed",
    "stray_files",
)


def _reply_ids(segments: str) -> list[str]:
    try:
        parsed = json.loads(segments)
    except ValueError:
        return []
    return [
        str(s["data"]["id"])
        for s in parsed
        if isinstance(s, dict)
        and s.get("type") == "reply"
        and isinstance(s.get("data"), dict)
        and "id" in s["data"]
    ]


def related_set(store: Store, retention: Retention) -> set[int]:
    """Seqs in the related set (see module docstring)."""
    con = store.con
    by_id: dict[tuple[str, str, str], tuple[int, str]] = {}
    commands: set[int] = set()
    for row in con.execute("SELECT seq,bot_id,group_id,message_id,segments,kind FROM messages"):
        if row["kind"] == "owner_command":
            commands.add(row["seq"])  # D-19 `#新版` command: not related through quotes
            continue
        by_id[(row["bot_id"], row["group_id"], row["message_id"])] = (row["seq"], row["segments"])
    related: set[int] = set()
    for request in con.execute("SELECT bot_id,group_id,message_id FROM reply_requests"):
        bot_id, group_id = request["bot_id"], request["group_id"]
        anchor = by_id.get((bot_id, group_id, request["message_id"]))
        if anchor is None:
            continue
        branch = {anchor[0]}
        frontier = [(anchor[1], 0)]
        while frontier and len(branch) < retention.max_branch_messages:
            segments, depth = frontier.pop(0)
            if depth >= retention.max_branch_depth:
                continue
            for rid in _reply_ids(segments):
                target = by_id.get((bot_id, group_id, rid))
                if target is None or target[0] in branch:
                    continue
                branch.add(target[0])
                if len(branch) >= retention.max_branch_messages:
                    break
                frontier.append((target[1], depth + 1))
        related |= branch
    for seq, relation in con.execute("SELECT DISTINCT seq, relation FROM related_marks"):
        # A command message is kept only as the anchor of its own new_version run (its
        # outbound row is the once-per-question key), never through context marks.
        if seq in commands and relation != "new_version_command":
            continue
        related.add(seq)
    return related


def plan_cleanup(store: Store, retention: Retention, now: float) -> dict[str, Any]:
    """Decide which messages go. Returns seq lists and counts; changes nothing."""
    related = related_set(store, retention)
    hard = retention.related_days * DAY
    recent = retention.recent_hours * HOUR
    remove: dict[int, str] = {}
    kept: dict[tuple[str, str], list[tuple[int, float, int]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in store.con.execute(
        "SELECT seq,bot_id,group_id,COALESCE(platform_time,received_at) AS t FROM messages"
    ):
        counts["messages_scanned"] += 1
        seq, t = row["seq"], row["t"]
        is_related = seq in related
        if t is None:
            counts["kept_no_time"] += 1
        elif now - t > hard:
            remove[seq] = "expired_hard"
            continue
        elif now - t > recent and not is_related:
            remove[seq] = "expired_recent"
            continue
        sort_time = -1.0 if t is None else float(t)
        kept[(row["bot_id"], row["group_id"])].append((int(is_related), sort_time, seq))
    for rows in kept.values():
        excess = len(rows) - retention.max_messages_per_group
        if excess > 0:
            for _, _, seq in sorted(rows)[:excess]:
                remove[seq] = "over_capacity"
    counts.update(remove.values())
    counts["kept_related"] = sum(
        1 for rows in kept.values() for is_rel, _, seq in rows if is_rel and seq not in remove
    )
    request_cutoff = now - hard
    counts["requests_expired"] = store.con.execute(
        "SELECT count(*) FROM reply_requests WHERE COALESCE(platform_time,received_at) < ?",
        (request_cutoff,),
    ).fetchone()[0]
    return {"remove": remove, "counts": counts, "request_cutoff": request_cutoff}


def _media_plan(store: Store, removed: set[int], max_bytes: int) -> tuple[set[str], list[str]]:
    """Assets left unreferenced after removing `removed`, then oldest assets over the cap."""
    refs: dict[str, set[int]] = defaultdict(set)
    for row in store.con.execute("SELECT seq, sha256 FROM message_media WHERE sha256 IS NOT NULL"):
        refs[row["sha256"]].add(row["seq"])
    assets = store.con.execute(
        "SELECT sha256, bytes, created_at FROM media_assets ORDER BY created_at, sha256"
    ).fetchall()
    orphans = {a["sha256"] for a in assets if not (refs.get(a["sha256"], set()) - removed)}
    total = sum(a["bytes"] for a in assets if a["sha256"] not in orphans)
    evict = []
    for asset in assets:
        if total <= max_bytes:
            break
        if asset["sha256"] in orphans:
            continue
        evict.append(asset["sha256"])
        total -= asset["bytes"]
    return orphans, evict


def _stray_files(root: Path | None, known: set[str], now: float, grace: float) -> list[Path]:
    if root is None or not root.is_dir():
        return []
    stray = []
    for path in root.iterdir():
        if not path.is_file() or not MEDIA_FILE.fullmatch(path.name) or path.name in known:
            continue
        try:
            if now - path.stat().st_mtime > grace:
                stray.append(path)
        except OSError:
            continue
    return stray


def run_cleanup(
    store: Store,
    retention: Retention,
    *,
    now: float | None = None,
    media_root: Path | None = None,
    dry_run: bool = False,
    file_grace_seconds: float = 600.0,
) -> dict[str, Any]:
    """Apply (or with `dry_run`, only count) the retention rules. Returns counts only."""
    now = time.time() if now is None else now
    plan = plan_cleanup(store, retention, now)
    remove: dict[int, str] = plan["remove"]
    counts: Counter[str] = plan["counts"]
    removed = set(remove)
    orphans, evict = _media_plan(store, removed, retention.media_max_mb * 1024 * 1024)
    con = store.con
    paths = {r["sha256"]: r["path"] for r in con.execute("SELECT sha256, path FROM media_assets")}
    counts["media_orphaned"] = len(orphans)
    counts["media_evicted"] = len(evict)
    counts["media_bytes_freed"] = sum(
        r["bytes"]
        for r in con.execute("SELECT sha256, bytes FROM media_assets")
        if r["sha256"] in orphans or r["sha256"] in set(evict)
    )
    stray = _stray_files(media_root, set(paths.values()), now, file_grace_seconds)
    counts["stray_files"] = len(stray)
    report = {"dry_run": dry_run, "now": now, **{k: counts[k] for k in REPORT_KEYS}}
    if dry_run:
        return report
    seqs = [(s,) for s in sorted(removed)]
    with con:
        con.executemany(
            "DELETE FROM outbound WHERE run_id IN (SELECT run_id FROM runs WHERE message_seq=?)",
            seqs,
        )
        con.executemany("DELETE FROM runs WHERE message_seq=?", seqs)
        # Owner control confirmations have no run: they age out with the request rows
        # (the control_events audit rows hold no content and are kept).
        con.execute(
            "DELETE FROM outbound WHERE kind='owner_control' AND created_at < ?",
            (plan["request_cutoff"],),
        )
        con.executemany("DELETE FROM messages WHERE seq=?", seqs)  # media/marks cascade
        con.execute(
            "DELETE FROM reply_requests WHERE COALESCE(platform_time,received_at) < ?",
            (plan["request_cutoff"],),
        )
        con.executemany(
            "UPDATE message_media SET status='unavailable', reason='evicted', sha256=NULL"
            " WHERE sha256=?",
            [(sha,) for sha in evict],
        )
    names = store.drop_orphan_media()
    files = remove_media_files(media_root, names) if media_root is not None else 0
    for path in stray:
        try:
            path.unlink()
            files += 1
        except OSError:
            pass
    report["files_deleted"] = files
    return report


async def periodic_cleanup(
    stop: asyncio.Event,
    interval_seconds: float,
    cleanup: Callable[[], dict[str, Any]],
    *,
    on_report: Callable[[dict[str, Any]], None] = lambda _: None,
    log: Callable[[str], None] = lambda _: None,
) -> int:
    """Run `cleanup` now and then every `interval_seconds` until `stop`; returns runs."""
    runs = 0
    while not stop.is_set():
        try:
            on_report(cleanup())
            runs += 1
        except Exception as exc:
            log(f"retention cleanup failed: {type(exc).__name__}")
        try:
            await asyncio.wait_for(
                stop.wait(), interval_seconds() if callable(interval_seconds) else interval_seconds
            )
        except TimeoutError:
            pass
    return runs
