"""Read-only sync of the bot account's own group messages from SnowLuma's message store.

SnowLuma does not push the bot account's own messages to WebSocket clients
(`reportSelfMessage=false`), so the old pipeline's (AstrBot's) replies only reached this
store when a member quoted one (`get_msg` backfill). During the shadow trial strategy C
then could not see the bot's latest answer to an asker ("continues"), nor which earlier
requests were answered. `SelfSync` closes that gap:

- every `interval` seconds it opens SnowLuma's `messages.db` read-only (`mode=ro` URI,
  `PRAGMA query_only`, `temp_store=MEMORY`, short connection in a worker thread; WAL
  readers never block SnowLuma's writer) and reads, per group this store collects, the
  rows after a per-group cursor on SnowLuma's group `sequence` (indexed; a small overlap
  catches rows written slightly out of order). A group seen for the first time is read
  from `lookback` seconds back;
- the bot account's messages become `Message`s stored with `source='self_sync'`
  (deduplicated by message id, so a `〔新版〕` send already stored by the send path is not
  stored twice). They never pass through reply admission and never start a run;
- each fresh message gets its `answered_by` mark (`Store.link_bot_message`): a reply
  segment to a stored member message links directly; an unquoted message is paired by
  the `shadow_report` / `legacy_link` rule (same group, 300 s, first unclaimed request);
- `catch_up(message, seq)` (called by the orchestrator before a run builds its context):
  when the asker has an unanswered request in the group within 300 s, one immediate sync
  runs, bounded to `catch_up_seconds` (1 s), because the previous answer may not be here
  yet (the delay is at most one interval).

An unreadable SnowLuma store (missing, locked, corrupt) logs one warning and the serve
loop carries on; the next successful read logs once that it recovered. Logs and counts
carry no ids, paths or content. Nothing here writes to SnowLuma or holds a send path.
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from .contracts import Message, parse_group_message
from .replay.vclock import real_time
from .store import Store

SNOWLUMA_DATA = Path(__file__).resolve().parents[3] / "SnowLuma" / "data"
REPLY_WINDOW_SECONDS = 300.0  # = scripts/replay/legacy_link.REPLY_WINDOW_SECONDS


def default_snowluma_db(bot_id: str, root: Path = SNOWLUMA_DATA) -> Path:
    return root / str(bot_id) / "messages.db"


def connect_ro(path: Path, timeout: float = 0.5) -> sqlite3.Connection:
    """Read-only connection that never writes the database or a temp file."""
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def read_rows(
    path: Path,
    groups: Iterable[str],
    cursors: dict[str, int],
    since: float,
    overlap: int,
    timeout: float = 0.5,
) -> tuple[list[tuple[str, int, int, str]], dict[str, int]]:
    """SnowLuma group rows after each group's cursor (`sequence > cursor - overlap`), or
    from `since` for a group without one. Returns `(group, sequence, timestamp, data)`
    rows in sequence order per group, and the new cursors. Runs in a worker thread."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("SnowLuma message store not found")
    con = connect_ro(path, timeout)
    rows: list[tuple[str, int, int, str]] = []
    new: dict[str, int] = {}
    try:
        for group in groups:
            if not group.isdigit():
                continue  # SnowLuma keys group sessions by the numeric group id
            if group in cursors:
                found = con.execute(
                    "SELECT sequence, timestamp, data FROM messages WHERE is_group=1"
                    " AND session_id=? AND sequence>? ORDER BY sequence",
                    (int(group), cursors[group] - overlap),
                ).fetchall()
            else:
                found = con.execute(
                    "SELECT sequence, timestamp, data FROM messages WHERE is_group=1"
                    " AND session_id=? AND timestamp>=? ORDER BY sequence",
                    (int(group), int(since)),
                ).fetchall()
            top = cursors.get(group, 0)
            for sequence, timestamp, data in found:
                rows.append((group, int(sequence), int(timestamp or 0), data))
                top = max(top, int(sequence))
            new[group] = top
    finally:
        con.close()
    return rows, new


class _Seen(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def add(self, key) -> None:
        self[key] = None
        self.move_to_end(key)
        while len(self) > self.capacity:
            self.popitem(last=False)


class SelfSync:
    def __init__(
        self,
        store: Store,
        bot_id: str,
        path: Path,
        *,
        groups: Callable[[], Iterable[str]] | None = None,
        excluded_groups: Callable[[], Iterable[str]] = lambda: (),
        interval: float = 5.0,
        lookback: float = 1800.0,
        overlap: int = 32,
        window: float = REPLY_WINDOW_SECONDS,
        catch_up_seconds: float = 1.0,
        on_stored: Callable[[int, Message], Any] | None = None,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = lambda _: None,
        reader: Callable[..., tuple[list, dict]] = read_rows,
    ):
        self.store, self.bot_id, self.path = store, str(bot_id), Path(path)
        self.interval, self.lookback, self.overlap = interval, lookback, overlap
        self.window, self.catch_up_seconds = window, catch_up_seconds
        self.on_stored, self.clock, self.log, self.reader = on_stored, clock, log, reader
        self.excluded_groups = excluded_groups
        self._groups = groups
        self._known: set[str] = set(store.groups(self.bot_id)) if groups is None else set()
        self.cursors: dict[str, int] = {}
        self.seen = _Seen(8192)
        self.counts: Counter[str] = Counter()
        self._task: asyncio.Task | None = None
        self._failing = False

    # -- groups -------------------------------------------------------------------------
    def observe(self, message: Message | None) -> None:
        """A message of a group was stored here (intake): that group is collected."""
        if message is not None and message.bot_id == self.bot_id:
            self._known.add(message.group_id)

    def groups(self) -> list[str]:
        known = set(self._groups()) if self._groups is not None else set(self._known)
        return sorted(known - set(self.excluded_groups()))

    # -- one poll ---------------------------------------------------------------------------
    async def poll(self) -> int:
        """Read once and store the new bot messages; returns how many were stored."""
        groups = self.groups()
        self.counts["polls"] += 1
        if not groups:
            return 0
        since = self.clock() - self.lookback
        try:
            async with real_time():  # live file I/O: wall time on a replay's virtual loop
                rows, cursors = await asyncio.to_thread(
                    self.reader, self.path, groups, dict(self.cursors), since, self.overlap
                )
        except (sqlite3.Error, OSError, ValueError) as exc:
            self.counts["read_errors"] += 1
            if not self._failing:
                self._failing = True
                self.log(
                    f"self-sync: SnowLuma message store not readable ({type(exc).__name__});"
                    " serve continues without it"
                )
            return 0
        if self._failing:
            self._failing = False
            self.log("self-sync: SnowLuma message store readable again")
        self.cursors.update(cursors)
        return self.apply(rows)

    def apply(self, rows: list[tuple[str, int, int, str]]) -> int:
        stored = 0
        now = self.clock()
        excluded = set(self.excluded_groups())
        for group, _sequence, timestamp, data in rows:
            if group in excluded:
                continue
            try:
                event = json.loads(data)
            except (TypeError, ValueError):
                self.counts["invalid"] += 1
                continue
            if not isinstance(event, dict) or str(event.get("user_id")) != self.bot_id:
                continue
            key = (group, str(event.get("message_id")))
            if key in self.seen:
                continue
            message = parse_group_message(event, self.bot_id)
            if message is None or message.group_id != group:
                self.seen.add(key)
                self.counts["invalid"] += 1
                continue
            t = event.get("time")
            platform_time = t if isinstance(t, int) and not isinstance(t, bool) else timestamp
            try:
                seq, fresh = self.store.ingest(message, platform_time, now, source="self_sync")
                self.seen.add(key)
                if not fresh:
                    self.counts["duplicate"] += 1
                    continue
                stored += 1
                self.counts["synced"] += 1
                outcome, _ = self.store.link_bot_message(seq, message, platform_time, self.window)
                self.counts[f"paired_{outcome}" if outcome != "unpaired" else "unpaired"] += 1
            except sqlite3.Error as exc:
                self.counts["store_errors"] += 1
                self.log(f"self-sync: store write failed ({type(exc).__name__})")
                continue
            if self.on_stored is not None:
                try:
                    self.on_stored(seq, message)
                except Exception as exc:  # media scheduling must not stop the sync
                    self.log(f"self-sync: on_stored failed ({type(exc).__name__})")
        return stored

    def _start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self.poll())
        return self._task

    # -- immediate sync ---------------------------------------------------------------
    async def sync_now(self, timeout: float) -> bool:
        """One fresh poll within `timeout` seconds (a poll already running is awaited
        first: it may have read before the answer was written). False on timeout; the
        poll then finishes in the background."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        if self._task is not None and not self._task.done():
            await asyncio.wait({self._task}, timeout=max(0.0, end - loop.time()))
            if not self._task.done():
                return False
        task = self._start()
        done, _ = await asyncio.wait({task}, timeout=max(0.0, end - loop.time()))
        return bool(done)

    async def catch_up(self, message: Message, seq: int) -> str:
        """Before a run builds its context: sync once when the asker's previous request
        (same group, 300 s) has no answer here yet. Returns skipped / synced / timeout /
        error; never raises."""
        try:
            self.observe(message)  # a run's group is collected
            if not self.store.unanswered_request_before(message, seq, self.window):
                return "skipped"
            self.counts["catch_up"] += 1
            if await self.sync_now(self.catch_up_seconds):
                return "synced"
            self.counts["catch_up_timeouts"] += 1
            return "timeout"
        except Exception as exc:
            self.counts["catch_up_errors"] += 1
            self.log(f"self-sync: catch-up failed ({type(exc).__name__})")
            return "error"

    # -- loop -------------------------------------------------------------------------------
    async def run(self, stop: asyncio.Event) -> int:
        """Poll every `interval` seconds until `stop`; returns the number of stored rows."""
        total = 0
        while not stop.is_set():
            task = self._start()
            await asyncio.wait({task})
            if not task.cancelled() and task.exception() is None:
                total += task.result()
            elif not task.cancelled():
                self.counts["poll_errors"] += 1
                self.log(f"self-sync: poll failed ({type(task.exception()).__name__})")
            try:
                await asyncio.wait_for(stop.wait(), self.interval)
            except TimeoutError:
                pass
        return total

    async def aclose(self, timeout: float = 2.0) -> None:
        if self._task is not None and not self._task.done():
            await asyncio.wait({self._task}, timeout=timeout)
