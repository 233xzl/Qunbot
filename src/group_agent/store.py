"""Prototype SQLite store. One connection, short transactions."""

import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .contracts import Message


from . import received_groups

SCHEMA_VERSION = 13
# Where a stored message came from: a pushed event ("live"), a get_msg lookup for a
# referenced message that was missing locally ("backfill"; never a reply trigger), or the
# read-only sync of the bot account's own messages from SnowLuma's message store
# ("self_sync", `self_sync.py`; SnowLuma does not push them, never a reply trigger).
MESSAGE_SOURCES = ("live", "backfill", "self_sync")
# Sources stored after the run boundary but placed by platform time (`history`).
LATE_SOURCES = ("backfill", "self_sync")
# Message kinds: "chat", or "owner_command" (the owner's `#新版` command message, D-19:
# never context material, `owner_command.NewVersionCommand.observe`).
MESSAGE_KINDS = ("chat", "owner_command")
NEW_VERSION_PREFIX = "〔新版〕"  # = owner_command.PREFIX
MESSAGES_COLUMNS_V6 = (
    "seq,bot_id,group_id,message_id,user_id,segments,revoked,platform_time,received_at,source"
)


def _messages_ddl(name: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {name} (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, message_id TEXT NOT NULL,
    user_id TEXT NOT NULL, segments TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
    platform_time INTEGER, received_at REAL,
    source TEXT NOT NULL DEFAULT 'live' CHECK(source IN ('live', 'backfill', 'self_sync')),
    kind TEXT NOT NULL DEFAULT 'chat' CHECK(kind IN ('chat', 'owner_command')),
    UNIQUE(bot_id, group_id, message_id)
);
"""


# Reply request states: "recorded" (logged only, e.g. shadow mode), "queued" (handed to the
# scheduler), "dropped" (the scheduler queue refused it).
REQUEST_STATES = ("recorded", "queued", "dropped")
MEDIA_STATES = ("pending", "available", "unavailable")
# Run states (T3.4, RUN-02..05). "recorded" is the offline Pipeline's terminal state; the
# online orchestrator ends in succeeded / failed / cancelled / expired / rejected.
RUN_STATES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "expired",
    "rejected",
    "recorded",
)
RUN_ACTIVE_STATES = ("queued", "running")
# Why a run exists: "reply" (one per admitted request message, AC-15) or "new_version"
# (D-19: the owner's `#新版` command; its `message_seq` is the command message, its
# `message_id` the quoted question).
RUN_PURPOSES = ("reply", "new_version")
# Outbound states (T3.7, D-19, cutover). Record-only answers end `recorded_not_sent`. Live
# rows (mode `live_send`: the owner's `#新版` command, `owner_command.py`, or every answer
# and busy notice when `[serve] live_send` is on, `live_send.py`) go pending -> sending ->
# sent | uncertain | failed. `uncertain` (timeout / connection lost after the frame was
# written, or a process that ended mid-send) is never retried automatically (RUN-12).
OUTBOUND_STATES = ("pending", "recorded_not_sent", "sending", "sent", "uncertain", "failed")
OUTBOUND_MODES = ("offline", "record_only", "live_send")
# `owner_control` (v10): the fixed confirmation of an owner remote-control command (`#暂停`,
# `#恢复`, `#状态`, `owner_control.py`); always `live_send`, one row per command message, and
# no run behind it (so `outbound.run_id` no longer references `runs`).
OUTBOUND_KINDS = ("answer", "busy_notice", "new_version", "owner_control")
# Owner control actions (`control_events.action`).
CONTROL_ACTIONS = ("pause", "resume", "status")
RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, message_seq INTEGER NOT NULL UNIQUE REFERENCES messages(seq),
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'succeeded', 'failed',
        'cancelled', 'expired', 'rejected', 'recorded')),
    error_kind TEXT,
    bot_id TEXT, group_id TEXT, message_id TEXT, user_id TEXT,
    cost_mode TEXT, route TEXT, queued INTEGER NOT NULL DEFAULT 0,
    created_at REAL, started_at REAL, ended_at REAL, updated_at REAL,
    detail TEXT,
    purpose TEXT NOT NULL DEFAULT 'reply' CHECK(purpose IN ('reply', 'new_version'))
);
"""
OUTBOUND_DDL = """
CREATE TABLE IF NOT EXISTS outbound (
    run_id TEXT PRIMARY KEY,
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, reply_to TEXT NOT NULL,
    requester TEXT NOT NULL, text TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending', 'recorded_not_sent', 'sending', 'sent',
        'uncertain', 'failed')),
    mode TEXT NOT NULL CHECK(mode IN ('offline', 'record_only', 'live_send')),
    kind TEXT NOT NULL DEFAULT 'answer' CHECK(kind IN ('answer', 'busy_notice',
        'new_version', 'owner_control')),
    idempotency_key TEXT UNIQUE,
    error_kind TEXT, created_at REAL, scheduled_at REAL, recorded_at REAL,
    qq_message_id TEXT, sent_at REAL
);
"""
OUTBOUND_V5_COLUMNS = (
    "run_id,bot_id,group_id,reply_to,requester,text,state,mode,kind,idempotency_key,"
    "error_kind,created_at,scheduled_at,recorded_at,qq_message_id"
)
SCHEMA = (
    _messages_ddl("messages")
    + RUNS_DDL
    + OUTBOUND_DDL
    + """
CREATE TABLE IF NOT EXISTS context_snapshots (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    anchor_seq INTEGER NOT NULL, strategy TEXT NOT NULL,
    message_ids TEXT NOT NULL, reasons TEXT NOT NULL, gaps TEXT NOT NULL,
    degraded INTEGER NOT NULL DEFAULT 0, created_at REAL
);
CREATE TABLE IF NOT EXISTS usage_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, group_id TEXT,
    day TEXT NOT NULL, created_at REAL NOT NULL,
    provider TEXT NOT NULL, model TEXT NOT NULL, task TEXT, mode TEXT, status TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0, thinking_tokens INTEGER,
    requests INTEGER NOT NULL DEFAULT 0, cost_cny REAL NOT NULL DEFAULT 0, latency_s REAL,
    channel TEXT, service TEXT, list_cost REAL, list_currency TEXT, google_queries INTEGER
);
CREATE TABLE IF NOT EXISTS group_names (
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, name TEXT NOT NULL, updated_at REAL NOT NULL,
    PRIMARY KEY(bot_id, group_id)
);
CREATE TABLE IF NOT EXISTS request_members (
    seq INTEGER PRIMARY KEY REFERENCES messages(seq) ON DELETE CASCADE, name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_run ON usage_ledger(run_id);
CREATE INDEX IF NOT EXISTS usage_time ON usage_ledger(created_at);
CREATE INDEX IF NOT EXISTS runs_time ON runs(created_at);
CREATE INDEX IF NOT EXISTS usage_group_day ON usage_ledger(group_id,day);
CREATE INDEX IF NOT EXISTS runs_group_time ON runs(group_id,created_at);
CREATE INDEX IF NOT EXISTS usage_ledger_day ON usage_ledger(day);
CREATE TABLE IF NOT EXISTS reply_requests (
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, message_id TEXT NOT NULL,
    seq INTEGER, user_id TEXT NOT NULL, platform_time INTEGER, received_at REAL,
    state TEXT NOT NULL DEFAULT 'recorded' CHECK(state IN ('recorded', 'queued', 'dropped')),
    reaction TEXT,
    PRIMARY KEY(bot_id, group_id, message_id)
);
CREATE TABLE IF NOT EXISTS related_marks (
    seq INTEGER NOT NULL REFERENCES messages(seq) ON DELETE CASCADE,
    anchor_seq INTEGER NOT NULL, relation TEXT NOT NULL,
    PRIMARY KEY(seq, anchor_seq)
);
CREATE TABLE IF NOT EXISTS media_assets (
    sha256 TEXT PRIMARY KEY, mime TEXT NOT NULL, bytes INTEGER NOT NULL,
    width INTEGER, height INTEGER, path TEXT NOT NULL, created_at REAL
);
CREATE TABLE IF NOT EXISTS message_media (
    seq INTEGER NOT NULL REFERENCES messages(seq) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'available', 'unavailable')),
    reason TEXT, declared_bytes INTEGER,
    sha256 TEXT REFERENCES media_assets(sha256),
    updated_at REAL,
    PRIMARY KEY(seq, segment_index)
);
CREATE INDEX IF NOT EXISTS message_media_sha ON message_media(sha256);
CREATE INDEX IF NOT EXISTS messages_group_time ON messages(bot_id, group_id, platform_time);
CREATE TABLE IF NOT EXISTS control_state (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS control_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, message_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('pause', 'resume', 'status')),
    effect TEXT NOT NULL,
    UNIQUE(bot_id, group_id, message_id)
);
"""
)
# v1 -> v2: keep platform time and local receive time for ordering, retention and shadow checks.
MIGRATE_1_TO_2 = """
ALTER TABLE messages ADD COLUMN platform_time INTEGER;
ALTER TABLE messages ADD COLUMN received_at REAL;
"""
# v2 -> v3: mark messages fetched by reference lookup (T3.1, MIG-02).
MIGRATE_2_TO_3 = """
ALTER TABLE messages ADD COLUMN source TEXT NOT NULL DEFAULT 'live'
    CHECK(source IN ('live', 'backfill'));
"""
# v3 -> v4 (T3.2/T3.3): reply_requests, related_marks, media_assets, message_media are new
# tables created by SCHEMA; no existing table changes.
# v4 -> v5 (T3.4/T3.7/T3.9): runs and outbound get the online state machine (new CHECK
# constraints, so the tables are rebuilt, see REBUILD_RUNS_OUTBOUND); context_snapshots
# and usage_ledger are new.
# v5 -> v6 (D-19): runs get `purpose`; outbound gets the live-send states, kind
# `new_version`, mode `live_send` and `sent_at` (new CHECK constraints: rebuilt; nothing
# references outbound, so dropping it touches no other table).
MIGRATE_5_TO_6 = (
    """
ALTER TABLE runs ADD COLUMN purpose TEXT NOT NULL DEFAULT 'reply'
    CHECK(purpose IN ('reply', 'new_version'));
CREATE TABLE _outbound_v5 AS SELECT * FROM outbound;
DROP TABLE outbound;
"""
    + OUTBOUND_DDL
    + f"""
INSERT INTO outbound({OUTBOUND_V5_COLUMNS}) SELECT {OUTBOUND_V5_COLUMNS} FROM _outbound_v5;
DROP TABLE _outbound_v5;
"""
)
MIGRATIONS = {
    0: "",
    1: MIGRATE_1_TO_2 + MIGRATE_2_TO_3,
    2: MIGRATE_2_TO_3,
    3: "",
    4: "",
    5: MIGRATE_5_TO_6,
    6: "",
    7: "",
    8: "",
    9: "",
    10: "",
    11: "",
    12: "",
    SCHEMA_VERSION: "",
}
# v9 -> v10 (owner remote control): new tables `control_state` / `control_events` (SCHEMA);
# outbound gets kind `owner_control` (new CHECK) and loses its reference to runs (a control
# confirmation has no run), so the table is rebuilt; nothing references outbound.
OUTBOUND_COLUMNS = (
    "run_id,bot_id,group_id,reply_to,requester,text,state,mode,kind,idempotency_key,"
    "error_kind,created_at,scheduled_at,recorded_at,qq_message_id,sent_at"
)
REBUILD_OUTBOUND_V10 = (
    """
CREATE TABLE _outbound_v9 AS SELECT * FROM outbound;
DROP TABLE outbound;
"""
    + OUTBOUND_DDL
    + f"""
INSERT INTO outbound({OUTBOUND_COLUMNS}) SELECT {OUTBOUND_COLUMNS} FROM _outbound_v9;
DROP TABLE _outbound_v9;
"""
)
# v7 -> v8 (D-20): reply_requests get `reaction` (NULL = no "processing" reaction tried;
# pending / ok / failed_<kind>), added to any existing reply_requests table (see Store).
ADD_REACTION = "ALTER TABLE reply_requests ADD COLUMN reaction TEXT;\n"
# v8 -> v9 (B1): usage_ledger gets the billing identity of a call; old rows keep NULLs.
USAGE_V9_COLUMNS = {
    "channel": "TEXT",
    "service": "TEXT",
    "list_cost": "REAL",
    "list_currency": "TEXT",
    "google_queries": "INTEGER",
}
# v6 -> v7 (self-sync, owner command kind): messages get source `self_sync` (new CHECK) and
# `kind`; the table is rebuilt with foreign keys off (SQLite's documented procedure: new
# table, copy, drop, rename), keeping every seq and the AUTOINCREMENT high-water mark, so
# runs, marks, media slots and snapshots still point at the same rows. Applied to every
# older database that has a messages table, after the column migrations above.
REBUILD_MESSAGES = (
    _messages_ddl("_messages_v7")
    + f"""
INSERT INTO _messages_v7({MESSAGES_COLUMNS_V6}) SELECT {MESSAGES_COLUMNS_V6} FROM messages;
DROP TABLE messages;
ALTER TABLE _messages_v7 RENAME TO messages;
"""
)
# Applied before SCHEMA when an older database has the v1-v4 runs/outbound tables.
REBUILD_RUNS_OUTBOUND = (
    """
CREATE TABLE _runs_v4 AS SELECT * FROM runs;
CREATE TABLE _outbound_v4 AS SELECT * FROM outbound;
DROP TABLE outbound;
DROP TABLE runs;
"""
    + RUNS_DDL
    + OUTBOUND_DDL
    + """
INSERT INTO runs(run_id, message_seq, state, error_kind)
    SELECT run_id, message_seq, state, error_kind FROM _runs_v4;
INSERT INTO outbound(run_id, bot_id, group_id, reply_to, requester, text, state, mode)
    SELECT run_id, bot_id, group_id, reply_to, requester, text, state, mode FROM _outbound_v4;
DROP TABLE _runs_v4;
DROP TABLE _outbound_v4;
"""
)


def outbound_key(bot_id: str, group_id: str, message_id: str, kind: str) -> str:
    return f"{bot_id}:{group_id}:{message_id}:{kind}"


def _message(row: sqlite3.Row) -> Message:
    return Message(
        row["bot_id"],
        row["group_id"],
        row["message_id"],
        row["user_id"],
        tuple(json.loads(row["segments"])),
    )


def _is_new_version(message: Message) -> bool:
    """The new framework's own `〔新版〕` send (D-19), not an old-pipeline reply."""
    return message.text.strip().startswith(NEW_VERSION_PREFIX)


class Store:
    def __init__(self, path: Path | str = ":memory:"):
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        version = self.con.execute("PRAGMA user_version").fetchone()[0]
        if version not in MIGRATIONS:
            self.con.close()
            raise ValueError("unsupported database schema version")

        def has_table(name: str) -> bool:
            return bool(
                self.con.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
                ).fetchone()[0]
            )

        rebuild = ""
        if version < 5:
            rebuild = REBUILD_RUNS_OUTBOUND if has_table("runs") else ""
        if version < 7 and has_table("messages"):
            # Foreign keys stay off (the connection default) while messages is rebuilt.
            high = 0
            if has_table("sqlite_sequence"):
                row = self.con.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name='messages'"
                ).fetchone()
                high = int(row[0]) if row and row[0] is not None else 0
            rebuild += REBUILD_MESSAGES + (
                f"UPDATE sqlite_sequence SET seq=max(seq, {high}) WHERE name='messages';\n"
                f"INSERT INTO sqlite_sequence(name, seq) SELECT 'messages', {high}"
                " WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name='messages');\n"
            )
        if version < 8 and has_table("reply_requests"):
            columns = {r[1] for r in self.con.execute("PRAGMA table_info(reply_requests)")}
            if "reaction" not in columns:
                rebuild += ADD_REACTION
        if version < 9 and has_table("usage_ledger"):
            columns = {r[1] for r in self.con.execute("PRAGMA table_info(usage_ledger)")}
            for column, kind in USAGE_V9_COLUMNS.items():
                if column not in columns:
                    rebuild += f"ALTER TABLE usage_ledger ADD COLUMN {column} {kind};\n"
        if 6 <= version < 10 and has_table("outbound"):
            rebuild += REBUILD_OUTBOUND_V10  # (v5 databases get the new table in 5 -> 6)
        if version < 11 and has_table("usage_ledger"):
            columns = {r[1] for r in self.con.execute("PRAGMA table_info(usage_ledger)")}
            if "group_id" not in columns:
                rebuild += "ALTER TABLE usage_ledger ADD COLUMN group_id TEXT;\n"
            rebuild += (
                "UPDATE usage_ledger SET group_id=(SELECT group_id FROM runs "
                "WHERE runs.run_id=usage_ledger.run_id) WHERE group_id IS NULL;\n"
            )
        # Schema creation, migration and version bump succeed or roll back together.
        try:
            self.con.executescript(
                f"BEGIN IMMEDIATE;\n{MIGRATIONS[version]}\n{rebuild}\n{SCHEMA}\n"
                f"{received_groups.SCHEMA}\n{received_groups.BACKFILL if version < 13 else ''}\n"
                f"PRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
            )
        except Exception:
            self.con.rollback()
            self.con.close()
            raise
        self.con.execute("PRAGMA foreign_keys=ON")

    def allows_group_reply(self, bot, group, seq=None):
        return received_groups.allows(self.con, bot, group, seq)

    def close(self):
        self.con.close()

    def _insert_message(
        self, message, platform_time, received_at, source, kind="chat"
    ) -> tuple[int, bool]:
        if source not in MESSAGE_SOURCES:
            raise ValueError("unknown message source")
        if kind not in MESSAGE_KINDS:
            raise ValueError("unknown message kind")
        cursor = self.con.execute(
            "INSERT OR IGNORE INTO messages"
            "(bot_id,group_id,message_id,user_id,segments,platform_time,received_at,source,kind)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                message.bot_id,
                message.group_id,
                message.message_id,
                message.user_id,
                json.dumps(message.segments, ensure_ascii=False),
                platform_time,
                received_at,
                source,
                kind,
            ),
        )
        fresh = cursor.rowcount == 1
        if fresh and source == "live" and message.user_id != message.bot_id:
            received_groups.observe(self.con, message.bot_id, message.group_id, received_at)
        row = self.con.execute(
            "SELECT seq FROM messages WHERE bot_id=? AND group_id=? AND message_id=?",
            (message.bot_id, message.group_id, message.message_id),
        ).fetchone()
        return row[0], fresh

    def ingest(
        self,
        message: Message,
        platform_time: int | None = None,
        received_at: float | None = None,
        source: str = "live",
        kind: str = "chat",
    ) -> tuple[int, bool]:
        """Insert once per (bot, group, message). A recall tombstone also blocks re-insertion.
        `kind` "owner_command" stores an owner command message already flagged."""
        if source not in MESSAGE_SOURCES:
            raise ValueError("unknown message source")
        with self.con:
            return self._insert_message(message, platform_time, received_at, source, kind)

    def ingest_request(
        self,
        message: Message,
        platform_time: int | None = None,
        received_at: float | None = None,
        *,
        member_name: str = "",
    ) -> tuple[int, bool, bool]:
        """Store a live reply candidate and its reply request in one transaction (RUN-01).

        The request row is keyed by (bot, group, message) and outlives the message row
        until the related-set period ends, so a redelivered event cannot request twice
        even after retention removed the message.

        Returns:
            `(seq, fresh_message, fresh_request)`; a reply run may start only when both
            are true.
        """
        with self.con:
            seq, fresh = self._insert_message(message, platform_time, received_at, "live")
            if not fresh:
                return seq, False, False
            if member_name:
                self.con.execute(
                    "INSERT INTO request_members VALUES (?,?)", (seq, member_name[:128])
                )
            cursor = self.con.execute(
                "INSERT OR IGNORE INTO reply_requests"
                "(bot_id,group_id,message_id,seq,user_id,platform_time,received_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    message.bot_id,
                    message.group_id,
                    message.message_id,
                    seq,
                    message.user_id,
                    platform_time,
                    received_at,
                ),
            )
            return seq, True, cursor.rowcount == 1

    def set_request_state(self, bot_id: str, group_id: str, message_id: str, state: str):
        if state not in REQUEST_STATES:
            raise ValueError("unknown request state")
        with self.con:
            self.con.execute(
                "UPDATE reply_requests SET state=? WHERE bot_id=? AND group_id=? AND message_id=?",
                (state, bot_id, group_id, message_id),
            )

    def claim_reaction(
        self, bot_id: str, group_id: str, message_id: str, *, now: float, max_age_s: float
    ) -> bool:
        """D-20: claim the one "processing" reaction of a request. Only a queued live
        request received within `max_age_s` that was never claimed qualifies; the claim is
        persistent, so a redelivered event or a restart never reacts twice."""
        with self.con:
            cursor = self.con.execute(
                "UPDATE reply_requests SET reaction='pending'"
                " WHERE bot_id=? AND group_id=? AND message_id=? AND reaction IS NULL"
                " AND state='queued' AND received_at IS NOT NULL AND received_at >= ?"
                " AND EXISTS (SELECT 1 FROM messages m WHERE m.seq=reply_requests.seq"
                " AND m.source='live' AND m.kind='chat')",
                (bot_id, group_id, message_id, now - max_age_s),
            )
            return cursor.rowcount == 1

    def set_reaction(self, bot_id: str, group_id: str, message_id: str, outcome: str) -> None:
        with self.con:
            self.con.execute(
                "UPDATE reply_requests SET reaction=? WHERE bot_id=? AND group_id=? AND message_id=?",
                (outcome, bot_id, group_id, message_id),
            )

    def reaction_state(self, bot_id: str, group_id: str, message_id: str) -> str | None:
        row = self.con.execute(
            "SELECT reaction FROM reply_requests WHERE bot_id=? AND group_id=? AND message_id=?",
            (bot_id, group_id, message_id),
        ).fetchone()
        return row[0] if row else None

    def mark_related(self, anchor_seq: int, seqs, relation: str) -> int:
        """Record messages used for one request (context snapshot, answers, ...).

        Marked messages join the related set kept for `related_days` (CONTEXT-RETENTION);
        a mark never extends a message's life beyond that hard limit.
        """
        with self.con:
            cursor = self.con.executemany(
                "INSERT OR IGNORE INTO related_marks(seq,anchor_seq,relation)"
                " SELECT seq,?,? FROM messages WHERE seq=?",
                [(anchor_seq, relation, seq) for seq in seqs],
            )
            return cursor.rowcount

    def revoke(
        self,
        bot_id: str,
        group_id: str,
        message_id: str,
        user_id: str,
        received_at: float | None = None,
    ) -> str:
        """Drop recalled content and media links; keep id, sender and the flag (D-02).

        Media files left without references are removed by `drop_orphan_media` callers.

        Returns:
            "revoked" when stored content was removed, "tombstone" when the recall
            arrived first (a placeholder blocks later insertion), "duplicate" otherwise.
        """
        with self.con:
            cursor = self.con.execute(
                "UPDATE messages SET revoked=1, segments='[]'"
                " WHERE bot_id=? AND group_id=? AND message_id=? AND revoked=0",
                (bot_id, group_id, message_id),
            )
            if cursor.rowcount:
                self.con.execute(
                    "DELETE FROM message_media WHERE seq=(SELECT seq FROM messages"
                    " WHERE bot_id=? AND group_id=? AND message_id=?)",
                    (bot_id, group_id, message_id),
                )
                self.con.execute(
                    "DELETE FROM related_marks WHERE seq=(SELECT seq FROM messages"
                    " WHERE bot_id=? AND group_id=? AND message_id=?)",
                    (bot_id, group_id, message_id),
                )
                return "revoked"
            cursor = self.con.execute(
                "INSERT OR IGNORE INTO messages"
                "(bot_id,group_id,message_id,user_id,segments,revoked,received_at)"
                " VALUES (?,?,?,?,'[]',1,?)",
                (bot_id, group_id, message_id, user_id, received_at),
            )
            return "tombstone" if cursor.rowcount else "duplicate"

    def lookup(self, bot_id: str, group_id: str, message_id: str) -> dict | None:
        """Reference lookup within one group: sender, time, source and recall flag, no content."""
        row = self.con.execute(
            "SELECT seq,user_id,revoked,platform_time,source FROM messages"
            " WHERE bot_id=? AND group_id=? AND message_id=?",
            (bot_id, group_id, message_id),
        ).fetchone()
        return dict(row) if row else None

    def history(self, message: Message, boundary: int) -> list[Message]:
        """Visible messages of the same bot and group before the run boundary.

        The boundary is the current message's seq. Backfilled and self-synced messages
        (stored after the current message arrived, so with a larger seq) are included
        when their platform time is not after the current message's. Order is platform
        time (local receive time when missing), with seq as the tie-breaker, so such a
        message appears at its real position. The owner's `#新版` command messages are
        never included.
        """
        current = self.con.execute(
            "SELECT platform_time FROM messages WHERE seq=? AND bot_id=? AND group_id=?",
            (boundary, message.bot_id, message.group_id),
        ).fetchone()
        current_time = current["platform_time"] if current else None
        rows = self.con.execute(
            f"SELECT * FROM messages WHERE {self._VISIBLE}"
            " ORDER BY COALESCE(platform_time, received_at), seq",
            (message.bot_id, message.group_id, boundary, boundary, current_time, current_time),
        ).fetchall()
        return [_message(r) for r in rows]

    # History-eligible rows: same bot and group, not recalled, not the owner's `#新版`
    # command (D-19: never context material), before the boundary seq - or stored later by
    # a lookup / self-sync with a platform time not after the current message's.
    _VISIBLE = (
        "bot_id=? AND group_id=? AND revoked=0 AND kind<>'owner_command' AND seq<>? AND"
        " (seq<? OR (source IN ('backfill', 'self_sync') AND platform_time IS NOT NULL"
        " AND ? IS NOT NULL AND platform_time<=?))"
    )

    def _visible_args(self, message: Message, boundary: int) -> tuple:
        current = self.con.execute(
            "SELECT platform_time FROM messages WHERE seq=? AND bot_id=? AND group_id=?",
            (boundary, message.bot_id, message.group_id),
        ).fetchone()
        t = current["platform_time"] if current else None
        return (message.bot_id, message.group_id, boundary, boundary, t, t)

    def history_window(
        self, message: Message, boundary: int, since: float | None, limit: int
    ) -> list[sqlite3.Row]:
        """Rows `history()` would return, restricted to production time >= `since` and the
        latest `limit`, in history order (platform time, then seq). For the online run's
        context window, so a run never scans a whole group's buffer."""
        args = self._visible_args(message, boundary)
        where = self._VISIBLE
        params: tuple = args
        if since is not None:
            where += " AND COALESCE(platform_time, received_at) >= ?"
            params = (*args, since)
        rows = self.con.execute(
            f"SELECT * FROM messages WHERE {where}"
            " ORDER BY COALESCE(platform_time, received_at) DESC, seq DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return rows[::-1]

    def visible_by_ids(self, message: Message, boundary: int, ids) -> list[sqlite3.Row]:
        """Visible (history-eligible) rows of the same group with the given message ids."""
        ids = [str(i) for i in ids]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        return self.con.execute(
            f"SELECT * FROM messages WHERE {self._VISIBLE} AND message_id IN ({marks})",
            (*self._visible_args(message, boundary), *ids),
        ).fetchall()

    def history_range(
        self, message: Message, boundary: int, start: float, end: float, limit: int
    ) -> list[sqlite3.Row]:
        """Rows `history()` would return with production time in [start, end], oldest
        first, at most `limit` (the neighbourhood of an old quoted message)."""
        args = self._visible_args(message, boundary)
        rows = self.con.execute(
            f"SELECT * FROM messages WHERE {self._VISIBLE}"
            " AND COALESCE(platform_time, received_at) BETWEEN ? AND ?"
            " ORDER BY COALESCE(platform_time, received_at), seq LIMIT ?",
            (*args, start, end, limit),
        ).fetchall()
        return rows

    def request_run_status(self, bot_id: str, group_id: str, message_id: str) -> str:
        """What became of a request message's reply run (nudge handling): "active" (any
        non-final state), "answered" (an answer outbound row exists that did not fail: recorded
        in record-only mode, or sending / sent / uncertain live) or "open" (no run, or it
        ended without an answer: failed, expired, cancelled, rejected)."""
        run = self.con.execute(
            "SELECT run_id, state FROM runs WHERE bot_id=? AND group_id=? AND message_id=?"
            " AND purpose='reply'",
            (bot_id, group_id, message_id),
        ).fetchone()
        if run is None:
            return "open"
        if run["state"] in RUN_ACTIVE_STATES:
            return "active"
        sent = self.con.execute(
            "SELECT 1 FROM outbound WHERE run_id=? AND kind='answer' AND state<>'failed'",
            (run["run_id"],),
        ).fetchone()
        return "answered" if sent else "open"

    def admitted_runs(self, bot_id: str, group_id: str, since: float) -> int:
        """Reply runs of a group created since `since` that were not rejected (the per-group
        admission rate; cost-cap and rate-limited rejections do not count)."""
        return self.con.execute(
            "SELECT count(*) FROM runs WHERE bot_id=? AND group_id=? AND purpose='reply'"
            " AND created_at>=? AND state<>'rejected'",
            (bot_id, group_id, since),
        ).fetchone()[0]

    def revoked_among(self, bot_id: str, group_id: str, message_ids) -> bool:
        """True when any of the messages was recalled or is no longer stored."""
        ids = [str(i) for i in message_ids]
        if not ids:
            return False
        found = 0
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            row = self.con.execute(
                f"SELECT count(*), coalesce(sum(revoked), 0) FROM messages WHERE bot_id=?"
                f" AND group_id=? AND message_id IN ({marks})",
                (bot_id, group_id, *chunk),
            ).fetchone()
            if row[1]:
                return True
            found += row[0]
        return found < len(set(ids))

    def message_row(self, seq: int) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM messages WHERE seq=?", (seq,)).fetchone()

    def pending_media(self, seqs) -> int:
        seqs = list(seqs)
        if not seqs:
            return 0
        marks = ",".join("?" * len(seqs))
        return self.con.execute(
            f"SELECT count(*) FROM message_media WHERE status='pending' AND seq IN ({marks})",
            seqs,
        ).fetchone()[0]

    # -- media ----------------------------------------------------------------------
    def add_media_slots(
        self, seq: int, slots: list[tuple[int, int | None]], updated_at: float | None = None
    ) -> int:
        """Register image segments of a stored message as pending downloads."""
        with self.con:
            cursor = self.con.executemany(
                "INSERT OR IGNORE INTO message_media"
                "(seq,segment_index,status,declared_bytes,updated_at)"
                " SELECT seq,?, 'pending',?,? FROM messages WHERE seq=? AND revoked=0",
                [(index, declared, updated_at, seq) for index, declared in slots],
            )
            return cursor.rowcount

    def media_unavailable(
        self, seq: int, index: int, reason: str, updated_at: float | None = None
    ) -> bool:
        with self.con:
            cursor = self.con.execute(
                "UPDATE message_media SET status='unavailable',reason=?,updated_at=?"
                " WHERE seq=? AND segment_index=? AND status='pending'",
                (reason, updated_at, seq, index),
            )
            return cursor.rowcount == 1

    def retry_media(self, seq: int, index: int, updated_at: float) -> bool:
        """Reopen an existing slot only while its owning message remains valid."""
        with self.con:
            cursor = self.con.execute(
                "UPDATE message_media SET status='pending',reason=NULL,updated_at=?"
                " WHERE seq=? AND segment_index=? AND EXISTS"
                " (SELECT 1 FROM messages WHERE seq=? AND revoked=0)",
                (updated_at, seq, index, seq),
            )
            return cursor.rowcount == 1

    def attach_media(self, seq: int, index: int, asset: dict, updated_at: float | None = None):
        """Link a stored file to a pending slot. False when the slot is gone (recalled,
        cleaned up); the caller then garbage-collects the possibly orphaned asset."""
        with self.con:
            self.con.execute(
                "INSERT OR IGNORE INTO media_assets(sha256,mime,bytes,width,height,path,created_at)"
                " VALUES (:sha256,:mime,:bytes,:width,:height,:path,:created_at)",
                {**asset, "created_at": updated_at},
            )
            cursor = self.con.execute(
                "UPDATE message_media SET status='available',sha256=?,reason=NULL,updated_at=?"
                " WHERE seq=? AND segment_index=? AND status='pending'",
                (asset["sha256"], updated_at, seq, index),
            )
            return cursor.rowcount == 1

    def media_asset(self, sha256: str) -> dict | None:
        row = self.con.execute("SELECT * FROM media_assets WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None

    def message_media(self, seq: int) -> list[dict]:
        rows = self.con.execute(
            "SELECT mm.segment_index, mm.status, mm.reason, mm.declared_bytes, mm.sha256,"
            " a.mime, a.bytes, a.width, a.height, a.path FROM message_media mm"
            " LEFT JOIN media_assets a ON a.sha256 = mm.sha256 WHERE mm.seq=?"
            " ORDER BY mm.segment_index",
            (seq,),
        ).fetchall()
        return [dict(r) for r in rows]

    def drop_orphan_media(self) -> list[str]:
        """Delete asset rows no message references; returns their relative file paths."""
        with self.con:
            rows = self.con.execute(
                "SELECT sha256, path FROM media_assets WHERE sha256 NOT IN"
                " (SELECT sha256 FROM message_media WHERE sha256 IS NOT NULL)"
            ).fetchall()
            self.con.executemany(
                "DELETE FROM media_assets WHERE sha256=?", [(r["sha256"],) for r in rows]
            )
        return [r["path"] for r in rows]

    # -- runs -----------------------------------------------------------------------
    def start_run(self, seq: int) -> str:
        run_id = uuid4().hex
        with self.con:
            self.con.execute(
                "INSERT INTO runs(run_id,message_seq,state) VALUES (?,?,'running')", (run_id, seq)
            )
        return run_id

    def fail_run(self, run_id: str, kind: str, state="failed"):
        with self.con:
            self.con.execute(
                "UPDATE runs SET state=?,error_kind=? WHERE run_id=?", (state, kind, run_id)
            )

    def record_outbound(self, run_id: str, message: Message, text: str):
        """Offline Pipeline only: an intent row in state pending/offline."""
        with self.con:
            self.con.execute(
                "INSERT INTO outbound(run_id,bot_id,group_id,reply_to,requester,text,state,mode)"
                " VALUES (?,?,?,?,?,?,'pending','offline')",
                (
                    run_id,
                    message.bot_id,
                    message.group_id,
                    message.message_id,
                    message.user_id,
                    text,
                ),
            )
            self.con.execute("UPDATE runs SET state='recorded' WHERE run_id=?", (run_id,))

    # -- online runs (T3.4) -----------------------------------------------------------
    def create_run(
        self,
        run_id: str,
        seq: int,
        message: Message,
        *,
        state: str,
        cost_mode: str,
        now: float,
        error_kind: str | None = None,
        purpose: str = "reply",
    ) -> bool:
        """Insert the run row for one request; False when that message already has a run
        (one run per request, AC-15). A `new_version` run (D-19) is keyed by the owner's
        command message (`seq`) and records the quoted question as `message`."""
        if state not in RUN_STATES:
            raise ValueError("unknown run state")
        if purpose not in RUN_PURPOSES:
            raise ValueError("unknown run purpose")
        ended = now if state not in RUN_ACTIVE_STATES else None
        with self.con:
            cursor = self.con.execute(
                "INSERT OR IGNORE INTO runs(run_id,message_seq,state,error_kind,bot_id,group_id,"
                "message_id,user_id,cost_mode,created_at,updated_at,ended_at,purpose)"
                " SELECT ?,?,?,?,?,?,?,?,?,?,?,?,? WHERE EXISTS"
                " (SELECT 1 FROM messages WHERE seq=?)",
                (
                    run_id,
                    seq,
                    state,
                    error_kind,
                    message.bot_id,
                    message.group_id,
                    message.message_id,
                    message.user_id,
                    cost_mode,
                    now,
                    now,
                    ended,
                    purpose,
                    seq,
                ),
            )
            return cursor.rowcount == 1

    def update_run(self, run_id: str, now: float, **fields) -> None:
        allowed = {
            "state",
            "error_kind",
            "route",
            "queued",
            "started_at",
            "ended_at",
            "detail",
            "cost_mode",
        }
        if not set(fields) <= allowed:
            raise ValueError("unknown run field")
        if "state" in fields and fields["state"] not in RUN_STATES:
            raise ValueError("unknown run state")
        names = sorted(fields)
        sets = ",".join(f"{n}=?" for n in names)
        with self.con:
            self.con.execute(
                f"UPDATE runs SET {sets},updated_at=? WHERE run_id=?",
                (*(fields[n] for n in names), now, run_id),
            )

    def run(self, run_id: str) -> dict | None:
        row = self.con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def interrupt_active_runs(self, now: float) -> int:
        """A previous process ended while runs were active: they cannot resume (their
        in-memory state is gone), so they end as failed/interrupted with their outbound."""
        with self.con:
            self.con.execute(
                "UPDATE outbound SET state='failed', error_kind='interrupted' WHERE state="
                "'pending' AND run_id IN (SELECT run_id FROM runs WHERE state IN"
                " ('queued','running'))"
            )
            # A frame may have been written before the process ended: never resend (RUN-12).
            self.con.execute(
                "UPDATE outbound SET state='uncertain', error_kind='interrupted'"
                " WHERE state='sending'"
            )
            cursor = self.con.execute(
                "UPDATE runs SET state='failed', error_kind='interrupted', ended_at=?,"
                " updated_at=? WHERE state IN ('queued','running')",
                (now, now),
            )
            return cursor.rowcount

    def save_context_snapshot(
        self,
        run_id: str,
        anchor_seq: int,
        strategy: str,
        message_ids: list[str],
        reasons: dict[str, str],
        gaps: list[str],
        degraded: bool,
        now: float,
    ) -> None:
        """Context snapshot summary: ids and selection reasons only, no message content."""
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO context_snapshots VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    anchor_seq,
                    strategy,
                    json.dumps(message_ids, ensure_ascii=False),
                    json.dumps(reasons, ensure_ascii=False),
                    json.dumps(gaps, ensure_ascii=False),
                    int(degraded),
                    now,
                ),
            )

    def context_snapshot(self, run_id: str) -> dict | None:
        row = self.con.execute(
            "SELECT * FROM context_snapshots WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        for key in ("message_ids", "reasons", "gaps"):
            data[key] = json.loads(data[key])
        return data

    # -- outbound, record only (T3.7) ---------------------------------------------------
    def outbound_enqueue(
        self,
        run_id: str,
        kind: str,
        message: Message,
        text: str,
        now: float,
        mode: str = "record_only",
    ) -> tuple[dict, bool]:
        """Idempotent outbound intent (RUN-08): one row per run, keyed by run id and by
        (bot, group, request message, kind). Returns `(row, fresh)`. Kind `new_version`
        (the owner's `#新版` command, D-19) is always `live_send`; answers and busy notices
        are `live_send` only when the serve process sends (`[serve] live_send`). Kind
        `owner_control` (the confirmation of an owner remote-control command) is `live_send`
        when sent and `record_only` otherwise; no run row stands behind it."""
        if kind not in OUTBOUND_KINDS:
            raise ValueError("unknown outbound kind")
        if mode not in ("record_only", "live_send") or (
            kind == "new_version" and mode != "live_send"
        ):
            raise ValueError("outbound mode does not match kind")
        key = outbound_key(message.bot_id, message.group_id, message.message_id, kind)
        with self.con:
            cursor = self.con.execute(
                "INSERT OR IGNORE INTO outbound(run_id,bot_id,group_id,reply_to,requester,text,"
                "state,mode,kind,idempotency_key,created_at) VALUES"
                " (?,?,?,?,?,?,'pending',?,?,?,?)",
                (
                    run_id,
                    message.bot_id,
                    message.group_id,
                    message.message_id,
                    message.user_id,
                    text,
                    mode,
                    kind,
                    key,
                    now,
                ),
            )
        row = self.con.execute(
            "SELECT * FROM outbound WHERE run_id=? OR idempotency_key=?", (run_id, key)
        ).fetchone()
        return dict(row), cursor.rowcount == 1

    def outbound_update(self, run_id: str, now: float, **fields) -> None:
        allowed = {"state", "error_kind", "scheduled_at", "recorded_at", "qq_message_id", "sent_at"}
        if not set(fields) <= allowed:
            raise ValueError("unknown outbound field")
        if "state" in fields and fields["state"] not in OUTBOUND_STATES:
            raise ValueError("unknown outbound state")
        names = sorted(fields)
        sets = ",".join(f"{n}=?" for n in names)
        with self.con:
            self.con.execute(
                f"UPDATE outbound SET {sets} WHERE run_id=?",
                (*(fields[n] for n in names), run_id),
            )

    def claim_outbound(self, run_id: str, kinds: tuple[str, ...]) -> bool:
        """Atomically move one live row of `kinds` from pending to sending (exactly once:
        a second claim, a record-only row or a finished row is refused)."""
        marks = ",".join("?" for _ in kinds)
        with self.con:
            cursor = self.con.execute(
                "UPDATE outbound SET state='sending' WHERE run_id=? AND state='pending'"
                f" AND mode='live_send' AND kind IN ({marks})",
                (run_id, *kinds),
            )
        return cursor.rowcount == 1

    def fail_pending_outbound(self, run_id: str, reason: str | None) -> int:
        with self.con:
            cursor = self.con.execute(
                "UPDATE outbound SET state='failed', error_kind=? WHERE run_id=? AND"
                " state='pending'",
                (reason, run_id),
            )
            return cursor.rowcount

    def outbound(self, run_id: str) -> dict | None:
        row = self.con.execute("SELECT * FROM outbound WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def outbound_for(self, bot_id: str, group_id: str, message_id: str, kind: str):
        """The outbound row of one request message and kind (idempotency key), if any."""
        row = self.con.execute(
            "SELECT * FROM outbound WHERE idempotency_key=?",
            (outbound_key(bot_id, group_id, message_id, kind),),
        ).fetchone()
        return dict(row) if row else None

    def reply_run(self, seq: int) -> dict | None:
        """The reply run of one request message (by its seq) with its answer outbound row
        (`outbound_state`, `outbound_text`, `recorded_at`), if any."""
        row = self.con.execute(
            "SELECT r.*, o.state AS outbound_state, o.text AS outbound_text,"
            " o.recorded_at AS recorded_at FROM runs r LEFT JOIN outbound o"
            " ON o.run_id=r.run_id AND o.kind='answer'"
            " WHERE r.message_seq=? AND r.purpose='reply'",
            (seq,),
        ).fetchone()
        return dict(row) if row else None

    def answer_map(self, bot_id: str, group_id: str) -> dict[str, str]:
        """Bot message id -> answered request id (RUN-08): outbound rows with a QQ message
        id (the D-19 `#新版` sends) and `answered_by` marks (bot messages paired with a
        request, e.g. old-pipeline replies without a quote, `link_unquoted_bot_reply`)."""
        rows = self.con.execute(
            "SELECT qq_message_id, reply_to FROM outbound WHERE bot_id=? AND group_id=?"
            " AND qq_message_id IS NOT NULL AND kind<>'owner_control'",
            (bot_id, group_id),
        ).fetchall()
        mapping = {r[0]: r[1] for r in rows}
        marks = self.con.execute(
            "SELECT b.message_id, q.message_id FROM related_marks rm"
            " JOIN messages b ON b.seq=rm.seq JOIN messages q ON q.seq=rm.anchor_seq"
            " WHERE rm.relation='answered_by' AND b.bot_id=? AND b.group_id=?"
            " AND q.group_id=b.group_id",
            (bot_id, group_id),
        ).fetchall()
        for bot_message, question in marks:
            mapping.setdefault(bot_message, question)
        return mapping

    def answer_knowledge(self, bot_id: str, group_id: str, qq_message_id: str) -> dict[str, int]:
        """Knowledge `{id: revision}` a sent answer used (its run detail), for an owner
        correction quoting that answer (LEARN-03 expected revision); {} when unknown."""
        row = self.con.execute(
            "SELECT r.detail FROM outbound o JOIN runs r ON r.run_id=o.run_id"
            " WHERE o.bot_id=? AND o.group_id=? AND o.qq_message_id=?",
            (bot_id, group_id, qq_message_id),
        ).fetchone()
        try:
            used = json.loads(row[0]).get("knowledge") if row and row[0] else None
        except ValueError:
            used = None
        return {str(k): int(v) for k, v in (used or {}).items()}

    def _new_version_ids(self, bot_id: str, group_id: str) -> set[str]:
        return {
            r[0]
            for r in self.con.execute(
                "SELECT qq_message_id FROM outbound WHERE bot_id=? AND group_id=?"
                " AND kind='new_version' AND qq_message_id IS NOT NULL",
                (bot_id, group_id),
            )
        }

    def _answer_claims(self, bot_id: str, group_id: str) -> dict[int, list[sqlite3.Row]]:
        """Request seq -> the old-pipeline bot messages marked `answered_by` it. The new
        framework's own `〔新版〕` sends do not count as claims (shadow_report rule)."""
        new_version = self._new_version_ids(bot_id, group_id)
        claims: dict[int, list[sqlite3.Row]] = {}
        for row in self.con.execute(
            "SELECT rm.anchor_seq, b.seq, b.bot_id, b.group_id, b.message_id, b.user_id,"
            " b.segments, b.platform_time FROM related_marks rm JOIN messages b ON b.seq=rm.seq"
            " WHERE rm.relation='answered_by' AND b.bot_id=? AND b.group_id=? AND b.user_id=?",
            (bot_id, group_id, bot_id),
        ):
            if row["message_id"] in new_version or _is_new_version(_message(row)):
                continue
            claims.setdefault(row["anchor_seq"], []).append(row)
        return claims

    def link_unquoted_bot_reply(
        self, seq: int, message: Message, platform_time: int | None, window: float = 300.0
    ) -> int | None:
        """Pair a stored bot message that quotes nothing with the request it answered,
        by the old-pipeline rule (`scripts/replay/legacy_link.py`): the earliest request of
        the group within `window` seconds before it that no bot message has claimed yet
        (`〔新版〕` sends do not claim). Writes an `answered_by` mark (bot message ->
        request); returns the request seq."""
        if message.user_id != message.bot_id or message.reply_ids or platform_time is None:
            return None
        if _is_new_version(message):
            return None
        existing = self.con.execute(
            "SELECT anchor_seq FROM related_marks WHERE seq=? AND relation='answered_by'", (seq,)
        ).fetchone()
        if existing is not None:
            return existing[0]
        rows = self.con.execute(
            "SELECT rr.seq FROM reply_requests rr"
            " WHERE rr.bot_id=? AND rr.group_id=? AND rr.seq IS NOT NULL"
            " AND COALESCE(rr.platform_time, rr.received_at) BETWEEN ? AND ?"
            " ORDER BY COALESCE(rr.platform_time, rr.received_at), rr.seq",
            (message.bot_id, message.group_id, platform_time - window, platform_time),
        ).fetchall()
        claimed = self._answer_claims(message.bot_id, message.group_id)
        for row in rows:
            if row["seq"] in claimed or row["seq"] == seq:
                continue
            self.mark_related(row["seq"], [seq], "answered_by")
            return row["seq"]
        return None

    def link_bot_message(
        self, seq: int, message: Message, platform_time: int | None, window: float = 300.0
    ) -> tuple[str, int | None]:
        """Write `answered_by` for one stored bot message (self-sync, `self_sync.py`).

        A reply segment pointing at a stored member message links it directly (rule 1 of
        `legacy_link`); an unquoted claim on that request made earlier is released and
        that message is paired again by the unquoted rule. A message without a quote is
        paired by `link_unquoted_bot_reply` (same group, `window` seconds, first unclaimed
        request). Returns `(outcome, request seq)`: outcome `quote`, `unquoted`,
        `new_version` (the framework's own send, not paired) or `unpaired`."""
        if message.user_id != message.bot_id:
            return "unpaired", None
        if message.reply_ids:
            for rid in message.reply_ids:
                target = self.lookup(message.bot_id, message.group_id, rid)
                if target is None or target["user_id"] == message.bot_id:
                    continue
                displaced = [
                    r
                    for r in self._answer_claims(message.bot_id, message.group_id).get(
                        target["seq"], []
                    )
                    if r["seq"] != seq and not _message(r).reply_ids
                ]
                with self.con:
                    self.con.executemany(
                        "DELETE FROM related_marks WHERE seq=? AND anchor_seq=?"
                        " AND relation='answered_by'",
                        [(r["seq"], target["seq"]) for r in displaced],
                    )
                self.mark_related(target["seq"], [seq], "answered_by")
                for r in displaced:
                    self.link_unquoted_bot_reply(r["seq"], _message(r), r["platform_time"], window)
                return ("new_version" if _is_new_version(message) else "quote"), target["seq"]
            return "unpaired", None
        if _is_new_version(message):
            return "new_version", None
        request = self.link_unquoted_bot_reply(seq, message, platform_time, window)
        return ("unquoted", request) if request is not None else ("unpaired", None)

    def unanswered_request_before(self, message: Message, seq: int, window: float = 300.0) -> bool:
        """True when the same asker has an earlier request in this group within `window`
        seconds before `seq`'s message that no old-pipeline bot message answers yet (its
        answer may still be on its way through the self-sync)."""
        row = self.message_row(seq)
        if row is None:
            return False
        t = row["platform_time"] if row["platform_time"] is not None else row["received_at"]
        if t is None:
            return False
        rows = self.con.execute(
            "SELECT seq FROM reply_requests WHERE bot_id=? AND group_id=? AND user_id=?"
            " AND seq IS NOT NULL AND seq<>?"
            " AND COALESCE(platform_time, received_at) BETWEEN ? AND ?",
            (message.bot_id, message.group_id, message.user_id, seq, t - window, t),
        ).fetchall()
        if not rows:
            return False
        claimed = self._answer_claims(message.bot_id, message.group_id)
        return any(r["seq"] not in claimed for r in rows)

    def mark_owner_command(self, seq: int) -> bool:
        """Flag the owner's `#新版` command message (D-19): never context material and not
        in the related set through context or quotes (`_VISIBLE`, retention)."""
        with self.con:
            cursor = self.con.execute(
                "UPDATE messages SET kind='owner_command' WHERE seq=?", (seq,)
            )
            return cursor.rowcount == 1

    # -- owner remote control (v10) ---------------------------------------------------
    def control_state(self) -> tuple[bool, float | None]:
        """`(paused, since)`: the persisted pause state; `since` is when it last changed
        (None when it never did)."""
        rows = dict(self.con.execute("SELECT key, value FROM control_state").fetchall())
        since = rows.get("since")
        return rows.get("paused") == "1", float(since) if since is not None else None

    def apply_control(
        self,
        bot_id: str,
        group_id: str,
        message_id: str,
        action: str,
        now: float,
        *,
        apply: bool = True,
    ) -> str | None:
        """Record one owner control command and apply it to the persisted pause state in one
        transaction. Idempotent per command message: returns None for a command already
        recorded, else the effect: `paused`, `resumed`, `already_paused`, `already_running`,
        `status`, or `stale` (recorded only, `apply=False`)."""
        if action not in CONTROL_ACTIONS:
            raise ValueError("unknown control action")
        with self.con:
            paused, _ = self.control_state()
            if not apply:
                effect = "stale"
            elif action == "status":
                effect = "status"
            elif action == "pause":
                effect = "already_paused" if paused else "paused"
            else:
                effect = "resumed" if paused else "already_running"
            cursor = self.con.execute(
                "INSERT OR IGNORE INTO control_events"
                "(created_at,bot_id,group_id,message_id,action,effect) VALUES (?,?,?,?,?,?)",
                (now, bot_id, group_id, message_id, action, effect),
            )
            if cursor.rowcount != 1:
                return None
            if effect in ("paused", "resumed"):
                for key, value in (("paused", "1" if effect == "paused" else "0"), ("since", now)):
                    self.con.execute(
                        "INSERT INTO control_state(key,value,updated_at) VALUES (?,?,?)"
                        " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                        " updated_at=excluded.updated_at",
                        (key, str(value), now),
                    )
            return effect

    def control_events(self) -> list[dict]:
        rows = self.con.execute("SELECT * FROM control_events ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def groups(self, bot_id: str) -> list[str]:
        """Groups with stored messages of this bot."""
        return [
            r[0]
            for r in self.con.execute(
                "SELECT DISTINCT group_id FROM messages WHERE bot_id=?", (bot_id,)
            )
        ]

    # -- usage ledger (T3.9) ----------------------------------------------------------
    def record_usage(self, row: dict) -> None:
        with self.con:
            self.con.execute(
                "INSERT INTO usage_ledger(run_id,group_id,day,created_at,provider,model,task,mode,status,"
                "input_tokens,cached_tokens,output_tokens,thinking_tokens,requests,cost_cny,"
                "latency_s,channel,service,list_cost,list_currency,google_queries)"
                " VALUES (:run_id,COALESCE(:group_id,(SELECT group_id FROM runs WHERE run_id=:run_id)),:day,:created_at,:provider,:model,:task,:mode,"
                ":status,:input_tokens,:cached_tokens,:output_tokens,:thinking_tokens,:requests,"
                ":cost_cny,:latency_s,:channel,:service,:list_cost,:list_currency,"
                ":google_queries)",
                {
                    "group_id": None,
                    **dict.fromkeys(USAGE_V9_COLUMNS),
                    **row,
                },  # callers may omit the v9 keys
            )

    def usage_by_group(self, day: str) -> dict[str, float]:
        return dict(
            self.con.execute(
                "SELECT group_id,SUM(cost_cny) FROM usage_ledger "
                "WHERE day=? AND group_id IS NOT NULL GROUP BY group_id",
                (day,),
            )
        )

    def usage_total(self, day: str) -> float:
        value = self.con.execute(
            "SELECT COALESCE(SUM(cost_cny), 0) FROM usage_ledger WHERE day=?", (day,)
        ).fetchone()[0]
        return float(value)

    def usage_summary(self, day: str) -> dict:
        rows = self.con.execute(
            "SELECT provider, count(*) AS calls, SUM(cost_cny) AS cost,"
            " SUM(input_tokens) AS input, SUM(output_tokens) AS output FROM usage_ledger"
            " WHERE day=? GROUP BY provider",
            (day,),
        ).fetchall()
        return {
            r["provider"]: {
                "calls": r["calls"],
                "cost_cny": round(r["cost"] or 0.0, 6),
                "input": r["input"] or 0,
                "output": r["output"] or 0,
            }
            for r in rows
        }

    def _count(self, sql: str) -> int:
        return self.con.execute(sql).fetchone()[0]

    def summary(self) -> dict:
        return {
            "messages": self._count("SELECT count(*) FROM messages"),
            "revoked": self._count("SELECT count(*) FROM messages WHERE revoked=1"),
            "backfilled": self._count("SELECT count(*) FROM messages WHERE source='backfill'"),
            "self_synced": self._count("SELECT count(*) FROM messages WHERE source='self_sync'"),
            "owner_commands": self._count(
                "SELECT count(*) FROM messages WHERE kind='owner_command'"
            ),
            "reply_requests": self._count("SELECT count(*) FROM reply_requests"),
            "control_events": self._count("SELECT count(*) FROM control_events"),
            "media_slots": dict(
                self.con.execute(
                    "SELECT status,count(*) FROM message_media GROUP BY status"
                ).fetchall()
            ),
            "media_files": self._count("SELECT count(*) FROM media_assets"),
            "runs": self._count("SELECT count(*) FROM runs"),
            "outbound_records": self._count("SELECT count(*) FROM outbound"),
            "outbound_states": dict(
                self.con.execute("SELECT state,count(*) FROM outbound GROUP BY state").fetchall()
            ),
            # Only the owner's `#新版` command can reach `sent` (D-19, owner_command.py).
            "sent_to_qq": self._count("SELECT count(*) FROM outbound WHERE state='sent'"),
            "run_states": dict(
                self.con.execute("SELECT state,count(*) FROM runs GROUP BY state").fetchall()
            ),
        }
