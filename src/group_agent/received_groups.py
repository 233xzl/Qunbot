"""Local directory of groups that have delivered a message to this account."""

import asyncio
import time

from .gateway import GatewayError

SCHEMA = """
CREATE TABLE IF NOT EXISTS received_groups (
    bot_id TEXT NOT NULL, group_id TEXT NOT NULL, last_seen REAL NOT NULL,
    membership TEXT NOT NULL DEFAULT 'present', enabled INTEGER NOT NULL DEFAULT 1,
    stopped_through INTEGER NOT NULL DEFAULT 0, changed_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(bot_id,group_id)
);
"""
BACKFILL = """
INSERT OR IGNORE INTO received_groups(bot_id,group_id,last_seen,membership)
SELECT bot_id,group_id,COALESCE(MAX(received_at),0),'pending' FROM messages
WHERE source='live' AND revoked=0 AND user_id<>bot_id GROUP BY bot_id,group_id;
"""


def observe(con, bot, group, now):
    con.execute(
        """INSERT INTO received_groups(bot_id,group_id,last_seen,changed_at) VALUES (?,?,?,?)
        ON CONFLICT(bot_id,group_id) DO UPDATE SET last_seen=excluded.last_seen,
        changed_at=excluded.changed_at,membership='present'""",
        (bot, group, now or 0, time.time()),
    )


def directory(con, bot):
    if (
        con is None
        or not con.execute("SELECT 1 FROM sqlite_master WHERE name='received_groups'").fetchone()
    ):
        return []
    return [
        dict(r)
        for r in con.execute(
            """SELECT d.group_id, n.name, d.last_seen,d.membership,d.enabled
        FROM received_groups d LEFT JOIN group_names n USING(bot_id,group_id)
        WHERE d.bot_id=? AND d.membership!='left' ORDER BY d.last_seen DESC,d.group_id""",
            (bot,),
        )
    ]


def allows(con, bot, group, seq=None):
    row = con.execute(
        "SELECT enabled,membership,stopped_through FROM received_groups WHERE bot_id=? AND group_id=?",
        (bot, group),
    ).fetchone()
    return row is None or bool(row[0] and row[1] != "left" and (seq is None or seq > row[2]))


def stop_boundary(con, bot, group, *, left=False):
    con.execute(
        "UPDATE received_groups SET stopped_through=(SELECT COALESCE(MAX(seq),0) FROM messages),changed_at=?, "
        + ("membership='left'" if left else "enabled=0")
        + " WHERE bot_id=? AND group_id=?",
        (time.time(), bot, group),
    )
    con.execute(
        "UPDATE outbound SET state='failed',error_kind='group_stopped' "
        "WHERE bot_id=? AND group_id=? AND kind!='owner_control' AND state='pending'",
        (bot, group),
    )


async def sync_directory(stop, store, access, gateway, api, cancel):
    """Intersect a membership snapshot with observed groups. Never import unseen groups."""
    generation = None
    next_sync = 0
    while not stop.is_set():
        epoch = gateway.stats["connects"]
        if gateway.connected and (epoch != generation or time.monotonic() >= next_sync):
            generation = epoch
            start = time.time()
            try:
                members = await api.get_group_list()
            except GatewayError:
                with store.con:
                    store.con.execute(
                        "UPDATE received_groups SET membership='pending' WHERE bot_id=? AND membership='present' AND changed_at<=?",
                        (access.bot_id, start),
                    )
                next_sync = time.monotonic() + 30
            else:
                if gateway.connected and gateway.stats["connects"] == epoch:
                    with store.con:
                        groups = store.con.execute(
                            "SELECT group_id FROM received_groups WHERE bot_id=? AND membership!='left' AND changed_at<=?",
                            (access.bot_id, start),
                        ).fetchall()
                        by_id = {str(g["group_id"]): g.get("group_name") for g in members}
                        for (gid,) in groups:
                            if gid in access.snapshot.blocked_groups:
                                continue
                            if gid not in by_id:
                                stop_boundary(store.con, access.bot_id, gid, left=True)
                                cancel(gid)
                                continue
                            store.con.execute(
                                "UPDATE received_groups SET membership='present' WHERE bot_id=? AND group_id=?",
                                (access.bot_id, gid),
                            )
                            name = by_id[gid]
                            if isinstance(name, str) and name.strip():
                                store.con.execute(
                                    "INSERT OR REPLACE INTO group_names VALUES (?,?,?,?)",
                                    (access.bot_id, gid, name.strip()[:128], start),
                                )
                    next_sync = time.monotonic() + 60
        try:
            await asyncio.wait_for(stop.wait(), 2)
        except TimeoutError:
            pass
