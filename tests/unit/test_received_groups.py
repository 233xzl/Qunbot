import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from group_agent import received_groups as groups
from group_agent.config_service import schema
from group_agent.contracts import Message
from group_agent.gateway import ActionTimeout
from group_agent.store import Store


def test_directory_survives_retention_and_upgrades(tmp_path):
    path = tmp_path / "messages.db"
    db = Store(path)
    for bot, gid in [("bot", "known"), ("other", "secret")]:
        db.ingest(Message(bot, gid, "msg", "user", ()), received_at=50)
    db.con.executescript("DROP TABLE received_groups; PRAGMA user_version=12;")
    db.close()
    db = Store(path)
    assert [(g["group_id"], g["membership"]) for g in groups.directory(db.con, "bot")] == [
        ("known", "pending")
    ]
    db.con.execute("DELETE FROM messages")
    db.con.commit()
    db.close()
    db = Store(path)
    assert len(groups.directory(db.con, "bot")) == 1
    db.close()


async def test_snapshot_timeout_reconnect_and_rejoin():
    db = Store()
    bot = "bot"
    db.ingest(Message(bot, "A", "hello", "u", ()))
    db.ingest(Message(bot, "B", "hello", "u", ()))
    gateway = SimpleNamespace(connected=True, stats={"connects": 1})
    access = SimpleNamespace(bot_id=bot, snapshot=SimpleNamespace(blocked_groups=set()))
    cancelled = []

    async def sync(response):
        stop = asyncio.Event()

        async def snapshot():
            stop.set()
            if isinstance(response, Exception):
                raise response
            return response

        await groups.sync_directory(
            stop, db, access, gateway, SimpleNamespace(get_group_list=snapshot), cancelled.append
        )

    await sync(ActionTimeout("timeout"))
    assert {g["membership"] for g in groups.directory(db.con, bot)} == {"pending"}
    await sync(
        [{"group_id": "A", "group_name": "办公群"}, {"group_id": "unseen", "group_name": "other"}]
    )
    assert cancelled == ["B"]
    assert [(g["group_id"], g["name"]) for g in groups.directory(db.con, bot)] == [("A", "办公群")]
    db.ingest(Message(bot, "B", "back", "u", ()))
    assert len(groups.directory(db.con, bot)) == 2
    db.close()


def test_chinese_field_catalog_covers_editor_schema():
    catalog = json.loads(
        (Path(__file__).resolve().parents[2] / "src/group_agent/admin/static/zh-CN.json").read_text(
            "utf8"
        )
    )
    assert all(
        catalog.get(f"{section}.{key}") for section, fields in schema().items() for key in fields
    )
