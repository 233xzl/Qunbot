"""scripts/migrate_access.py with synthetic AstrBot files (no real ids)."""

import importlib.util
import json
from pathlib import Path
import tomllib

from group_agent.admission import AccessPolicy, AccessSnapshot, RequestQueue
from group_agent.intake import EventIntake
from group_agent.store import Store

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("migrate_access", ROOT / "scripts/migrate_access.py")
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)

SETTINGS = """secrets_file = "x.env"

[identity]
# local identity
bot_id = "10000001"
owner_id = "OWNER"
blocked_users = [
  "20000001",
]

[serve]
owner_command_send = true
"""


def _astrbot(tmp_path, *, whitelist=False, prefix=()):
    data = tmp_path / "astrbot"
    data.mkdir()
    access = {
        "owner_qq": "30000001",
        "bot_qq": "10000001",
        "blocked_qq_ids": ["20000002", 20000003],
        "blocked_group_ids": ["40000001"],
    }
    config = {
        "platform_settings": {
            "enable_id_white_list": whitelist,
            "id_whitelist": ["50000001"] if whitelist else [],
            "ignore_at_all": True,
            "reply_with_quote": True,
            "reply_with_mention": False,
            "reply_prefix": "",
            "segmented_reply": {"enable": False},
            "forward_threshold": 1500,
        },
        "wake_prefix": list(prefix),
        "admins_id": [],
    }
    (data / "strict_access.json").write_text(json.dumps(access), "utf-8")
    (data / "cmd_config.json").write_text(json.dumps(config), "utf-8")
    return data


def test_migrate_access_writes_identity_with_backup_and_reports_counts(tmp_path, capsys):
    data = _astrbot(tmp_path)
    settings = tmp_path / "settings.toml"
    settings.write_text(SETTINGS, "utf-8")
    backups = tmp_path / "backups"
    assert migrate.main(["--astrbot-data", str(data), "--settings", str(settings)]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["written"] is False and dry["up_to_date"] is False
    assert settings.read_text("utf-8") == SETTINGS
    argv = ["--astrbot-data", str(data), "--settings", str(settings), "--backup-dir", str(backups)]
    assert migrate.main([*argv, "--write"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["written"] is True
    assert report["identity"] == {
        "bot_id": "same",
        "owner_id": "changed",
        "blocked_users": {"count": 2, "added": 2, "removed": 1},
        "blocked_groups": {"count": 1, "added": 1, "removed": 0},
    }
    for value in ("30000001", "20000002", "40000001", "10000001"):
        assert value not in out  # counts and names only
    data_written = tomllib.loads(settings.read_text("utf-8"))
    assert data_written["identity"] == {
        "bot_id": "10000001",
        "owner_id": "30000001",
        "blocked_users": ["20000002", "20000003"],
        "blocked_groups": ["40000001"],
    }
    assert data_written["serve"] == {"owner_command_send": True}  # other tables kept
    assert "# local identity" in settings.read_text("utf-8")
    assert [p.read_text("utf-8") for p in backups.iterdir()] == [SETTINGS]
    assert report["differences"] == [d for d in report["differences"] if "long_reply" in d]
    assert migrate.main([*argv, "--write"]) == 0  # nothing left to change
    again = json.loads(capsys.readouterr().out)
    assert again["written"] is False and again["up_to_date"] is True
    assert len(list(backups.iterdir())) == 1


def test_migrate_access_lists_behaviour_differences(tmp_path, capsys):
    data = _astrbot(tmp_path, whitelist=True, prefix=["/"])
    settings = tmp_path / "settings.toml"
    settings.write_text(SETTINGS, "utf-8")
    migrate.main(["--astrbot-data", str(data), "--settings", str(settings)])
    names = [d.split(":")[0] for d in json.loads(capsys.readouterr().out)["differences"]]
    assert names == ["id_whitelist", "wake_prefix", "long_reply"]


def test_admission_matches_the_old_rules():
    """Private chat ignored, @all alone ignored, @all + real @bot answered, blocked silent."""
    store = Store()
    access = AccessPolicy(
        AccessSnapshot("BOT", "OWNER", blocked_users=frozenset({"U9"}), blocked_groups={"G9"})
    )
    queue = RequestQueue(8)
    intake = EventIntake(store, access, admit=True, sink=queue)

    def event(mid, segments, user="U1", group="G1", kind="group"):
        return {
            "post_type": "message",
            "message_type": kind,
            "self_id": "BOT",
            "group_id": group,
            "message_id": mid,
            "user_id": user,
            "message": segments,
        }

    at_bot = {"type": "at", "data": {"qq": "BOT"}}
    at_all = {"type": "at", "data": {"qq": "all"}}
    text = {"type": "text", "data": {"text": "在吗"}}
    statuses = [
        intake.intake(event("p1", [at_bot, text], kind="private")).status,
        intake.intake(event("a1", [at_all, text])).status,
        intake.intake(event("a2", [at_all, at_bot, text])).status,
        intake.intake(event("b1", [at_bot, text], user="U9")).status,
        intake.intake(event("b2", [at_bot, text], group="G9")).status,
        intake.intake(event("s1", [at_bot, text], user="BOT")).status,
    ]
    assert statuses == [
        "ignored_private",
        "stored",
        "trigger",
        "ignored_blocked",
        "ignored_blocked",
        "stored",
    ]
    assert [r.message.message_id for r in list(queue.queue._queue)] == ["a2"]
