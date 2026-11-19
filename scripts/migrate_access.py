"""Migrate identity and access rules from AstrBot to the new framework's settings (cutover).

Reads, read-only, AstrBot's `data/strict_access.json` (owner, bot account, blocked users and
groups) and the group-chat related items of `data/cmd_config.json` (id whitelist, @all,
wake prefixes, private chat, reply shape, admins), and writes the `[identity]` table of the
local, git-ignored `config/settings.toml` (`bot_id`, `owner_id`, `blocked_users`,
`blocked_groups`). The original file is copied to `data/config-backups/` first; other tables
and comments are kept as they are. The result is re-read with `config.load_settings`.

Output: a summary with field names, counts and same/changed flags only - never ids.
Behaviour items where the new framework differs from the old configuration are listed under
`differences` (by item name).

    .\\.venv\\Scripts\\python.exe scripts/migrate_access.py            # summary only
    .\\.venv\\Scripts\\python.exe scripts/migrate_access.py --write    # back up and write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group_agent.config import load_settings  # noqa: E402

KNOWN_ACCESS_KEYS = {"owner_qq", "bot_qq", "blocked_qq_ids", "blocked_group_ids"}
_ID = re.compile(r"\d{5,12}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:  # read-only
        return json.load(handle)


def _ids(values, field: str) -> list[str]:
    out = []
    for value in values or []:
        text = str(value).strip()
        if isinstance(value, bool) or not _ID.fullmatch(text):
            raise SystemExit(f"{field}: an entry is not a numeric QQ id")
        out.append(text)
    return sorted(set(out))


def legacy_identity(access: dict) -> dict:
    owner, bot = str(access.get("owner_qq") or ""), str(access.get("bot_qq") or "")
    if not _ID.fullmatch(owner) or not _ID.fullmatch(bot):
        raise SystemExit("strict_access.json: owner_qq / bot_qq missing or not numeric")
    return {
        "bot_id": bot,
        "owner_id": owner,
        "blocked_users": _ids(access.get("blocked_qq_ids"), "blocked_qq_ids"),
        "blocked_groups": _ids(access.get("blocked_group_ids"), "blocked_group_ids"),
    }


def behaviour_check(config: dict) -> tuple[dict, list[str]]:
    """Old group-chat settings (names and flags only) and the items that differ."""
    ps = config.get("platform_settings") or {}
    seg = ps.get("segmented_reply") or {}
    wake = [w for w in (config.get("wake_prefix") or []) if str(w).strip()]
    provider_wake = str((config.get("provider_settings") or {}).get("wake_prefix") or "").strip()
    old = {
        "id_whitelist_enabled": bool(ps.get("enable_id_white_list")),
        "id_whitelist_count": len(ps.get("id_whitelist") or []),
        "ignore_at_all": bool(ps.get("ignore_at_all")),
        "ignore_bot_self_message": bool(ps.get("ignore_bot_self_message")),
        "wake_prefix_count": len(wake) + (1 if provider_wake else 0),
        "friend_message_needs_wake_prefix": bool(ps.get("friend_message_needs_wake_prefix")),
        "reply_with_quote": bool(ps.get("reply_with_quote")),
        "reply_with_mention": bool(ps.get("reply_with_mention")),
        "reply_prefix_set": bool(str(ps.get("reply_prefix") or "")),
        "segmented_reply": bool(seg.get("enable")),
        "forward_threshold": ps.get("forward_threshold"),
        "admins_count": len(config.get("admins_id") or []),
        "t2i": bool(config.get("t2i")),
    }
    differences = []
    if old["id_whitelist_enabled"] and old["id_whitelist_count"]:
        differences.append("id_whitelist: old answers only listed ids; new has no whitelist")
    if not old["ignore_at_all"]:
        differences.append("at_all: old may wake on @all; new never treats @all as a trigger")
    if old["wake_prefix_count"]:
        differences.append("wake_prefix: old also wakes on a prefix; new only on a real @bot")
    if not old["reply_with_quote"] or old["reply_with_mention"] or old["reply_prefix_set"]:
        differences.append("reply_shape: new always sends reply + text, no mention, no prefix")
    if old["segmented_reply"]:
        differences.append("segmented_reply: old split replies; new sends one message")
    if isinstance(old["forward_threshold"], int) and old["forward_threshold"] > 0:
        differences.append(
            "long_reply: old turned replies above forward_threshold into a merged-forward"
            " message; new cuts at the same length with a visible mark"
        )
    if old["admins_count"]:
        differences.append("admins: old AstrBot admins have no role in the new framework")
    if old["t2i"]:
        differences.append("t2i: old rendered long text as an image; new sends text")
    return old, differences


def _toml_value(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(json.dumps(v) for v in value) + "]"
    return json.dumps(value)


def rewrite_identity(text: str, values: dict) -> str:
    """Replace (or add) the given keys in `[identity]`, keeping every other line."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "[identity]"), None)
    if start is None:
        block = ["[identity]"] + [f"{k} = {_toml_value(v)}" for k, v in values.items()]
        return "\n".join(block + [""] + lines) + "\n"
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    body, i, seen = [], start + 1, set()
    while i < end:
        line = lines[i]
        match = re.match(r"\s*([A-Za-z_]+)\s*=", line)
        if match and match.group(1) in values:
            key = match.group(1)
            # Skip a multi-line array value.
            depth = line.count("[") - line.count("]")
            while depth > 0 and i + 1 < end:
                i += 1
                depth += lines[i].count("[") - lines[i].count("]")
            body.append(f"{key} = {_toml_value(values[key])}")
            seen.add(key)
        else:
            body.append(line)
        i += 1
    missing = [f"{k} = {_toml_value(v)}" for k, v in values.items() if k not in seen]
    insert_at = len(body)
    while insert_at > 0 and not body[insert_at - 1].strip():
        insert_at -= 1
    body[insert_at:insert_at] = missing
    return "\n".join(lines[: start + 1] + body + lines[end:]) + "\n"


def current_identity(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text("utf-8")).get("identity") or {}
    return {
        "bot_id": str(data.get("bot_id", "")),
        "owner_id": str(data.get("owner_id", "")),
        "blocked_users": sorted({str(v) for v in data.get("blocked_users") or []}),
        "blocked_groups": sorted({str(v) for v in data.get("blocked_groups") or []}),
    }


def diff_summary(before: dict, after: dict) -> dict:
    out = {}
    for key in ("bot_id", "owner_id"):
        out[key] = "same" if before.get(key) == after[key] else "changed"
    for key in ("blocked_users", "blocked_groups"):
        old, new = set(before.get(key) or []), set(after[key])
        out[key] = {"count": len(new), "added": len(new - old), "removed": len(old - new)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--astrbot-data", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.toml")
    parser.add_argument("--backup-dir", type=Path, default=ROOT / "data/config-backups")
    parser.add_argument("--write", action="store_true", help="back up and write settings")
    args = parser.parse_args(argv)
    access = _read_json(args.astrbot_data / "strict_access.json")
    config = _read_json(args.astrbot_data / "cmd_config.json")
    target = legacy_identity(access)
    before = current_identity(args.settings)
    old, differences = behaviour_check(config)
    summary = {
        "source_fields": sorted(access),
        "unknown_source_fields": sorted(set(access) - KNOWN_ACCESS_KEYS),
        "identity": diff_summary(before, target),
        "legacy_group_settings": old,
        "same_behaviour": [
            "private chats: no reply (old strict patch accepts group messages only)",
            "@all alone: no reply; @all together with a real @bot: reply (both)",
            "blocked user or group: silent (both; new also does not keep their messages)",
            "bot's own messages: never a trigger (both)",
        ],
        "differences": differences,
        "written": False,
    }
    changed = before != target
    if args.write and changed:
        if not args.settings.is_file():
            raise SystemExit("settings file not found; copy config/settings.example.toml first")
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"settings-{time.strftime('%Y%m%dT%H%M%S')}.toml"
        shutil.copy2(args.settings, backup)
        new_text = rewrite_identity(args.settings.read_text("utf-8"), target)
        tomllib.loads(new_text)  # must stay valid TOML before it replaces the file
        args.settings.write_text(new_text, "utf-8")
        settings = load_settings(args.settings)
        loaded = {
            "bot_id": settings.bot_id,
            "owner_id": settings.owner_id,
            "blocked_users": sorted(settings.blocked_users),
            "blocked_groups": sorted(settings.blocked_groups),
        }
        if loaded != target:
            shutil.copy2(backup, args.settings)
            raise SystemExit("verification failed; the original settings were restored")
        summary["written"] = True
        summary["backup"] = str(backup.relative_to(ROOT)) if backup.is_relative_to(ROOT) else "set"
    summary["up_to_date"] = not changed or summary["written"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
