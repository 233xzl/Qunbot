"""Versioned configuration shared by the CLI and future management UI."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
import re
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import uuid

import tomllib
import tomli_w

from .config import (
    BillingSpec,
    Collection,
    ConfigError,
    ModelSpec,
    Retention,
    Settings,
    WebSpec,
    MODELS_SCALARS,
    parse_settings,
)
from .policy import RuntimeLimits


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [plain(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    return str(value) if isinstance(value, Path) else value


GROUPS = {
    "identity": {"bot_id", "owner_id", "blocked_users", "blocked_groups"},
    "budgets": {"deadline_seconds", "context_messages", "daily_cost_cny"},
    "serve": {
        "db_path",
        "owner_command_send",
        "fallback_reply",
        "self_sync",
        "self_sync_interval_s",
        "snowluma_db",
        "processing_reaction",
        "reaction_emoji_id",
        "live_send",
        "owner_control",
    },
    "agent": {
        "answer_tool_loop",
        "web_enabled",
        "decision_mode",
        "answer_thinking",
        "answer_same_model_retry",
    },
    "shadow": {"db_path"},
    "knowledge": {"db_path"},
    "retention": set(Retention.__dataclass_fields__),
    "collection": set(Collection.__dataclass_fields__),
    "billing": set(BillingSpec.__dataclass_fields__),
    "web": set(WebSpec.__dataclass_fields__),
    "runtime": set(RuntimeLimits.__dataclass_fields__)
    - {"deadline_seconds", "context_messages", "moderation_threshold", "retention_days"},
    "model_roles": {"answer", "decision", "fallback"},
    "prompts": {"persona", "evidence", "reply_contract", "decision", "tool_decision", "search"},
}
CONNECTION_KEYS = {
    "protocol",
    "base_url",
    "base_url_env",
    "api_key_env",
    "vertex",
    "location",
    "project_env",
    "api_version",
    "channel",
}
MODEL_KEYS = set(ModelSpec.__dataclass_fields__) - {"name", "cny_per_unit"}
PATH_DEFAULTS = {
    "secrets_file": "private/credentials.env",
    "serve.db_path": "data/serve/serve.sqlite3",
    "shadow.db_path": "data/shadow/shadow.sqlite3",
    "knowledge.db_path": "data/knowledge/knowledge.sqlite3",
    "billing.db_path": "data/billing/account.sqlite3",
}
PATH_FIELDS = {*PATH_DEFAULTS, "collection.media_dir", "serve.snowluma_db"}


def flatten(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def put(data, key, value):
    parts = key.split(".")
    for part in parts[:-1]:
        data = data.setdefault(part, {})
    data[parts[-1]] = value


def _keys(table, allowed, prefix):
    if not isinstance(table, dict):
        raise ConfigError(f"{prefix}: expected a table")
    for key in table:
        if key not in allowed:
            raise ConfigError(f"{prefix}.{key}: unknown field")


def validate_search_prompt(text):
    from string import Formatter

    try:
        for _, name, spec, conversion in Formatter().parse(text):
            if name is not None and (name != "query" or spec or conversion):
                raise ValueError
        text.format(query="test")
    except (ValueError, KeyError):
        raise ConfigError("prompts.search: only {query} is supported") from None


def validate_keys(data):
    _keys(
        data,
        {
            *GROUPS,
            "models",
            "connections",
            "group_profiles",
            "credentials",
            "secrets_file",
            "schema_version",
            "instance_root",
        },
        "settings",
    )
    profiles = data.get("group_profiles", {})
    if not isinstance(profiles, dict):
        raise ConfigError("group_profiles: expected a table")
    for gid, profile in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", gid):
            raise ConfigError("group_profiles: invalid group id")
        path = f"group_profiles.{gid}"
        _keys(profile, {"enabled", "prompts", "model_roles", "daily_cost_cny"}, path)
        if type(profile.get("enabled", False)) is not bool:
            raise ConfigError(f"{path}.enabled: expected a boolean")
        cap = profile.get("daily_cost_cny", 1)
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ConfigError(f"{path}.daily_cost_cny: expected a positive finite number")
        for key in ("prompts", "model_roles"):
            _keys(profile.get(key, {}), GROUPS[key], f"{path}.{key}")
        for name, text in profile.get("prompts", {}).items():
            if not isinstance(text, str) or not text.strip():
                raise ConfigError(f"{path}.prompts.{name}: expected text")
            if name == "search":
                validate_search_prompt(text)
    for group, allowed in GROUPS.items():
        _keys(data.get(group, {}), allowed, group)
    for name in ("models", "connections", "credentials"):
        if not isinstance(data.get(name, {}), dict):
            raise ConfigError(f"{name}: expected a table")
    for name, model in data.get("models", {}).items():
        if name not in MODELS_SCALARS:
            legacy = {"endpoint", "alternatives"} if data.get("schema_version", 0) == 0 else set()
            _keys(model, MODEL_KEYS | {"connection"} | legacy, f"models.{name}")
    for name, conn in data.get("connections", {}).items():
        _keys(conn, CONNECTION_KEYS, f"connections.{name}")
    for name, ref in data.get("credentials", {}).items():
        _keys(ref, {"source", "key", "path"}, f"credentials.{name}")
        if ref.get("source") not in {"env", "file", "managed"}:
            raise ConfigError(f"credentials.{name}.source: expected env, file or managed")
        from .config import ENV_NAME

        if (
            not ENV_NAME.fullmatch(name)
            or not isinstance(ref.get("key", name), str)
            or not ENV_NAME.fullmatch(ref.get("key", name))
        ):
            raise ConfigError(f"credentials.{name}.key: expected variable name")
        if (ref["source"] == "file" or "path" in ref) and (
            not isinstance(ref.get("path"), str) or not ref["path"]
        ):
            raise ConfigError(f"credentials.{name}.path: required")


def atomic_write(path: Path, text: str, *, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    tmp = Path(name)
    try:
        if private:
            from .credentials import restrict_file

            restrict_file(tmp)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            fd = -1
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if fd != -1:
            os.close(fd)
        tmp.unlink(missing_ok=True)


@contextmanager
def write_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+b") as file:
        file.seek(0)
        if os.name == "nt":
            import msvcrt

            if not file.read(1):
                file.write(b"0")
                file.flush()
            file.seek(0)
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ConfigError("configuration: another writer is active") from None
        else:
            import fcntl

            fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


@dataclass(frozen=True)
class ConfigBundle:
    data: Mapping
    settings: object
    revision: str
    path: Path


def load_bundle(path: Path, overrides=None) -> ConfigBundle:
    path = Path(path).resolve()
    service = None
    if path.is_dir():
        service = ConfigService(path)
        published = service.status()
        path = service.version_path(published["revision"])
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ConfigError("settings: cannot read valid TOML") from None
    bundle = build_bundle(data, path)
    if service:
        if bundle.revision != published["digest"]:
            raise ConfigError("revision: active files changed outside configuration service")
    if overrides:
        bundle = build_bundle(data, path, overrides)
    if service:
        settings = replace(
            bundle.settings, config_store=service.root, published_revision=published["revision"]
        )
        bundle = replace(bundle, settings=settings)
    return bundle


def build_bundle(data: dict, path: Path, overrides=None) -> ConfigBundle:
    data = plain(data)
    version = data.get("schema_version", 0)
    if type(version) is not int or version not in (0, 1):
        raise ConfigError("schema_version: supported versions are 0 and 1")
    validate_keys(data)
    root_value = data.get("instance_root", ".")
    if not isinstance(root_value, str) or not root_value:
        raise ConfigError("instance_root: expected path")
    root = (path.parent / root_value).resolve()
    supplied = set(flatten(data))
    if version == 0 and "secrets_file" not in data:
        data["secrets_file"] = str(path.parent.parent.parent / "secrets" / "group-agent.env")
    for key, value in (overrides or {}).items():
        put(data, key, value)
    validate_keys(data)
    for key in PATH_FIELDS:
        value = flatten(data).get(key)
        if value is None and version == 1:
            value = PATH_DEFAULTS.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ConfigError(f"{key}: expected path")
            put(data, key, str((root / value).resolve()))
    for ref in data.get("credentials", {}).values():
        if ref["source"] in {"file", "managed"}:
            ref["path"] = str((root / ref.get("path", "private/credentials.env")).resolve())
    resolved = plain(data)
    for name, model in resolved.get("models", {}).items():
        if name in MODELS_SCALARS or "connection" not in model:
            continue
        connection = model.pop("connection")
        if not isinstance(connection, str) or connection not in resolved.get("connections", {}):
            raise ConfigError(f"models.{name}.connection: unknown connection")
        for key, value in resolved["connections"][connection].items():
            if key in model and model[key] != value:
                raise ConfigError(f"models.{name}.{key}: conflicts with connection")
            model[key] = value
    texts = {}
    for name, value in data.get("prompts", {}).items():
        if not isinstance(value, str) or not value:
            raise ConfigError(f"prompts.{name}: expected resource path")
        resource = (root / value).resolve()
        try:
            texts[name] = resource.read_text(encoding="utf-8")
        except OSError:
            raise ConfigError(f"prompts.{name}: cannot read resource") from None
        if not texts[name].strip():
            raise ConfigError(f"prompts.{name}: empty text")
        data["prompts"][name] = str(resource)
    if "search" in texts:
        validate_search_prompt(texts["search"])
    from .config import _model_roles

    for profile in resolved.get("group_profiles", {}).values():
        _model_roles(
            {"models": resolved.get("models", {}), "model_roles": profile.get("model_roles", {})}
        )
    resolved["_prompt_text"] = texts
    settings = parse_settings(resolved, root / "settings.toml")
    # Preserve legacy sibling-layout lookup; new installations opt in with an explicit path.
    if settings.snowluma_db is None:
        if version == 0:
            settings = replace(
                settings,
                snowluma_db=path.parent.parent.parent
                / "SnowLuma"
                / "data"
                / settings.bot_id
                / "messages.db",
            )
        elif settings.self_sync:
            if data.get("serve", {}).get("self_sync") is True:
                raise ConfigError("serve.snowluma_db: required when self_sync is enabled")
            settings = replace(settings, self_sync=False)
    if version == 1 and settings.limits.finish_reserve_seconds >= settings.deadline_seconds:
        raise ConfigError("runtime.finish_reserve_seconds: must be less than deadline_seconds")
    from .persona import prompt_options
    from .gemini_search import SEARCH_PROMPT

    effective_data = plain(settings)
    if not settings.group_profiles:
        effective_data.pop("group_profiles")  # preserve pre-profile published digests
    digest = hashlib.sha256(
        json.dumps(
            {
                "data": data,
                "effective": effective_data,
                "prompts": dict(prompt_options(settings)),
                "search_prompt": texts.get("search", SEARCH_PROMPT),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    sources = {
        k: "cli" if k in (overrides or {}) else "file" for k in supplied | set(overrides or {})
    }
    settings = replace(settings, revision=digest, sources=sources)
    return ConfigBundle(freeze(data), settings, digest, path)


def effective(bundle):
    """Local inspection, with private values removed before serialization."""
    values = plain(bundle.settings)
    for key in (
        "bot_id",
        "owner_id",
        "blocked_users",
        "blocked_groups",
        "secrets_file",
        "credential_refs",
        "sources",
    ):
        values.pop(key, None)
    for model in values["models"].values():
        for key in ("base_url", "base_url_env", "api_key_env", "project_env"):
            model.pop(key, None)
    values["web"].pop("base_url", None)
    values["collection"].pop("disabled_groups", None)
    for key in ("serve_db", "shadow_db", "knowledge_db", "snowluma_db"):
        values.pop(key, None)
    values.pop("config_store", None)
    values["billing"].pop("db_path", None)
    values["collection"].pop("media_dir", None)
    return {
        "revision": bundle.revision,
        "apply": "restart",
        "values": values,
        "sources": dict(bundle.settings.sources),
        "unspecified_source": "default",
    }


def field_application(group, key):
    if group == "identity":
        return "maintenance" if key == "bot_id" else "immediate"
    if group in {"models", "models.*", "connections.*", "model_roles", "prompts", "web"}:
        return "new_run"
    if group == "budgets":
        return "immediate" if key == "daily_cost_cny" else "new_run"
    if group == "agent":
        return "immediate" if key == "web_enabled" else "new_run"
    if group == "runtime":
        from .runtime import IMMEDIATE_RUNTIME

        return "immediate" if key in IMMEDIATE_RUNTIME else "new_run"
    if group == "retention":
        return "next_cleanup"
    if group == "collection" and key == "disabled_groups":
        return "immediate"
    if group == "serve" and key == "fallback_reply":
        return "new_run"
    if group == "serve" and key == "live_send":
        return "disable_now_enable_maintenance"
    if group == "billing" and key in {"usd_cny", "usd_cny_date", "google_search_usd_per_1000"}:
        return "new_run"
    return "maintenance"


def schema():
    """Field inventory from the same dataclasses and accepted input keys."""
    defaults = parse_settings({}, Path.cwd() / "config" / "settings.toml")
    default_groups = {
        "identity": {k: getattr(defaults, k) for k in GROUPS["identity"]},
        "budgets": {k: getattr(defaults, k) for k in GROUPS["budgets"]},
        "runtime": plain(defaults.runtime),
        "retention": plain(defaults.retention),
        "collection": plain(defaults.collection),
        "billing": plain(defaults.billing),
        "web": plain(defaults.web),
        "agent": {k: getattr(defaults, k) for k in GROUPS["agent"]},
        "serve": {k: getattr(defaults, {"db_path": "serve_db"}.get(k, k)) for k in GROUPS["serve"]},
        "shadow": {"db_path": defaults.shadow_db},
        "knowledge": {"db_path": defaults.knowledge_db},
    }
    types = {
        "runtime": RuntimeLimits,
        "retention": Retention,
        "collection": Collection,
        "billing": BillingSpec,
        "web": WebSpec,
    }
    result = {}
    for group, keys in GROUPS.items():
        definitions = types.get(group, Settings).__dataclass_fields__
        result[group] = {}
        for key in sorted(keys):
            attr = (
                {"serve": "serve_db", "shadow": "shadow_db", "knowledge": "knowledge_db"}.get(group)
                if key == "db_path"
                else key
            )
            definition = definitions.get(attr or key)
            typ = definition.type if definition else str
            default = default_groups.get(group, {}).get(key)
            result[group][key] = {
                "default": PATH_DEFAULTS.get(f"{group}.{key}", plain(default)),
                "type": getattr(typ, "__name__", str(typ)),
                "apply": field_application(group, key),
                "source": "default < file < cli",
                "sensitive": key
                in {
                    "bot_id",
                    "owner_id",
                    "blocked_users",
                    "blocked_groups",
                    "disabled_groups",
                    "base_url",
                    "db_path",
                    "snowluma_db",
                    "media_dir",
                },
            }
    result["models.*"] = {
        key: {"type": str(ModelSpec.__dataclass_fields__[key].type), "apply": "new_run"}
        for key in sorted(MODEL_KEYS)
    }
    result["connections.*"] = {key: result["models.*"][key] for key in sorted(CONNECTION_KEYS)}
    result["models.*"]["connection"] = {"type": "str", "apply": "new_run"}
    result["credentials.*"] = {
        "source": ["env", "file", "managed"],
        "key": "variable name",
        "path": "file or managed store path",
    }
    return result


def materialize(bundle):
    data, settings = plain(bundle.data), bundle.settings
    paths = {
        "secrets_file": settings.secrets_file,
        "serve.db_path": settings.serve_db,
        "shadow.db_path": settings.shadow_db,
        "knowledge.db_path": settings.knowledge_db,
        "billing.db_path": settings.billing.db_path,
        "serve.snowluma_db": settings.snowluma_db,
        "collection.media_dir": settings.collection.media_dir,
    }
    for key, value in paths.items():
        if value is not None:
            put(data, key, str(value))
    return data


def migrate(bundle, output=None):
    data = materialize(bundle)
    data["schema_version"] = 1
    data["instance_root"] = str(bundle.path.parent)
    for name, model in data.get("models", {}).items():
        if name not in MODELS_SCALARS:
            model.pop("endpoint", None)
            model.pop("alternatives", None)
    target = Path(output).resolve() if output else bundle.path
    migrated = build_bundle(data, target)
    if output:
        if target.exists():
            raise ConfigError("output: already exists")
        atomic_write(target, tomli_w.dumps(data))
    return {"written": bool(output), "configuration": effective(migrated)}


class ConfigService:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def save_draft(self, data, expected=None):
        """Validated edits for a future UI; a draft never changes the active pointer."""
        target = self.root / "draft.toml"
        with write_lock(self.root):
            current = load_bundle(target).revision if target.exists() else None
            if current != expected:
                raise ConfigError("draft: changed; reload before saving")
            bundle = build_bundle(data, target)
            atomic_write(target, tomli_w.dumps(plain(data)))
            return bundle

    def save_editor(self, data, prompts, expected=None):
        """Keep draft prompt files immutable so a concurrent editor cannot change them."""
        with write_lock(self.root):
            target = self.root / "draft.toml"
            current = load_bundle(target).revision if target.exists() else None
            if current != expected:
                raise ConfigError("draft: changed; reload before saving")
            data = plain(data)
            resource_dir = self.root / "draft-prompts" / uuid.uuid4().hex
            for name, text in prompts.items():
                if name not in GROUPS["prompts"] or not isinstance(text, str) or not text.strip():
                    raise ConfigError("prompts: expected a known name and nonempty text")
                resource = resource_dir / f"{name}.md"
                atomic_write(resource, text)
                data.setdefault("prompts", {})[name] = str(resource)
            bundle = build_bundle(data, target)
            atomic_write(target, tomli_w.dumps(data))
            return bundle

    def status(self):
        path = self.root / "active.json"
        if not path.exists():
            return {"revision": None}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ConfigError("configuration: invalid active pointer") from None

    def active_path(self):
        return self.version_path(self.status()["revision"])

    def record_runtime(self, settings, state):
        import time

        record = {
            "state": state,
            "pid": os.getpid(),
            "at": time.time(),
            "published_revision": settings.published_revision,
            "effective_revision": settings.revision,
            "overrides": sorted(k for k, v in settings.sources.items() if v == "cli"),
        }
        atomic_write(self.root / "runtime.json", json.dumps(record))

    def report(self):
        result = self.status()
        path = self.root / "runtime.json"
        result["last_runtime"] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )
        result["pending"] = (
            not result["last_runtime"]
            or result["last_runtime"]["state"] == "failed"
            or result["last_runtime"]["published_revision"] != result["revision"]
        )
        return result

    def version_path(self, revision):
        if (
            not isinstance(revision, str)
            or len(revision) != 32
            or any(c not in "0123456789abcdef" for c in revision)
        ):
            raise ConfigError("revision: invalid identifier")
        return self.root / "versions" / revision / "settings.toml"

    def publish(self, source: Path, expected: str | None, *, rollback_of=None):
        bundle = load_bundle(source)
        with write_lock(self.root):
            current = self.status()
            if current["revision"] != expected:
                raise ConfigError("revision: changed; reload before publishing")
            if expected and load_bundle(self.active_path()).revision != current["digest"]:
                raise ConfigError("revision: active files changed outside configuration service")
            revision = uuid.uuid4().hex
            target = self.version_path(revision)
            target.parent.mkdir(parents=True)
            data = materialize(bundle)
            settings = bundle.settings
            data["instance_root"] = str(target.parent)
            for name, text in settings.prompts.items():
                resource = target.parent / "prompts" / f"{name}.md"
                atomic_write(resource, text)
                data["prompts"][name] = str(resource)
            atomic_write(target, tomli_w.dumps(data))
            validated = load_bundle(target)
            record = {
                "revision": revision,
                "digest": validated.revision,
                "previous": expected,
                "rollback_of": rollback_of,
                "apply": "pending",
            }
            atomic_write(target.parent / "revision.json", json.dumps(record))
            atomic_write(self.root / "active.json", json.dumps(record))
            return record

    def rollback(self, revision, expected):
        source = self.version_path(revision)
        # A rollback revalidates references but never writes the credential store or ledger.
        load_bundle(source).settings.credentials.check_references()
        return self.publish(source, expected, rollback_of=revision)

    def diff(self, source):
        before = (
            flatten(plain(load_bundle(self.active_path()).settings))
            if self.status()["revision"]
            else {}
        )
        after = flatten(plain(load_bundle(source).settings))
        result = {}
        for key in sorted(before.keys() | after.keys()):
            if key.split(".")[0] in {
                "revision",
                "published_revision",
                "config_store",
                "sources",
            } or before.get(key) == after.get(key):
                continue
            private = any(
                part in key
                for part in (
                    "bot_id",
                    "owner_id",
                    "blocked_",
                    "disabled_groups",
                    "_db",
                    "db_path",
                    "media_dir",
                    "secrets",
                    "credential",
                    "base_url",
                    "project_env",
                    "api_key",
                )
            )
            result[key] = {
                side: ("[redacted]" if private and values.get(key) is not None else values.get(key))
                for side, values in (("before", before), ("after", after))
            }
        return result
