"""Lazy credential references; ordinary configuration never stores credential values."""

from dataclasses import dataclass, field
import csv
import io
import os
from pathlib import Path
import subprocess

from .config import ConfigError, ENV_NAME


def restrict_file(path):
    if os.name == "nt":
        result = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True
        )
        sid = next(csv.reader(io.StringIO(result.stdout.strip())))[1]
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:F"],
            capture_output=True,
            check=True,
        )
    else:
        path.chmod(0o600)


@dataclass(frozen=True)
class CredentialStore:
    fallback: Path = field(repr=False)
    references: object = field(default_factory=dict, repr=False)

    def read(self):
        from .config import _read_env

        values = _read_env(self.fallback) if self.fallback.is_file() else {}
        for name, ref in self.references.items():
            key = ref.get("key", name)
            if ref["source"] == "env":
                value = os.environ.get(key)
            else:
                path = Path(ref["path"])
                value = _read_env(path).get(key) if path.is_file() else None
            values.pop(name, None)
            if value:
                values[name] = value
        return values

    def check_references(self):
        values = self.read()
        for name in self.references:
            if not values.get(name):
                raise ConfigError(f"credentials.{name}: unavailable")

    def status(self):
        values = self.read()
        return {
            name: {
                "source": self.references.get(name, {}).get("source", "file"),
                "configured": bool(values.get(name)),
            }
            for name in sorted(values.keys() | self.references.keys())
        }

    def update(self, name, action, value=None):
        from .config import _read_env
        from .config_service import atomic_write, write_lock

        ref = self.references.get(name)
        if ref is None or ref["source"] != "managed":
            raise ConfigError("credentials: only managed references are writable")
        if action == "keep":
            return
        if action not in {"replace", "clear"}:
            raise ConfigError("credentials: expected keep, replace or clear")
        if action == "replace" and (
            not isinstance(value, str)
            or not value
            or any(c in value for c in "\r\n\x00")
            or value != value.strip()
            or value.startswith(('"', "'"))
            or value.endswith(('"', "'"))
        ):
            raise ConfigError("credentials: expected a nonempty single-line value")
        path, key = Path(ref["path"]), ref.get("key", name)
        if not ENV_NAME.fullmatch(key):
            raise ConfigError("credentials: invalid key")
        with write_lock(path.parent):
            values = _read_env(path) if path.is_file() else {}
            if action == "clear":
                values.pop(key, None)
            else:
                values[key] = value
            atomic_write(
                path, "".join(f"{k}={v}\n" for k, v in sorted(values.items())), private=True
            )
