"""Single-administrator random key and bounded in-memory sessions."""

from collections import deque
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import time

from ..config import ConfigError
from ..config_service import atomic_write


def reset_key(instance):
    key = secrets.token_urlsafe(32)
    path = Path(instance) / "private" / "admin.json"
    atomic_write(
        path, json.dumps({"digest": hashlib.sha256(key.encode()).hexdigest()}), private=True
    )
    return key


class Auth:
    def __init__(self, instance):
        self.path = Path(instance) / "private" / "admin.json"
        self.digest = self.read_digest()
        self.sessions = {}
        self.attempts = deque(maxlen=10)

    def read_digest(self):
        if not self.path.is_file():
            raise ConfigError("admin: run admin init or admin reset-key first")
        return json.loads(self.path.read_text("utf-8"))["digest"]

    def refresh(self):
        digest = self.read_digest()
        if digest != self.digest:
            self.digest = digest
            self.sessions.clear()
        now = time.time()
        self.sessions = {k: v for k, v in self.sessions.items() if v > now}

    def login(self, key):
        self.refresh()
        now = time.time()
        if len(self.attempts) == 10 and now - self.attempts[0] < 60:
            return None, 429
        self.attempts.append(now)
        if not isinstance(key, str) or not hmac.compare_digest(
            hashlib.sha256(key.encode()).hexdigest(), self.digest
        ):
            return None, 401
        token = secrets.token_urlsafe(32)
        if len(self.sessions) >= 32:
            self.sessions.pop(next(iter(self.sessions)))
        self.sessions[token] = now + 12 * 3600
        return token, 200

    def valid(self, token):
        self.refresh()
        return token in self.sessions
