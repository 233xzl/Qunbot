"""Read-only image acquisition for local replay cases (stdlib only).

Images are resolved through SnowLuma's OneBot HTTP API (`get_image` returns a
refreshed download URL; stored segment URLs expire). Only an allowlist of read
actions can be called. Downloaded bytes are stored once per content hash under
the git-ignored raw directory. URLs, tokens, file names and source locators are
never written to outputs or logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Read actions only: anything that sends, sets, deletes or changes state is refused.
ALLOWED_ACTIONS = frozenset({"get_image", "get_status", "get_msg"})
MAX_IMAGE_BYTES = 10 * 1024 * 1024
INDEX_VERSION = "media-index-1"
# Failures worth retrying on a later run are not cached.
PERMANENT_REASONS = frozenset({"oversize", "not_image", "empty_content", "missing_file_key"})
EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
}


class OneBotError(RuntimeError):
    """Action failure; message never contains payloads, URLs or tokens."""


class OversizeError(ValueError):
    pass


class RateLimiter:
    """Minimum spacing between calls (<= per_second requests per second)."""

    def __init__(
        self,
        per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if per_second <= 0:
            raise ValueError("rate must be positive")
        self.interval = 1.0 / per_second
        self.clock = clock
        self.sleep = sleep
        self.next_at: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self.next_at is not None and now < self.next_at:
            self.sleep(self.next_at - now)
            now = self.next_at
        self.next_at = now + self.interval


class OneBotReadClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        opener: Callable[..., Any] | None = None,
        limiter: RateLimiter | None = None,
        timeout: float = 20,
    ):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OneBot base URL must be http(s)")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.opener = opener or urllib.request.urlopen
        self.limiter = limiter or RateLimiter(5)
        self.timeout = timeout

    @classmethod
    def from_config(cls, path: Path, **kwargs: Any) -> OneBotReadClient:
        """Read host/port/token from SnowLuma's onebot config (never echoed)."""
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        servers = config.get("networks", {}).get("httpServers", [])
        if not servers:
            raise ValueError("no OneBot HTTP server configured")
        server = servers[0]
        host = server.get("host") or "127.0.0.1"
        base = f"http://{host}:{int(server['port'])}{server.get('path') or '/'}"
        return cls(base, server.get("accessToken") or None, **kwargs)

    def call(self, action: str, payload: dict) -> dict:
        if action not in ALLOWED_ACTIONS:
            raise ValueError("refusing non-read OneBot action")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(
            f"{self.base_url}/{action}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        self.limiter.wait()
        try:
            with self.opener(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise OneBotError(f"{action} transport failure ({type(exc).__name__})") from None
        if not isinstance(body, dict) or body.get("status") != "ok" or body.get("retcode") != 0:
            raise OneBotError(f"{action} returned failure")
        data = body.get("data")
        return data if isinstance(data, dict) else {}


def default_download(
    url: str,
    max_bytes: int,
    *,
    opener: Callable[..., Any] | None = None,
    timeout: float = 30,
) -> bytes:
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("unsupported download scheme")
    opener = opener or urllib.request.urlopen
    with opener(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
        length = response.headers.get("Content-Length") if response.headers else None
        if length and str(length).isdigit() and int(length) > max_bytes:
            raise OversizeError("declared length over cap")
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise OversizeError("content over cap")
    return content


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    offset = 2
    while offset + 9 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            offset += 1 if marker == 0xFF else 2
            continue
        length = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE}:
            height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
            return width, height
        if length < 2:
            return None
        offset += 2 + length
    return None


def sniff_image(data: bytes) -> tuple[str, int | None, int | None] | None:
    """Return (mime, width, height) from magic bytes; None if not a known image."""
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return "image/png", width, height
        if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            return "image/gif", width, height
        if data.startswith(b"\xff\xd8"):
            size = _jpeg_size(data)
            return ("image/jpeg", *size) if size else ("image/jpeg", None, None)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            chunk = data[12:16]
            if chunk == b"VP8X" and len(data) >= 30:
                width = int.from_bytes(data[24:27], "little") + 1
                height = int.from_bytes(data[27:30], "little") + 1
                return "image/webp", width, height
            if chunk == b"VP8L" and len(data) >= 25:
                bits = int.from_bytes(data[21:25], "little")
                return "image/webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
            if chunk == b"VP8 " and len(data) >= 30:
                width, height = struct.unpack("<HH", data[26:30])
                return "image/webp", width & 0x3FFF, height & 0x3FFF
            return "image/webp", None, None
        if data.startswith(b"BM") and len(data) >= 26:
            width, height = struct.unpack("<ii", data[18:26])
            return "image/bmp", abs(width), abs(height)
    except struct.error:
        return None
    return None


def unavailable(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "sha256": None,
        "mime": None,
        "bytes": None,
        "width": None,
        "height": None,
        "path": None,
    }


class MediaFetcher:
    """Resolve image file keys to content-addressed local files with a rerun cache.

    The cache is keyed by the HMAC media_id (pseudonym of the file key), so it never
    stores raw locators. `path` values are relative to the raw replay root.
    """

    def __init__(
        self,
        client: OneBotReadClient | None,
        media_root: Path,
        *,
        max_bytes: int = MAX_IMAGE_BYTES,
        download: Callable[[str, int], bytes] | None = None,
        download_limiter: RateLimiter | None = None,
    ):
        self.client = client
        self.media_root = Path(media_root)
        self.max_bytes = max_bytes
        self.download = download or default_download
        self.download_limiter = download_limiter or RateLimiter(5)
        self.index_path = self.media_root / "index.json"
        self.index: dict[str, dict] = {}
        self.stats: Counter = Counter()
        if self.index_path.exists():
            stored = json.loads(self.index_path.read_text(encoding="utf-8"))
            if stored.get("version") == INDEX_VERSION:
                self.index = dict(stored.get("entries", {}))

    def _cached(self, media_id: str) -> dict | None:
        entry = self.index.get(media_id)
        if not entry:
            return None
        if entry["status"] == "available":
            if (self.media_root.parent / entry["path"]).is_file():
                return dict(entry)
            return None
        return dict(entry) if entry["reason"] in PERMANENT_REASONS else None

    def _remember(self, media_id: str, ref: dict) -> dict:
        if ref["status"] == "available" or ref["reason"] in PERMANENT_REASONS:
            self.index[media_id] = dict(ref)
        self.stats["result_" + ref["reason"]] += 1
        return ref

    def fetch(self, media_id: str, file_key: Any) -> dict:
        cached = self._cached(media_id)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        if not isinstance(file_key, str) or not file_key:
            return self._remember(media_id, unavailable("missing_file_key"))
        if self.client is None:
            return self._remember(media_id, unavailable("fetch_disabled"))
        try:
            self.stats["get_image_calls"] += 1
            info = self.client.call("get_image", {"file": file_key})
        except OneBotError:
            return self._remember(media_id, unavailable("get_image_failed"))
        url = info.get("url")
        if not isinstance(url, str) or not url:
            return self._remember(media_id, unavailable("no_download_url"))
        declared = str(info.get("file_size") or "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            return self._remember(media_id, unavailable("oversize"))
        try:
            self.download_limiter.wait()
            self.stats["downloads"] += 1
            content = self.download(url, self.max_bytes)
        except OversizeError:
            return self._remember(media_id, unavailable("oversize"))
        except urllib.error.HTTPError:
            return self._remember(media_id, unavailable("download_http_error"))
        except (urllib.error.URLError, OSError, ValueError):
            return self._remember(media_id, unavailable("download_failed"))
        if not content:
            return self._remember(media_id, unavailable("empty_content"))
        if len(content) > self.max_bytes:
            return self._remember(media_id, unavailable("oversize"))
        sniffed = sniff_image(content)
        if sniffed is None:
            return self._remember(media_id, unavailable("not_image"))
        mime, width, height = sniffed
        digest = hashlib.sha256(content).hexdigest()
        relative = f"{self.media_root.name}/{digest}.{EXTENSIONS[mime]}"
        target = self.media_root / f"{digest}.{EXTENSIONS[mime]}"
        if target.is_file():
            self.stats["dedup_hits"] += 1
        else:
            self.media_root.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(content)
            os.replace(partial, target)
            self.stats["files_written"] += 1
        return self._remember(
            media_id,
            {
                "status": "available",
                "reason": "fetched_via_onebot_get_image",
                "sha256": digest,
                "mime": mime,
                "bytes": len(content),
                "width": width,
                "height": height,
                "path": relative,
            },
        )

    def save_index(self) -> None:
        self.media_root.mkdir(parents=True, exist_ok=True)
        partial = self.index_path.with_suffix(".json.part")
        partial.write_text(
            json.dumps(
                {"version": INDEX_VERSION, "entries": dict(sorted(self.index.items()))},
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(partial, self.index_path)
