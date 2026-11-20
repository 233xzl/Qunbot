"""Immediate image collection for stored group messages (T3.3, CTX-06).

For every image segment of a newly stored message, `MediaCollector.schedule` registers a
pending media slot and starts a background task, so the event loop that receives
messages never waits for a download. Each task:

1. asks SnowLuma for the image with the read-only `get_image` action (a refreshed URL or
   a local cached file), falling back to the URL carried by the segment itself;
2. reads a local file (absolute path, regular file) or downloads the URL under
   `ImageSourcePolicy`: http/https only, no credentials, default port, the host must be a
   QQ image domain (suffix allowlist), and it must resolve only to public addresses; at
   most two redirects, each re-checked. This mirrors WEB-04 and keeps the downloader from
   being steered at local or internal services (SSRF);
3. caps the size (10 MB; a declared size over the cap is never fetched and only the
   metadata is kept), checks the bytes are a known image type (mime, width, height),
   and stores the file once per SHA-256 under the media directory;
4. links the stored file to the slot. A recall or cleanup that removed the slot first
   wins: the file is then garbage-collected if nothing else references it.

Failures mark the slot `unavailable` with a short reason. Rate (token bucket), concurrent
tasks and pending tasks are bounded; over the pending bound a slot is marked `queue_full`.
URLs, file keys and file names are never logged.
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import struct
import time
from typing import Any
import httpx
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

from .contracts import Message
from .gateway import GatewayError, TokenBucket
from .store import Store

MAX_IMAGE_BYTES = 10 * 1024 * 1024
EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
}
# Hosts SnowLuma builds image URLs for (multimedia.nt.qq.com.cn, gchat.qpic.cn, ...).
DEFAULT_IMAGE_HOST_SUFFIXES = ("qpic.cn", "nt.qq.com.cn")
MEDIA_FILE = re.compile(r"[0-9a-f]{64}\.(?:jpg|png|gif|webp|bmp)(?:\.[0-9a-f]{8}\.part)?")


class MediaUnavailable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# -- image sniffing (same rules as scripts/replay/replay_media.py) ----------------------


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


# -- source policy --------------------------------------------------------------------

Resolver = Callable[[str, int], Awaitable[list[str]]]


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def _is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class ImageSourcePolicy:
    """Where image bytes may come from: QQ image hosts over the network, or a local file."""

    def __init__(
        self,
        host_suffixes: tuple[str, ...] = DEFAULT_IMAGE_HOST_SUFFIXES,
        resolver: Resolver = system_resolver,
    ):
        self.host_suffixes = tuple(s.lower().strip(".") for s in host_suffixes if s.strip("."))
        self.resolver = resolver

    def host_allowed(self, host: str) -> bool:
        host = host.lower().rstrip(".")
        return any(host == s or host.endswith("." + s) for s in self.host_suffixes)

    async def check_url(self, url: Any) -> None:
        if not isinstance(url, str) or not 0 < len(url) <= 8192 or any(c.isspace() for c in url):
            raise MediaUnavailable("unsafe_url")
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            raise MediaUnavailable("unsafe_url") from None
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise MediaUnavailable("unsafe_url")
        if parts.username is not None or parts.password is not None:
            raise MediaUnavailable("unsafe_url")
        default_port = 443 if parts.scheme == "https" else 80
        if port not in (None, default_port):
            raise MediaUnavailable("unsafe_url")
        host = parts.hostname
        try:
            ipaddress.ip_address(host.strip("[]"))
            raise MediaUnavailable("unsafe_url")  # IP literals are never QQ image hosts
        except ValueError:
            pass
        if not self.host_allowed(host):
            raise MediaUnavailable("host_not_allowed")
        try:
            addresses = await self.resolver(host, default_port)
        except OSError:
            raise MediaUnavailable("dns_failed") from None
        if not addresses or not all(_is_public_ip(a) for a in addresses):
            raise MediaUnavailable("unsafe_url")

    @staticmethod
    def check_local(path: Any) -> Path:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise MediaUnavailable("no_source")
        candidate = Path(path)
        if not candidate.is_absolute():
            raise MediaUnavailable("no_source")  # a bare cache key, not a file path
        try:
            if candidate.is_symlink() or not candidate.is_file():
                raise MediaUnavailable("no_source")
        except OSError:
            raise MediaUnavailable("no_source") from None
        return candidate


Downloader = Callable[[str, int, float], Awaitable[tuple[int, bytes, str | None]]]


async def httpx_download(url: str, max_bytes: int, timeout: float) -> tuple[int, bytes, str | None]:
    """One GET without following redirects. Returns (status, body, redirect location)."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            location = response.headers.get("location") if response.is_redirect else None
            if location:
                return response.status_code, b"", location
            length = response.headers.get("content-length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise MediaUnavailable("oversize")
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise MediaUnavailable("oversize")
                chunks.append(chunk)
            return response.status_code, b"".join(chunks), None


def _read_capped(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise MediaUnavailable("oversize")
    return content


def _write_atomic(target: Path, content: bytes) -> bool:
    if target.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    # Unique partial name: concurrent downloads of the same content must not collide.
    partial = target.with_name(f"{target.name}.{uuid4().hex[:8]}.part")
    try:
        partial.write_bytes(content)
        try:
            os.replace(partial, target)
        except OSError:
            if not target.is_file():  # Windows: a concurrent replace may hold the target
                raise
            return False
    finally:
        partial.unlink(missing_ok=True)
    return True


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


class MediaCollector:
    def __init__(
        self,
        store: Store,
        api: Any | None,
        root: Path,
        *,
        policy: ImageSourcePolicy | None = None,
        max_bytes: int = MAX_IMAGE_BYTES,
        rate_per_second: float = 2.0,
        burst: int = 2,
        max_concurrency: int = 2,
        max_pending: int = 200,
        timeout: float = 30.0,
        max_redirects: int = 2,
        download: Downloader = httpx_download,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = lambda _: None,
    ):
        self.store, self.api, self.root = store, api, Path(root)
        self.policy = policy or ImageSourcePolicy()
        self.max_bytes, self.timeout, self.max_redirects = max_bytes, timeout, max_redirects
        self.max_pending = max_pending
        self.download, self.clock, self.log = download, clock, log
        self._bucket = TokenBucket(rate_per_second, burst)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task] = set()
        self._by_key: OrderedDict[tuple, str] = OrderedDict()  # bot/group/file -> sha256
        self._flights: dict[tuple, asyncio.Task] = {}
        self._slots: dict[tuple[int, int], asyncio.Task] = {}
        self._cycles: OrderedDict[tuple, dict] = OrderedDict()
        self.counts: Counter[str] = Counter()

    # -- scheduling -----------------------------------------------------------------
    def schedule(self, seq: int, message: Message) -> int:
        """Queue downloads for the image segments of a stored message; returns the count."""
        images = [
            (index, segment["data"])
            for index, segment in enumerate(message.segments)
            if segment.get("type") == "image" and isinstance(segment.get("data"), dict)
        ]
        if not images:
            return 0
        now = self.clock()
        self.store.add_media_slots(
            seq, [(index, _int(data.get("file_size"))) for index, data in images], now
        )
        for index, data in images:
            self.counts["images"] += 1
            if len(self._tasks) >= self.max_pending:
                self._finish_unavailable(seq, index, "queue_full")
                continue
            self._schedule_slot(seq, index, dict(data))
        return len(images)

    def _schedule_slot(self, seq: int, index: int, data: dict, deadline=None) -> None:
        slot = (seq, index)
        if slot in self._slots:
            return
        task = asyncio.create_task(self._collect(seq, index, data, deadline))
        self._tasks.add(task)
        self._slots[slot] = task

        def done(task):
            self._tasks.discard(task)
            self._slots.pop(slot, None)
            if not self._tasks:
                self.gc()

        task.add_done_callback(done)

    def _resource_key(self, seq: int, index: int, data: dict) -> tuple:
        row = self.store.message_row(seq)
        # URL-only images stay independent; file keys are scoped to the receiving group.
        return (row["bot_id"], row["group_id"], data.get("file") or (seq, index))

    @staticmethod
    def _recoverable(reason: str | None) -> bool:
        return reason in {
            "download_timeout",
            "download_connection",
            "dns_failed",
            "download_failed",
            "http_403",
            "http_404",
            "http_429",
            "no_source",
            "read_failed",
            "file_missing",
            "interrupted",
            "queue_full",
        } or bool(reason and reason.startswith("http_5"))

    async def prepare(self, selected: Callable[[], list[int]], deadline: float) -> None:
        """One recovery cycle for relevant slots; cancellation only stops this waiter.

        Initial ingestion owns its downloads. Recovery is bounded by this preparation
        deadline; other runs can join it without resetting either deadline or attempts.
        """
        loop = asyncio.get_running_loop()
        considered: set[tuple[int, int]] = set()
        while loop.time() < deadline:
            waiting = []
            for seq in selected():
                row = self.store.message_row(seq)
                if row is None or row["revoked"]:
                    continue
                segments = json.loads(row["segments"])
                for slot in self.store.message_media(seq):
                    index = slot["segment_index"]
                    identity = (seq, index)
                    task = self._slots.get(identity)
                    if task is not None:
                        waiting.append(task)
                        continue
                    if identity in considered:
                        continue
                    reason = slot["reason"]
                    if slot["status"] == "available":
                        if slot["path"] and (self.root / slot["path"]).is_file():
                            continue
                        reason = "file_missing"
                    if not self._recoverable(reason) or index >= len(segments):
                        continue
                    data = dict(segments[index].get("data") or {})
                    key = self._resource_key(seq, index, data)
                    cycle = self._cycles.get(key)
                    if cycle and cycle["attempts"] >= 2 and self.clock() < cycle["retry_after"]:
                        considered.add(identity)
                        continue
                    if cycle and cycle["attempts"] >= 2:
                        self._cycles.pop(key)
                    considered.add(identity)
                    if len(self._tasks) >= self.max_pending:
                        continue
                    if not self.store.retry_media(seq, index, self.clock()):
                        continue
                    self._schedule_slot(seq, index, data, deadline)
                    waiting.append(self._slots[identity])
                    self.counts["recovery_requested"] += 1
            if not waiting:
                return
            # Poll permissions/recalls too; asyncio.wait never cancels shared downloads.
            await asyncio.wait(waiting, timeout=min(0.05, max(0, deadline - loop.time())))

    @property
    def pending(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout: float | None = None) -> None:
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)

    async def aclose(self, drain_timeout: float = 5.0) -> None:
        await self.drain(drain_timeout)
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        flights = list(self._flights.values())
        for task in flights:
            task.cancel()
        await asyncio.gather(*flights, return_exceptions=True)
        # Slots still pending after shutdown would never finish.
        for row in self.store.con.execute(
            "SELECT seq, segment_index FROM message_media WHERE status='pending'"
        ).fetchall():
            self._finish_unavailable(row[0], row[1], "interrupted")

    # -- one image ------------------------------------------------------------------
    def _finish_unavailable(self, seq: int, index: int, reason: str) -> None:
        self.store.media_unavailable(seq, index, reason, self.clock())
        self.counts[f"unavailable_{reason}"] += 1

    async def _collect(self, seq: int, index: int, data: dict, deadline=None) -> None:
        try:
            key = data.get("file") if isinstance(data.get("file"), str) else None
            resource = self._resource_key(seq, index, data)
            if key and resource in self._by_key:
                asset = self.store.media_asset(self._by_key[resource])
                if asset is not None and (self.root / asset["path"]).is_file():
                    self._link(seq, index, asset, "dedup_key")
                    return
            declared = _int(data.get("file_size"))
            if declared is not None and declared > self.max_bytes:
                raise MediaUnavailable("oversize")
            flight = self._flights.get(resource)
            if flight is None:
                cycle = self._cycles.setdefault(
                    resource,
                    {
                        "attempts": 0,
                        "reason": None,
                        "url": None,
                        "retry_after": 0.0,
                    },
                )
                if cycle["attempts"] >= 2 or (
                    cycle["reason"] and not self._recoverable(cycle["reason"])
                ):
                    raise MediaUnavailable(cycle["reason"] or "attempts_exhausted")
                if cycle["reason"] and deadline is None:
                    raise MediaUnavailable(cycle["reason"])
                flight = asyncio.create_task(self._fetch(data, key, cycle, deadline))
                self._flights[resource] = flight

                def finished(task):
                    self._flights.pop(resource, None)
                    if not task.cancelled():
                        task.exception()  # retrieve even if every slot waiter was cancelled

                flight.add_done_callback(finished)
                while len(self._cycles) > 4096:
                    oldest = next((k for k in self._cycles if k not in self._flights), None)
                    if oldest is None:
                        break
                    self._cycles.pop(oldest)
            else:
                self.counts["inflight_shared"] += 1
            asset = await asyncio.shield(flight)
            if key:
                self._by_key[resource] = asset["sha256"]
                while len(self._by_key) > 4096:
                    self._by_key.popitem(last=False)
            self._link(seq, index, asset, "stored")
        except MediaUnavailable as exc:
            self._finish_unavailable(seq, index, exc.reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let one image break collection
            self.log(f"image collection failed: {type(exc).__name__}")
            self._finish_unavailable(seq, index, "error")

    async def _fetch(self, data: dict, key: str | None, cycle: dict, deadline) -> dict:
        started = time.monotonic()
        cycle["attempts"] += 1
        self.counts["acquire_attempts"] += 1
        try:
            limit = (
                self.timeout
                if deadline is None
                else max(0, deadline - asyncio.get_running_loop().time())
            )
            async with asyncio.timeout(limit):
                async with self._semaphore:
                    await self._bucket.acquire()
                    content = await self._acquire(data, key, cycle)
                asset = await self._store_file(content)
            cycle["reason"] = None
            return asset
        except TimeoutError:
            cycle["reason"] = "download_timeout"
            raise MediaUnavailable("download_timeout") from None
        except MediaUnavailable as exc:
            cycle["reason"] = exc.reason
            raise
        finally:
            cycle["retry_after"] = self.clock() + 60
            cycle["elapsed_ms"] = int((time.monotonic() - started) * 1000)
            self.log(
                f"image acquire attempt={cycle['attempts']} "
                f"reason={cycle['reason'] or 'ok'} elapsed_ms={int((time.monotonic() - started) * 1000)}"
            )

    def diagnostics(self, seq: int, index: int, data: dict) -> dict:
        cycle = self._cycles.get(self._resource_key(seq, index, data), {})
        return {k: cycle[k] for k in ("attempts", "elapsed_ms") if k in cycle}

    def _link(self, seq: int, index: int, asset: dict, outcome: str) -> None:
        if self.store.attach_media(seq, index, asset, self.clock()):
            self.counts[outcome] += 1
        else:
            self.counts["slot_gone"] += 1  # recalled or cleaned while downloading
            # A shared download may still be linking another live slot. The final
            # slot completion performs GC after all consumers have attached.

    async def forward_image(self, data: dict) -> bytes:
        if (_int(data.get("file_size")) or 0) > self.max_bytes:
            raise MediaUnavailable("oversize")
        key = data.get("file") if isinstance(data.get("file"), str) else None
        async with self._semaphore:
            await self._bucket.acquire()
            content = await self._acquire(data, key, {"reason": None, "url": None})
        if len(content) > self.max_bytes or sniff_image(content) is None:
            raise MediaUnavailable("invalid_image")
        return content

    async def _acquire(self, data: dict, key: str | None, cycle: dict) -> bytes:
        sources: list[tuple[str, str]] = []
        if self.api is not None and key:
            try:
                self.counts["get_image_calls"] += 1
                info = await self.api.get_image(key)
            except GatewayError:
                self.counts["get_image_failed"] += 1
            else:
                if info.file_size is not None and info.file_size > self.max_bytes:
                    raise MediaUnavailable("oversize")
                if info.local_path:
                    sources.append(("local", info.local_path))
                if info.url:
                    sources.append(("url", info.url))
        segment_url = data.get("url")
        if isinstance(segment_url, str) and segment_url.startswith(("http://", "https://")):
            if ("url", segment_url) not in sources:
                sources.append(("url", segment_url))
        if not sources:
            raise MediaUnavailable("no_source")
        last = MediaUnavailable(cycle["reason"] or "no_source")
        for kind, location in sources:
            try:
                if kind == "local":
                    path = self.policy.check_local(location)
                    content = await asyncio.to_thread(_read_capped, path, self.max_bytes)
                    self.counts["local_reads"] += 1
                else:
                    # A stale signed URL must be refreshed, never requested twice.
                    if cycle["reason"] in {"http_403", "http_404"} and location == cycle["url"]:
                        continue
                    cycle["url"] = location
                    content = await self._download(location)
                    self.counts["downloads"] += 1
            except MediaUnavailable as exc:
                if kind == "url" or exc.reason != "no_source":
                    raise
                last = exc
                continue
            except OSError:
                last = MediaUnavailable("read_failed")
                continue
            if not content:
                if kind == "url":
                    raise MediaUnavailable("empty")
                last = MediaUnavailable("empty")
                continue
            return content
        raise last

    async def _download(self, url: str) -> bytes:
        current = url
        for _ in range(self.max_redirects + 1):
            await self.policy.check_url(current)
            try:
                status, body, location = await self.download(current, self.max_bytes, self.timeout)
            except MediaUnavailable:
                raise
            except (TimeoutError, httpx.TimeoutException):
                raise MediaUnavailable("download_timeout") from None
            except (ConnectionError, httpx.TransportError):
                raise MediaUnavailable("download_connection") from None
            except Exception:
                raise MediaUnavailable("download_failed") from None
            if location:
                current = urljoin(current, location)
                continue
            if status != 200:
                raise MediaUnavailable(f"http_{status}")
            return body
        raise MediaUnavailable("too_many_redirects")

    async def _store_file(self, content: bytes) -> dict:
        if len(content) > self.max_bytes:
            raise MediaUnavailable("oversize")
        sniffed = sniff_image(content)
        if sniffed is None:
            raise MediaUnavailable("not_image")
        mime, width, height = sniffed
        digest = hashlib.sha256(content).hexdigest()
        name = f"{digest}.{EXTENSIONS[mime]}"
        written = await asyncio.to_thread(_write_atomic, self.root / name, content)
        self.counts["files_written" if written else "dedup_hash"] += 1
        return {
            "sha256": digest,
            "mime": mime,
            "bytes": len(content),
            "width": width,
            "height": height,
            "path": name,
        }

    # -- garbage collection -----------------------------------------------------------
    def gc(self) -> int:
        """Delete asset rows and files that no message references any more."""
        return remove_media_files(self.root, self.store.drop_orphan_media())


def remove_media_files(root: Path, names: list[str]) -> int:
    removed = 0
    for name in names:
        if not MEDIA_FILE.fullmatch(name):
            continue  # only files this module wrote
        try:
            (Path(root) / name).unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return removed
