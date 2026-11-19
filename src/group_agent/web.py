"""Per-run web tools with shared budgets, timeouts, dedup, ledger and source records.

Gemini main mode uses isolated google_search and url_context requests. Missing sources,
failed URL retrieval, empty results and provider failures may fall back to DeepSeek's
official search/fetch endpoints. Each backend attempt consumes a separate budget slot
and cost record. Legacy DeepSeek main mode retains its optional Gemini supplement.

All returned material is untrusted and neutralised in the web section. Direct HTTP for
short links and Bilibili applies public-address checks per redirect and body limits.
Provider error text is never stored, only safe class names and status codes.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import socket
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

from .cards import bilibili_video_ref, is_bilibili, is_short_link
from .evidence import Evidence, render_evidence
from .materials import MESSAGE_CONTINUATION, neutralize, section
from .models import BudgetExceeded, CallRecord, Ledger, Usage
from .policy import RuntimeLimits, ToolBudget, tool_timeout

DEEPSEEK_WEB_BASE = "https://api.deepseek.com"
BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
CN_TZ = timezone(timedelta(hours=8))
MAX_URL = 8192


class WebError(Exception):
    """A failed web call; `kind` is safe to record (no provider text). `cost_cny` is the
    known cost of a failed but billed call (e.g. a Gemini search without sources), with its
    original-currency `list_cost`/`list_currency` and the `google_queries` actually issued
    when the response said so (B1)."""

    def __init__(
        self,
        kind: str,
        cost_cny: float | None = None,
        *,
        list_cost: float | None = None,
        list_currency: str | None = None,
        google_queries: int | None = None,
    ):
        super().__init__(kind)
        self.kind = kind
        self.cost_cny = cost_cny
        self.list_cost, self.list_currency = list_cost, list_currency
        self.google_queries = google_queries


class UnsafeUrl(WebError):
    """WEB-04 refusal: scheme, credentials, or a non-public address."""


def guard(text: Any, **kw: Any) -> str:
    """Untrusted web text for the `<web>` section (`materials.neutralize`)."""
    return neutralize(text, **kw)


def _lines(text: Any, limit: int | None = None) -> str:
    """Multi-line web text: continuation lines are indented so they cannot pose as a new
    source or search block."""
    return neutralize(text, continuation=MESSAGE_CONTINUATION, limit=limit)


def check_url(url: str) -> tuple[str, int]:
    """Syntax part of WEB-04; returns (host, port)."""
    if not isinstance(url, str) or not 0 < len(url) <= MAX_URL or any(c.isspace() for c in url):
        raise UnsafeUrl("bad_url")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise UnsafeUrl("bad_url") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnsafeUrl("bad_scheme")
    if parts.username is not None or parts.password is not None:
        raise UnsafeUrl("credentials_in_url")
    return parts.hostname, port or (443 if parts.scheme == "https" else 80)


def is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


Resolver = Callable[[str, int], Awaitable[list[str]]]


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


async def check_public(url: str, resolver: Resolver) -> None:
    """WEB-04: refuse a URL whose host is, or resolves to, any non-public address."""
    host, port = check_url(url)
    try:
        ipaddress.ip_address(host.strip("[]"))
        addresses = [host.strip("[]")]
    except ValueError:
        if host.endswith(".localhost") or host == "localhost":
            raise UnsafeUrl("private_address") from None
        try:
            addresses = await resolver(host, port)
        except OSError:
            raise UnsafeUrl("dns_failed") from None
    if not addresses or not all(is_public_ip(a) for a in addresses):
        raise UnsafeUrl("private_address")


@dataclass
class HttpResult:
    url: str
    status: int
    body: bytes
    truncated: bool
    location: str | None = None


class SafeHttp:
    """Direct GET under WEB-04 rules; `transport`/`resolver` are injectable for tests."""

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        transport=None,
        max_bytes: int = 1_000_000,
        max_redirects: int = 3,
    ):
        self.resolver, self.transport = resolver, transport
        self.max_bytes, self.max_redirects = max_bytes, max_redirects

    async def get(
        self, url: str, timeout: float, *, follow: bool = True, params: dict | None = None
    ) -> HttpResult:
        import httpx

        current = url
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=self.transport,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(self.max_redirects + 1):
                await check_public(current, self.resolver)
                async with client.stream("GET", current, params=params) as response:
                    location = response.headers.get("location")
                    if response.is_redirect and location:
                        target = urljoin(current, location)
                        if not follow:
                            await check_public(target, self.resolver)
                            return HttpResult(current, response.status_code, b"", False, target)
                        current, params = target, None
                        continue
                    chunks, size, truncated = [], 0, False
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > self.max_bytes:
                            truncated = True
                            break
                    body = b"".join(chunks)[: self.max_bytes]
                    return HttpResult(current, response.status_code, body, truncated)
        raise WebError("too_many_redirects")


@dataclass(frozen=True)
class WebCredentials:
    api_key: str = field(repr=False)
    base_url: str = DEEPSEEK_WEB_BASE


class DeepSeekWebClient:
    """Official DeepSeek search/fetch endpoints (legacy AstrBot request/response shape)."""

    def __init__(self, creds: WebCredentials, transport=None):
        self.creds, self.transport = creds, transport

    @property
    def channel(self) -> str:
        """Billing channel (B1): the official endpoint host, e.g. official:api.deepseek.com."""
        return f"official:{(urlsplit(self.creds.base_url).hostname or 'unknown').lower()}"

    async def _post(self, path: str, body: dict, timeout: float) -> dict:
        import httpx

        try:
            async with httpx.AsyncClient(
                base_url=self.creds.base_url, timeout=timeout, transport=self.transport
            ) as client:
                response = await client.post(
                    path, json=body, headers={"Authorization": f"Bearer {self.creds.api_key}"}
                )
        except httpx.TimeoutException:
            raise WebError("timeout") from None
        except httpx.HTTPError as exc:
            raise WebError(type(exc).__name__) from None
        if response.status_code != 200:
            raise WebError(f"http_{response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise WebError("invalid_json") from None
        if not isinstance(data, dict):
            raise WebError("invalid_json")
        return data

    async def search(self, query: str, timeout: float, count: int = 5) -> list[dict]:
        data = await self._post(
            "/v1/tools/search",
            {"query": query[:128], "count": count, "need_content": False},
            timeout,
        )
        out = []
        for item in data.get("results") or []:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            out.append(
                {
                    "title": str(item.get("title") or "")[:200],
                    "url": str(item.get("url"))[:MAX_URL],
                    "site": str(item.get("site_name") or "")[:60],
                    "date": str(item.get("date") or "")[:32],
                    "summary": str(item.get("summary") or item.get("snippet") or "")[:600],
                    "truncated": len(str(item.get("summary") or item.get("snippet") or "")) > 600,
                }
            )
        return out

    async def fetch(self, url: str, timeout: float) -> dict:
        data = await self._post("/v1/tools/fetch", {"url": url}, timeout)
        content = str(data.get("content") or "")
        if not content.strip():
            raise WebError("empty_content")
        return {"title": str(data.get("title") or "")[:200], "content": content}


def _gemini_search_client(settings, fetcher, purpose: str):
    from .config import ConfigError, load_model_credentials
    from .gemini_search import GeminiSearchClient, SEARCH_PROMPT

    spec = settings.web
    gemini = settings.models.get(spec.gemini_credentials_model)
    if gemini is None or gemini.protocol != "google-genai":
        raise ConfigError(
            f"web.gemini_credentials_model: requires a google-genai model for {purpose}"
        )
    return GeminiSearchClient(
        gemini,
        load_model_credentials(settings.credentials, gemini),
        search_prompt=settings.prompts.get("search", SEARCH_PROMPT),
        model=spec.gemini_model,
        thinking_level=spec.gemini_thinking_level,
        fee_per_query=spec.gemini_fee_per_query,
        fetcher=fetcher,
        use_url_context=spec.gemini_url_context and fetcher is not None,
        fallback_client=fetcher if spec.deepseek_fallback else None,
    )


@dataclass
class WebBackends:
    """V3: search and fetch bound independently (`[web] search/fetch`, each with an
    optional fallback). WebTool runs every attempt under the same run budget, dedup and
    evidence structure; switching one operation's backend never touches the other."""

    search: Any
    fetch: Any
    search_fallback: Any = None
    fetch_fallback: Any = None

    @property
    def provider_name(self) -> str:
        return getattr(self.search, "provider_name", "deepseek_web")

    def route(self, op: str) -> tuple[Any, Any]:
        if op == "search":
            return self.search, self.search_fallback
        return self.fetch, self.fetch_fallback


def _price_unit(client: Any, op: str) -> tuple[str, float]:
    unit = getattr(client, "price_unit", None)
    return unit(op) if unit is not None else ("CNY", 1.0)


def _google_web_client(settings, alias: str, key: str, *, url_context: bool):
    """A google_search-only (search) or url_context-only (fetch) client priced and
    credentialed by its own model table."""
    from .config import ConfigError, load_model_credentials
    from .gemini_search import GeminiSearchClient, SEARCH_PROMPT

    spec = settings.models.get(alias)
    if spec is None or spec.protocol != "google-genai":
        raise ConfigError(f"web.{key}: requires a google-genai model alias")
    fee = {"fee_per_query": settings.web.gemini_fee_per_query}  # CNY, relay/legacy path
    if spec.channel == "vertex":
        # Vertex bills grounding at Google's USD list price per query; keep the USD figure
        # and its CNY conversion at the configured rate.
        usd = settings.billing.google_search_usd_per_1000 / 1000
        fee = {
            "fee_per_query": usd * settings.billing.usd_cny,
            "fee_list_per_query": usd,
            "fee_currency": "USD",
        }
    return GeminiSearchClient(
        spec,
        load_model_credentials(settings.credentials, spec),
        search_prompt=settings.prompts.get("search", SEARCH_PROMPT),
        model=spec.model,
        thinking_level=settings.web.gemini_thinking_level,
        use_url_context=url_context,
        **fee,
    )


def build_web_backends(settings) -> WebBackends:
    from .config import load_web_api_key

    spec = settings.web
    bindings = {
        key: getattr(spec, key) for key in ("search", "search_fallback", "fetch", "fetch_fallback")
    }
    deepseek = None
    if "deepseek" in bindings.values():
        deepseek = DeepSeekWebClient(
            WebCredentials(load_web_api_key(settings.credentials, spec), spec.base_url)
        )
    clients = {}
    for key, alias in bindings.items():
        if not alias:
            clients[key] = None
        elif alias == "deepseek":
            clients[key] = deepseek
        else:
            clients[key] = _google_web_client(
                settings, alias, key, url_context=key.startswith("fetch")
            )
    return WebBackends(**clients)


def build_web_client(settings) -> Any:
    """Build the primary client and optional fallback; load keys only via config."""
    from .config import load_web_api_key

    spec = settings.web
    if spec.search is not None:
        return build_web_backends(settings)
    deepseek = DeepSeekWebClient(
        WebCredentials(load_web_api_key(settings.credentials, spec), spec.base_url)
    )
    if spec.search_provider != "gemini":
        return deepseek
    return _gemini_search_client(settings, deepseek, "web.search_provider = 'gemini'")


def build_supplement_client(settings) -> Any:
    """D-18 supplement search client (Gemini, google_search only), or None when
    `[web] gemini_supplement` is off or Gemini is already the main search backend."""
    spec = settings.web
    if spec.search is not None:  # V3 bindings: the legacy supplement does not apply
        return None
    if not spec.gemini_supplement or spec.search_provider == "gemini":
        return None
    return _gemini_search_client(settings, None, "web.gemini_supplement = true")


BACKEND_LABELS = {
    "deepseek_web": "DeepSeek",
    "gemini_search": "Gemini",
    "gemini_url_context": "Gemini URL context",
}
# D-18 supplement outcomes recorded per run (see WebTool.supplement_search).
SUPPLEMENT_STATUSES = (
    "found",
    "no_sources",
    "timeout",
    "skipped_deadline",
    "skipped_breaker",
    "budget",
    "error",
    "off",
)
SUPPLEMENT_FAILURES = frozenset({"no_sources", "timeout", "error"})


class SupplementBreaker:
    """Process-wide circuit breaker for the supplement backend (shared by all runs).

    After `failures` consecutive failed supplement calls (no sources, timeout, error) the
    supplement is skipped for `cooldown_s`; afterwards one call is let through, and a
    further failure reopens it at once. A found result closes it. Thread-safe: the live
    replay runs cases on worker threads with their own event loops."""

    def __init__(self, failures: int = 3, cooldown_s: float = 1800.0, clock=time.monotonic):
        self.threshold, self.cooldown_s, self.clock = failures, cooldown_s, clock
        self.consecutive, self.open_until, self.opened = 0, 0.0, 0
        self._lock = threading.Lock()

    def remaining_s(self) -> float:
        """Seconds the breaker stays open; 0 when calls are allowed."""
        with self._lock:
            return max(0.0, self.open_until - self.clock())

    def record(self, status: str) -> None:
        with self._lock:
            if status == "found":
                self.consecutive, self.open_until = 0, 0.0
            elif status in SUPPLEMENT_FAILURES:
                self.consecutive += 1
                if self.consecutive >= self.threshold:
                    self.open_until = self.clock() + self.cooldown_s
                    self.opened += 1


def _clock(ts: Any) -> str:
    if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0:
        return datetime.fromtimestamp(ts, CN_TZ).strftime("%Y-%m-%d")
    return ""


@dataclass
class WebTool:
    """Per-run web access with budget, timeouts, dedup and source records."""

    client: DeepSeekWebClient | None
    http: SafeHttp
    budget: ToolBudget
    remaining: Callable[[], float]
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    ledger: Ledger | None = None
    price_per_call_cny: float = 0.003
    # D-18: Gemini supplement search client (None = off) and the least per-call tool time
    # it needs; outcomes are appended to `supplements`.
    supplement: Any = None
    supplement_min_seconds: float = 15.0
    supplement_breaker: SupplementBreaker | None = None
    calls: list[dict] = field(default_factory=list)
    records: list[CallRecord] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    supplements: list[dict] = field(default_factory=list)
    _cache: dict = field(default_factory=dict)
    _entries: dict = field(default_factory=dict)

    # -- bookkeeping ----------------------------------------------------------------
    async def _call(
        self,
        tool: str,
        key: str,
        args: dict,
        work,
        *,
        billed: bool,
        price: float | None = None,
        model: str = "deepseek_web",
        channel: str | None = None,
        unit: tuple[str, float] = ("CNY", 1.0),
        model_id: str | None = None,
    ):
        """Dedup, budget, timeout and records around one tool call; None on failure.

        `price` is the reserved estimate (default `price_per_call_cny`). A result or
        WebError carrying `cost_cny` (token-billed Gemini search) settles at that cost;
        otherwise, and for timeouts, the estimate is billed. `channel` (B1) is the backend's
        billing channel; the record's service is "search" or "fetch" by tool. A result or
        WebError may also carry `list_cost`/`list_currency` (original-currency list price) and
        `google_queries`; without them the CNY cost is shown in `unit` = (currency, CNY per
        unit): CNY for the per-call DeepSeek tools, the model table's currency for an
        estimate billed after a failed Google request."""
        if self.ledger is not None and not getattr(self.ledger, "web_allowed", lambda: True)():
            self.calls.append({"name": tool, "status": "disabled"})
            return None
        cache_key = (tool, key)
        if cache_key in self._cache:
            previous = self._entries.get(cache_key, {})
            self.calls.append(
                {
                    "name": tool,
                    "status": "dedup",
                    "evidence_ids": previous.get("evidence_ids", []),
                    "cached_status": previous.get("status"),
                    **args,
                }
            )
            return self._cache[cache_key]
        entry = {
            "name": tool,
            "status": "ok",
            **args,
            "fetched_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        }
        self.calls.append(entry)
        self._entries[cache_key] = entry
        if not self.budget.try_consume(tool):
            entry["status"] = "budget_exhausted"
            return None
        timeout = tool_timeout(self.remaining(), self.limits)
        if timeout <= 1:
            entry["status"] = "no_time_left"
            return None
        reserved = (self.price_per_call_cny if price is None else price) if billed else 0.0
        if billed and self.ledger is not None:
            try:
                self.ledger.reserve(reserved)
            except BudgetExceeded:
                entry["status"] = "budget_cny"
                raise
        started = time.perf_counter()
        result, status, known_cost = None, "ok", None
        cancelled = None
        accounting: Any = None  # object carrying list_cost / google_queries (result or WebError)
        try:
            async with asyncio.timeout(timeout):
                result = await work(timeout)
            known_cost = getattr(result, "cost_cny", None)
            accounting = result
        except TimeoutError:
            status = "timeout"
        except asyncio.CancelledError as exc:
            status, cancelled = "cancelled", exc
        except WebError as exc:
            status, known_cost, accounting = exc.kind, exc.cost_cny, exc
        except Exception as exc:  # noqa: BLE001 - recorded by class name only
            status = type(exc).__name__
        latency = time.perf_counter() - started
        entry.update(status=status, latency_s=round(latency, 3))
        if billed:
            # Failed attempts are billed conservatively (the request may have been served).
            record = CallRecord(
                task=tool,
                model=model,
                model_id=model_id,
                mode="http",
                status=status if status in ("ok", "cancelled") else "error",
                error=None if status == "ok" else status,
                latency_s=latency,
                cost_cny=reserved if known_cost is None else known_cost,
                responses=1,
                channel=channel,
                service="search" if tool == "web_search" else "fetch",
                google_queries=getattr(accounting, "google_queries", None),
            )
            listed = getattr(accounting, "list_cost", None)
            if listed is None:  # per-call tools, or an estimate billed on failure
                record.list_cost, record.list_currency = record.cost_cny / unit[1], unit[0]
            else:
                record.list_cost = listed
                record.list_currency = getattr(accounting, "list_currency", None) or unit[0]
            usage = getattr(result, "usage", None)
            if isinstance(usage, Usage):
                record.usage = usage
            entry["cost_cny"] = round(record.cost_cny, 6)
            self.records.append(record)
            if self.ledger is not None:
                self.ledger.settle(reserved, record)
        if cancelled is not None:
            # Preserve cancellation after settling this possibly served request. The
            # caller must not continue into a fallback or cache a cancelled result.
            raise cancelled
        self._cache[cache_key] = result
        return result

    def summary(self) -> dict:
        statuses = Counter(f"{c['name']}:{c['status']}" for c in self.calls)
        return {
            "searches": sum(
                1
                for c in self.calls
                if c["name"] == "web_search"
                and c["status"] != "dedup"
                and c["status"] != "budget_exhausted"
            ),
            "reads": sum(
                1
                for c in self.calls
                if c["name"] in ("web_fetch", "bilibili_video_info")
                and c["status"] not in ("dedup", "budget_exhausted")
            ),
            "cost_cny": round(sum(r.cost_cny for r in self.records), 6),
            "sources": self.sources,
            "statuses": dict(statuses),
            "calls": self.calls,
            "supplements": self.supplements,
        }

    def search_status(self, query: str, backend: str = "deepseek_web") -> str | None:
        """Status of the first call for `query` on `backend` ("ok", "timeout", ...)."""
        query = " ".join(str(query).split())[:128]
        key = query if backend == "deepseek_web" else f"{backend}:{query}"
        entry = self._entries.get(("web_search", key))
        return None if entry is None else entry["status"]

    def material(self) -> str:
        return render_evidence(self.evidence, self.sources)

    @property
    def blocks(self) -> list[str]:
        return [item.render(self.sources) for item in self.evidence]

    def tool_result(self, call_start: int) -> str:
        """Only the evidence associated with this invocation, including cache hits."""
        calls = self.calls[call_start:]
        ids = {i for call in calls for i in call.get("evidence_ids", [])}
        material = render_evidence([e for e in self.evidence if e.id in ids], self.sources)
        statuses = [
            f"{c['name']}: {c['status']}"
            + (f"（缓存原状态：{c['cached_status']}）" if c.get("cached_status") else "")
            for c in calls
        ]
        return material + section(
            "notes",
            "\n".join(statuses) or "无有效工具输入",
        )

    def _evidence(self, kind, source_ids, content, entry, *, query="", limit=3000, truncated=False):
        item = Evidence(
            len(self.evidence) + 1,
            kind,
            tuple(source_ids),
            str(content)[:limit],
            entry.get("fetched_at", ""),
            query,
            truncated or len(str(content)) > limit,
        )
        self.evidence.append(item)
        entry.setdefault("evidence_ids", []).append(item.id)

    def _source(
        self, kind: str, title: str, url: str, site: str = "", date: str = "", backend: str = ""
    ) -> int:
        for n, s in enumerate(self.sources, 1):
            if s["url"] == url or s.get("canonical_url") == url:
                return n
        source = {"kind": kind, "title": title, "url": url, "site": site, "date": date}
        if backend:
            source["backend"] = backend
        self.sources.append(source)
        return len(self.sources)

    def _seen(self, tool: str, key: str) -> bool:
        return (tool, key) in self._cache

    # -- tools ----------------------------------------------------------------------
    def _route(self, op: str) -> tuple[Any, Any]:
        """(primary, fallback) for "search" or "fetch": V3 `WebBackends` binds them
        separately; a single legacy client serves both with its `fallback_client`."""
        route = getattr(self.client, "route", None)
        if route is not None:
            return route(op)
        return self.client, getattr(self.client, "fallback_client", None)

    async def search(self, query: str) -> list[dict] | None:
        query = " ".join(str(query).split())[:128]
        primary, fallback = self._route("search") if self.client is not None else (None, None)
        if not query or primary is None:
            return None
        results = await self._search(query, primary, supplement=False)
        if not results and fallback is not None:
            return await self._search(query, fallback, supplement=False)
        return results

    async def _search(self, query: str, client: Any, *, supplement: bool) -> list[dict] | None:
        backend = getattr(client, "provider_name", "deepseek_web")
        # The main backend keeps plain query keys; other backends get their own dedup key.
        key = query if client is self._route("search")[0] else f"{backend}:{query}"
        args = {"query": query, "backend": backend}
        if supplement:
            args["supplement"] = True
        if self._seen("web_search", key):  # material already added once
            return await self._call("web_search", key, args, None, billed=True)
        price = getattr(client, "search_estimate_cny", None)
        quota = getattr(self.ledger, "quota", None)
        estimate = getattr(client, "search_estimate", None)
        if quota is not None and estimate is not None and client.channel == quota.channel:
            # B3: reserve no query fee while the free monthly quota covers the estimate.
            price = estimate(quota.remaining())
        results = await self._call(
            "web_search",
            key,
            args,
            lambda t: client.search(query, t),
            billed=True,
            price=price,
            model=backend,
            channel=getattr(client, "channel", None),
            unit=_price_unit(client, "search"),
            model_id=getattr(client, "model", "deepseek-web-tools"),
        )
        if not results:
            return results
        entry = self._entries[("web_search", key)]
        source_ids = []
        answer = getattr(results, "answer", "")
        for item in results[:5]:
            n = self._source(
                "search", item["title"], item["url"], item["site"], item["date"], backend
            )
            source_ids.append(n)
            self._evidence(
                item.get("evidence_kind", "search_snippet"),
                [n],
                item["summary"] or "仅返回来源链接，未取得片段内容。",
                entry,
                query=query,
                limit=600,
                truncated=item.get("truncated", False),
            )
        if answer:
            self._evidence("search_summary", source_ids, answer, entry, query=query)
        return results

    async def supplement_search(self, query: str, reason: str) -> dict | None:
        """D-18: one Gemini search that supplements the main (DeepSeek) search.

        Counts against the run's search budget like any search, is reserved and billed at
        its estimate / actual cost, and runs under `policy.tool_timeout`. Skipped (status
        `skipped_deadline`) when that timeout is below `supplement_min_seconds`, `budget`
        when no search is left or the CNY budget refuses the reservation, `skipped_breaker`
        while `supplement_breaker` is open (the main results stay usable either way).
        Returns the outcome record, also kept in `supplements`."""
        query = " ".join(str(query).split())[:128]
        if not query:
            return None
        rec: dict = {"reason": reason, "triggered": False, "query_chars": len(query)}
        self.supplements.append(rec)
        if self.supplement is None:
            rec["status"] = "off"
            return rec
        breaker = self.supplement_breaker
        if breaker is not None and (wait := breaker.remaining_s()) > 0:
            rec.update(status="skipped_breaker", breaker_open_s=round(wait))
            return rec
        if self.budget.left.get("search", 0) <= 0:
            rec.update(status="budget", detail="search_budget")
            return rec
        left = tool_timeout(self.remaining(), self.limits)
        if left < self.supplement_min_seconds:
            rec.update(status="skipped_deadline", timeout_s=round(left, 1))
            return rec
        rec["triggered"] = True
        try:
            results = await self._search(query, self.supplement, supplement=True)
        except BudgetExceeded:
            rec.update(status="budget", detail="budget_cny")
            return rec
        backend = getattr(self.supplement, "provider_name", "supplement")
        entry = self._entries.get(("web_search", f"{backend}:{query}")) or {}
        status = entry.get("status")
        if results:
            rec.update(status="found", results=len(results))
        elif status in ("no_sources", "timeout"):
            rec["status"] = status
        elif status == "budget_exhausted":
            rec.update(status="budget", detail="search_budget")
        elif status == "no_time_left":
            rec["status"] = "skipped_deadline"
        else:
            rec.update(status="error", detail=str(status))
        rec.update(
            latency_s=entry.get("latency_s"),
            cost_cny=entry.get("cost_cny", 0.0),
            queries=len(getattr(results, "queries", []) or []),
        )
        if breaker is not None:
            breaker.record(rec["status"])
        return rec

    async def fetch(self, url: str, max_chars: int = 3000) -> dict | None:
        """Read using the selected provider, accounting each fallback as a separate read."""
        primary, fallback = self._route("fetch") if self.client is not None else (None, None)
        if primary is None:
            return None
        try:
            check_url(url)
        except UnsafeUrl as exc:
            self.calls.append({"name": "web_fetch", "status": f"refused:{exc.kind}", "url": url})
            return None
        parts = urlsplit(url)
        if (
            parts.scheme == "https"
            and parts.hostname == "vertexaisearch.cloud.google.com"
            and parts.port in (None, 443)
            and parts.path.startswith("/grounding-api-redirect/")
            and (
                self.budget.left.get("read", 0) > 0
                or ("grounding_redirect_resolve", url) in self._cache
            )
        ):
            original = url
            url = await self._resolve_grounding_url(original)
            if url is None:
                return None
            # Preserve the grounded redirect as provenance and separately record the
            # verified destination. Model-written links never populate this field.
            for source in self.sources:
                if source["url"] == original:
                    source["canonical_url"] = url
        page = await self._fetch(url, primary, max_chars)
        if not page and fallback is not None:
            return await self._fetch(url, fallback, max_chars)
        return page

    async def _resolve_grounding_url(self, url: str) -> str | None:
        async def work(timeout):
            result = await self.http.get(url, timeout)
            if result.status != 200:
                raise WebError(f"http_{result.status}")
            if result.url == url:
                raise WebError("grounding_redirect_unresolved")
            return result.url

        canonical = await self._call(
            "grounding_redirect_resolve", url, {"url": url}, work, billed=False
        )
        if canonical:
            self._entries[("grounding_redirect_resolve", url)]["canonical_url"] = canonical
        return canonical

    async def _fetch(self, url: str, client: Any, max_chars: int) -> dict | None:
        backend = getattr(client, "fetch_provider_name", "deepseek_web")
        key = url if client is self._route("fetch")[0] else f"{backend}:{url}"
        args = {"url": url, "backend": backend}
        if self._seen("web_fetch", key):
            return await self._call("web_fetch", key, args, None, billed=True)
        page = await self._call(
            "web_fetch",
            key,
            args,
            lambda t: client.fetch(url, t),
            billed=True,
            price=getattr(client, "fetch_estimate_cny", None),
            model=backend,
            channel=getattr(client, "fetch_channel", getattr(client, "channel", None)),
            unit=_price_unit(client, "fetch"),
            model_id=getattr(client, "model", "deepseek-web-tools")
            if getattr(client, "use_url_context", True)
            else "deepseek-web-tools",
        )
        if page:
            kind = page.get("evidence_kind", "page")
            n = self._source(kind, page["title"], url, backend=backend)
            self._evidence(
                kind, [n], page["content"], self._entries[("web_fetch", key)], limit=max_chars
            )
        return page

    async def expand_short_link(self, url: str) -> str | None:
        async def work(timeout):
            result = await self.http.get(url, timeout, follow=False)
            return result.location

        return await self._call("short_link_expand", url, {"url": url}, work, billed=False)

    async def bilibili_video(self, url: str) -> dict | None:
        """LINK-01: expand b23.tv, then the public video info API (title, desc, owner, ...)."""
        target = url
        if is_short_link(url):
            target = await self.expand_short_link(url) or ""
        ref = bilibili_video_ref(target) if target else None
        if ref is None:
            if target:
                self.calls.append({"name": "bilibili_video_info", "status": "not_a_video"})
            return None
        key, value = ref
        if self._seen("bilibili_video_info", value):
            return await self._call(
                "bilibili_video_info", value, {"video": value}, None, billed=False
            )

        async def work(timeout):
            result = await self.http.get(BILIBILI_VIEW_API, timeout, params={key: value})
            if result.status != 200 or result.truncated:
                raise WebError(f"http_{result.status}" if result.status != 200 else "too_large")
            body = json.loads(result.body.decode("utf-8", "replace"))
            if body.get("code") != 0 or not isinstance(body.get("data"), dict):
                raise WebError(f"api_code_{body.get('code')}")
            d = body["data"]
            return {
                "bvid": str(d.get("bvid") or value),
                "title": str(d.get("title") or "")[:200],
                "desc": str(d.get("desc") or "")[:800],
                "owner": str((d.get("owner") or {}).get("name") or "")[:60],
                "pubdate": _clock(d.get("pubdate")),
                "duration_s": int(d.get("duration") or 0),
            }

        info = await self._call("bilibili_video_info", value, {"video": value}, work, billed=False)
        if info:
            page = f"https://www.bilibili.com/video/{info['bvid']}"
            n = self._source("bilibili", info["title"], page, "哔哩哔哩", info["pubdate"])
            minutes, seconds = divmod(info["duration_s"], 60)
            self._evidence(
                "bilibili",
                [n],
                f"B 站视频《{info['title']}》，"
                f"UP 主 {guard(info['owner'], single_line=True)}，"
                f"发布于 {info['pubdate'] or '未知'}，时长 {minutes}:{seconds:02d}。\n"
                f"简介：{info['desc'] or '（无）'}",
                self._entries[("bilibili_video_info", value)],
            )
        return info

    async def read_link(self, url: str) -> dict | None:
        """Bilibili videos via the video API, other links via page fetch."""
        if is_bilibili(url):
            return await self.bilibili_video(url)
        return await self.fetch(url)
