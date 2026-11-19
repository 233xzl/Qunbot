"""Independent Gemini built-in tools, isolated from the answer model's function tools.

Search carries only google_search and requires grounding sources. URL-context reading
carries only url_context and requires successful retrieval metadata for the requested
URL. Its output is a page extraction summary, rather than raw page text. WebTool owns
fallback attempts so every backend request has its own budget and cost record.

2026-11-13 relay probes: flash search returned sources 5/5; URL context returned HTTP
400 for flash and hydrogen. Fetch errors therefore fall back to DeepSeek when enabled.
Provider error text is never recorded.
"""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

from .config import ModelCredentials, ModelSpec
from .models import Usage, cost_cny, list_cost
from .web import WebError

SEARCH_PROMPT = (
    "联网检索下面的问题，用中文简要回答（不超过 300 字）。只写检索结果支持的事实，"
    "涉及时间的写明日期；检索不到就直说。\n问题：{query}"
)
DOMAIN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)
MAX_ANSWER = 1500


class GroundedResults(list):
    """Source dicts plus the grounded answer and accounting of one search request."""

    def __init__(
        self,
        items=(),
        *,
        answer: str = "",
        queries: list[str] | None = None,
        usage: Usage | None = None,
        cost_cny: float = 0.0,
        list_cost: float | None = None,
        list_currency: str | None = None,
    ):
        super().__init__(items)
        self.answer = answer
        self.queries = list(queries or [])
        self.usage = usage or Usage()
        self.cost_cny = cost_cny
        self.list_cost, self.list_currency = list_cost, list_currency

    @property
    def google_queries(self) -> int:
        return len(self.queries)


def response_usage(response: Any) -> Usage:
    """Usage of one generateContent response: input includes tool-use prompt tokens,
    output includes thinking (both billed)."""
    meta = getattr(response, "usage_metadata", None)

    def n(name: str) -> int:
        return int(getattr(meta, name, None) or 0) if meta is not None else 0

    thoughts = n("thoughts_token_count")
    return Usage(
        input=n("prompt_token_count") + n("tool_use_prompt_token_count"),
        cached=n("cached_content_token_count"),
        output=n("candidates_token_count") + thoughts,
        thinking=thoughts,
        requests=1,
    )


def _text(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    content = getattr(candidates[0], "content", None) if candidates else None
    parts = getattr(content, "parts", None) or []
    return "".join(
        p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)
    )


def parse_grounded(response: Any, count: int = 5) -> tuple[str, list[str], list[dict]]:
    """(answer, search queries, sources) from a response; sources are empty when the
    candidate has no grounding metadata."""
    candidates = getattr(response, "candidates", None) or []
    meta = getattr(candidates[0], "grounding_metadata", None) if candidates else None
    answer = _text(response).strip()
    if meta is None:
        return answer, [], []
    sources: list[dict | None] = []
    for chunk in meta.grounding_chunks or []:
        web = getattr(chunk, "web", None)
        uri = str(getattr(web, "uri", None) or "") if web is not None else ""
        if not uri.startswith(("https://", "http://")):
            sources.append(None)
            continue
        title = str(getattr(web, "title", None) or "").strip()[:200]
        domain = str(getattr(web, "domain", None) or "").strip()[:60]
        site = domain or (title if DOMAIN.match(title) else "")
        sources.append(
            {
                "title": title or site or "（无标题）",
                "url": uri,
                "site": site,
                "date": "",
                "hits": 0,
            }
        )
    summaries: dict[int, list[str]] = {}
    for support in meta.grounding_supports or []:
        segment = getattr(support, "segment", None)
        text = str(getattr(segment, "text", None) or "").strip()
        for idx in support.grounding_chunk_indices or []:
            if isinstance(idx, int) and 0 <= idx < len(sources) and sources[idx] is not None:
                sources[idx]["hits"] += 1
                if text and text not in summaries.setdefault(idx, []):
                    summaries[idx].append(text)
    ordered, seen = [], set()
    ranked = sorted(
        ((i, s) for i, s in enumerate(sources) if s is not None), key=lambda p: -p[1]["hits"]
    )
    for i, source in ranked:
        if source["url"] in seen:
            continue
        seen.add(source["url"])
        ordered.append(
            {
                "title": source["title"],
                "url": source["url"],
                "site": source["site"],
                "date": "",
                "summary": " ".join(summaries.get(i, []))[:600],
                # groundingSupports.segment is generated answer text, not a page quote.
                "evidence_kind": "grounded_summary",
                "truncated": len(" ".join(summaries.get(i, []))) > 600,
            }
        )
    queries = [str(q)[:200] for q in meta.web_search_queries or []]
    return answer, queries, ordered[:count]


class GeminiSearchClient:
    """Search through a Gemini request carrying only google_search; fetch via `fetcher`."""

    provider_name = "gemini_search"

    def __init__(
        self,
        price_spec: ModelSpec,
        creds: ModelCredentials | None,
        *,
        model: str,
        thinking_level: str | None = "LOW",
        fee_per_query: float = 0.1,
        max_output_tokens: int = 2048,
        fetcher: Any = None,
        use_url_context: bool = False,
        fallback_client: Any = None,
        client_factory: Callable[[float], Any] | None = None,
        fee_list_per_query: float | None = None,
        fee_currency: str = "CNY",
        search_prompt: str = SEARCH_PROMPT,
    ):
        self.search_prompt = search_prompt
        self.price_spec, self.creds, self.model = price_spec, creds, model
        self.thinking_level, self.fee_per_query = thinking_level, fee_per_query
        # `fee_per_query` is the CNY fee every gate uses; `fee_list_per_query` is the same fee
        # in `fee_currency` as the vendor lists it (default: the CNY fee itself).
        self.fee_list_per_query = (
            fee_per_query if fee_list_per_query is None else fee_list_per_query
        )
        self.fee_currency = fee_currency if fee_list_per_query is not None else "CNY"
        self.max_output_tokens, self.fetcher = max_output_tokens, fetcher
        self.use_url_context, self.fallback_client = use_url_context, fallback_client
        self._factory = client_factory or self._default_client

    @property
    def channel(self) -> str:
        """Billing channel of the search requests: the channel of the model table."""
        return self.price_spec.channel

    @property
    def fetch_channel(self) -> str | None:
        if self.use_url_context:
            return self.price_spec.channel
        return getattr(self.fetcher, "channel", None)

    def price_unit(self, op: str) -> tuple[str, float]:
        """(currency, CNY per unit) of the estimate this client reserves for "search" or
        "fetch"; a plain page fetch through `fetcher` is a per-call CNY price."""
        if op == "search" or self.use_url_context:
            return self.price_spec.price_currency, self.price_spec.cny_per_unit
        return "CNY", 1.0

    def _list_cost(self, usage: Usage, queries: int = 0) -> float:
        """Original-currency list cost: tokens plus per-query fees in the table's currency
        (a fee listed in another currency is converted through the CNY rates)."""
        spec = self.price_spec
        fee = self.fee_list_per_query * queries
        if self.fee_currency != spec.price_currency:
            fee = self.fee_per_query * queries / spec.cny_per_unit
        return list_cost(spec, usage) + fee

    ESTIMATE_QUERIES = 4  # probe maximum per search request

    def search_estimate(self, free_queries: int = 0) -> float:
        """Reservation per search: 4 queries (probe max) plus ~600 input tokens and the
        output cap. The queries cost nothing while `free_queries` (the account's remaining
        monthly free quota) covers all of them, else the list price (B3)."""
        usage = Usage(input=600, output=self.max_output_tokens, requests=1)
        fee = (
            0.0
            if free_queries >= self.ESTIMATE_QUERIES
            else self.ESTIMATE_QUERIES * self.fee_per_query
        )
        return fee + cost_cny(self.price_spec, usage)

    @property
    def search_estimate_cny(self) -> float:
        """Reservation at list price (no free quota known)."""
        return self.search_estimate()

    def _default_client(self, timeout: float):
        from .provider_adapters import google_client

        if self.creds is None:
            raise WebError("no_credentials")
        # Shared factory: Vertex (v1, resource path) or relay (google-genai>=3 defaults to
        # /v1; the relay serves /v1beta only).
        return google_client(self.price_spec, self.creds, timeout, relay_api_version="v1beta")

    @property
    def request_model(self) -> str:
        from .provider_adapters import google_model_name

        if self.creds is None:
            return self.model
        return google_model_name(self.price_spec, self.creds, self.model)

    def request_config(self):
        """Only the built-in search tool: no function tools, schema, MIME type or system
        instruction (vendor advice; re-check 2)."""
        from google.genai import types

        return types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=self.max_output_tokens,
            thinking_config=(
                types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.thinking_level
                else None
            ),
        )

    async def search(self, query: str, timeout: float, count: int = 5) -> GroundedResults:
        prompt = self.search_prompt.format(query=" ".join(str(query).split())[:300])
        response = await self._generate(prompt, timeout, self.request_config())
        usage = response_usage(response)
        answer, queries, sources = parse_grounded(response, count)
        cost = cost_cny(self.price_spec, usage) + self.fee_per_query * len(queries)
        listed = self._list_cost(usage, len(queries))
        currency = self.price_spec.price_currency
        if not sources:
            raise WebError(
                "no_sources",
                cost_cny=cost,
                list_cost=listed,
                list_currency=currency,
                google_queries=len(queries),
            )
        return GroundedResults(
            sources,
            answer=answer[:MAX_ANSWER],
            queries=queries,
            usage=usage,
            cost_cny=cost,
            list_cost=listed,
            list_currency=currency,
        )

    async def _generate(self, prompt: str, timeout: float, config):
        client = self._factory(timeout)
        try:
            return await client.aio.models.generate_content(
                model=self.request_model, contents=prompt, config=config
            )
        except WebError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider text may hold prompt or host
            code = getattr(exc, "code", None)
            if isinstance(code, int):
                raise WebError(f"http_{code}") from None
            name = type(exc).__name__
            raise WebError("timeout" if "Timeout" in name else name) from None
        finally:
            close = getattr(getattr(client, "aio", None), "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 - closing must not mask the result
                    pass

    @property
    def fetch_provider_name(self) -> str:
        return "gemini_url_context" if self.use_url_context else "deepseek_web"

    @property
    def fetch_estimate_cny(self) -> float | None:
        if not self.use_url_context:
            return None
        # URL context input can be larger than a search prompt; reserve a page allowance.
        return cost_cny(self.price_spec, Usage(input=16000, output=self.max_output_tokens))

    async def fetch(self, url: str, timeout: float) -> dict:
        if self.use_url_context:
            from google.genai import types

            config = self.request_config()
            config.tools = [types.Tool(url_context=types.UrlContext())]
            response = await self._generate(
                f"读取以下 URL 的页面，提取与页面内容有关的事实和主要段落，保留日期和标题。"
                f"只依据页面，无法访问时说明。URL：{url}",
                timeout,
                config,
            )
            usage = response_usage(response)
            cost = cost_cny(self.price_spec, usage)
            candidates = getattr(response, "candidates", None) or []
            meta = getattr(candidates[0], "url_context_metadata", None) if candidates else None
            retrieved = getattr(meta, "url_metadata", None) or []
            success = any(
                getattr(item, "retrieved_url", None) == url
                and str(getattr(item, "url_retrieval_status", "")).split(".")[-1]
                == "URL_RETRIEVAL_STATUS_SUCCESS"
                for item in retrieved
            )
            text = _text(response).strip()
            currency = self.price_spec.price_currency
            if not success or not text:
                raise WebError(
                    "url_context_unverified" if not success else "empty_content",
                    cost,
                    list_cost=list_cost(self.price_spec, usage),
                    list_currency=currency,
                )
            page = ContextPage(title=url, content=text, evidence_kind="url_context_summary")
            page.usage, page.cost_cny = usage, cost
            page.list_cost, page.list_currency = list_cost(self.price_spec, usage), currency
            return page
        if self.fetcher is None:
            raise WebError("fetch_unavailable")
        return await self.fetcher.fetch(url, timeout)


class ContextPage(dict):
    """A verified URL-context extraction with response accounting attached."""

    usage: Usage
    cost_cny: float
    list_cost: float
    list_currency: str
