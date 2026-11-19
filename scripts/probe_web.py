"""Probe web search options for the live agent (WEB-01): Gemini built-in tools and DeepSeek.

Usage:
    python scripts/probe_web.py gemini [--budget-cny 0.05]
    python scripts/probe_web.py deepseek [--budget-cny 0.05]
    python scripts/probe_web.py gemini-isolated [--list-models] [--reps 2]
        [--extra-models a,b] [--only substring] [--budget-cny 0.05]

gemini-isolated (re-check 2, relay vendor advice): google-genai SDK, one request per call
that carries only `Tool(google_search=GoogleSearch())` -- no function tools, no response
schema / MIME type; variants with/without system_instruction and thinking LOW / HIGH /
default; serial with 429 backoff. Records include answers and grounding sources
(data/probes/web-gemini-isolated-*.json, relay host and keys redacted).

gemini: Pydantic AI `WebSearchTool` (google_search grounding) and `WebFetchTool`
(url_context) on the configured Gemini model through the relay, alone and combined with
a function tool / native structured output; plus one raw REST call with and without
google_search to compare `usageMetadata` (per-search fees, tool-use prompt tokens).
deepseek: the official DeepSeek `/v1/tools/search` and `/v1/tools/fetch` endpoints with
DEEPSEEK_SEARCH_API_KEY (same request shape as the legacy AstrBot tool).

Keys, the relay URL and host are never printed; error texts are reduced to class names
and status codes. Records (usage, latency, source counts, no keys) go to the git-ignored
data/probes/ directory. Findings: docs/MODEL-ACCESS.md.
"""

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group_agent.config import (  # noqa: E402
    _read_env,
    load_model_credentials,
    load_settings,
)
from group_agent.models import Usage, _pai_usage, _redacted_error, cost_cny  # noqa: E402

SEARCH_QUESTIONS = [
    "2026 年英雄联盟全球总决赛的冠军是哪支队伍？",
    "Python 目前最新的稳定版本号是多少？",
    "《原神》现在最新的版本号是多少，什么时候上线的？",
    "DeepSeek 最近一次发布的模型叫什么？",
    "《黑神话：钟馗》正式发售了吗？",
]
FETCH_URL = "https://www.python.org/downloads/"


def record_sources(messages) -> tuple[list[dict], list[str], list[str]]:
    """Sources (title/uri) and queries from native tool return parts; part kinds trace."""
    sources, queries, kinds = [], [], []
    for message in messages:
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", type(part).__name__)
            kinds.append(f"{kind}:{getattr(part, 'tool_name', '') or ''}")
            if kind == "native-tool-call":
                args = getattr(part, "args", None)
                if isinstance(args, dict):
                    queries += [str(q) for q in args.get("queries") or []]
            if kind == "native-tool-return":
                content = getattr(part, "content", None)
                items = content if isinstance(content, list) else []
                for item in items:
                    if isinstance(item, dict):
                        sources.append(
                            {
                                "title": str(item.get("title") or "")[:80],
                                "domain": str(item.get("domain") or "")[:60],
                                "has_uri": bool(item.get("uri") or item.get("url")),
                                "status": item.get("url_retrieval_status"),
                            }
                        )
    return sources, queries, kinds


async def gemini_probe(args, settings, out: list[dict]) -> None:
    from pydantic import BaseModel
    from pydantic_ai import Agent, NativeOutput, WebFetchTool, WebSearchTool, capture_run_messages
    from pydantic_ai.capabilities import NativeTool

    from group_agent.models import build_pai_model

    spec = settings.models["gemini"]
    creds = load_model_credentials(settings.credentials, spec)
    model = build_pai_model(spec, creds)
    base = {"timeout": 90, "max_tokens": 4096, "google_thinking_config": {"thinking_level": "LOW"}}
    spent = 0.0

    class Verdict(BaseModel):
        answer: str
        confident: bool

    def group_size() -> int:
        """查询本群当前人数"""
        return 42

    async def run(name, prompt, *, tools=(), output=str, function_tools=()):
        nonlocal spent
        if spent > args.budget_cny:
            out.append({"probe": name, "status": "skipped_budget"})
            return
        agent = Agent(
            model,
            output_type=output,
            instructions="用中文简短回答，并说明依据。",
            capabilities=[NativeTool(t) for t in tools],
            tools=list(function_tools),
            model_settings=base,
        )
        started = time.perf_counter()
        row = {"probe": name, "status": "ok"}
        result = None
        with capture_run_messages() as messages:
            try:
                result = await agent.run(prompt)
            except Exception as exc:  # noqa: BLE001 - redacted
                row.update(status="error", error=_redacted_error(exc))
            msgs = result.all_messages() if result is not None else list(messages)
        usage, responses = _pai_usage(msgs, True)
        sources, queries, kinds = record_sources(msgs)
        row.update(
            latency_s=round(time.perf_counter() - started, 2),
            usage=usage.__dict__,
            responses=responses,
            cost_cny=round(cost_cny(spec, usage), 6),
            sources=len(sources),
            source_sample=sources[:3],
            queries=len(queries),
            kinds=kinds,
            answer_chars=len(str(result.output)) if result is not None else 0,
        )
        if result is not None and output is not str:
            row["structured_ok"] = isinstance(result.output, Verdict)
        if result is not None and output is str:
            row["answer_head"] = str(result.output)[:60]  # public-fact probes only
        spent += row["cost_cny"]
        out.append(row)
        print(json.dumps({k: row[k] for k in row if k not in ("kinds",)}, ensure_ascii=False))

    for i, q in enumerate(SEARCH_QUESTIONS):
        await run(f"search_{i + 1}", q, tools=[WebSearchTool()])
    await run("baseline_no_tool", SEARCH_QUESTIONS[0])
    await run(
        "search+native_output",
        SEARCH_QUESTIONS[1],
        tools=[WebSearchTool()],
        output=NativeOutput(Verdict),
    )
    await run(
        "search+function_tool",
        "本群现在多少人？另外 Python 最新稳定版本号是多少？",
        tools=[WebSearchTool()],
        function_tools=[group_size],
    )
    await run(
        "url_context",
        f"读取 {FETCH_URL} ，告诉我页面上列出的最新 Python 版本号。",
        tools=[WebFetchTool()],
    )
    await run(
        "search+url_context",
        f"读取 {FETCH_URL} 并联网确认 Python 最新稳定版本号。",
        tools=[WebSearchTool(), WebFetchTool()],
    )
    client = getattr(model, "client", None)
    aio = getattr(client, "aio", None)
    if aio is not None:
        await aio.aclose()

    # Raw REST: compare usageMetadata and billing-like fields with and without search.
    import httpx

    url = f"{creds.base_url}/{spec.api_version}/models/{spec.model}:generateContent"
    for name, tools in (("raw_no_tool", None), ("raw_google_search", [{"google_search": {}}])):
        if spent > args.budget_cny:
            break
        body = {
            "contents": [{"role": "user", "parts": [{"text": SEARCH_QUESTIONS[0]}]}],
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": "LOW"},
                "maxOutputTokens": 4096,
            },
        }
        if tools:
            body["tools"] = tools
        started = time.perf_counter()
        row = {"probe": name}
        try:
            async with httpx.AsyncClient(timeout=90) as http:
                r = await http.post(url, json=body, headers={"x-goog-api-key": creds.api_key})
            row["http_status"] = r.status_code
            data = r.json() if r.status_code == 200 else {}
        except Exception as exc:  # noqa: BLE001
            row["error"], data = _redacted_error(exc), {}
        meta = data.get("usageMetadata") or {}
        cand = (data.get("candidates") or [{}])[0]
        grounding = cand.get("groundingMetadata") or {}
        thoughts = int(meta.get("thoughtsTokenCount") or 0)
        usage = Usage(
            input=int(meta.get("promptTokenCount") or 0)
            + int(meta.get("toolUsePromptTokenCount") or 0),
            cached=int(meta.get("cachedContentTokenCount") or 0),
            output=int(meta.get("candidatesTokenCount") or 0) + thoughts,
            thinking=thoughts,
            requests=1,
        )
        row.update(
            latency_s=round(time.perf_counter() - started, 2),
            top_level_keys=sorted(data),
            usage_metadata=meta,
            grounding_keys=sorted(grounding),
            web_search_queries=len(grounding.get("webSearchQueries") or []),
            grounding_chunks=len(grounding.get("groundingChunks") or []),
            billing_headers=sorted(
                k
                for k in r.headers
                if any(w in k.lower() for w in ("cost", "bill", "quota", "price"))
            )
            if "http_status" in row
            else [],
            cost_cny=round(cost_cny(spec, usage), 6),
        )
        spent += row["cost_cny"]
        out.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print(f"gemini probe spent ~{spent:.4f} CNY (tokens x configured price)")


async def deepseek_probe(args, settings, out: list[dict]) -> None:
    import httpx

    key = _read_env(settings.secrets_file).get("DEEPSEEK_SEARCH_API_KEY")
    if not key:
        raise SystemExit("secrets_file.DEEPSEEK_SEARCH_API_KEY: required")
    async with httpx.AsyncClient(base_url="https://api.deepseek.com", timeout=30) as http:
        calls = [
            ("search", "/v1/tools/search", {"query": q, "count": 5, "need_content": False})
            for q in SEARCH_QUESTIONS[:3]
        ] + [("fetch", "/v1/tools/fetch", {"url": FETCH_URL})]
        for kind, path, body in calls:
            started = time.perf_counter()
            row = {"probe": f"deepseek_{kind}"}
            try:
                r = await http.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
                row["http_status"] = r.status_code
                data = r.json() if r.status_code == 200 else {}
            except Exception as exc:  # noqa: BLE001
                row["error"], data = _redacted_error(exc), {}
            results = data.get("results") or []
            row.update(
                latency_s=round(time.perf_counter() - started, 2),
                top_level_keys=sorted(data),
                result_keys=sorted(results[0]) if results else [],
                results=len(results),
                titles=[str(x.get("title") or "")[:50] for x in results[:3]],
                usage=data.get("usage"),
                content_chars=len(str(data.get("content") or "")),
            )
            out.append(row)
            print(json.dumps(row, ensure_ascii=False))


ISOLATED_QUESTIONS = [
    "今天（2026 年 11 月 8 日）国内外有哪些重要新闻？列出 3 条。",
    "Python 目前最新的稳定版本号是多少？什么时候发布的？",
    "《原神》当前最新版本是多少，什么时候上线的？",
    "2026 年英雄联盟全球总决赛现在进行到哪个阶段了？",
]
ISOLATED_SYSTEM = "你是检索助手。先用 Google 搜索核实，再用中文简洁回答，并说明依据。"


def _grounding_row(candidate) -> dict:
    """Grounding fields of one candidate (sources keep the URI as returned)."""
    g = getattr(candidate, "grounding_metadata", None) if candidate is not None else None
    if g is None:
        return {"grounding": False}
    chunks = []
    for chunk in g.grounding_chunks or []:
        web = getattr(chunk, "web", None)
        if web is not None:
            chunks.append({"title": web.title, "domain": web.domain, "uri": web.uri})
    supports = []
    for s in g.grounding_supports or []:
        seg = getattr(s, "segment", None)
        supports.append(
            {
                "text": (getattr(seg, "text", "") or "")[:80],
                "chunks": list(s.grounding_chunk_indices or []),
            }
        )
    return {
        "grounding": True,
        "grounding_fields": sorted(g.model_dump(exclude_none=True)),
        "web_search_queries": list(g.web_search_queries or []),
        "chunks": chunks,
        "supports": len(supports),
        "support_sample": supports[:3],
        "search_entry_point": g.search_entry_point is not None,
    }


def gemini_isolated_probe(args, settings, out: list[dict]) -> None:
    """Built-in google_search alone in its own request (relay vendor advice): no function
    tools, no response schema / MIME type; serial calls with 429 backoff."""
    from google import genai
    from google.genai import errors, types

    spec = settings.models["gemini"]
    creds = load_model_credentials(settings.credentials, spec)
    client = genai.Client(
        api_key=creds.api_key,
        http_options=types.HttpOptions(
            base_url=creds.base_url, api_version=spec.api_version or "v1beta", timeout=90_000
        ),
    )
    if args.list_models:
        names = sorted(
            m.name.split("/")[-1] for m in client.models.list() if "gemini" in (m.name or "")
        )
        print(json.dumps(names, ensure_ascii=False))
        out.append({"probe": "list_models", "gemini_models": names})
        return
    main = args.main_model or spec.model
    models = [main] + [m for m in (args.extra_models or "").split(",") if m.strip()]
    # (name, model, system_instruction?, thinking level, google_search?); the "notool" row is
    # a control: same questions without any tool, to tell searched facts from model memory.
    variants = [
        (f"{main}|nosys|LOW", main, False, "LOW", True),
        (f"{main}|sys|LOW", main, True, "LOW", True),
        (f"{main}|nosys|HIGH", main, False, "HIGH", True),
        (f"{main}|sys|HIGH", main, True, "HIGH", True),
        (f"{main}|nosys|default", main, False, None, True),
        (f"{main}|notool|LOW", main, False, "LOW", False),
    ] + [(f"{m.strip()}|nosys|default", m.strip(), False, None, True) for m in models[1:]]
    if args.only:
        variants = [v for v in variants if args.only in v[0]]
    spent, n = 0.0, 0
    for rep in range(args.reps):
        for vi, (name, model, with_sys, level, search) in enumerate(variants):
            question = ISOLATED_QUESTIONS[(vi + rep) % len(ISOLATED_QUESTIONS)]
            if spent > args.budget_cny:
                out.append({"variant": name, "rep": rep + 1, "status": "skipped_budget"})
                continue
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())] if search else None,
                max_output_tokens=4096,
                system_instruction=ISOLATED_SYSTEM if with_sys else None,
                thinking_config=types.ThinkingConfig(thinking_level=level) if level else None,
                # Raw body, so fields the SDK might not map are still visible.
                should_return_http_response=True,
            )
            row = {
                "variant": name,
                "rep": rep + 1,
                "model": model,
                "question": question,
                "request_config": config.model_dump(
                    exclude_none=True, mode="json", exclude={"should_return_http_response"}
                ),
            }
            response, attempts = None, 0
            started = time.perf_counter()
            for delay in (0, 8, 20, 45):
                if delay:
                    time.sleep(delay)
                attempts += 1
                started = time.perf_counter()
                try:
                    raw_response = client.models.generate_content(
                        model=model, contents=question, config=config
                    )
                    body = raw_response.sdk_http_response.body or "{}"
                    raw = json.loads(body)
                    raw_cand = (raw.get("candidates") or [{}])[0]
                    row["raw_top_keys"] = sorted(raw)
                    row["raw_candidate_keys"] = sorted(raw_cand)
                    # The relay now returns serviceTier, which this SDK's strict usage
                    # schema does not know. Preserve it in the probe, then parse usage.
                    raw_usage = raw.get("usageMetadata") or {}
                    if "serviceTier" in raw_usage:
                        row["service_tier"] = raw_usage.pop("serviceTier")
                    response = types.GenerateContentResponse.model_validate(raw)
                    row["status"] = "ok"
                    break
                except errors.APIError as exc:
                    row.update(status="error", error=f"{type(exc).__name__}:{exc.code}")
                    if exc.code != 429:
                        break
                except Exception as exc:  # noqa: BLE001 - redacted
                    row.update(status="error", error=_redacted_error(exc))
                    break
            row["attempts"] = attempts
            row["latency_s"] = round(time.perf_counter() - started, 2)
            if response is not None:
                cand = (response.candidates or [None])[0]
                meta = response.usage_metadata
                meta_dict = meta.model_dump(exclude_none=True, mode="json") if meta else {}
                thoughts = int(meta_dict.get("thoughts_token_count") or 0)
                usage = Usage(
                    input=int(meta_dict.get("prompt_token_count") or 0)
                    + int(meta_dict.get("tool_use_prompt_token_count") or 0),
                    cached=int(meta_dict.get("cached_content_token_count") or 0),
                    output=int(meta_dict.get("candidates_token_count") or 0) + thoughts,
                    thinking=thoughts,
                    requests=1,
                )
                row.update(
                    model_version=response.model_version,
                    finish_reason=str(getattr(cand, "finish_reason", None)),
                    candidate_fields=sorted(cand.model_dump(exclude_none=True)) if cand else [],
                    usage_metadata=meta_dict,
                    cost_cny=round(cost_cny(spec, usage), 6),
                    **_grounding_row(cand),
                    text=response.text or "",
                )
                spent += row["cost_cny"]
            n += 1
            out.append(row)
            brief = {
                k: row.get(k)
                for k in ("variant", "rep", "status", "error", "latency_s", "grounding")
            }
            brief.update(
                queries=len(row.get("web_search_queries") or []),
                chunks=len(row.get("chunks") or []),
                tool_tokens=(row.get("usage_metadata") or {}).get("tool_use_prompt_token_count"),
                cost=row.get("cost_cny"),
            )
            print(json.dumps(brief, ensure_ascii=False), flush=True)
            time.sleep(1)
    print(f"gemini-isolated: {n} calls, ~{spent:.4f} CNY (tokens x configured price)")


def _redact_record(text: str, settings) -> str:
    """Remove the relay URL/host and keys from a serialised record."""
    values = _read_env(settings.secrets_file)
    secrets = [v for k, v in values.items() if v and ("KEY" in k or "TOKEN" in k or "URL" in k)]
    for name in ("RELAY_BASE_URL",):
        url = values.get(name, "")
        host = url.split("://", 1)[-1].split("/", 1)[0]
        if host:
            secrets.append(host)
    for value in sorted(set(secrets), key=len, reverse=True):
        if len(value) >= 6:
            text = text.replace(value, "<redacted>")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["gemini", "gemini-isolated", "deepseek"])
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.toml")
    parser.add_argument("--budget-cny", type=float, default=0.05)
    parser.add_argument("--reps", type=int, default=2, help="gemini-isolated: rounds")
    parser.add_argument("--extra-models", default="", help="gemini-isolated: comma list")
    parser.add_argument("--only", default="", help="gemini-isolated: variant substring")
    parser.add_argument("--main-model", default="", help="gemini-isolated: model of main set")
    parser.add_argument("--list-models", action="store_true", help="gemini-isolated")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    out: list[dict] = []
    if args.target == "gemini-isolated":
        try:
            gemini_isolated_probe(args, settings, out)
        finally:
            path = ROOT / f"data/probes/web-gemini-isolated-{datetime.now():%Y%m%d-%H%M%S}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(out, ensure_ascii=False, indent=2, default=str)
            path.write_text(_redact_record(text, settings), encoding="utf-8")
            print(f"records: {path.relative_to(ROOT)}")
        return 0
    probe = gemini_probe if args.target == "gemini" else deepseek_probe
    try:
        asyncio.run(probe(args, settings, out))
    finally:
        path = ROOT / f"data/probes/web-{args.target}-{datetime.now():%Y%m%d-%H%M%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"records: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
