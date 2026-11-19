"""Framework-level Vertex AI capability probe (task V4); synthetic public content only.

Unlike scripts/probe_vertex.py (raw SDK), every case here runs the real framework path:
`LiveModel` (Pydantic AI + `provider_adapters.build_pai_model`) for generation and
`GeminiSearchClient` for the independent Google search / URL context requests. Cases run
strictly one after another under one shared `Ledger(budget)`; the first `BudgetExceeded`
stops the remaining cases. No gateway, QQ or store module is imported.

Real run (needs the configured credentials file with the Vertex key and project):

    .\\.venv\\Scripts\\python.exe scripts/probe_vertex_framework.py --budget-cny 2

Offline self-test (no credentials, no network; Pydantic AI FunctionModel and a fake
google-genai client):

    .\\.venv\\Scripts\\python.exe scripts/probe_vertex_framework.py --budget-cny 1 --dry

Records go to data/probes/vertex-framework-<UTC>/results.json (git-ignored). A record holds
only class names, HTTP status codes and WebError kinds: never exception text, URLs with
keys, keys or the project id. The exit code is 0 when every required case passed;
flash_url_context is optional (it decides whether URL context becomes a fetch backend).
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group_agent.billing import open_account_book, open_google_quota  # noqa: E402
from group_agent.config import (  # noqa: E402
    ConfigError,
    ModelCredentials,
    ModelSpec,
    load_model_credentials,
    load_settings,
)
from group_agent.gemini_search import GeminiSearchClient  # noqa: E402
from group_agent.models import (  # noqa: E402
    BudgetExceeded,
    CallRecord,
    ImagePart,
    Ledger,
    LiveModel,
    ModelCallError,
    Usage,
)
from group_agent.provider_adapters import build_pai_model  # noqa: E402
from group_agent.web import WebError  # noqa: E402

CASES = (
    "answer_text",
    "answer_image",
    "answer_tool_roundtrip",
    "answer_structured",
    "flash_structured",
    "flash_search",
    "flash_url_context",
)
OPTIONAL = frozenset({"flash_url_context"})
FLASH_CASES = frozenset({"flash_structured", "flash_search", "flash_url_context"})

TOOL_CODE = "ZX-4471"
SEARCH_QUERY = "Python 3.13 release date"
URL_CONTEXT_URL = "https://www.python.org/downloads/"
PROBE_INSTRUCTIONS = "You are a capability probe. Answer briefly and follow the request exactly."
STRUCTURED_REQUEST = (
    "<request>\nWhen was Python 3.13 released? Answer for a public chat.\n</request>"
)
DECISION_INSTRUCTIONS = (
    "You route one chat question. Fill the schema only; do not answer the question. "
    "Use route web when the answer needs an external fact that may have changed."
)
DRY_CREDENTIALS = ModelCredentials("", "dry-run-api-key-0000", "dry-run-project-0000")


class ProbeDecision(BaseModel):
    """Routing decision of a chat question.

    Mirrors the shape of live_agent.RouteDecision that matters for native schema output
    (Literal enum, bools, a length-limited list, extra="forbid" -> additionalProperties
    false). live_agent itself is not imported: it pulls in the gateway/outbox modules.
    """

    model_config = ConfigDict(extra="forbid")
    route: Literal["chat", "local_knowledge", "web", "clarify", "teaching"] = Field(
        description="Handling route"
    )
    should_clarify: bool = Field(description="Whether to ask the user a question first")
    clarify_question: str = Field(default="", description="One short question, else empty")
    search_terms: list[str] = Field(default_factory=list, max_length=3, description="Up to 3")
    fact_check: bool = Field(default=False, description="Whether a real-world fact needs checking")
    time_sensitive: bool = Field(default=False, description="Whether the answer changes over time")
    reason: str = Field(description="One-sentence reason, at most 40 characters")


# -- synthetic inputs ------------------------------------------------------------------


def red_png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(out, format="PNG")
    return out.getvalue()


# -- offline fakes (--dry) -------------------------------------------------------------


def _dry_respond(messages: list, info: AgentInfo) -> ModelResponse:
    usage = RequestUsage(input_tokens=400, output_tokens=60, details={"thoughts_tokens": 20})
    if info.model_request_parameters.output_mode == "native":
        body = {"route": "web", "should_clarify": False, "reason": "dry-run decision"}
        return ModelResponse(parts=[TextPart(json.dumps(body))], usage=usage)
    if info.function_tools:
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            call = ToolCallPart(info.function_tools[0].name, {"name": "widget"}, "dry-call")
            return ModelResponse(parts=[call], usage=usage)
        return ModelResponse(parts=[TextPart(f"The code is {returns[-1].content}.")], usage=usage)
    prompts = [p for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
    has_image = any(
        isinstance(item, BinaryContent)
        for p in prompts
        if not isinstance(p.content, str)
        for item in p.content
    )
    return ModelResponse(parts=[TextPart("Red" if has_image else "OK")], usage=usage)


def dry_model_builder(spec: ModelSpec, creds: Any) -> FunctionModel:
    return FunctionModel(_dry_respond)


def _dry_response(text: str, **candidate: Any) -> SimpleNamespace:
    parts = [SimpleNamespace(text=text, thought=False)]
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts), **candidate)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=300,
            tool_use_prompt_token_count=0,
            cached_content_token_count=0,
            candidates_token_count=80,
            thoughts_token_count=20,
        ),
    )


class DryGenaiClient:
    """Stands in for `google.genai.Client`: only `aio.models.generate_content` and `aclose`."""

    def __init__(self) -> None:
        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=self._generate), aclose=self._aclose
        )

    async def _aclose(self) -> None:
        return None

    async def _generate(self, *, model: str, contents: str, config: Any) -> SimpleNamespace:
        tool = config.tools[0]
        if getattr(tool, "url_context", None) is not None:
            meta = SimpleNamespace(
                url_metadata=[
                    SimpleNamespace(
                        retrieved_url=URL_CONTEXT_URL,
                        url_retrieval_status="URL_RETRIEVAL_STATUS_SUCCESS",
                    )
                ]
            )
            return _dry_response(
                "Python downloads page: latest releases.", url_context_metadata=meta
            )
        chunks = [
            SimpleNamespace(
                web=SimpleNamespace(
                    uri=f"https://example.com/{n}", title="Example", domain="example.com"
                )
            )
            for n in (1, 2)
        ]
        grounding = SimpleNamespace(
            grounding_chunks=chunks,
            grounding_supports=[
                SimpleNamespace(
                    segment=SimpleNamespace(text="Python 3.13 was released in 2024."),
                    grounding_chunk_indices=[0, 1],
                )
            ],
            web_search_queries=["python 3.13 release date"],
        )
        return _dry_response(
            "Python 3.13 was released in October 2024.", grounding_metadata=grounding
        )


def dry_client_factory(timeout: float) -> DryGenaiClient:
    return DryGenaiClient()


# -- configuration ---------------------------------------------------------------------


def dry_spec(alias: str, model: str, *, timeout: float, tokens: int, **prices: float) -> ModelSpec:
    return ModelSpec(
        alias,
        "google-genai",
        model,
        "",
        "DRY_API_KEY_ENV",
        timeout,
        thinking_level="LOW",
        max_output_tokens=tokens,
        vertex=True,
        location="global",
        **prices,
    )


def resolve_models(args: argparse.Namespace) -> tuple[ModelSpec, Any, ModelSpec, Any]:
    """(answer spec, creds, flash spec, creds). `--dry` never reads the secrets file; it
    falls back to synthetic specs when the settings file is absent."""
    settings_path = Path(args.settings)
    if args.dry and not settings_path.is_file():
        return (
            dry_spec(
                "answer", "dry-answer", timeout=90, tokens=8192, price_input=10.8, price_output=54.0
            ),
            DRY_CREDENTIALS,
            dry_spec(
                "flash", "dry-flash", timeout=30, tokens=2048, price_input=5.4, price_output=27.0
            ),
            DRY_CREDENTIALS,
        )
    settings = load_settings(settings_path)
    specs = []
    for alias in (args.answer_alias, args.flash_alias):
        if alias not in settings.models:
            raise ConfigError(f"models.{alias}: not defined in settings")
        specs.append(settings.models[alias])
    answer, flash = specs
    if flash.protocol != "google-genai":
        raise ConfigError(f"models.{flash.name}: search and URL context need google-genai")
    if args.dry:
        return answer, DRY_CREDENTIALS, flash, DRY_CREDENTIALS
    return (
        answer,
        load_model_credentials(settings.credentials, answer),
        flash,
        load_model_credentials(settings.credentials, flash),
    )


# -- case plumbing ---------------------------------------------------------------------


@dataclass
class Context:
    ledger: Ledger
    timeout_s: float
    fee_per_query: float
    answer_spec: ModelSpec
    flash_spec: ModelSpec
    answer: LiveModel
    flash: LiveModel
    flash_creds: Any
    client_factory: Callable[[float], Any] | None

    def search_client(self, *, use_url_context: bool) -> GeminiSearchClient:
        return GeminiSearchClient(
            self.flash_spec,
            self.flash_creds,
            model=self.flash_spec.model,
            thinking_level=self.flash_spec.thinking_level or "LOW",
            fee_per_query=self.fee_per_query,
            use_url_context=use_url_context,
            client_factory=self.client_factory,
        )


def call_error_kind(record: CallRecord | None, fallback: str) -> str:
    """`http_<code>` or a class name from the redacted `CallRecord.error` ("Name:code")."""
    error = record.error if record is not None else None
    if not error:
        return fallback
    name, _, status = error.partition(":")
    return f"http_{status}" if status.isdigit() else name


def error_kind(exc: BaseException) -> str:
    if isinstance(exc, WebError):
        return exc.kind
    if isinstance(exc, BudgetExceeded):
        return "BudgetExceeded"
    if isinstance(exc, ModelCallError):
        return call_error_kind(exc.record, exc.kind)
    if isinstance(exc, TimeoutError):
        return "TimeoutError"
    return type(exc).__name__


def fill_call(rec: dict, live: LiveModel, record: CallRecord) -> None:
    rec["usage"] = asdict(record.usage)
    rec["call_mode"] = record.mode
    rec["responses"] = record.responses
    rec["thinking_level"] = record.thinking_level
    rec["error_causes"] = list(record.error_causes)
    try:
        rec["provider"] = getattr(live.model, "system", None)
    except Exception:  # noqa: BLE001 - provider name is informational only
        rec["provider"] = None


def preview(text: str) -> str:
    return " ".join(text.split())[:120]


async def metered_web(
    ctx: Context,
    task: str,
    estimate: float,
    call: Callable[[float], Any],
    service: str = "search",
    unit: tuple[str, float] = ("CNY", 1.0),
) -> Any:
    """One independent search/fetch request: reserve, run, settle the actual cost (like
    `web.WebTool._call`: a failed request without a known cost bills the estimate)."""
    ctx.ledger.reserve(estimate)
    started = time.perf_counter()
    result, error, cost = None, None, None
    accounting: Any = None  # result or WebError: list cost and Google query count
    channel = ctx.flash_spec.channel
    try:
        async with asyncio.timeout(ctx.timeout_s):
            result = await call(ctx.timeout_s)
        cost, accounting = getattr(result, "cost_cny", None), result
    except TimeoutError:
        error = "timeout"
    except WebError as exc:
        error, cost, accounting = exc.kind, exc.cost_cny, exc
    except asyncio.CancelledError:
        ctx.ledger.settle(
            estimate,
            CallRecord(
                task,
                ctx.flash_spec.model,
                "http",
                "cancelled",
                "CancelledError",
                cost_cny=estimate,
                channel=channel,
                service=service,
                list_cost=estimate / unit[1],
                list_currency=unit[0],
            ),
        )
        raise
    except Exception as exc:  # noqa: BLE001 - recorded by class name only
        error = type(exc).__name__
    record = CallRecord(
        task,
        ctx.flash_spec.model,
        "http",
        status="ok" if error is None else "error",
        error=error,
        latency_s=time.perf_counter() - started,
        cost_cny=estimate if cost is None else cost,
        responses=1,
        channel=channel,
        service=service,
        google_queries=getattr(accounting, "google_queries", None),
    )
    listed = getattr(accounting, "list_cost", None)
    record.list_cost = record.cost_cny / unit[1] if listed is None else listed
    record.list_currency = getattr(accounting, "list_currency", None) or unit[0]
    usage = getattr(result, "usage", None)
    if isinstance(usage, Usage):
        record.usage = usage
    ctx.ledger.settle(estimate, record)
    if error is not None:
        raise WebError(error, cost)
    return result


# -- cases -----------------------------------------------------------------------------


async def case_answer_text(ctx: Context, rec: dict) -> None:
    text, record = await ctx.answer.generate(
        "probe_text", PROBE_INSTRUCTIONS, ["Reply with exactly OK."], ctx.timeout_s
    )
    fill_call(rec, ctx.answer, record)
    rec["answer"] = preview(text)
    rec["passed"] = bool(text.strip())
    rec["check"] = None if rec["passed"] else "empty_answer"


async def case_answer_image(ctx: Context, rec: dict) -> None:
    parts = ["What colour is this image? One word.", ImagePart(red_png(), "image/png")]
    text, record = await ctx.answer.generate(
        "probe_image", PROBE_INSTRUCTIONS, parts, ctx.timeout_s
    )
    fill_call(rec, ctx.answer, record)
    rec["answer"] = preview(text)
    rec["passed"] = "red" in text.lower() or "红" in text
    rec["check"] = None if rec["passed"] else "red_not_mentioned"


async def case_answer_tool_roundtrip(ctx: Context, rec: dict) -> None:
    def lookup_code(name: str) -> str:
        """Return the access code registered for a synthetic part name."""
        rec["tool_calls"] += 1
        rec["tool_called"] = True
        return TOOL_CODE

    prompt = "Call the lookup_code tool with name 'widget', then report the code it returned."
    # Same call shape as live_agent's BoundCall: tools + request / tool-call limits.
    text, record = await ctx.answer.generate(
        "probe_tool",
        PROBE_INSTRUCTIONS,
        [prompt],
        ctx.timeout_s,
        tools=[lookup_code],
        max_requests=6,
        max_tool_calls=8,
    )
    fill_call(rec, ctx.answer, record)
    rec["answer"] = preview(text)
    rec["passed"] = rec["tool_called"] and TOOL_CODE in text
    rec["check"] = (
        None if rec["passed"] else ("tool_not_called" if not rec["tool_called"] else "code_missing")
    )


async def structured(ctx: Context, live: LiveModel, rec: dict) -> None:
    # live_agent._structured_decision: native schema output when the protocol has it.
    mode = "native" if live.supports_native_output else "tool"
    output, record = await live.structured(
        "probe_structured",
        DECISION_INSTRUCTIONS,
        [STRUCTURED_REQUEST],
        ProbeDecision,
        mode,
        ctx.timeout_s,
        tool_name="submit_route",
        tool_description="Submit the routing decision",
    )
    fill_call(rec, live, record)
    rec["first_try"] = record.first_try
    rec["passed"] = isinstance(output, ProbeDecision)
    rec["route"] = output.route if rec["passed"] else None
    rec["check"] = None if rec["passed"] else "not_parsed"


async def case_answer_structured(ctx: Context, rec: dict) -> None:
    await structured(ctx, ctx.answer, rec)


async def case_flash_structured(ctx: Context, rec: dict) -> None:
    await structured(ctx, ctx.flash, rec)


async def case_flash_search(ctx: Context, rec: dict) -> None:
    client = ctx.search_client(use_url_context=False)
    results = await metered_web(
        ctx,
        "probe_search",
        client.search_estimate_cny,
        lambda timeout: client.search(SEARCH_QUERY, timeout),
        "search",
        client.price_unit("search"),
    )
    rec["usage"] = asdict(results.usage)
    rec["call_mode"] = "http"
    rec["sources"], rec["queries"] = len(results), len(results.queries)
    rec["passed"] = len(results) > 0
    rec["check"] = None if rec["passed"] else "no_sources"


async def case_flash_url_context(ctx: Context, rec: dict) -> None:
    client = ctx.search_client(use_url_context=True)
    page = await metered_web(
        ctx,
        "probe_url_context",
        client.fetch_estimate_cny or 0.0,
        lambda timeout: client.fetch(URL_CONTEXT_URL, timeout),
        "fetch",
        client.price_unit("fetch"),
    )
    rec["usage"] = asdict(page.usage)
    rec["call_mode"] = "http"
    rec["content_chars"] = len(page.get("content", ""))
    rec["passed"] = rec["content_chars"] > 0
    rec["check"] = None if rec["passed"] else "empty_content"


CASE_FUNCTIONS = {
    "answer_text": case_answer_text,
    "answer_image": case_answer_image,
    "answer_tool_roundtrip": case_answer_tool_roundtrip,
    "answer_structured": case_answer_structured,
    "flash_structured": case_flash_structured,
    "flash_search": case_flash_search,
    "flash_url_context": case_flash_url_context,
}


def new_record(name: str, ctx: Context) -> dict:
    spec = ctx.flash_spec if name in FLASH_CASES else ctx.answer_spec
    rec: dict[str, Any] = {
        "case": name,
        "alias": spec.name,
        "model": spec.model,
        "mode": "vertex" if spec.vertex else "relay",
        "optional": name in OPTIONAL,
        "passed": False,
        "skipped": False,
        "latency_s": None,
        "usage": None,
        "cost_cny": 0.0,
        "error": None,
        "check": None,
        "provider": None,
    }
    if name == "answer_tool_roundtrip":
        rec.update(tool_called=False, tool_calls=0)
    if name == "flash_search":
        rec.update(sources=0, queries=0)
    return rec


async def run_case(name: str, ctx: Context) -> tuple[dict, bool]:
    """(record, budget_exhausted). Failures become an `error` kind, never exception text."""
    rec = new_record(name, ctx)
    spent_before, started = ctx.ledger.spent, time.perf_counter()
    exhausted = False
    try:
        # Outer guard above the per-call timeouts, so a hang cannot stall the probe.
        async with asyncio.timeout(ctx.timeout_s + 15):
            await CASE_FUNCTIONS[name](ctx, rec)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified by kind below
        rec["passed"], rec["error"] = False, error_kind(exc)
        exhausted = isinstance(exc, BudgetExceeded)
        record = getattr(exc, "record", None)
        if isinstance(record, CallRecord):
            rec["usage"] = asdict(record.usage)
            rec["call_mode"] = record.mode
            rec["error_causes"] = list(record.error_causes)
    rec["latency_s"] = round(time.perf_counter() - started, 3)
    rec["cost_cny"] = round(ctx.ledger.spent - spent_before, 7)
    return rec, exhausted


def skipped_record(name: str, ctx: Context) -> dict:
    rec = new_record(name, ctx)
    rec.update(skipped=True, error="budget_stop", check="not_run")
    return rec


# -- driver ----------------------------------------------------------------------------


def summarize(records: list[dict]) -> dict:
    return {
        "passed": sum(1 for r in records if r["passed"]),
        "failed": sum(1 for r in records if not r["passed"] and not r["skipped"]),
        "skipped": sum(1 for r in records if r["skipped"]),
    }


def write_results(out: Path, results: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def run_probe(
    args: argparse.Namespace,
    *,
    ledger: Ledger | None = None,
    model_builder: Callable[[ModelSpec, Any], Any] | None = None,
    client_factory: Callable[[float], Any] | None = None,
) -> dict:
    """Run the selected cases serially; returns the results dict (also written to `out`)."""
    answer_spec, answer_creds, flash_spec, flash_creds = resolve_models(args)
    if args.dry:
        model_builder = model_builder or dry_model_builder
        client_factory = client_factory or dry_client_factory
    build = model_builder or build_pai_model
    book = None
    if ledger is None:
        # Probes spend the real account, so real runs append to the account book; --dry and
        # --no-account-book write nowhere.
        quota = None
        if not args.dry and args.account_book:
            billing_settings = load_settings(Path(args.settings))
            book = open_account_book(billing_settings, "probe")
            quota = open_google_quota(billing_settings, book)  # shares the free monthly quota
        ledger = Ledger(
            args.budget_cny, account=book, source="probe" if book else None, quota=quota
        )
    answer = LiveModel(answer_spec, lambda: build(answer_spec, answer_creds), ledger)
    # The decider shape of live_agent: LOW thinking, small expected output.
    flash = LiveModel(
        flash_spec,
        lambda: build(flash_spec, flash_creds),
        ledger,
        thinking_level="LOW",
        expected_output_tokens=800,
    )
    ctx = Context(
        ledger,
        args.timeout_s,
        args.fee_per_query,
        answer_spec,
        flash_spec,
        answer,
        flash,
        flash_creds,
        client_factory,
    )
    selected = [c for c in CASES if args.cases is None or c in args.cases]
    out = Path(args.out)
    results: dict[str, Any] = {
        "at_utc": args.stamp,
        "dry": bool(args.dry),
        "budget_cny": args.budget_cny,
        "answer_alias": answer_spec.name,
        "flash_alias": flash_spec.name,
        "spent_cny": 0.0,
        "budget_stopped": False,
        "totals": {},
        "required_passed": False,
        "cases": [],
    }
    records: list[dict] = results["cases"]
    try:
        for name in selected:
            if results["budget_stopped"]:
                records.append(skipped_record(name, ctx))
            else:
                rec, exhausted = await run_case(name, ctx)
                records.append(rec)
                results["budget_stopped"] = exhausted
            results["spent_cny"] = round(ledger.spent, 6)
            results["totals"] = summarize(records)
            write_results(out, results)
    finally:
        await answer.aclose()
        await flash.aclose()
        if book is not None:
            book.close()
    results["pending_cny"] = round(ledger.pending, 6)
    results["required_passed"] = all(r["passed"] for r in records if r["case"] not in OPTIONAL)
    write_results(out, results)
    return results


def print_table(results: dict, out: Path) -> None:
    print(f"{'case':<24}{'alias':<18}{'ok':<6}{'latency_s':>10}{'cost_cny':>11}  error")
    for r in results["cases"]:
        ok = "skip" if r["skipped"] else ("PASS" if r["passed"] else "FAIL")
        latency = "-" if r["latency_s"] is None else f"{r['latency_s']:.1f}"
        error = r["error"] or r["check"] or ""
        print(f"{r['case']:<24}{r['alias']:<18}{ok:<6}{latency:>10}{r['cost_cny']:>11.5f}  {error}")
    totals = results["totals"]
    print(
        f"passed {totals['passed']}  failed {totals['failed']}  skipped {totals['skipped']}  "
        f"spent {results['spent_cny']:.5f} / {results['budget_cny']} CNY"
        + ("  [budget stop]" if results["budget_stopped"] else "")
    )
    print(f"results: {out / 'results.json'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.toml")
    parser.add_argument("--answer-alias", default="vertex_hydrogen")
    parser.add_argument("--flash-alias", default="vertex_flash")
    parser.add_argument(
        "--budget-cny", type=float, required=True, help="hard cap, shared by all cases"
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, help="subset (default: all)")
    parser.add_argument("--out", type=Path, help="default data/probes/vertex-framework-<UTC>/")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="per case, before spec caps")
    parser.add_argument("--fee-per-query", type=float, default=0.1, help="CNY per Google query")
    parser.add_argument(
        "--dry", action="store_true", help="offline fakes; no credentials or network"
    )
    parser.add_argument(
        "--no-account-book",
        dest="account_book",
        action="store_false",
        help="do not append calls to the account-wide usage book (--dry never does)",
    )
    args = parser.parse_args(argv)
    if args.budget_cny <= 0:
        parser.error("--budget-cny must be positive")
    args.stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.out is None:
        args.out = ROOT / "data" / "probes" / f"vertex-framework-{args.stamp}"
    return args


def main(
    argv: list[str] | None = None,
    *,
    model_builder: Callable[[ModelSpec, Any], Any] | None = None,
    client_factory: Callable[[float], Any] | None = None,
) -> int:
    args = parse_args(argv)
    try:
        results = asyncio.run(
            run_probe(args, model_builder=model_builder, client_factory=client_factory)
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print_table(results, Path(args.out))
    return 0 if results["required_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
