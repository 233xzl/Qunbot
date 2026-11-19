"""Live model runners: Gemini and DeepSeek through Pydantic AI.

Access decisions follow docs/MODEL-ACCESS.md:

- Gemini: `GoogleModel` with a `GoogleProvider(client=genai.Client(...api_version=v1beta))`,
  thinking level from config (HIGH), structured output with `NativeOutput`; when a function
  call is required, `ToolOutput(name=...)` pinned with `tool_choice=ToolOrOutput([])`.
- DeepSeek: `AnthropicModel` + `AnthropicProvider(base_url=relay)`, no thinking parameter;
  extraction with `ToolOutput`, `PromptedOutput` as fallback; never `NativeOutput`.

Every call has a timeout, captures usage (input incl. cached, cached, billed output incl.
thinking, separately reported thinking) and computes cost from configured CNY prices:
non-cached input x input price + cached x cached price + output x output price. A shared
`Ledger` reserves an estimate before each call and refuses calls that could exceed the
budget. Provider error texts are never stored: only class names and HTTP status codes,
because they can contain prompt material, keys or the relay host.
"""

from __future__ import annotations

import asyncio

from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import inspect
import threading
import time
from typing import Any

from .config import ModelSpec
from .contracts import Context
from .provider_adapters import build_pai_model, protocol_adapter  # noqa: F401 - public export


class BudgetExceeded(Exception):
    """The projected spend of the next call would exceed the configured budget."""

    reason = "budget_exceeded"


class ModelCallError(Exception):
    """A live call failed; `record` keeps usage already billed for failed attempts."""

    def __init__(self, kind: str, record: "CallRecord"):
        super().__init__(kind)
        self.kind = kind
        self.record = record
        self.reason = f"model_{kind}"


class ModelSkipped(ModelCallError):
    """The model's breaker is open: the call was skipped without a request (D-20)."""


# Exception class names (redacted `CallRecord.error`) that mean the model endpoint was not
# reachable: httpx / anthropic / aiohttp transport errors, and Pydantic AI's
# `ModelAPIError` without an HTTP status (how it wraps Anthropic connection errors).
CONNECTION_ERRORS = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "ReadError",
        "WriteError",
        "NetworkError",
        "TransportError",
        "RemoteProtocolError",
        "ProtocolError",
        "APIConnectionError",
        "APITimeoutError",
        "ModelAPIError",
        "ServerError",
        "ClientConnectorError",
        "ServerDisconnectedError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "OSError",
    }
)


def infrastructure_failure(record: "CallRecord", min_timeout_s: float = 0.0) -> bool:
    """Whether a failed call says the model service is unhealthy (D-20): a timeout (when the
    caller allowed at least `min_timeout_s`), a connection error, HTTP 429 or 5xx. Output
    validation failures, refusals and other 4xx answers are not counted."""
    if record.status == "timeout":
        return record.timeout_s is None or record.timeout_s >= min_timeout_s
    if record.status != "error" or not record.error:
        return False
    name, _, status = record.error.partition(":")
    if status.isdigit():
        code = int(status)
        return code == 429 or code >= 500
    return name in CONNECTION_ERRORS


class ModelBreaker:
    """Process-wide circuit breaker per model key (D-20), shared by every run and thread.

    `failures` consecutive infrastructure failures (`infrastructure_failure`) open the key
    for `cooldown_s`; calls are skipped meanwhile. When the cooldown ends, one call is let
    through as a probe (half-open; others keep being skipped): a response closes the key, an
    infrastructure failure reopens it, a cancelled probe frees the slot for the next call.
    A response of any call closes the key and resets the count; failures that are not
    infrastructure failures neither count nor reset. Timeouts shorter than `min_timeout_s`
    (the caller's budget, not the model, was short) are not counted either.
    """

    def __init__(
        self,
        failures: int = 2,
        cooldown_s: float = 300.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = lambda _: None,
        min_timeout_s: float = 8.0,
        probe_timeout_s: float = 300.0,
    ):
        if failures < 1 or cooldown_s <= 0:
            raise ValueError("breaker needs failures >= 1 and cooldown_s > 0")
        self.threshold, self.cooldown_s = failures, cooldown_s
        self.clock, self.log = clock, log
        self.min_timeout_s, self.probe_timeout_s = min_timeout_s, probe_timeout_s
        self._keys: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _entry(self, key: str) -> dict:
        entry = self._keys.get(key)
        if entry is None:
            entry = self._keys[key] = {
                "state": "closed",
                "consecutive": 0,
                "open_until": 0.0,
                "opened_at": None,
                "probe_since": None,
                "opened": 0,
                "open_seconds": 0.0,
                "skipped": 0,
                "probes": 0,
            }
        return entry

    def _state(self, entry: dict, now: float) -> str:
        if entry["state"] == "open" and now >= entry["open_until"]:
            return "half_open"
        return entry["state"]

    def state(self, key: str) -> str:
        """closed | open (cooling down) | half_open (cooldown over, probe due or running)."""
        with self._lock:
            return self._state(self._entry(key), self.clock())

    def available(self, key: str, count_skip: bool = False) -> bool:
        """Whether a call would be let through now (closed, or a probe slot is free);
        `count_skip`: the caller skips the model when not, count it like a skipped call."""
        with self._lock:
            entry, now = self._entry(key), self.clock()
            state = self._state(entry, now)
            ok = state == "closed" or (state == "half_open" and not self._probing(entry, now))
            if not ok and count_skip:
                entry["skipped"] += 1
            return ok

    def _probing(self, entry: dict, now: float) -> bool:
        since = entry["probe_since"]
        return since is not None and now - since < self.probe_timeout_s

    def acquire(self, key: str) -> str | None:
        """Ticket for one call ("closed" or "probe"); None = skip the call (counted)."""
        with self._lock:
            entry, now = self._entry(key), self.clock()
            state = self._state(entry, now)
            if state == "closed":
                return "closed"
            if state == "half_open" and not self._probing(entry, now):
                entry["probe_since"] = now
                entry["probes"] += 1
                self._log(key, "half_open", "cooldown over, one probe call")
                return "probe"
            entry["skipped"] += 1
            return None

    def record(self, key: str, ticket: str | None, outcome: str) -> None:
        """`outcome`: ok (the model answered), infra, other (not counted) or cancelled."""
        with self._lock:
            entry, now = self._entry(key), self.clock()
            probe = ticket == "probe"
            if probe:
                entry["probe_since"] = None
            if outcome == "ok" or (probe and outcome == "other"):
                entry["consecutive"] = 0
                if entry["state"] != "closed":
                    self._close(key, entry, now, "probe answered" if probe else "call answered")
            elif outcome == "infra":
                if entry["state"] == "closed":
                    entry["consecutive"] += 1
                    if entry["consecutive"] >= self.threshold:
                        entry["state"], entry["opened_at"] = "open", now
                        self._open(key, entry, now, f"{entry['consecutive']} failures in a row")
                elif probe:
                    self._open(key, entry, now, "probe failed")

    def _open(self, key: str, entry: dict, now: float, why: str) -> None:
        entry["open_until"] = now + self.cooldown_s
        entry["opened"] += 1
        self._log(key, "open", f"{why}; skipped for {self.cooldown_s:.0f}s")

    def _close(self, key: str, entry: dict, now: float, why: str) -> None:
        if entry["opened_at"] is not None:
            entry["open_seconds"] += now - entry["opened_at"]
        entry.update(state="closed", consecutive=0, open_until=0.0, opened_at=None)
        entry["probe_since"] = None
        self._log(key, "closed", why)

    def _log(self, key: str, state: str, why: str) -> None:
        try:
            self.log(f"model breaker {key}: {state} ({why})")
        except Exception:  # noqa: BLE001 - logging never breaks a model call
            pass

    def summary(self) -> dict:
        """Per key: state, times opened, seconds open (incl. the current period), skips."""
        with self._lock:
            now, out = self.clock(), {}
            for key, entry in self._keys.items():
                seconds = entry["open_seconds"]
                if entry["opened_at"] is not None:
                    seconds += now - entry["opened_at"]
                out[key] = {
                    "state": self._state(entry, now),
                    "opened": entry["opened"],
                    "open_seconds": round(seconds, 1),
                    "skipped": entry["skipped"],
                    "probes": entry["probes"],
                }
            return out


@dataclass
class Usage:
    """Normalised usage: `input` includes cached input; `output` is billed output."""

    input: int = 0
    cached: int = 0
    output: int = 0
    thinking: int | None = None
    requests: int = 0

    def add(self, other: "Usage") -> None:
        self.input += other.input
        self.cached += other.cached
        self.output += other.output
        if other.thinking is not None:
            self.thinking = (self.thinking or 0) + other.thinking
        self.requests += other.requests


def list_cost(spec: ModelSpec, usage: Usage) -> float:
    """List price of the tokens in the table's own currency (`spec.price_currency`)."""
    cached_price = spec.price_cached_input if spec.price_cached_input is not None else 0.0
    fresh = max(usage.input - usage.cached, 0)
    return (
        fresh * spec.price_input + usage.cached * cached_price + usage.output * spec.price_output
    ) / 1_000_000


def cost_cny(spec: ModelSpec, usage: Usage) -> float:
    """CNY estimate: the list price times the rate the spec resolved at settings load
    (`ModelSpec.cny_per_unit`, 1.0 for CNY tables)."""
    return list_cost(spec, usage) * spec.cny_per_unit


def _fill_list_cost(record: "CallRecord", spec: ModelSpec) -> None:
    """Original-currency list cost for a call whose cost is known only in CNY (estimates
    billed for failed or cancelled requests)."""
    if record.list_cost is None:
        record.list_cost = record.cost_cny / spec.cny_per_unit
    record.list_currency = spec.price_currency


@dataclass
class CallRecord:
    task: str
    model: str
    mode: str  # text | native | tool | prompted | systemone
    status: str = "ok"  # ok | error | timeout | budget
    error: str | None = None
    latency_s: float = 0.0
    usage: Usage = field(default_factory=Usage)
    cost_cny: float = 0.0
    responses: int = 0
    # Structured output validated from the first model response, without a retry.
    first_try: bool | None = None
    thinking_level: str | None = None  # Gemini level actually requested
    # Seconds the caller allowed this call (min of the caller's budget and the spec timeout).
    timeout_s: float | None = None
    # D-20 breaker ticket: closed | probe | open (skipped); None = no breaker on this model.
    breaker: str | None = None
    error_causes: list[str] = field(default_factory=list)
    failure_phase: str | None = None
    # B1 billing identity: the channel that bills the call (`ModelSpec.channel`), the service
    # (generate | search | fetch | judge), the original-currency list price of the call
    # (`list_cost` in `list_currency`; `cost_cny` stays the CNY figure every gate uses) and
    # the Google search queries actually issued (None = unknown or not a Google search).
    channel: str | None = None
    service: str | None = None
    list_cost: float | None = None
    list_currency: str | None = None
    google_queries: int | None = None
    # B3: queries of this call estimated to fall in the account's monthly free quota. Their
    # per-query fee is already left out of `cost_cny`; `list_cost` stays the full list price.
    # None = no quota applied (relay, no account book, unknown query count).
    free_queries: int | None = None
    model_id: str | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["latency_s"] = round(self.latency_s, 3)
        data["cost_cny"] = round(self.cost_cny, 7)
        if self.list_cost is not None:
            data["list_cost"] = round(self.list_cost, 9)
        return data


class Ledger:
    """Thread-safe usage ledger for one replay run with a hard CNY budget.

    `account` (a `billing.AccountBook`) makes every settled call also append one row to the
    account-wide usage book, tagged with `source` ("serve" | "replay" | "probe"). Only this
    base `settle` writes it, so subclasses must call `super().settle` and never write again.
    A failed append never breaks the call; it is counted in `counts`.

    `quota` (a `billing.GoogleFreeQuota`, B3) takes the free monthly Google search queries
    out of a settled Vertex search call's `cost_cny` before every total and the book row see
    it. Without one (relay, dry runs, `--no-account-book`) calls keep their list price."""

    def __init__(
        self,
        budget_cny: float | None = None,
        *,
        account: Any = None,
        source: str | None = None,
        model_ids: dict[str, str] | None = None,
        quota: Any = None,
    ):
        if account is not None and source is None:
            raise ValueError("an account book needs a usage source")
        self.quota = quota
        self.budget_cny = budget_cny
        self.spent = 0.0
        self.pending = 0.0
        self.records: list[CallRecord] = []
        self.account, self.source = account, source
        self.model_ids = dict(model_ids or {})  # configured alias -> real model id
        self.counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def reserve(self, estimate: float) -> None:
        with self._lock:
            if (
                self.budget_cny is not None
                and self.spent + self.pending + estimate > self.budget_cny
            ):
                raise BudgetExceeded("projected spend exceeds budget")
            self.pending += estimate

    def settle(self, estimate: float, record: CallRecord) -> None:
        # The quota's lock spans the free-query split and the book append, so concurrent
        # settles cannot both take the last free queries.
        with self.quota.settling(record) if self.quota is not None else nullcontext():
            with self._lock:
                self.pending = max(0.0, self.pending - estimate)
                self.spent += record.cost_cny
                self.records.append(record)
            if self.account is not None:
                try:
                    self.account.append(
                        record,
                        source=self.source,
                        model_id=record.model_id or self.model_ids.get(record.model),
                    )
                    outcome = "account_rows"
                except Exception:  # noqa: BLE001 - the in-memory total still counts the call
                    outcome = "account_write_failed"
                with self._lock:
                    self.counts[outcome] += 1

    def totals(self) -> dict:
        with self._lock:
            by_model: dict[str, dict] = {}
            for r in self.records:
                t = by_model.setdefault(
                    r.model,
                    {"calls": 0, "input": 0, "cached": 0, "output": 0, "thinking": 0, "cost": 0.0},
                )
                t["calls"] += 1
                t["input"] += r.usage.input
                t["cached"] += r.usage.cached
                t["output"] += r.usage.output
                t["thinking"] += r.usage.thinking or 0
                t["cost"] += r.cost_cny
            return {"spent_cny": round(self.spent, 6), "by_model": by_model}


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str


def estimate_input_tokens(parts: Sequence[str | ImagePart], instructions: str = "") -> int:
    """Rough upper estimate for budgeting: ~1 token per character, ~1,300 per image."""
    text = len(instructions) + sum(len(p) for p in parts if isinstance(p, str))
    images = sum(1 for p in parts if isinstance(p, ImagePart))
    return text + 1300 * images


def _input_token_bound(value: Any) -> int:
    """Bound text using UTF-8 bytes; image payload bytes are not text tokens."""
    from pydantic_ai import BinaryContent

    if isinstance(value, (BinaryContent, ImagePart)):
        # Keep the existing conservative visual allowance, independent of PNG/JPEG
        # compression. Counting repr(binary) charged the same image a second time.
        return 20_000
    if is_dataclass(value) and not isinstance(value, type):
        return 64 + sum(
            len(f.name) + _input_token_bound(getattr(value, f.name)) for f in fields(value)
        )
    if isinstance(value, dict):
        return 8 + sum(_input_token_bound(k) + _input_token_bound(v) + 4 for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return 8 + sum(_input_token_bound(v) + 4 for v in value)
    return len(repr(value).encode("utf-8")) + 4


def _redacted_error(exc: BaseException) -> str:
    status = next(
        (
            getattr(e, "status_code", None)
            for e in _exception_chain(exc)
            if isinstance(getattr(e, "status_code", None), int)
        ),
        None,
    )
    return f"{type(exc).__name__}:{status}" if status else type(exc).__name__


def _exception_chain(exc: BaseException) -> list[BaseException]:
    result = []
    while exc is not None and len(result) < 5 and all(exc is not e for e in result):
        result.append(exc)
        exc = exc.__cause__ or exc.__context__
    return result


def _error_causes(exc: BaseException) -> list[str]:
    # Never include exception messages, request/response bodies or addresses.
    return [_redacted_error(e) for e in _exception_chain(exc)]


def _pai_usage(messages, google: bool) -> tuple[Usage, int]:
    """Sum per-response usage. Pydantic AI's Google `output_tokens` already includes
    thinking; `details["thoughts_tokens"]` is a subset (measured in T2.4)."""
    total, responses = Usage(), 0
    for message in messages:
        if getattr(message, "kind", "") != "response":
            continue
        responses += 1
        u = message.usage
        details = dict(u.details or {})
        total.add(
            Usage(
                input=u.input_tokens,
                cached=u.cache_read_tokens,
                output=u.output_tokens,
                thinking=details.get("thoughts_tokens", 0) if google else None,
                requests=1,
            )
        )
    return total, responses


class LiveModel:
    """One configured generation model. `model_factory` returns a Pydantic AI model and is
    called lazily, on the event loop that uses it (HTTP clients are loop-bound)."""

    def __init__(
        self,
        spec: ModelSpec,
        model_factory: Callable[[], Any],
        ledger: Ledger,
        *,
        thinking_level: str | None = "default",
        expected_output_tokens: int | None = None,
        breaker: ModelBreaker | None = None,
    ):
        self.spec = spec
        self._factory = model_factory
        self._model = None
        self.ledger = ledger
        # D-20: shared breaker keyed by the config name (`[models.<name>]`); None = none.
        self.breaker = breaker
        self.adapter = protocol_adapter(spec.protocol)
        self.google = self.adapter.google_thinking
        self.thinking_level = spec.thinking_level if thinking_level == "default" else thinking_level
        self.expected_output = expected_output_tokens or (3000 if self.google else 800)

    @property
    def model(self):
        if self._model is None:
            self._model = self._factory()
        return self._model

    async def aclose(self) -> None:
        """Close the provider HTTP client on the loop that created it."""
        if self._model is None:
            return
        client = getattr(self._model, "client", None)
        try:
            aio = getattr(client, "aio", None)
            if aio is not None and hasattr(aio, "aclose"):
                await aio.aclose()
            elif client is not None and hasattr(client, "close"):
                result = client.close()
                if inspect.isawaitable(result):
                    await result
        except Exception:  # noqa: BLE001 - closing is best effort
            pass
        self._model = None

    def settings(
        self, pin_output_tool: bool = False, thinking_level: str | None = "default"
    ) -> dict:
        level = self.thinking_level if thinking_level == "default" else thinking_level
        return self.adapter.settings(self.spec, level, pin_output_tool)

    def estimate_cost(self, parts, instructions: str) -> float:
        usage = Usage(input=estimate_input_tokens(parts, instructions), output=self.expected_output)
        return cost_cny(self.spec, usage)

    async def generate(
        self,
        task: str,
        instructions: str,
        parts: Sequence[str | ImagePart],
        timeout: float,
        thinking_level: str | None = "default",
        *,
        tools: Sequence[Callable] = (),
        max_requests: int = 6,
        max_tool_calls: int = 8,
    ) -> tuple[str, CallRecord]:
        """`thinking_level` overrides the model's level for this call (Gemini only)."""
        return await self._run(
            task,
            "text",
            instructions,
            parts,
            str,
            timeout,
            thinking_level,
            tools=tools,
            max_requests=max_requests,
            max_tool_calls=max_tool_calls,
        )

    async def structured(
        self,
        task: str,
        instructions: str,
        parts: Sequence[str | ImagePart],
        output_type: type,
        mode: str,
        timeout: float,
        tool_name: str = "submit",
        tool_description: str = "",
    ) -> tuple[Any, CallRecord]:
        from pydantic_ai import NativeOutput, PromptedOutput, ToolOutput

        if mode == "native":
            if not self.adapter.native_output:
                raise ValueError("native schema output unsupported by configured protocol")
            spec = NativeOutput(output_type)
        elif mode == "tool":
            spec = ToolOutput(output_type, name=tool_name, description=tool_description)
        elif mode == "prompted":
            spec = PromptedOutput(output_type)
        else:
            raise ValueError(mode)
        return await self._run(task, mode, instructions, parts, spec, timeout)

    async def _run(
        self,
        task,
        mode,
        instructions,
        parts,
        output,
        timeout,
        thinking="default",
        *,
        tools=(),
        max_requests=6,
        max_tool_calls=8,
    ):
        record = CallRecord(
            task=task,
            model=self.spec.name,
            model_id=self.spec.model,
            mode=mode,
            channel=self.spec.channel,
            service="generate",
            list_cost=0.0,
            list_currency=self.spec.price_currency,
        )
        if self.google:
            record.thinking_level = self.thinking_level if thinking == "default" else thinking
        breaker, key = self.breaker, self.spec.name
        ticket = None
        if breaker is not None:
            ticket = breaker.acquire(key)
            if ticket is None:
                # No request and no ledger entry; the caller logs the skipped record.
                record.status, record.error, record.breaker = "skipped", "BreakerOpen", "open"
                raise ModelSkipped("breaker_open", record)
            record.breaker = ticket
        outcome = "cancelled"
        try:
            if tools:
                result = await self._call_tools(
                    record,
                    instructions,
                    parts,
                    timeout,
                    thinking,
                    tools,
                    max_requests,
                    max_tool_calls,
                )
            else:
                result = await self._call(record, instructions, parts, output, timeout, thinking)
            outcome = "ok"
            return result
        except ModelCallError as exc:
            outcome = (
                "infra"
                if breaker is not None and infrastructure_failure(exc.record, breaker.min_timeout_s)
                else "other"
            )
            raise
        finally:
            if breaker is not None:
                breaker.record(key, ticket, outcome)

    @property
    def supports_native_output(self) -> bool:
        return self.adapter.native_output

    async def _call_tools(
        self,
        record,
        instructions,
        parts,
        timeout,
        thinking,
        tools,
        max_requests,
        max_tool_calls,
        output=str,
    ):
        """Run Pydantic AI with per-request accounting, including validation retries."""
        from pydantic_ai import Agent, BinaryContent
        from pydantic_ai.models.wrapper import WrapperModel
        from pydantic_ai.usage import UsageLimits

        if max_requests < 1 or max_tool_calls < 1:
            raise ValueError("tool loop limits must be positive")
        owner = self
        requests: list[CallRecord] = []
        settings = self.settings(pin_output_tool=record.mode == "tool", thinking_level=thinking)
        # A real output ceiling is required for the pending-request reservation.
        settings.setdefault("max_tokens", self.expected_output)

        class MeteredModel(WrapperModel):
            async def request(self, messages, model_settings, model_request_parameters):
                # UTF-8 bytes of the complete history and tool schemas conservatively
                # bound text tokens, including tool results and subsequent round trips.
                input_bound = (
                    _input_token_bound(messages)
                    + _input_token_bound(model_request_parameters)
                    + len(instructions.encode("utf-8"))
                )
                estimate = cost_cny(
                    owner.spec,
                    Usage(
                        input=input_bound,
                        output=settings["max_tokens"],
                    ),
                )
                item = CallRecord(
                    record.task,
                    record.model,
                    record.mode,
                    timeout_s=record.timeout_s,
                    thinking_level=record.thinking_level,
                    breaker=record.breaker,
                    channel=record.channel,
                    service=record.service,
                    model_id=record.model_id,
                )
                owner.ledger.reserve(estimate)
                requests.append(item)
                started = time.perf_counter()
                try:
                    response = await self.wrapped.request(
                        messages,
                        model_settings,
                        model_request_parameters,
                    )
                    item.usage, item.responses = _pai_usage([response], owner.google)
                    item.cost_cny = cost_cny(owner.spec, item.usage)
                    item.list_cost = list_cost(owner.spec, item.usage)
                    return response
                except asyncio.CancelledError:
                    item.status, item.error = "cancelled", "CancelledError"
                    item.cost_cny = estimate
                    item.usage.requests = 1
                    raise
                except Exception as exc:
                    item.status, item.error = "error", _redacted_error(exc)
                    item.error_causes = _error_causes(exc)
                    item.failure_phase = "model_request"
                    # Unknown provider failures may occur after generation has begun.
                    item.cost_cny = estimate
                    item.usage.requests = 1
                    raise
                finally:
                    item.latency_s = time.perf_counter() - started
                    _fill_list_cost(item, owner.spec)
                    owner.ledger.settle(estimate, item)

        prompt = [
            p if isinstance(p, str) else BinaryContent(data=p.data, media_type=p.media_type)
            for p in parts
        ]
        limit = max(0.001, min(timeout, self.spec.timeout_seconds))
        record.timeout_s = round(limit, 3)
        started = time.perf_counter()
        result, cancelled, budget_error = None, False, None
        try:
            async with asyncio.timeout(limit):
                agent = Agent(
                    MeteredModel(self.model),
                    output_type=output,
                    instructions=instructions,
                    tools=list(tools),
                    model_settings=settings,
                    retries={"output": 1},
                )
                result = await agent.run(
                    prompt,
                    usage_limits=UsageLimits(
                        request_limit=max_requests,
                        tool_calls_limit=max_tool_calls,
                    ),
                )
        except TimeoutError:
            record.status, record.error = "timeout", "TimeoutError"
        except asyncio.CancelledError:
            record.status, record.error, cancelled = "cancelled", "CancelledError", True
        except BudgetExceeded as exc:
            record.status, record.error = "budget", "BudgetExceeded"
            budget_error = exc
        except Exception as exc:
            record.status, record.error = "error", _redacted_error(exc)
            record.error_causes = _error_causes(exc)
            record.failure_phase = (
                "model_request" if requests and requests[-1].status == "error" else "tool_loop"
            )
        finally:
            record.latency_s = time.perf_counter() - started
            for item in requests:
                record.usage.add(item.usage)
                record.responses += item.responses
                record.cost_cny += item.cost_cny
                record.list_cost = (record.list_cost or 0.0) + (item.list_cost or 0.0)
        if record.mode != "text":
            record.first_try = result is not None and record.responses == 1
        if cancelled:
            exc = asyncio.CancelledError()
            exc.record = record
            raise exc
        if budget_error is not None:
            if not requests:
                # Preserve a diagnostic entry for a call rejected before transmission.
                self.ledger.settle(0.0, record)
            budget_error.record = record
            raise budget_error
        if result is None:
            raise ModelCallError(record.status, record)
        return result.output, record

    async def _call(self, record, instructions, parts, output, timeout, thinking):
        return await self._call_tools(
            record,
            instructions,
            parts,
            timeout,
            thinking,
            (),
            6,
            8,
            output=output,
        )


class NoTimeLeft(Exception):
    """Too little run time remains for this attempt; skipped without a model call."""


class BoundCall:
    """Adapter so a prepared call can sit in `contracts.FallbackModel` (ModelRunner).

    `timeout()` is evaluated when the attempt starts, so later attempts see the time that
    earlier ones used; an attempt with less than `min_seconds` left is skipped.
    """

    def __init__(
        self,
        model: LiveModel,
        task: str,
        instructions: str,
        parts,
        timeout,
        sink,
        thinking_level: str | None = "default",
        min_seconds: float = 5.0,
        *,
        tools: Sequence[Callable] = (),
        max_requests: int = 6,
        max_tool_calls: int = 8,
        parts_factory: Callable[[], Sequence[str | ImagePart]] | None = None,
    ):
        self.model, self.task, self.instructions = model, task, instructions
        self.parts, self.timeout, self.sink = parts, timeout, sink
        self.thinking_level, self.min_seconds = thinking_level, min_seconds
        self.tools = tools
        self.parts_factory = parts_factory
        self.max_requests, self.max_tool_calls = max_requests, max_tool_calls

    async def run(self, context: Context) -> str:
        budget = self.timeout()
        if budget < self.min_seconds:
            raise NoTimeLeft(f"{budget:.1f}s")
        parts = self.parts_factory() if self.parts_factory is not None else self.parts
        try:
            text, record = await self.model.generate(
                self.task,
                self.instructions,
                parts,
                budget,
                self.thinking_level,
                **(
                    {
                        "tools": self.tools,
                        "max_requests": self.max_requests,
                        "max_tool_calls": self.max_tool_calls,
                    }
                    if self.tools
                    else {}
                ),
            )
        except (ModelCallError, BudgetExceeded, asyncio.CancelledError) as exc:
            record = getattr(exc, "record", None)
            if record is not None:
                self.sink.append(record)
            raise
        self.sink.append(record)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty model output")
        return text
