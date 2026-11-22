"""Validated configuration; offline replay never requires production credentials."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import math
import re
from urllib.parse import urlsplit

from .policy import RuntimeLimits


GROUP_TOOL_LIMITS = {"web_searches", "web_reads"}
# Admission rate per group (`Orchestrator.submit`): at most `group_admit_max` admitted
# requests per `group_admit_window_minutes`; 0 = no limit.
GROUP_ADMIT_LIMITS = {"group_admit_window_minutes", "group_admit_max"}
# `group_profiles.<id>.runtime` keys a group may override (whole numbers >= 0).
GROUP_RUNTIME_KEYS = GROUP_TOOL_LIMITS | GROUP_ADMIT_LIMITS


class ConfigError(ValueError):
    """Errors contain field names, never supplied configuration values."""


@dataclass(frozen=True)
class Retention:
    """Retention cleanup parameters (docs/CONTEXT-RETENTION.md proposal)."""

    recent_hours: int = 24  # ordinary buffer per group
    related_days: int = 7  # hard limit; related-set messages are kept until then
    max_messages_per_group: int = 15000
    max_branch_messages: int = 300  # per request branch in the related set
    max_branch_depth: int = 16  # reply-chain traversal depth from a request
    media_max_mb: int = 2048  # total media cache
    cleanup_interval_minutes: int = 60


# QQ image hosts SnowLuma returns download URLs for (multimedia.nt.qq.com.cn, gchat.qpic.cn).
IMAGE_HOST_SUFFIXES = ("qpic.cn", "nt.qq.com.cn")
# D-21 failure fallback: the old bot's (AstrBot persona) error message wording.
DEFAULT_FALLBACK_REPLY = "这个史暂时没解析成功，稍后再 @ 我试试。"
FALLBACK_REPLY_MAX_CHARS = 100
HOST_SUFFIX = re.compile(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+")


@dataclass(frozen=True)
class Collection:
    """`[collection]` table: ordinary message collection and image caching (CTX-05/06)."""

    disabled_groups: frozenset[str] = frozenset()  # groups whose ordinary messages are not kept
    media_dir: Path | None = field(default=None, repr=False)  # default: <db>.media beside the db
    max_image_mb: int = 10
    image_host_suffixes: tuple[str, ...] = IMAGE_HOST_SUFFIXES
    download_per_second: float = 2.0
    max_concurrent_downloads: int = 2
    max_pending_downloads: int = 200


@dataclass(frozen=True)
class SessionSpec:
    """`[session]` table: per-group multi-turn answer sessions (`session.py`).

    Off by default. A session restarts after `idle_minutes` without a committed turn, once
    its last request reached `max_tokens`, or `max_age_minutes` after it started."""

    enabled: bool = False
    idle_minutes: int = 20
    max_tokens: int = 8000
    max_age_minutes: int = 120
    max_new_messages: int = 40  # group messages added per turn (the most recent ones)


SESSION_MINIMUMS = {"idle_minutes": 1, "max_tokens": 1000, "max_age_minutes": 1, "max_new_messages": 0}


@dataclass(frozen=True)
class Settings:
    bot_id: str = "BOT"
    owner_id: str | None = "OWNER"
    blocked_users: frozenset[str] = frozenset()
    blocked_groups: frozenset[str] = frozenset()
    deadline_seconds: float = 120
    context_messages: int = 80
    # D-04: daily model + search cost cap in CNY (80 % degrade, 100 % owner only).
    daily_cost_cny: float = 3.0
    retention: Retention = field(default_factory=Retention)
    collection: Collection = field(default_factory=Collection)
    secrets_file: Path = field(default=Path("private/credentials.env"), repr=False)
    shadow_db: Path = field(default=Path("data/shadow/shadow.sqlite3"), repr=False)
    serve_db: Path = field(default=Path("data/serve/serve.sqlite3"), repr=False)
    # T4.1/T4.3: the knowledge base (`knowledge_base.py`); None = no knowledge, teaching
    # recognised but not written. `load_settings` defaults to data/knowledge/.
    knowledge_db: Path | None = field(default=None, repr=False)
    # D-19: the owner's `#新版` command may send (the only send path); off by default.
    owner_command_send: bool = False
    # Fixed reply when every answer attempt of a run fails (recorded like an answer, run
    # marked `fallback_reply`); "" = no reply. The old bot's error message wording.
    fallback_reply: str = DEFAULT_FALLBACK_REPLY
    # Read-only sync of the bot account's own messages from SnowLuma's message store
    # (self_sync.py); `snowluma_db` None = <SnowLuma data>/<bot_id>/messages.db.
    self_sync: bool = True
    self_sync_interval_s: float = 5.0
    snowluma_db: Path | None = field(default=None, repr=False)
    # D-20: "processing" reaction on admitted requests (`reaction.py`); off in the shadow
    # trial (AstrBot reacts), on after the cutover. Emoji id is a QQ face id.
    processing_reaction: bool = False
    reaction_emoji_id: str = "424"
    # Cutover (D-21, RUN-07..09): send every answer (`live_send.py`); off by default. When
    # on, the owner's `#新版` command is switched off (it would answer twice).
    live_send: bool = False
    # Owner remote control (`owner_control.py`): the owner's `#暂停` / `#恢复` / `#状态`
    # in any group; the pause is persisted. On by default.
    owner_control: bool = True
    models: dict[str, "ModelSpec"] = field(default_factory=dict)
    model_roles: dict[str, str] = field(default_factory=dict)
    answer_tool_loop: bool = True
    web_enabled: bool = True
    decision_mode: str = "lightweight"
    answer_thinking: str = "complex_only"
    # V2 fixed roles: an answer runs on the answer role, then the fallback role. `true`
    # restores the historical same-model HIGH -> LOW retry in between.
    answer_same_model_retry: bool = False
    # D-20: `[models] breaker_failures` consecutive infrastructure failures (timeout,
    # connection, 429/5xx) skip Gemini for `breaker_cooldown_s` (models.ModelBreaker).
    model_breaker_failures: int = 2
    model_breaker_cooldown_s: float = 300.0
    web: "WebSpec" = field(default_factory=lambda: WebSpec())
    billing: "BillingSpec" = field(default_factory=lambda: BillingSpec())
    runtime: RuntimeLimits = field(default_factory=RuntimeLimits)
    session: SessionSpec = field(default_factory=SessionSpec)
    prompts: dict[str, str] = field(default_factory=dict, repr=False)
    group_profiles: dict = field(default_factory=dict, repr=False)
    credential_refs: dict = field(default_factory=dict, repr=False)
    revision: str = ""
    published_revision: str | None = None
    config_store: Path | None = field(default=None, repr=False)
    sources: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from .config_service import freeze

        for name in (
            "models",
            "model_roles",
            "prompts",
            "group_profiles",
            "credential_refs",
            "sources",
        ):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def runtime_for_group(self, group_id, base: RuntimeLimits) -> RuntimeLimits:
        from dataclasses import replace

        profile = self.group_profiles.get(group_id, {})
        overrides = profile.get("runtime") if profile.get("enabled") else None
        return replace(base, **overrides) if overrides else base

    def session_for_group(self, group_id) -> SessionSpec:
        from dataclasses import replace

        profile = self.group_profiles.get(group_id, {})
        overrides = profile.get("session") if profile.get("enabled") else None
        return replace(self.session, **overrides) if overrides else self.session

    def for_group(self, group_id):
        from dataclasses import replace

        profile = self.group_profiles.get(group_id, {})
        if not profile.get("enabled"):
            return self
        roles = dict(self.model_roles)
        roles.setdefault("decision", roles.get("answer", "gemini"))
        return replace(
            self,
            runtime=self.runtime_for_group(group_id, self.runtime),
            prompts={**self.prompts, **profile.get("prompts", {})},
            model_roles={**roles, **profile.get("model_roles", {})},
        )

    @property
    def credentials(self):
        from .credentials import CredentialStore

        return CredentialStore(self.secrets_file, self.credential_refs)

    @property
    def limits(self) -> RuntimeLimits:
        from dataclasses import replace

        return replace(
            self.runtime,
            deadline_seconds=self.deadline_seconds,
            context_messages=self.context_messages,
        )


@dataclass(frozen=True)
class ModelCredentials:
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    # Vertex mode only: project ID or number for the full model resource path.
    project: str | None = field(default=None, repr=False)


MODEL_PROTOCOLS = {"systemone", "google-genai", "anthropic", "openai-compatible", "openai-responses"}
GOOGLE_THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
API_VERSION = re.compile(r"v[0-9]+(?:alpha|beta)?[0-9]*")
EMOJI_ID = re.compile(r"[0-9]{1,6}")
# Google Cloud location (global, us-central1, ...) and project ID or number.
VERTEX_LOCATION = re.compile(r"[a-z]+(?:-[a-z0-9]+)*")
VERTEX_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,61}[a-z0-9]|[0-9]{1,20}")
PRICE_CURRENCIES = ("CNY", "USD")
CHANNEL_NAME = re.compile(r"[a-z0-9:._-]+")
ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class ModelSpec:
    """One `[models.<name>]` table; credentials stay in the secrets file."""

    name: str
    protocol: str
    model: str
    base_url_env: str
    api_key_env: str
    timeout_seconds: float
    price_input: float
    price_output: float
    price_cached_input: float | None = None
    # Google GenAI: thinking level; Anthropic: budget tokens, 0 disables extended thinking.
    thinking_level: str | None = None
    thinking_budget_tokens: int = 0
    max_output_tokens: int | None = None
    # Google GenAI path version; the relay serves v1beta only, google-genai>=3 defaults to v1.
    api_version: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = field(default=None, repr=False)
    # Vertex mode (google-genai only): vertexai=True, API key, `api_version` (default v1)
    # and the full resource path projects/<project>/locations/<location>/publishers/google/
    # models/<model>. `model` stays the short name for capabilities, prices and diagnostics;
    # the project comes from the secrets file (`project_env`).
    vertex: bool = False
    location: str | None = None
    project_env: str | None = None
    # B1 billing identity. `channel` says who bills the call (vertex | relay |
    # official:<host>); empty = derived by `default_channel` in `__post_init__`. Prices
    # above are `price_currency` per 1M tokens, valid since `price_effective` (YYYY-MM-DD,
    # informational; the table is replaced when the vendor price changes).
    channel: str = ""
    price_currency: str = "CNY"
    price_effective: str | None = None
    # CNY per one unit of `price_currency`, resolved once from `[billing] usd_cny` when the
    # settings load (1.0 for CNY), so `models.cost_cny(spec, usage)` needs no FX argument and
    # an alias can never pick up another alias's rate. None = resolve in `__post_init__`:
    # 1.0 for CNY; a USD table built by hand must pass the rate explicitly.
    cny_per_unit: float | None = None

    def __post_init__(self) -> None:
        if not self.channel:
            object.__setattr__(self, "channel", default_channel(self))
        if self.price_currency not in PRICE_CURRENCIES:
            raise ValueError("price_currency: expected CNY or USD")
        if self.cny_per_unit is None:
            if self.price_currency != "CNY":
                raise ValueError("cny_per_unit: required for a non-CNY price table")
            object.__setattr__(self, "cny_per_unit", 1.0)


def default_channel(spec: "ModelSpec") -> str:
    """Billing channel when `[models.<alias>] channel` is omitted: Vertex, the shared relay
    (a `base_url_env` gateway) or an official endpoint named by its host. A literal
    `base_url` is the endpoint credentials use, so it wins over a stray `base_url_env`."""
    if spec.vertex:
        return "vertex"
    host = urlsplit(spec.base_url).hostname if spec.base_url else None
    if host:
        return f"official:{host.lower()}"
    return "relay" if spec.base_url_env else "unknown"


@dataclass(frozen=True)
class BillingSpec:
    """`[billing]` table (B1/B2): FX rate, Google search list price and the account book."""

    # CNY per USD for converting USD price tables; `usd_cny_date` documents the rate's day.
    usd_cny: float = 7.2
    usd_cny_date: str | None = None
    # Google Search grounding list price, USD per 1,000 queries (Gemini 3/4, beyond the
    # free monthly quota that B3 will model). Used for V3 search backends on a Vertex channel.
    google_search_usd_per_1000: float = 14.0
    # Free Google Search grounding queries per billing account and month on Vertex (this bot
    # is the account's only user), and the day of month the free quota restarts (1..28). The
    # month boundary is the Asia/Shanghai calendar date; Google counts per billing month in
    # Pacific time, so the quota counter is an estimate (billing.GoogleFreeQuota).
    google_free_queries_per_month: int = 5000
    billing_cycle_day: int = 1
    # Account-wide usage book (billing.AccountBook): its own file, never the serve store.
    db_path: Path = field(default=Path("data/billing/account.sqlite3"), repr=False)


@dataclass(frozen=True)
class WebSpec:
    """`[web]` table: search/fetch provider (docs/MODEL-ACCESS.md, web search section)."""

    provider: str = "deepseek"
    api_key_env: str = "DEEPSEEK_SEARCH_API_KEY"
    # Official endpoint, not a secret; the relay has no search endpoint.
    base_url: str = "https://api.deepseek.com"
    # Official CNY price per search/fetch, checked 2026-11-14: 5 / 1,000.
    price_per_call: float = 0.005
    # Search backend: "deepseek" (official endpoint) or "gemini" (a separate Gemini request
    # carrying only google_search, via [models.gemini] credentials; page reading stays on
    # DeepSeek). Re-check 2 in docs/MODEL-ACCESS.md: only some relay models return sources.
    search_provider: str = "deepseek"
    gemini_model: str = "gemini-4-argon"
    gemini_thinking_level: str | None = "LOW"
    # CNY per Google search query issued by the model, on top of tokens. Unverified: Google
    # lists USD 14 per 1,000 queries, but the relay's price list is CNY 6 per 1,000 searches.
    # Whether the relay counts requests or queries is unstated; charge per query (upper
    # bound). One request issued 2-4 queries in the probe.
    gemini_fee_per_query: float = 0.006
    # D-18: with DeepSeek as the search backend, one Gemini search per run supplements it
    # when a DeepSeek query returns nothing or fails, or when the route decision flags
    # recent_news. It counts against the run's search budget (D-11) and is skipped when the
    # per-call tool timeout left is below `supplement_min_seconds` (argon took 12.6-19.3 s).
    gemini_supplement: bool = True
    supplement_min_seconds: float = 15.0
    # Breaker: after this many consecutive supplement failures (no sources, timeout,
    # error) the supplement is skipped in this process for `supplement_breaker_seconds`.
    supplement_breaker_failures: int = 3
    supplement_breaker_seconds: float = 1800.0
    # Gemini main mode tries URL context first and falls back per attempt in WebTool.
    gemini_url_context: bool = True
    deepseek_fallback: bool = True
    # Credential and pricing role for independent Gemini web requests.
    gemini_credentials_model: str = "gemini"
    # V3 separate bindings. `search` None = the legacy keys above decide. Otherwise each
    # value is "deepseek" (official web tools, `price_per_call`) or a google-genai model
    # alias whose credentials, model and prices are used (search: google_search only;
    # fetch: URL context only). "" = no fallback. Legacy keys `search_provider`,
    # `gemini_model`, `gemini_credentials_model`, `gemini_url_context`, `deepseek_fallback`
    # and `gemini_supplement` are not used then.
    search: str | None = None
    search_fallback: str = ""
    fetch: str = "deepseek"
    fetch_fallback: str = ""


@dataclass(frozen=True)
class SnowLumaCredentials:
    ws_url: str = field(repr=False)
    token: str = field(repr=False)


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _table(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: expected a table")
    return value


def _number(data: dict, name: str, default: float, prefix: str, integer=False):
    value = data.get(name, default)
    kind = (int,) if integer else (int, float)
    if (
        isinstance(value, bool)
        or not isinstance(value, kind)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(
            f"{prefix}.{name}: expected a positive finite {'integer' if integer else 'number'}"
        )
    return value


def _identifier(value, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ConfigError(f"{name}: expected an identifier")
    return str(value)


def _identifiers(data, name, prefix="identity"):
    value = data.get(name, [])
    if not isinstance(value, list):
        raise ConfigError(f"{prefix}.{name}: expected an array")
    return frozenset(_identifier(item, f"{prefix}.{name}") for item in value)


def _collection(data: dict, settings_path: Path) -> Collection:
    table = _table(data, "collection")
    default = Collection()
    media_dir = table.get("media_dir")
    if media_dir is not None:
        if not isinstance(media_dir, str) or not media_dir:
            raise ConfigError("collection.media_dir: expected a path")
        media_dir = Path(media_dir)
        if not media_dir.is_absolute():
            media_dir = (settings_path.resolve().parent / media_dir).resolve()
    suffixes = table.get("image_host_suffixes", list(default.image_host_suffixes))
    if (
        not isinstance(suffixes, list)
        or not suffixes
        or not all(isinstance(s, str) and HOST_SUFFIX.fullmatch(s) for s in suffixes)
    ):
        raise ConfigError("collection.image_host_suffixes: expected lower-case domain names")
    prefix = "collection"
    return Collection(
        disabled_groups=_identifiers(table, "disabled_groups", prefix),
        media_dir=media_dir,
        max_image_mb=_number(table, "max_image_mb", default.max_image_mb, prefix, integer=True),
        image_host_suffixes=tuple(suffixes),
        download_per_second=float(
            _number(table, "download_per_second", default.download_per_second, prefix)
        ),
        max_concurrent_downloads=_number(
            table, "max_concurrent_downloads", default.max_concurrent_downloads, prefix, True
        ),
        max_pending_downloads=_number(
            table, "max_pending_downloads", default.max_pending_downloads, prefix, True
        ),
    )


def _price(table: dict, name: str, prefix: str, required=True) -> float | None:
    if name not in table and not required:
        return None
    value = table.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigError(f"{prefix}.{name}: expected a non-negative finite number")
    return float(value)


MODELS_SCALARS = ("breaker_failures", "breaker_cooldown_s")  # [models] keys, not tables


def _model_roles(data: dict) -> dict[str, str]:
    roles = _table(data, "model_roles")
    models = _table(data, "models")
    for role, name in roles.items():
        if role not in {"answer", "decision", "fallback"}:
            raise ConfigError("model_roles: expected answer, decision or fallback")
        if not isinstance(name, str) or not isinstance(models.get(name), dict):
            raise ConfigError(f"model_roles.{role}: expected a configured model alias")
        if models[name].get("protocol") == "systemone":
            raise ConfigError(f"model_roles.{role}: expected a generation model")
    return dict(roles)


def _iso_date(value, name: str) -> str | None:
    if value is None:
        return None
    try:
        if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
            raise ValueError
        date.fromisoformat(value)
    except ValueError:
        raise ConfigError(f"{name}: expected a date as YYYY-MM-DD") from None
    return value


def _billing(data: dict, settings_path: Path) -> BillingSpec:
    table = _table(data, "billing")
    default = BillingSpec()
    db_path = table.get("db_path", "../data/billing/account.sqlite3")
    if not isinstance(db_path, str) or not db_path:
        raise ConfigError("billing.db_path: expected a path")
    path = Path(db_path)
    if not path.is_absolute():
        path = (settings_path.resolve().parent / path).resolve()
    return BillingSpec(
        usd_cny=float(_number(table, "usd_cny", default.usd_cny, "billing")),
        usd_cny_date=_iso_date(table.get("usd_cny_date"), "billing.usd_cny_date"),
        google_search_usd_per_1000=float(
            _number(
                table, "google_search_usd_per_1000", default.google_search_usd_per_1000, "billing"
            )
        ),
        google_free_queries_per_month=_whole(
            table,
            "google_free_queries_per_month",
            default.google_free_queries_per_month,
            "billing",
            minimum=0,
        ),
        billing_cycle_day=_whole(
            table, "billing_cycle_day", default.billing_cycle_day, "billing", minimum=1, maximum=28
        ),
        db_path=path,
    )


def _whole(data: dict, name: str, default: int, prefix: str, *, minimum: int, maximum=None) -> int:
    """An integer setting within [minimum, maximum]."""
    value = data.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ConfigError(f"{prefix}.{name}: expected an integer {bound}")
    return value


def parse_session(table: dict, prefix: str, base: SessionSpec) -> SessionSpec:
    """`[session]` or a group's `session` override; missing keys keep `base`."""
    if not isinstance(table, dict):
        raise ConfigError(f"{prefix}: expected a table")
    for key in table:
        if key not in SessionSpec.__dataclass_fields__:
            raise ConfigError(f"{prefix}.{key}: unknown field")
    enabled = table.get("enabled", base.enabled)
    if type(enabled) is not bool:
        raise ConfigError(f"{prefix}.enabled: expected a boolean")
    values = {
        key: _whole(table, key, getattr(base, key), prefix, minimum=minimum)
        for key, minimum in SESSION_MINIMUMS.items()
    }
    return SessionSpec(enabled=enabled, **values)


def _models(data: dict, usd_cny: float = BillingSpec().usd_cny) -> dict[str, ModelSpec]:
    specs = {}
    for name, table in _table(data, "models").items():
        if name in MODELS_SCALARS:
            continue
        prefix = f"models.{name}"
        if not isinstance(table, dict):
            raise ConfigError(f"{prefix}: expected a table")
        protocol = table.get("protocol")
        if protocol not in MODEL_PROTOCOLS:
            raise ConfigError(f"{prefix}.protocol: expected one of {sorted(MODEL_PROTOCOLS)}")
        model = table.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ConfigError(f"{prefix}.model: expected a model name")
        vertex = table.get("vertex", False)
        if not isinstance(vertex, bool) or (vertex and protocol != "google-genai"):
            raise ConfigError(f"{prefix}.vertex: expected true or false for google-genai")
        location = project_env = None
        if vertex:
            for key in ("base_url", "base_url_env"):
                if key in table:
                    raise ConfigError(f"{prefix}.{key}: not used with vertex = true")
            location = table.get("location", "global")
            if not isinstance(location, str) or not VERTEX_LOCATION.fullmatch(location):
                raise ConfigError(f"{prefix}.location: expected a Google Cloud location")
            project_env = table.get("project_env", "GOOGLE_CLOUD_PROJECT")
            if not isinstance(project_env, str) or not ENV_NAME.fullmatch(project_env):
                raise ConfigError(f"{prefix}.project_env: expected an upper-case variable name")
        else:
            for key in ("location", "project_env"):
                if key in table:
                    raise ConfigError(f"{prefix}.{key}: requires vertex = true")
        base_url = table.get("base_url")
        if base_url is not None:
            u = urlsplit(base_url) if isinstance(base_url, str) else None
            if (
                not u
                or u.scheme != "https"
                or not u.hostname
                or u.username
                or u.password
                or u.query
                or u.fragment
            ):
                raise ConfigError(f"{prefix}.base_url: expected a credential-free HTTPS URL")
        envs = {}
        for key in ("base_url_env", "api_key_env"):
            value = table.get(key)
            if key == "base_url_env" and (vertex or (base_url and value is None)):
                envs[key] = ""
                continue
            if not isinstance(value, str) or not ENV_NAME.fullmatch(value):
                raise ConfigError(f"{prefix}.{key}: expected an upper-case variable name")
            envs[key] = value
        level = table.get("thinking_level")
        if level is not None and (
            protocol != "google-genai"
            or not isinstance(level, str)
            or level not in GOOGLE_THINKING_LEVELS
        ):
            raise ConfigError(
                f"{prefix}.thinking_level: expected one of {sorted(GOOGLE_THINKING_LEVELS)} "
                "for google-genai"
            )
        budget = table.get("thinking_budget_tokens", 0)
        if (
            isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget < 0
            or (budget and protocol != "anthropic")
        ):
            raise ConfigError(
                f"{prefix}.thinking_budget_tokens: expected a non-negative integer for anthropic"
            )
        max_output = table.get("max_output_tokens")
        effort = table.get("reasoning_effort")
        if effort is not None and (
            protocol not in {"openai-compatible", "openai-responses"}
            or not isinstance(effort, str)
            or effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
        ):
            raise ConfigError(f"{prefix}.reasoning_effort: invalid effort for openai-compatible")
        if max_output is not None:
            max_output = _number(table, "max_output_tokens", 1, prefix, integer=True)
        api_version = table.get("api_version")
        if api_version is not None and (
            protocol != "google-genai"
            or not isinstance(api_version, str)
            or not API_VERSION.fullmatch(api_version)
        ):
            raise ConfigError(f"{prefix}.api_version: expected e.g. v1beta for google-genai")
        if vertex and api_version is None:
            api_version = "v1"  # verified with global resource paths (2026-11-14 probe)
        channel = table.get("channel", "")
        if not isinstance(channel, str) or (channel and not CHANNEL_NAME.fullmatch(channel)):
            raise ConfigError(f"{prefix}.channel: expected a lower-case name like [a-z0-9:._-]+")
        currency = table.get("price_currency", "CNY")
        if currency not in PRICE_CURRENCIES:
            raise ConfigError(f"{prefix}.price_currency: expected CNY or USD")
        specs[name] = ModelSpec(
            name=name,
            protocol=protocol,
            model=model,
            timeout_seconds=_number(table, "timeout_seconds", 60, prefix),
            price_input=_price(table, "price_input", prefix),
            price_output=_price(table, "price_output", prefix),
            price_cached_input=_price(table, "price_cached_input", prefix, required=False),
            thinking_level=level,
            thinking_budget_tokens=budget,
            max_output_tokens=max_output,
            api_version=api_version,
            reasoning_effort=effort,
            base_url=base_url,
            vertex=vertex,
            location=location,
            project_env=project_env,
            channel=channel,
            price_currency=currency,
            price_effective=_iso_date(table.get("price_effective"), f"{prefix}.price_effective"),
            cny_per_unit=usd_cny if currency == "USD" else 1.0,
            **envs,
        )
    return specs


def _web(data: dict) -> WebSpec:
    table = _table(data, "web")
    default = WebSpec()
    provider = table.get("provider", default.provider)
    if provider != "deepseek":
        raise ConfigError("web.provider: expected 'deepseek'")
    key_env = table.get("api_key_env", default.api_key_env)
    if not isinstance(key_env, str) or not ENV_NAME.fullmatch(key_env):
        raise ConfigError("web.api_key_env: expected an upper-case variable name")
    base_url = table.get("base_url", default.base_url)
    try:
        u = urlsplit(base_url) if isinstance(base_url, str) else None
        if u is None or u.scheme != "https" or not u.hostname or u.username or u.password:
            raise ValueError
    except ValueError:
        raise ConfigError("web.base_url: expected a credential-free HTTPS URL") from None
    price = (
        _price(table, "price_per_call", "web")
        if "price_per_call" in table
        else default.price_per_call
    )
    search_provider = table.get("search_provider", default.search_provider)
    if search_provider not in ("deepseek", "gemini"):
        raise ConfigError("web.search_provider: expected 'deepseek' or 'gemini'")
    gemini_model = table.get("gemini_model", default.gemini_model)
    if not isinstance(gemini_model, str) or not gemini_model.strip():
        raise ConfigError("web.gemini_model: expected a model name")
    level = table.get("gemini_thinking_level", default.gemini_thinking_level)
    if level == "":
        level = None
    if level is not None and (not isinstance(level, str) or level not in GOOGLE_THINKING_LEVELS):
        raise ConfigError(
            f"web.gemini_thinking_level: expected one of {sorted(GOOGLE_THINKING_LEVELS)} or ''"
        )
    fee = (
        _price(table, "gemini_fee_per_query", "web")
        if "gemini_fee_per_query" in table
        else default.gemini_fee_per_query
    )
    supplement = table.get("gemini_supplement", default.gemini_supplement)
    if not isinstance(supplement, bool):
        raise ConfigError("web.gemini_supplement: expected true or false")
    min_seconds = _number(table, "supplement_min_seconds", default.supplement_min_seconds, "web")
    breaker_failures = _number(
        table,
        "supplement_breaker_failures",
        default.supplement_breaker_failures,
        "web",
        integer=True,
    )
    breaker_seconds = _number(
        table, "supplement_breaker_seconds", default.supplement_breaker_seconds, "web"
    )
    for key in ("gemini_url_context", "deepseek_fallback"):
        if not isinstance(table.get(key, getattr(default, key)), bool):
            raise ConfigError(f"web.{key}: expected true or false")
    credentials_model = table.get("gemini_credentials_model", default.gemini_credentials_model)
    if not isinstance(credentials_model, str) or not credentials_model.strip():
        raise ConfigError("web.gemini_credentials_model: expected a model alias")
    bindings = _web_bindings(data, table)
    return WebSpec(
        provider,
        key_env,
        base_url.rstrip("/"),
        price,
        search_provider,
        gemini_model,
        level,
        fee,
        supplement,
        float(min_seconds),
        breaker_failures,
        float(breaker_seconds),
        table.get("gemini_url_context", default.gemini_url_context),
        table.get("deepseek_fallback", default.deepseek_fallback),
        credentials_model.strip(),
        **bindings,
    )


WEB_BINDING_KEYS = ("search", "search_fallback", "fetch", "fetch_fallback")
LEGACY_WEB_KEYS = (
    "search_provider",
    "gemini_model",
    "gemini_credentials_model",
    "gemini_url_context",
    "deepseek_fallback",
    "gemini_supplement",
)


def _web_bindings(data: dict, table: dict) -> dict:
    """V3: `[web] search/fetch (+ _fallback)` bind each operation to "deepseek" or a
    google-genai model alias. Absent `search` keeps the legacy keys."""
    if "search" not in table:
        for key in WEB_BINDING_KEYS[1:]:
            if key in table:
                raise ConfigError(f"web.{key}: requires web.search")
        return {}
    legacy = [key for key in LEGACY_WEB_KEYS if key in table]
    if legacy:
        raise ConfigError(f"web.{legacy[0]}: not used with web.search; remove it")
    models = _table(data, "models")
    out = {}
    for key in WEB_BINDING_KEYS:
        value = table.get(key, "deepseek" if key == "fetch" else "")
        if not isinstance(value, str) or (key in ("search", "fetch") and not value):
            raise ConfigError(f"web.{key}: expected 'deepseek' or a google-genai model alias")
        if value and value != "deepseek":
            model = models.get(value)
            if not isinstance(model, dict) or model.get("protocol") != "google-genai":
                raise ConfigError(f"web.{key}: expected 'deepseek' or a google-genai model alias")
        out[key] = value
    for key in ("search", "fetch"):
        if out[f"{key}_fallback"] == out[key]:
            raise ConfigError(f"web.{key}_fallback: must differ from web.{key}")
    return out


def load_settings(path: Path) -> Settings:
    from .config_service import load_bundle

    return load_bundle(path).settings


def parse_settings(data: dict, path: Path) -> Settings:
    """Single semantic validator used by files, revisions and CLI previews."""
    runtime = _table(data, "runtime")
    defaults = RuntimeLimits()
    runtime_values = {
        k: (
            _whole(runtime, k, getattr(defaults, k), "runtime", minimum=0)
            if k in GROUP_RUNTIME_KEYS
            else _number(
                runtime,
                k,
                getattr(defaults, k),
                "runtime",
                integer=type(getattr(defaults, k)) is int,
            )
        )
        for k in defaults.__dataclass_fields__
    }
    limits = RuntimeLimits(**runtime_values)
    if not 0 < limits.degrade_ratio < limits.cap_ratio <= 1:
        raise ConfigError("runtime.degrade_ratio: expected 0 < degrade_ratio < cap_ratio <= 1")
    if limits.group_admit_max and limits.group_admit_window_minutes < 1:
        raise ConfigError("runtime.group_admit_window_minutes: expected at least 1")
    session = parse_session(_table(data, "session"), "session", SessionSpec())
    identity, budgets, retention = (_table(data, n) for n in ("identity", "budgets", "retention"))
    r = Retention(
        **{
            k: _number(retention, k, getattr(Retention(), k), "retention", integer=True)
            for k in Retention.__dataclass_fields__
        }
    )
    if r.recent_hours > r.related_days * 24:
        raise ConfigError("retention.recent_hours: must not exceed related_days")
    if r.max_branch_messages > r.max_messages_per_group:
        raise ConfigError("retention.max_branch_messages: must not exceed max_messages_per_group")
    secrets_file = data.get("secrets_file", "../private/credentials.env")
    if not isinstance(secrets_file, str) or not secrets_file:
        raise ConfigError("secrets_file: expected a path")
    secrets_path = Path(secrets_file)
    if not secrets_path.is_absolute():
        secrets_path = path.resolve().parent / secrets_path
    shadow = _table(data, "shadow")
    shadow_db = shadow.get("db_path", "../data/shadow/shadow.sqlite3")
    if not isinstance(shadow_db, str) or not shadow_db:
        raise ConfigError("shadow.db_path: expected a path")
    shadow_path = Path(shadow_db)
    if not shadow_path.is_absolute():
        shadow_path = (path.resolve().parent / shadow_path).resolve()
    serve = _table(data, "serve")
    serve_db = serve.get("db_path", "../data/serve/serve.sqlite3")
    if not isinstance(serve_db, str) or not serve_db:
        raise ConfigError("serve.db_path: expected a path")
    serve_path = Path(serve_db)
    if not serve_path.is_absolute():
        serve_path = (path.resolve().parent / serve_path).resolve()
    knowledge = _table(data, "knowledge")
    knowledge_db = knowledge.get("db_path", "../data/knowledge/knowledge.sqlite3")
    if not isinstance(knowledge_db, str) or not knowledge_db:
        raise ConfigError("knowledge.db_path: expected a path")
    knowledge_path = Path(knowledge_db)
    if not knowledge_path.is_absolute():
        knowledge_path = (path.resolve().parent / knowledge_path).resolve()
    owner_send = serve.get("owner_command_send", False)
    if not isinstance(owner_send, bool):
        raise ConfigError("serve.owner_command_send: expected true or false")
    self_sync = serve.get("self_sync", True)
    if not isinstance(self_sync, bool):
        raise ConfigError("serve.self_sync: expected true or false")
    sync_interval = float(_number(serve, "self_sync_interval_s", 5.0, "serve"))
    fallback_reply = serve.get("fallback_reply", DEFAULT_FALLBACK_REPLY)
    if not isinstance(fallback_reply, str) or len(fallback_reply) > FALLBACK_REPLY_MAX_CHARS:
        raise ConfigError("serve.fallback_reply: expected text of at most 100 characters")
    reaction = serve.get("processing_reaction", False)
    if not isinstance(reaction, bool):
        raise ConfigError("serve.processing_reaction: expected true or false")
    live_send = serve.get("live_send", False)
    if not isinstance(live_send, bool):
        raise ConfigError("serve.live_send: expected true or false")
    owner_control = serve.get("owner_control", True)
    if not isinstance(owner_control, bool):
        raise ConfigError("serve.owner_control: expected true or false")
    emoji = serve.get("reaction_emoji_id", "424")
    if (
        isinstance(emoji, bool)
        or not isinstance(emoji, (str, int))
        or not EMOJI_ID.fullmatch(str(emoji))
    ):
        raise ConfigError("serve.reaction_emoji_id: expected a numeric face id")
    models_table = _table(data, "models")
    answer_loop = _table(data, "agent").get("answer_tool_loop", True)
    if not isinstance(answer_loop, bool):
        raise ConfigError("agent.answer_tool_loop: expected a boolean")
    agent_table = _table(data, "agent")
    if not isinstance(agent_table.get("web_enabled", True), bool):
        raise ConfigError("agent.web_enabled: expected true or false")
    decision_mode = agent_table.get("decision_mode", "lightweight")
    if decision_mode not in ("model", "lightweight"):
        raise ConfigError("agent.decision_mode: expected model or lightweight")
    answer_thinking = agent_table.get("answer_thinking", "complex_only")
    if not isinstance(answer_thinking, str) or answer_thinking not in (
        "auto",
        "complex_only",
        "default",
        *GOOGLE_THINKING_LEVELS,
    ):
        raise ConfigError(
            "agent.answer_thinking: expected auto, complex_only, default or a thinking level"
        )
    same_model_retry = agent_table.get("answer_same_model_retry", False)
    if not isinstance(same_model_retry, bool):
        raise ConfigError("agent.answer_same_model_retry: expected true or false")
    breaker_failures = _number(models_table, "breaker_failures", 2, "models", integer=True)
    breaker_cooldown = float(_number(models_table, "breaker_cooldown_s", 300.0, "models"))
    billing = _billing(data, path)
    snowluma_db = serve.get("snowluma_db")
    snowluma_path = None
    if snowluma_db is not None:
        if not isinstance(snowluma_db, str) or not snowluma_db:
            raise ConfigError("serve.snowluma_db: expected a path")
        snowluma_path = Path(snowluma_db)
        if not snowluma_path.is_absolute():
            snowluma_path = (path.resolve().parent / snowluma_path).resolve()
    return Settings(
        bot_id=_identifier(identity.get("bot_id", "BOT"), "identity.bot_id"),
        owner_id=_identifier(identity.get("owner_id", "OWNER"), "identity.owner_id"),
        blocked_users=_identifiers(identity, "blocked_users"),
        blocked_groups=_identifiers(identity, "blocked_groups"),
        deadline_seconds=_number(budgets, "deadline_seconds", 120, "budgets"),
        context_messages=_number(budgets, "context_messages", 80, "budgets", integer=True),
        daily_cost_cny=float(_number(budgets, "daily_cost_cny", 3.0, "budgets")),
        retention=r,
        collection=_collection(data, path),
        secrets_file=secrets_path,
        shadow_db=shadow_path,
        serve_db=serve_path,
        knowledge_db=knowledge_path,
        owner_command_send=owner_send,
        fallback_reply=fallback_reply.strip(),
        self_sync=self_sync,
        self_sync_interval_s=sync_interval,
        snowluma_db=snowluma_path,
        processing_reaction=reaction,
        reaction_emoji_id=str(emoji),
        live_send=live_send,
        owner_control=owner_control,
        models=_models(data, billing.usd_cny),
        model_roles=_model_roles(data),
        group_profiles=data.get("group_profiles", {}),
        answer_tool_loop=answer_loop,
        web_enabled=agent_table.get("web_enabled", True),
        decision_mode=decision_mode,
        answer_thinking=answer_thinking,
        answer_same_model_retry=same_model_retry,
        model_breaker_failures=breaker_failures,
        model_breaker_cooldown_s=breaker_cooldown,
        web=_web(data),
        billing=billing,
        runtime=limits,
        session=session,
        prompts=data.get("_prompt_text", {}),
        credential_refs=data.get("credentials", {}),
    )


def _read_env(path: Path) -> dict[str, str]:
    from .credentials import CredentialStore

    if isinstance(path, CredentialStore):
        return path.read()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        raise ConfigError("secrets_file: cannot read") from None
    values = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError("secrets_file: expected KEY=VALUE")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_snowluma_credentials(path: Path) -> SnowLumaCredentials:
    """Shadow ingestion may only reach a SnowLuma WebSocket on this machine."""
    values = _read_env(path)
    for name in ("SNOWLUMA_WS_URL", "SNOWLUMA_WS_TOKEN"):
        if not values.get(name):
            raise ConfigError(f"secrets_file.{name}: required")
    try:
        u = urlsplit(values["SNOWLUMA_WS_URL"])
        if (
            u.scheme != "ws"
            or u.hostname not in LOOPBACK_HOSTS
            or u.username
            or u.password
            or u.query
            or u.fragment
        ):
            raise ValueError
        _ = u.port
    except ValueError:
        raise ConfigError(
            "secrets_file.SNOWLUMA_WS_URL: expected ws:// URL on a loopback address"
        ) from None
    return SnowLumaCredentials(values["SNOWLUMA_WS_URL"], values["SNOWLUMA_WS_TOKEN"])


def _relay_credentials(path: Path, url_name: str, key_name: str) -> tuple[str, str]:
    values = _read_env(path)
    for name in (url_name, key_name):
        if not values.get(name):
            raise ConfigError(f"secrets_file.{name}: required")
    try:
        u = urlsplit(values[url_name])
        if (
            u.scheme != "https"
            or not u.hostname
            or u.username
            or u.password
            or u.query
            or u.fragment
        ):
            raise ValueError
        _ = u.port
    except ValueError:
        raise ConfigError(
            f"secrets_file.{url_name}: expected credential-free HTTPS base URL"
        ) from None
    return values[url_name].rstrip("/"), values[key_name]


def load_web_api_key(path: Path, spec: WebSpec) -> str:
    """Search/fetch key from the secrets file; the value never appears in errors."""
    value = _read_env(path).get(spec.api_key_env)
    if not value:
        raise ConfigError(f"secrets_file.{spec.api_key_env}: required")
    return value


def load_model_credentials(path: Path, spec: ModelSpec) -> ModelCredentials:
    """Resolve the env names of one model table; values never appear in repr or errors."""
    if spec.vertex:
        values = _read_env(path)
        key = values.get(spec.api_key_env)
        if not key:
            raise ConfigError(f"secrets_file.{spec.api_key_env}: required")
        project_env = spec.project_env or "GOOGLE_CLOUD_PROJECT"
        project = values.get(project_env, "")
        if not VERTEX_PROJECT.fullmatch(project):
            raise ConfigError(f"secrets_file.{project_env}: expected a project ID or number")
        return ModelCredentials("", key, project)
    if spec.base_url:
        key = _read_env(path).get(spec.api_key_env)
        if not key:
            raise ConfigError(f"secrets_file.{spec.api_key_env}: required")
        return ModelCredentials(spec.base_url, key)
    return ModelCredentials(*_relay_credentials(path, spec.base_url_env, spec.api_key_env))
