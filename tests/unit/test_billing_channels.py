"""B1: channel identity, original-currency prices and the call-record billing fields.

Fakes only: Pydantic AI FunctionModel, a fake google-genai client and an httpx MockTransport."""

import asyncio
from dataclasses import replace

import httpx
import pytest

from group_agent.config import ConfigError, JevCredentials, ModelSpec, default_channel
from group_agent.gemini_search import GeminiSearchClient
from group_agent.models import JevClient, Ledger, LiveModel, Usage, cost_cny, list_cost
from group_agent.store import SCHEMA_VERSION, Store
from group_agent.web import DeepSeekWebClient, WebBackends, WebCredentials, build_web_client
from test_gemini_search import GROUNDED, SPEC, UNGROUNDED, FakeGenai, make_tool
from test_live_replay import GEMINI, JEV, text_model
from test_vertex_routing import FIXED, VERTEX_TABLES, WEB_FIXED, write

USD_TABLE = """
[billing]
usd_cny = 7.0
usd_cny_date = "2026-11-14"
google_search_usd_per_1000 = 10.0

[models.usd_pro]
protocol = "google-genai"
vertex = true
model = "pro-model"
api_key_env = "K"
timeout_seconds = 30
price_currency = "USD"
price_effective = "2026-11-14"
price_input = 1.5
price_output = 7.5

[models.usd_flash]
protocol = "google-genai"
vertex = true
model = "flash-model"
api_key_env = "K"
timeout_seconds = 30
price_currency = "USD"
price_input = 0.75
price_output = 3.75
"""


# -- channel defaults and validation --------------------------------------------------


def test_default_channels(tmp_path):
    models = write(tmp_path, VERTEX_TABLES + FIXED).models
    assert models["vertex_hydrogen"].channel == "vertex"
    assert models["relay_gemini"].channel == "relay"
    assert models["deepseek_official"].channel == "official:api.deepseek.com"
    assert all(m.price_currency == "CNY" and m.cny_per_unit == 1.0 for m in models.values())
    assert all(m.price_effective is None for m in models.values())
    # A hand-built spec gets the same defaults.
    assert GEMINI.channel == "relay" and GEMINI.price_currency == "CNY"
    assert default_channel(
        replace(GEMINI, base_url_env="", base_url="https://API.Example.com/v1")
    ) == ("official:api.example.com")


def test_explicit_channel_and_validation(tmp_path):
    body = VERTEX_TABLES.replace(
        'model = "gemini-4-hydrogen"\napi_key_env = "GOOGLE_CLOUD_API_KEY"',
        'model = "gemini-4-hydrogen"\napi_key_env = "GOOGLE_CLOUD_API_KEY"\nchannel = "vertex:cn_1"',
        1,
    )
    assert write(tmp_path, body).models["vertex_hydrogen"].channel == "vertex:cn_1"
    for extra, field in (
        ('channel = "Vertex"', "channel"),
        ('channel = "a b"', "channel"),
        ("channel = 3", "channel"),
        ('price_currency = "EUR"', "price_currency"),
        ('price_currency = "usd"', "price_currency"),
        ('price_effective = "2026-13-01"', "price_effective"),
        ('price_effective = "11/14/2026"', "price_effective"),
        ("price_effective = 20261114", "price_effective"),
    ):
        bad = VERTEX_TABLES.replace("timeout_seconds = 90", f"timeout_seconds = 90\n{extra}")
        with pytest.raises(ConfigError, match=field):
            write(tmp_path, bad)


def test_billing_table_defaults_and_validation(tmp_path):
    default = write(tmp_path, VERTEX_TABLES).billing
    assert (default.usd_cny, default.usd_cny_date, default.google_search_usd_per_1000) == (
        7.2,
        None,
        14.0,
    )
    assert default.db_path == (tmp_path / "../data/billing/account.sqlite3").resolve()
    settings = write(tmp_path, USD_TABLE)
    assert (settings.billing.usd_cny, settings.billing.usd_cny_date) == (7.0, "2026-11-14")
    for body, field in (
        ("[billing]\nusd_cny = 0", "billing.usd_cny"),
        ("[billing]\nusd_cny = -7", "billing.usd_cny"),
        ('[billing]\nusd_cny = "7"', "billing.usd_cny"),
        ('[billing]\nusd_cny_date = "soon"', "billing.usd_cny_date"),
        ("[billing]\ngoogle_search_usd_per_1000 = 0", "billing.google_search_usd_per_1000"),
        ("[billing]\ndb_path = 3", "billing.db_path"),
    ):
        with pytest.raises(ConfigError, match=field):
            write(tmp_path, body)
    abs_path = tmp_path / "book.sqlite3"
    assert (
        write(tmp_path, f'[billing]\ndb_path = "{abs_path.as_posix()}"').billing.db_path == abs_path
    )


def test_configs_without_billing_keys_keep_their_cny_costs(tmp_path):
    spec = write(tmp_path, VERTEX_TABLES).models["vertex_flash"]  # CNY table, no new keys
    usage = Usage(input=1000, cached=0, output=500, requests=1)
    assert cost_cny(spec, usage) == pytest.approx((1000 * 5.4 + 500 * 27.0) / 1e6)
    assert list_cost(spec, usage) == cost_cny(spec, usage)


# -- USD prices and alias isolation ---------------------------------------------------


def test_usd_prices_convert_with_the_load_time_rate(tmp_path):
    models = write(tmp_path, USD_TABLE).models
    pro, flash = models["usd_pro"], models["usd_flash"]
    assert (pro.price_currency, pro.price_effective, pro.cny_per_unit) == ("USD", "2026-11-14", 7.0)
    usage = Usage(input=4000, cached=0, output=500, requests=1)
    assert list_cost(pro, usage) == pytest.approx((4000 * 1.5 + 500 * 7.5) / 1e6)  # 0.00975 USD
    assert cost_cny(pro, usage) == pytest.approx(0.00975 * 7.0)
    # Cached tokens without a cached price bill at zero, as for CNY tables (the caller sets
    # a cached price when the vendor lists one).
    cached = Usage(input=4000, cached=1000, output=0, requests=1)
    assert list_cost(pro, cached) == pytest.approx(3000 * 1.5 / 1e6)
    # The same usage on another alias uses that alias's table only.
    assert list_cost(flash, usage) == pytest.approx((4000 * 0.75 + 500 * 3.75) / 1e6)
    assert cost_cny(flash, usage) != cost_cny(pro, usage)


def test_usd_spec_built_by_hand_needs_a_rate():
    kwargs = dict(
        name="x",
        protocol="google-genai",
        model="m",
        base_url_env="",
        api_key_env="K",
        timeout_seconds=1,
        price_input=1.0,
        price_output=1.0,
        vertex=True,
    )
    with pytest.raises(ValueError, match="cny_per_unit"):
        ModelSpec(price_currency="USD", **kwargs)
    assert ModelSpec(price_currency="USD", cny_per_unit=7.2, **kwargs).channel == "vertex"


def test_switching_alias_never_reuses_another_aliass_prices(tmp_path):
    models = write(tmp_path, USD_TABLE).models
    ledger = Ledger(10.0)
    results = {}
    for alias in ("usd_pro", "usd_flash"):
        spec = models[alias]
        live = LiveModel(spec, lambda: text_model(), ledger)
        _, record = asyncio.run(live.generate("t", "指令", ["材料"], timeout=5))
        results[alias] = record
    pro, flash = results["usd_pro"], results["usd_flash"]
    assert pro.usage == flash.usage  # same fake usage, different tables
    assert pro.list_cost == pytest.approx(list_cost(models["usd_pro"], pro.usage))
    assert flash.list_cost == pytest.approx(list_cost(models["usd_flash"], flash.usage))
    assert pro.list_cost == pytest.approx(2 * flash.list_cost)  # Hydrogen is 2x Flash per token
    assert pro.cost_cny == pytest.approx(pro.list_cost * 7.0)
    assert flash.cost_cny == pytest.approx(flash.list_cost * 7.0)
    assert ledger.spent == pytest.approx(pro.cost_cny + flash.cost_cny)


# -- call records ---------------------------------------------------------------------


def test_live_model_records_channel_service_and_list_cost(tmp_path):
    spec = write(tmp_path, USD_TABLE).models["usd_pro"]
    ledger = Ledger(10.0)
    live = LiveModel(spec, lambda: text_model(), ledger)
    _, record = asyncio.run(live.generate("answer", "指令", ["材料"], timeout=5))
    assert (record.channel, record.service, record.list_currency) == ("vertex", "generate", "USD")
    assert record.google_queries is None
    settled = ledger.records[0]
    assert (settled.channel, settled.service, settled.list_currency) == (
        "vertex",
        "generate",
        "USD",
    )
    assert settled.list_cost > 0
    assert settled.list_cost == pytest.approx(list_cost(spec, settled.usage))
    assert settled.cost_cny == pytest.approx(settled.list_cost * 7.0)


def test_failed_live_call_keeps_original_currency_estimate(tmp_path):
    spec = write(tmp_path, USD_TABLE).models["usd_pro"]
    ledger = Ledger(10.0)
    live = LiveModel(spec, lambda: text_model(fail=True), ledger)
    with pytest.raises(Exception):
        asyncio.run(live.generate("answer", "指令", ["材料"], timeout=5))
    failed = ledger.records[0]
    assert failed.status == "error" and failed.list_currency == "USD"
    assert failed.list_cost > 0
    assert failed.list_cost == pytest.approx(failed.cost_cny / 7.0)


def test_jev_client_records_judge_service():
    body = {
        "answers": {
            "route": {
                "type": "choice",
                "choice": "web",
                "confidence": 0.4,
                "probabilities": {"chat": 0.3, "web": 0.7},
            }
        },
        "usage": {"input_tokens": 500, "output_tokens": 40},
    }
    ledger = Ledger(1.0)
    client = JevClient(
        JEV,
        JevCredentials("https://relay.invalid", "k"),
        ledger,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )
    questions = {
        "route": {"type": "choice", "criteria": {"chat": "chat", "web": "web"}, "question": "?"}
    }
    _, record = asyncio.run(client.judge("jev", {"current": "x"}, questions))
    for item in (record, ledger.records[0]):
        assert (item.channel, item.service, item.list_currency) == ("relay", "judge", "CNY")
        assert item.list_cost == pytest.approx(500 * 0.21 / 1e6)


def test_deepseek_web_records_official_channel_and_per_call_price():
    def handler(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={"results": [{"title": "T", "url": "https://a.example/x", "summary": "S"}]},
            )
        return httpx.Response(200, json={"title": "页", "content": "正文"})

    client = DeepSeekWebClient(WebCredentials("k-test"), transport=httpx.MockTransport(handler))
    assert client.channel == "official:api.deepseek.com"
    ledger = Ledger(1.0)
    tool = make_tool(client, ledger=ledger)
    asyncio.run(tool.search("x"))
    asyncio.run(tool.fetch("https://a.example/x"))
    search, fetch = tool.records[0], tool.records[-1]
    assert (search.channel, search.service) == ("official:api.deepseek.com", "search")
    assert (fetch.channel, fetch.service) == ("official:api.deepseek.com", "fetch")
    for record in (search, fetch):
        assert record.list_currency == "CNY" and record.list_cost == pytest.approx(0.003)
        assert record.google_queries is None
    assert [r.service for r in ledger.records] == ["search", "fetch"]
    custom = DeepSeekWebClient(WebCredentials("k", "https://API.Example.com"))
    assert custom.channel == "official:api.example.com"


def make_vertex_client(fake, spec=None, **kw):
    spec = spec or replace(SPEC, channel="vertex", price_currency="USD", cny_per_unit=7.0)
    return GeminiSearchClient(
        spec,
        None,
        model="m",
        client_factory=lambda t: fake,
        fee_per_query=0.014 * 7.0,
        fee_list_per_query=0.014,
        fee_currency="USD",
        **kw,
    )


def test_google_search_records_queries_and_usd_list_cost():
    client = make_vertex_client(FakeGenai(GROUNDED))
    ledger = Ledger(10.0)
    tool = make_tool(client, ledger=ledger)
    assert asyncio.run(tool.search("python"))
    record = tool.records[0]
    assert (record.channel, record.service, record.model) == ("vertex", "search", "gemini_search")
    assert record.google_queries == 2
    tokens = (250 * 0.225 + 500 * 1.125) / 1e6  # SPEC prices in USD here
    assert record.list_currency == "USD"
    assert record.list_cost == pytest.approx(tokens + 2 * 0.014)
    assert record.cost_cny == pytest.approx((tokens + 2 * 0.014) * 7.0)
    assert ledger.records[0] is record


def test_google_search_without_sources_still_counts_queries():
    # A grounded response whose sources are all unusable: 1 query issued, no source kept.
    body = {
        "candidates": [
            {
                "content": {"parts": [{"text": "x"}]},
                "groundingMetadata": {"webSearchQueries": ["only one"]},
            }
        ],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 100},
    }
    tool = make_tool(make_vertex_client(FakeGenai(body)), ledger=Ledger(10.0))
    assert asyncio.run(tool.search("q")) is None
    record = tool.records[0]
    assert record.error == "no_sources" and record.google_queries == 1
    assert record.list_cost == pytest.approx((20 * 0.225 + 100 * 1.125) / 1e6 + 0.014)
    assert record.cost_cny == pytest.approx(record.list_cost * 7.0)
    # Without any grounding metadata the response proves zero queries were issued.
    tool = make_tool(make_vertex_client(FakeGenai(UNGROUNDED)), ledger=Ledger(10.0))
    assert asyncio.run(tool.search("q")) is None
    assert tool.records[0].google_queries == 0


def test_google_search_timeout_bills_estimate_in_table_currency_with_unknown_queries():
    tool = make_tool(
        make_vertex_client(FakeGenai(GROUNDED, delay=5)),
        ledger=Ledger(10.0),
        remaining=31.5,
    )
    asyncio.run(tool.search("q"))
    record = tool.records[0]
    assert record.error == "timeout" and record.google_queries is None
    assert record.list_currency == "USD" and record.list_cost == pytest.approx(
        record.cost_cny / 7.0
    )


def test_url_context_fetch_records_fetch_service_on_its_channel():
    page_body = {
        "candidates": [
            {
                "content": {"parts": [{"text": "摘要"}]},
                "urlContextMetadata": {
                    "urlMetadata": [
                        {
                            "retrievedUrl": "https://a.example/p",
                            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS",
                        }
                    ]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 100},
    }
    client = make_vertex_client(FakeGenai(page_body), use_url_context=True)
    tool = make_tool(client, ledger=Ledger(10.0))
    asyncio.run(tool.fetch("https://a.example/p"))
    record = tool.records[0]
    assert (record.channel, record.service, record.model) == (
        "vertex",
        "fetch",
        "gemini_url_context",
    )
    assert record.google_queries is None and record.list_currency == "USD"
    assert record.list_cost == pytest.approx((1000 * 0.225 + 100 * 1.125) / 1e6)


def test_v3_vertex_search_fee_comes_from_billing_table(tmp_path):
    body = (
        VERTEX_TABLES
        + FIXED
        + WEB_FIXED
        + "\n[billing]\nusd_cny = 7.0\ngoogle_search_usd_per_1000 = 10.0\n"
    )
    settings = write(tmp_path, body)
    backends = build_web_client(settings)
    assert isinstance(backends, WebBackends)
    search = backends.search
    assert search.channel == "vertex"
    assert search.fee_per_query == pytest.approx(0.01 * 7.0)
    assert (search.fee_list_per_query, search.fee_currency) == (0.01, "USD")
    assert search.price_unit("search") == ("CNY", 1.0)  # this table is CNY
    # The DeepSeek fallback is the official channel, priced per call in CNY.
    assert backends.search_fallback.channel == "official:api.deepseek.com"


def test_relay_search_keeps_the_legacy_cny_query_fee(tmp_path):
    body = (
        VERTEX_TABLES
        + '[web]\nsearch = "relay_gemini"\nsearch_fallback = "deepseek"\nfetch = "deepseek"\n'
        + "gemini_fee_per_query = 0.05\n[billing]\ngoogle_search_usd_per_1000 = 99.0\n"
    )
    search = build_web_client(write(tmp_path, body)).search
    assert search.channel == "relay"
    assert (search.fee_per_query, search.fee_list_per_query, search.fee_currency) == (
        0.05,
        0.05,
        "CNY",
    )


# -- usage_ledger (serve store schema v9) ---------------------------------------------


def test_store_v8_database_migrates_to_v9_and_keeps_rows(tmp_path):
    import sqlite3

    path = tmp_path / "v8.db"
    Store(path).close()
    con = sqlite3.connect(path)
    for column in ("channel", "service", "list_cost", "list_currency", "google_queries"):
        con.execute(f"ALTER TABLE usage_ledger DROP COLUMN {column}")
    con.execute(
        "INSERT INTO usage_ledger(run_id,day,created_at,provider,model,task,mode,status,"
        "input_tokens,cached_tokens,output_tokens,thinking_tokens,requests,cost_cny,latency_s)"
        " VALUES ('r1','2026-11-13',1.0,'gemini','m','answer','text','ok',10,0,5,0,1,0.25,1.0)"
    )
    con.execute("PRAGMA user_version=8")
    con.commit()
    con.close()
    store = Store(path)
    assert store.con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    old = store.con.execute("SELECT * FROM usage_ledger").fetchone()
    assert old["cost_cny"] == 0.25 and old["provider"] == "gemini"
    assert all(old[c] is None for c in ("channel", "service", "list_cost", "google_queries"))
    assert store.usage_total("2026-11-13") == pytest.approx(0.25)
    store.record_usage(
        {
            "run_id": "r2",
            "day": "2026-11-13",
            "created_at": 2.0,
            "provider": "gemini_search",
            "model": "m",
            "task": "web_search",
            "mode": "http",
            "status": "ok",
            "input_tokens": 1,
            "cached_tokens": 0,
            "output_tokens": 1,
            "thinking_tokens": 0,
            "requests": 1,
            "cost_cny": 0.5,
            "latency_s": 1.0,
            "channel": "vertex",
            "service": "search",
            "list_cost": 0.07,
            "list_currency": "USD",
            "google_queries": 2,
        }
    )
    assert store.usage_total("2026-11-13") == pytest.approx(0.75)
    new = store.con.execute("SELECT * FROM usage_ledger WHERE run_id='r2'").fetchone()
    assert (new["channel"], new["service"], new["list_currency"], new["google_queries"]) == (
        "vertex",
        "search",
        "USD",
        2,
    )
    assert store.usage_summary("2026-11-13")["gemini_search"]["calls"] == 1
    store.close()


def test_store_record_usage_accepts_rows_without_the_v9_keys():
    store = Store()
    store.record_usage(
        {
            "run_id": None,
            "day": "2026-11-14",
            "created_at": 1.0,
            "provider": "p",
            "model": "m",
            "task": "t",
            "mode": "text",
            "status": "ok",
            "input_tokens": 1,
            "cached_tokens": 0,
            "output_tokens": 1,
            "thinking_tokens": 0,
            "requests": 1,
            "cost_cny": 0.1,
            "latency_s": 0.1,
        }
    )
    assert store.usage_total("2026-11-14") == pytest.approx(0.1)


def test_usage_ledger_persists_billing_columns():
    from group_agent.usage import UsageLedger

    store = Store()
    ledger = UsageLedger(store, clock=lambda: 1_700_000_000.0)
    live = LiveModel(GEMINI, lambda: text_model(), ledger)
    asyncio.run(live.generate("answer", "指令", ["材料"], timeout=5))
    row = store.con.execute("SELECT * FROM usage_ledger").fetchone()
    assert (row["channel"], row["service"], row["list_currency"]) == ("relay", "generate", "CNY")
    assert row["list_cost"] == pytest.approx(row["cost_cny"]) and row["google_queries"] is None
    assert ledger.counts["rows"] == 1 and ledger.counts["write_failed"] == 0
