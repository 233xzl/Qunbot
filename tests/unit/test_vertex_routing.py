"""V1-V3: Vertex factory, fixed generation roles and separate search/fetch bindings.

No network: model builds are inspected offline, web backends are fakes."""

import asyncio
from dataclasses import replace

from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
import pytest

from group_agent.config import (
    ConfigError,
    ModelCredentials,
    load_model_credentials,
    load_settings,
)
from group_agent.gemini_search import GeminiSearchClient
from group_agent.provider_adapters import build_pai_model, google_client, google_model_name
from group_agent.serve import model_ids
from group_agent.web import WebBackends, build_supplement_client, build_web_client
from test_gemini_search import DS_HIT, GROUNDED, FakeDeepSeek, FakeGenai, make_tool
from test_live_replay import live_config, make_case, run_live, text_model

PROJECT = "demo-project-1"
SECRETS = (
    "GOOGLE_CLOUD_API_KEY=v-secret\n"
    f"GOOGLE_CLOUD_PROJECT={PROJECT}\n"
    "DEEPSEEK_SEARCH_API_KEY=d-secret\n"
    "RELAY_BASE_URL=https://relay.example.invalid\n"
    "GEMINI_API_KEY=g-secret\n"
)
VERTEX_TABLES = """
[models.vertex_hydrogen]
protocol = "google-genai"
vertex = true
model = "gemini-4-hydrogen"
api_key_env = "GOOGLE_CLOUD_API_KEY"
timeout_seconds = 90
price_input = 10.8
price_output = 54.0

[models.vertex_flash]
protocol = "google-genai"
vertex = true
location = "global"
project_env = "GOOGLE_CLOUD_PROJECT"
model = "gemini-3.8-flash"
api_key_env = "GOOGLE_CLOUD_API_KEY"
timeout_seconds = 30
thinking_level = "LOW"
price_input = 5.4
price_output = 27.0

[models.deepseek_official]
protocol = "openai-compatible"
model = "deepseek-flash"
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_SEARCH_API_KEY"
timeout_seconds = 45
price_input = 2.0
price_output = 8.0
reasoning_effort = "low"

[models.relay_gemini]
protocol = "google-genai"
base_url_env = "RELAY_BASE_URL"
api_key_env = "GEMINI_API_KEY"
model = "gemini-4-hydrogen"
api_version = "v1beta"
timeout_seconds = 60
price_input = 0.225
price_output = 1.125
"""
FIXED = """
[model_roles]
answer = "vertex_hydrogen"
decision = "vertex_flash"
fallback = "deepseek_official"
"""


def write(tmp_path, body: str):
    env = tmp_path / "secrets.env"
    env.write_text(SECRETS, encoding="utf-8")
    path = tmp_path / "settings.toml"
    path.write_text(f'secrets_file = "{env.as_posix()}"\n' + body, encoding="utf-8")
    return load_settings(path)


# -- V1 -----------------------------------------------------------------------------


def test_vertex_table_defaults_and_credentials(tmp_path):
    settings = write(tmp_path, VERTEX_TABLES + FIXED)
    spec = settings.models["vertex_hydrogen"]
    assert (spec.vertex, spec.location, spec.project_env) == (
        True,
        "global",
        "GOOGLE_CLOUD_PROJECT",
    )
    assert spec.api_version == "v1" and spec.base_url_env == "" and spec.base_url is None
    assert spec.model == "gemini-4-hydrogen"  # short name for prices and records
    relay = settings.models["relay_gemini"]
    assert not relay.vertex and relay.location is None and relay.api_version == "v1beta"
    creds = load_model_credentials(settings.secrets_file, spec)
    assert (creds.api_key, creds.project) == ("v-secret", PROJECT)
    assert "v-secret" not in repr(creds) and PROJECT not in repr(creds)
    assert settings.model_roles == {
        "answer": "vertex_hydrogen",
        "decision": "vertex_flash",
        "fallback": "deepseek_official",
    }


@pytest.mark.parametrize(
    "extra,field",
    [
        ('vertex = true\nbase_url_env = "RELAY_BASE_URL"', "base_url_env"),
        ('vertex = true\nbase_url = "https://relay.example.invalid"', "base_url"),
        ('vertex = true\nlocation = "Global"', "location"),
        ('vertex = true\nproject_env = "lower"', "project_env"),
        ('vertex = "yes"', "vertex"),
        ('location = "global"', "location"),
        ('project_env = "GOOGLE_CLOUD_PROJECT"', "project_env"),
    ],
)
def test_vertex_table_errors_name_the_field(tmp_path, extra, field):
    body = (
        '[models.v]\nprotocol = "google-genai"\nmodel = "m"\napi_key_env = "K"\n'
        "price_input = 0\nprice_output = 0\n"
    )
    if "base_url" not in extra and "vertex = true" not in extra:
        body += 'base_url_env = "RELAY_BASE_URL"\n'
    with pytest.raises(ConfigError, match=f"models.v.{field}"):
        write(tmp_path, body + extra + "\n")


def test_vertex_only_for_google_genai(tmp_path):
    body = (
        '[models.v]\nprotocol = "anthropic"\nvertex = true\nmodel = "m"\napi_key_env = "K"\n'
        'base_url_env = "RELAY_BASE_URL"\nprice_input = 0\nprice_output = 0\n'
    )
    with pytest.raises(ConfigError, match="models.v.vertex"):
        write(tmp_path, body)


@pytest.mark.parametrize("project", ["", "Bad_Project", "zz9"])
def test_vertex_project_is_validated_without_echoing_it(tmp_path, project):
    settings = write(tmp_path, VERTEX_TABLES)
    env = tmp_path / "bad.env"
    env.write_text(f"GOOGLE_CLOUD_API_KEY=k\nGOOGLE_CLOUD_PROJECT={project}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="GOOGLE_CLOUD_PROJECT") as error:
        load_model_credentials(env, settings.models["vertex_flash"])
    assert not project or project not in str(error.value)


def test_vertex_model_uses_resource_path_v1_and_single_attempt(tmp_path):
    settings = write(tmp_path, VERTEX_TABLES)
    spec = settings.models["vertex_hydrogen"]
    model = build_pai_model(spec, load_model_credentials(settings.secrets_file, spec))
    assert model.system == "google-cloud"
    assert model.model_name == (
        f"projects/{PROJECT}/locations/global/publishers/google/models/gemini-4-hydrogen"
    )
    api = model.client._api_client
    options = api._http_options
    assert api.vertexai and options.api_version == "v1"
    assert options.base_url.startswith("https://aiplatform.googleapis.com")
    assert options.retry_options.attempts == 1
    assert model.profile.supports_json_schema_output  # native structured output kept


def test_relay_google_model_is_unchanged(tmp_path):
    settings = write(tmp_path, VERTEX_TABLES)
    spec = settings.models["relay_gemini"]
    model = build_pai_model(spec, load_model_credentials(settings.secrets_file, spec))
    options = model.client._api_client._http_options
    assert model.system == "google" and model.model_name == "gemini-4-hydrogen"
    assert options.base_url.startswith("https://relay.example.invalid")
    assert options.api_version == "v1beta" and not model.client._api_client.vertexai


def test_shared_factory_serves_web_search_requests(tmp_path):
    settings = write(tmp_path, VERTEX_TABLES)
    spec = settings.models["vertex_flash"]
    creds = load_model_credentials(settings.secrets_file, spec)
    client = google_client(spec, creds, 7)
    assert client._api_client.vertexai and client._api_client._http_options.timeout == 7000
    fake = FakeGenai(GROUNDED)
    search = GeminiSearchClient(
        spec, creds, model=spec.model, fee_per_query=0.1, client_factory=lambda t: fake
    )
    asyncio.run(search.search("q", 10))
    assert fake.calls[0]["model"] == google_model_name(spec, creds)
    assert fake.calls[0]["model"].endswith("/publishers/google/models/gemini-3.8-flash")
    # The relay search client still defaults to v1beta.
    relay = settings.models["relay_gemini"]
    relay_client = google_client(
        replace(relay, api_version=None),
        ModelCredentials("https://relay.example.invalid", "k"),
        relay_api_version="v1beta",
    )
    assert relay_client._api_client._http_options.api_version == "v1beta"


# -- V2 -----------------------------------------------------------------------------


def test_fixed_roles_skip_same_model_retry_and_keep_images(tmp_path):
    seen = []

    def deepseek_fn(messages, info):
        seen.append(messages)
        return ModelResponse(parts=[TextPart("备用回答")])

    config = live_config(
        tmp_path,
        lambda: text_model(fail=True),
        lambda: FunctionModel(deepseek_fn),
        decider="deepseek",
        decision_mode="lightweight",
        answer_tools=True,
        answer_same_model_retry=False,
    )
    _, anchor, record = run_live(make_case(tmp_path), config)
    answer_calls = [c for c in record["calls"] if c["stage"] == "answer"]
    assert [(c["model"], c["status"]) for c in answer_calls] == [
        ("gemini", "error"),
        ("deepseek", "ok"),
    ]
    assert record["answer_chain"] == [f"gemini:{record['answer_thinking']}", "deepseek"]
    assert record["answer_source"] == "deepseek" and record["answer_fallback_used"]
    assert "answer_fallback:deepseek" in anchor["degraded"]
    # The fallback receives the same picture as the primary attempt.
    contents = [p.content for m in seen[-1] for p in m.parts if hasattr(p, "content")]
    flat = [c for item in contents for c in (item if isinstance(item, list) else [item])]
    assert any(isinstance(c, BinaryContent) for c in flat)


def test_same_model_retry_remains_available_when_enabled(tmp_path):
    config = live_config(
        tmp_path,
        lambda: text_model(fail=True),
        lambda: text_model("备用"),
        answer_thinking="HIGH",
        answer_same_model_retry=True,
    )
    _, _, record = run_live(make_case(tmp_path), config)
    assert record["answer_chain"] == ["gemini:HIGH", "gemini:LOW", "deepseek"]


def test_settings_default_to_fixed_roles(tmp_path):
    settings = write(tmp_path, VERTEX_TABLES + FIXED)
    assert settings.answer_same_model_retry is False
    on = write(tmp_path, VERTEX_TABLES + FIXED + "[agent]\nanswer_same_model_retry = true\n")
    assert on.answer_same_model_retry is True
    with pytest.raises(ConfigError, match="agent.answer_same_model_retry"):
        write(tmp_path, VERTEX_TABLES + "[agent]\nanswer_same_model_retry = 1\n")


def test_breaker_skip_is_recorded_in_chain(tmp_path):
    from group_agent.models import ModelBreaker

    breaker = ModelBreaker(1, 300.0)
    breaker.record("gemini", "closed", "infra")
    config = live_config(
        tmp_path,
        lambda: text_model(fail=True),
        lambda: text_model("备用"),
        model_breaker=breaker,
        answer_same_model_retry=False,
    )
    _, _, record = run_live(make_case(tmp_path), config)
    assert record["answer_skipped"] == ["gemini:breaker_open"]
    # Images first, then the text-only DeepSeek attempt; Gemini is not called.
    assert record["answer_chain"] == ["deepseek", "deepseek:no_images"]
    assert not any(c["model"] == "gemini" and c["stage"] == "answer" for c in record["calls"])


# -- V3 -----------------------------------------------------------------------------

WEB_FIXED = """
[web]
search = "vertex_flash"
search_fallback = "deepseek"
fetch = "deepseek"
"""


def test_search_and_fetch_bind_separately(tmp_path):
    settings = write(tmp_path, VERTEX_TABLES + FIXED + WEB_FIXED)
    web = settings.web
    assert (web.search, web.search_fallback, web.fetch, web.fetch_fallback) == (
        "vertex_flash",
        "deepseek",
        "deepseek",
        "",
    )
    backends = build_web_client(settings)
    assert isinstance(backends, WebBackends)
    search = backends.search
    # Flash prices and model come from its own table, not from the answer model.
    assert search.price_spec is settings.models["vertex_flash"]
    assert search.model == "gemini-3.8-flash" and not search.use_url_context
    assert backends.search_fallback is backends.fetch  # one DeepSeek client
    assert backends.fetch_fallback is None and backends.provider_name == "gemini_search"
    assert build_supplement_client(settings) is None
    ids = model_ids(settings, dry_models=False)
    assert ids["gemini_search"] == "gemini-3.8-flash" and "gemini_url_context" not in ids
    # A model table named "deepseek" (the relay alias) is not the web keyword.
    relay = replace(settings.models["deepseek_official"], name="deepseek", model="relay-ds")
    clash = replace(settings, models={**settings.models, "deepseek": relay})
    ids = model_ids(clash, dry_models=False)
    assert ids["gemini_search"] == "gemini-3.8-flash" and "gemini_url_context" not in ids


def test_live_config_wires_fixed_roles_and_backends(tmp_path):
    from group_agent.models import Ledger
    from group_agent.serve import build_live_config

    settings = write(tmp_path, VERTEX_TABLES + FIXED + WEB_FIXED)
    cfg = build_live_config(
        settings,
        ledger=Ledger(1.0),
        media_root=tmp_path,
        dry_models=False,
        web=True,
        group_info=None,
        knowledge_path=None,
    )
    assert cfg.gemini.name == "vertex_hydrogen" and cfg.decision_model.name == "vertex_flash"
    assert cfg.deepseek.name == "deepseek_official"
    assert cfg.gemini_creds.project == PROJECT and cfg.decision_creds.project == PROJECT
    assert cfg.answer_same_model_retry is False
    assert isinstance(cfg.web_client, WebBackends) and cfg.web_supplement is None


def test_fetch_can_switch_without_touching_search(tmp_path):
    body = VERTEX_TABLES + WEB_FIXED.replace(
        'fetch = "deepseek"', 'fetch = "vertex_flash"\nfetch_fallback = "deepseek"'
    )
    backends = build_web_client(write(tmp_path, body))
    assert backends.fetch.use_url_context and backends.fetch is not backends.search
    assert backends.fetch.fetch_provider_name == "gemini_url_context"
    assert backends.fetch_fallback.__class__.__name__ == "DeepSeekWebClient"
    assert not backends.search.use_url_context


@pytest.mark.parametrize(
    "web,field",
    [
        ('[web]\nsearch = "relay_gemini"\nsearch_provider = "gemini"', "web.search_provider"),
        ('[web]\nsearch = "deepseek_official"', "web.search"),
        ('[web]\nsearch = ""', "web.search"),
        ('[web]\nsearch = "missing"', "web.search"),
        ('[web]\nsearch = "deepseek"\nsearch_fallback = "deepseek"', "web.search_fallback"),
        ('[web]\nsearch = "deepseek"\nfetch = 1', "web.fetch"),
        ('[web]\nfetch = "deepseek"', "web.fetch"),
        ('[web]\nsearch = "deepseek"\ngemini_supplement = true', "web.gemini_supplement"),
    ],
)
def test_web_binding_errors(tmp_path, web, field):
    with pytest.raises(ConfigError, match=field):
        write(tmp_path, VERTEX_TABLES + web + "\n")


def test_backends_share_run_budget_and_fall_back_per_operation():
    class FakeSearch(FakeDeepSeek):
        provider_name = "gemini_search"

    class Fetcher:
        fetch_provider_name = "deepseek_web"

        def __init__(self):
            self.urls = []

        async def fetch(self, url, timeout):
            self.urls.append(url)
            return {"title": "页", "content": "正文"}

    primary, fallback, fetcher = FakeSearch(fail="http_503"), FakeDeepSeek(DS_HIT), Fetcher()
    tool = make_tool(WebBackends(search=primary, fetch=fetcher, search_fallback=fallback))
    results = asyncio.run(tool.search("vertex"))
    assert results and primary.queries == ["vertex"] and fallback.queries == ["vertex"]
    page = asyncio.run(tool.fetch("https://example.com/a"))
    assert page["content"] == "正文" and fetcher.urls == ["https://example.com/a"]
    # Both search attempts and the fetch spent the run budget and left records.
    summary = tool.summary()
    assert summary["searches"] == 2 and summary["reads"] == 1
    assert [r.model for r in tool.records] == ["gemini_search", "deepseek_web", "deepseek_web"]
    assert tool.search_status("vertex") == "http_503"
