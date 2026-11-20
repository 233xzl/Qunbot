from pathlib import Path

import pytest

from group_agent.config import (
    ConfigError,
    load_model_credentials,
    Settings,
    load_settings,
)


def test_example_is_offline():
    settings = load_settings(Path("config/settings.example.toml"))
    assert settings.bot_id == "BOT"
    assert settings.retention.related_days == 7


@pytest.mark.parametrize(
    "body,field",
    [
        ("[budgets]\ndeadline_seconds = -1", "budgets.deadline_seconds"),
        ("[budgets]\ncontext_messages = true", "budgets.context_messages"),
        ("[budgets]\ndeadline_seconds = nan", "budgets.deadline_seconds"),
        ("[retention]\nrecent_hours = 200", "retention.recent_hours"),
        ('[identity]\nblocked_users = "secret"', "identity.blocked_users"),
        ('[collection]\ndisabled_groups = "secret"', "collection.disabled_groups"),
        ('[collection]\nimage_host_suffixes = ["secret/x"]', "collection.image_host_suffixes"),
        ("[collection]\nmax_image_mb = 0", "collection.max_image_mb"),
        ("[retention]\ncleanup_interval_minutes = 0", "retention.cleanup_interval_minutes"),
    ],
)
def test_bad_settings_report_field_only(tmp_path, body, field):
    path = tmp_path / "settings.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_settings(path)
    assert field in str(error.value)
    assert "secret" not in str(error.value)


def test_breaker_and_reaction_settings(tmp_path):
    """D-20: [models] breaker scalars sit beside the model tables; [serve] reaction."""
    default = load_settings(Path("config/settings.example.toml"))
    assert (default.model_breaker_failures, default.model_breaker_cooldown_s) == (2, 300.0)
    assert default.processing_reaction is True and default.reaction_emoji_id == "424"
    assert default.daily_cost_cny == 20 and default.runtime.degrade_ratio == 0.9
    path = tmp_path / "settings.toml"
    path.write_text(
        "[models]\nbreaker_failures = 3\nbreaker_cooldown_s = 60\n"
        "[serve]\nprocessing_reaction = true\nreaction_emoji_id = 76\n",
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert (settings.model_breaker_failures, settings.model_breaker_cooldown_s) == (3, 60.0)
    assert settings.models == {}
    assert settings.processing_reaction is True and settings.reaction_emoji_id == "76"
    for body, field in (
        ("[models]\nbreaker_failures = 0", "models.breaker_failures"),
        ("[models]\nbreaker_cooldown_s = -5", "models.breaker_cooldown_s"),
        ('[serve]\nprocessing_reaction = "yes"', "serve.processing_reaction"),
        ('[serve]\nreaction_emoji_id = "x1"', "serve.reaction_emoji_id"),
        ('[serve]\nowner_control = "yes"', "serve.owner_control"),
    ):
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ConfigError) as error:
            load_settings(path)
        assert field in str(error.value)
    assert default.owner_control is True and Settings().owner_control is True
    path.write_text("[serve]\nowner_control = false\n", encoding="utf-8")
    assert load_settings(path).owner_control is False


def test_collection_settings(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        '[collection]\ndisabled_groups = [123, "G2"]\nmedia_dir = "media"\n'
        'image_host_suffixes = ["qpic.cn"]\ndownload_per_second = 0.5\n'
        "[retention]\nmedia_max_mb = 100\n",
        encoding="utf-8",
    )
    settings = load_settings(path)
    collection = settings.collection
    assert collection.disabled_groups == frozenset({"123", "G2"})
    assert collection.media_dir == (tmp_path / "media").resolve()
    assert collection.image_host_suffixes == ("qpic.cn",)
    assert collection.download_per_second == 0.5 and collection.max_image_mb == 10
    assert settings.retention.media_max_mb == 100
    assert settings.retention.max_branch_depth == 16
    default = load_settings(Path("config/settings.example.toml")).collection
    assert default.media_dir is None and default.disabled_groups == frozenset()


def test_secrets_not_in_repr_or_errors(tmp_path):
    path = tmp_path / "test.env"
    path.write_text(
        "RELAY_BASE_URL=https://example.invalid\nGEMINI_API_KEY=secret-test-value", encoding="utf-8"
    )
    credentials = load_model_credentials(
        path, load_settings(Path("config/settings.example.toml")).models["gemini"]
    )
    assert credentials.api_key == "secret-test-value"
    assert "secret-test-value" not in repr(credentials)
    assert "example.invalid" not in repr(credentials)
    path.write_text(
        "RELAY_BASE_URL=https://user:secret@example.invalid\nGEMINI_API_KEY=secret-test-value",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as error:
        load_model_credentials(
            path, load_settings(Path("config/settings.example.toml")).models["gemini"]
        )
    assert "secret" not in str(error.value).replace("secrets_file", "")


def test_missing_key(tmp_path):
    path = tmp_path / "test.env"
    path.write_text("RELAY_BASE_URL=https://example.invalid", encoding="utf-8")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_model_credentials(
            path, load_settings(Path("config/settings.example.toml")).models["gemini"]
        )


def test_example_model_tables():
    models = load_settings(Path("config/settings.example.toml")).models
    assert set(models) == {"gemini", "deepseek"}
    gemini, deepseek = models["gemini"], models["deepseek"]
    assert (gemini.protocol, gemini.api_key_env, gemini.thinking_level) == (
        "google-genai",
        "GEMINI_API_KEY",
        "HIGH",
    )
    assert gemini.api_version == "v1beta"
    assert gemini.price_cached_input == 0.0225
    assert (deepseek.protocol, deepseek.api_key_env, deepseek.thinking_budget_tokens) == (
        "anthropic",
        "DEEPSEEK_API_KEY",
        0,
    )
    assert deepseek.price_input == 2 * gemini.price_input


MODEL_TABLE = """
[models.m]
protocol = "anthropic"
model = "synthetic-model"
base_url_env = "SYN_BASE_URL"
api_key_env = "SYN_API_KEY"
price_input = 1
price_output = 2
"""


@pytest.mark.parametrize(
    "extra,field",
    [
        ('protocol = "openai"', "models.m.protocol"),
        ('thinking_level = "HIGH"', "models.m.thinking_level"),
        ('api_version = "v1beta"', "models.m.api_version"),
        ("thinking_budget_tokens = -1", "models.m.thinking_budget_tokens"),
        ("price_cached_input = -0.1", "models.m.price_cached_input"),
        ("max_output_tokens = 0", "models.m.max_output_tokens"),
        ("timeout_seconds = true", "models.m.timeout_seconds"),
    ],
)
def test_bad_model_table_reports_field(tmp_path, extra, field):
    body = (
        MODEL_TABLE.replace('protocol = "anthropic"\n', "") if "protocol" in extra else MODEL_TABLE
    )
    path = tmp_path / "settings.toml"
    path.write_text(body + extra + "\n", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_settings(path)
    assert field in str(error.value)


@pytest.mark.parametrize(
    "old,new,field",
    [
        ('api_key_env = "SYN_API_KEY"', 'api_key_env = "syn-key"', "models.m.api_key_env"),
        ("price_output = 2", "", "models.m.price_output"),
        ('model = "synthetic-model"', 'model = " "', "models.m.model"),
    ],
)
def test_bad_model_values_report_field(tmp_path, old, new, field):
    path = tmp_path / "settings.toml"
    path.write_text(MODEL_TABLE.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_settings(path)
    assert field in str(error.value)
    assert "syn-key" not in str(error.value)


def _spec(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(MODEL_TABLE, encoding="utf-8")
    return load_settings(path).models["m"]


def test_model_credentials_hidden(tmp_path):
    spec = _spec(tmp_path)
    env = tmp_path / "test.env"
    env.write_text(
        "SYN_BASE_URL=https://relay.example.invalid/\nSYN_API_KEY=secret-model-key\n",
        encoding="utf-8",
    )
    credentials = load_model_credentials(env, spec)
    assert credentials.base_url == "https://relay.example.invalid"
    assert credentials.api_key == "secret-model-key"
    assert "secret-model-key" not in repr(credentials)
    assert "relay.example.invalid" not in repr(credentials)


@pytest.mark.parametrize(
    "body,field",
    [
        ("SYN_BASE_URL=https://relay.example.invalid", "secrets_file.SYN_API_KEY"),
        ("SYN_API_KEY=secret-model-key", "secrets_file.SYN_BASE_URL"),
        (
            "SYN_BASE_URL=http://relay.example.invalid\nSYN_API_KEY=secret-model-key",
            "secrets_file.SYN_BASE_URL",
        ),
        (
            "SYN_BASE_URL=https://u:secret-pw@relay.example.invalid\nSYN_API_KEY=secret-model-key",
            "secrets_file.SYN_BASE_URL",
        ),
    ],
)
def test_model_credential_errors_name_field_only(tmp_path, body, field):
    spec = _spec(tmp_path)
    env = tmp_path / "test.env"
    env.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_model_credentials(env, spec)
    message = str(error.value)
    assert field in message
    assert "secret-model-key" not in message and "secret-pw" not in message
    assert "relay.example.invalid" not in message


def test_web_table_defaults_and_key(tmp_path):
    from group_agent.config import load_web_api_key

    web = load_settings(Path("config/settings.example.toml")).web
    assert (web.provider, web.api_key_env, web.price_per_call) == (
        "deepseek",
        "DEEPSEEK_SEARCH_API_KEY",
        0.005,
    )
    env = tmp_path / "test.env"
    env.write_text("OTHER=x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="DEEPSEEK_SEARCH_API_KEY"):
        load_web_api_key(env, web)
    env.write_text("DEEPSEEK_SEARCH_API_KEY=secret-search-key\n", encoding="utf-8")
    assert load_web_api_key(env, web) == "secret-search-key"


@pytest.mark.parametrize(
    "body,field",
    [
        ('[web]\nprovider = "gemini"', "web.provider"),
        ('[web]\napi_key_env = "lower"', "web.api_key_env"),
        ('[web]\nbase_url = "http://x.invalid"', "web.base_url"),
        ("[web]\nprice_per_call = -1", "web.price_per_call"),
    ],
)
def test_bad_web_table_reports_field(tmp_path, body, field):
    path = tmp_path / "settings.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_settings(path)
    assert field in str(error.value)


def test_role_bindings_and_openai_protocol(tmp_path):
    from group_agent.models import Ledger
    from group_agent.serve import build_live_config

    source = Path("config/settings.example.toml").read_text(encoding="utf-8")
    source = source.replace("[models.gemini]", "[models.primary]")
    source = source.replace('answer = "gemini"', 'answer = "primary"')
    source = source.replace('decision = "gemini"', 'decision = "deepseek"')
    path = tmp_path / "roles.toml"
    path.write_text(source, encoding="utf-8")
    settings = load_settings(path)
    cfg = build_live_config(
        settings,
        ledger=Ledger(1),
        media_root=tmp_path,
        dry_models=True,
        web=False,
        group_info=None,
        knowledge_path=None,
    )
    assert cfg.gemini.name == "primary" and cfg.decision_model.name == "deepseek"
    assert cfg.answer_tools is True
    source = source.replace('protocol = "anthropic"', 'protocol = "openai-compatible"')
    path.write_text(source, encoding="utf-8")
    assert load_settings(path).models["deepseek"].protocol == "openai-compatible"
    path.write_text(source.replace('answer = "primary"', 'answer = "missing"'), encoding="utf-8")
    with pytest.raises(ConfigError, match="model_roles.answer"):
        load_settings(path)
