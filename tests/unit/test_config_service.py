import json

import pytest

from group_agent.config import ConfigError, load_model_credentials, load_settings
from group_agent.config_service import ConfigService, effective, load_bundle, migrate


def draft(root, text="", persona="first persona"):
    (root / "persona.md").write_text(persona, encoding="utf-8")
    path = root / "settings.toml"
    path.write_text(
        'schema_version = 1\n[prompts]\npersona = "persona.md"\n' + text, encoding="utf-8"
    )
    return path


def test_publish_snapshot_conflict_and_rollback(tmp_path):
    source = draft(tmp_path, "[runtime]\nweb_searches = 1\n")
    service = ConfigService(tmp_path / "versions-store")
    first = service.publish(source, None)
    (tmp_path / "persona.md").write_text("second persona", encoding="utf-8")
    assert load_bundle(service.root).settings.prompts["persona"] == "first persona"
    assert "prompts.persona" in service.diff(source)
    with pytest.raises(ConfigError, match="reload"):
        service.publish(source, "stale")
    second = service.publish(source, first["revision"])
    restored = service.rollback(first["revision"], second["revision"])
    settings = load_settings(service.root)
    assert settings.prompts["persona"] == "first persona"
    assert settings.limits.web_searches == 1
    assert settings.serve_db == tmp_path / "data/serve/serve.sqlite3"
    assert restored["revision"] != first["revision"]
    with pytest.raises(TypeError):
        settings.prompts["persona"] = "mutable"
    service.record_runtime(settings, "adopted")
    assert service.report()["pending"] is False
    assert service.report()["last_runtime"]["effective_revision"] == settings.revision


def test_group_tool_limits_inherit_and_round_trip(tmp_path):
    source = draft(
        tmp_path,
        """
[runtime]
web_searches = 4
web_reads = 2
[group_profiles.A]
enabled = true
[group_profiles.A.runtime]
web_searches = 0
[group_profiles.B]
enabled = false
[group_profiles.B.runtime]
web_reads = 9
""",
    )
    service = ConfigService(tmp_path / "versions-store")
    service.publish(source, None)
    settings = load_settings(service.root)
    a = settings.for_group("A").limits
    assert (a.web_searches, a.web_reads) == (0, 2)
    for gid in ("B", "C"):
        assert settings.for_group(gid).limits == settings.limits


@pytest.mark.parametrize(
    "field", ["web_reads = -1", "web_searches = 1.5", "web_reads = true", "deadline_seconds = 1"]
)
def test_group_tool_limits_validate(tmp_path, field):
    with pytest.raises(ConfigError, match=r"group_profiles.A.runtime\."):
        load_settings(draft(tmp_path, f"[group_profiles.A.runtime]\n{field}\n"))


def test_failed_publish_and_external_edit_leave_pointer(tmp_path, monkeypatch):
    import group_agent.config_service as module

    source = draft(tmp_path)
    service = ConfigService(tmp_path / "store")
    first = service.publish(source, None)
    original = module.atomic_write

    def fail_pointer(path, text, **kwargs):
        if path.name == "active.json":
            raise OSError("interrupted")
        original(path, text, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_write", fail_pointer)
        with pytest.raises(OSError):
            service.publish(source, first["revision"])
    assert service.status() == first
    active_prompt = service.active_path().parent / "prompts/persona.md"
    active_prompt.write_text("external edit", encoding="utf-8")
    with pytest.raises(ConfigError, match="outside"):
        load_bundle(service.root)
    with pytest.raises(ConfigError, match="outside"):
        service.publish(source, first["revision"])


def test_migration_paths_and_explicit_cli_precedence(tmp_path, monkeypatch):
    from group_agent.app import main
    import group_agent.app as app

    old = tmp_path / "old.toml"
    old.write_text(
        '[identity]\nbot_id="TEST"\n[serve]\ndb_path="local.sqlite3"\nlive_send=true\n',
        encoding="utf-8",
    )
    target = tmp_path / "moved/settings.toml"
    migrate(load_bundle(old), target)
    assert load_settings(old).serve_db == load_settings(target).serve_db
    seen = []
    monkeypatch.setattr(
        app, "serve_main", lambda parser, settings, args: seen.append(settings) or 0
    )
    main(["--settings", str(target), "--serve"])
    assert seen[-1].live_send is True
    main(["--settings", str(target), "--serve", "--no-live-send"])
    assert seen[-1].live_send is False
    assert seen[-1].sources["serve.live_send"] == "cli"
    assert load_settings(target).live_send is True
    assert seen[-1].revision != seen[0].revision


def test_connections_and_managed_credentials_are_lazy_and_redacted(tmp_path, monkeypatch):
    source = draft(
        tmp_path,
        """
[connections.official]
protocol = "openai-compatible"
base_url = "https://example.invalid"
api_key_env = "TEST_KEY"
[models.answer]
connection = "official"
model = "example"
price_input = 1
price_output = 2
[model_roles]
answer = "answer"
[credentials.TEST_KEY]
source = "managed"
""",
    )
    bundle = load_bundle(source)
    assert bundle.settings.models["answer"].protocol == "openai-compatible"
    store = bundle.settings.credentials
    store.update("TEST_KEY", "replace", "test-private-value")
    credentials = load_model_credentials(store, bundle.settings.models["answer"])
    assert credentials.api_key == "test-private-value"
    assert "test-private-value" not in json.dumps(effective(bundle)) + repr(store) + repr(
        credentials
    )
    store.update("TEST_KEY", "clear")
    monkeypatch.setenv("TEST_KEY", "must-not-fall-back-to-environment")
    with pytest.raises(ConfigError, match="TEST_KEY"):
        load_model_credentials(store, bundle.settings.models["answer"])


@pytest.mark.parametrize(
    "body, field",
    [
        ("[runtime]\nglobal_actve=1", "global_actve"),
        ("[runtime]\ndegrade_ratio=1", "degrade_ratio"),
        ("[budgets]\ndeadline_seconds=10", "finish_reserve_seconds"),
        ("[serve]\nself_sync=true", "snowluma_db"),
    ],
)
def test_invalid_configuration_rejected_before_publication(tmp_path, body, field):
    source = draft(tmp_path, body)
    with pytest.raises(ConfigError, match=field):
        ConfigService(tmp_path / "store").publish(source, None)
    assert not (tmp_path / "store/active.json").exists()


def test_configured_prompt_reaches_model_and_run_record(tmp_path):
    from pydantic_ai.messages import TextPart
    from pydantic_ai.models.function import FunctionModel
    from group_agent.persona import prompt_options
    from test_answer_tool_loop import response, run_loop
    from test_live_replay import live_config, text_only_case

    source = draft(tmp_path, persona="Use a distinctive detailed voice.")
    settings = load_settings(source)
    captured = []

    def answer(messages, info):
        captured.append(str(messages))
        return response(TextPart("configured answer"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(answer),
        lambda: FunctionModel(answer),
        decision_mode="lightweight",
        answer_tools=True,
        config_revision=settings.revision,
        prompt_text=prompt_options(settings),
    )
    _, _, record = run_loop(text_only_case(tmp_path), cfg)
    assert any("Use a distinctive detailed voice." in text for text in captured)
    assert record["config_revision"] == settings.revision
