import json
import secrets
import importlib

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic_ai import Agent

from group_agent.admin.cli import initialize
from group_agent.admin.server import create_app
from group_agent.config import ModelCredentials, ModelSpec, load_model_credentials
from group_agent.config_service import load_bundle
from group_agent.provider_adapters import build_pai_model


def test_local_key_keep_rotate_restart_and_conflict(tmp_path):
    key = initialize(tmp_path)
    secret1, secret2 = secrets.token_urlsafe(24), secrets.token_urlsafe(24)

    def client():
        return TestClient(
            create_app(tmp_path, demo=True, allowed_hosts={"testserver"}),
            headers={"X-Qunbot-Request": "1"},
        )

    with client() as c:
        c.post("/api/login", json={"key": key}).raise_for_status()
        cfg = c.get("/api/config").json()
        data = cfg["data"]
        data["connections"] = {
            "relay": {"protocol": "openai-responses", "base_url": "https://unit.invalid/v1"}
        }
        data["models"]["unit"] = {
            "connection": "relay",
            "model": "test-model",
            "price_input": 1,
            "price_output": 2,
        }

        def save(value, expected):
            return c.post(
                "/api/config/draft",
                json={
                    "data": data,
                    "expected": expected,
                    "provider_keys": [{"group": "connections", "name": "relay", "value": value}],
                },
            )

        r = save(secret1, cfg["draft_revision"])
        assert r.is_success, r.text
        rev = r.json()["draft_revision"]
        cfg = c.get("/api/config").json()
        data = cfg["data"]
        ref = data["connections"]["relay"]["api_key_env"]
        assert cfg["credentials"][ref]["configured"]
        stored = tmp_path / "private" / "provider-keys.env"
        assert secret1 in stored.read_text()
        assert save(secret2, "stale").status_code == 400
        assert secret1 in stored.read_text()
        save("", rev).raise_for_status()
        rev = c.get("/api/config").json()["draft_revision"]
        r = save(secret2, rev)
        r.raise_for_status()
        c.post(
            "/api/config/apply",
            json={
                "draft_revision": r.json()["draft_revision"],
                "expected": cfg["published_revision"],
            },
        ).raise_for_status()
        assert secret2 in stored.read_text() and secret1 not in stored.read_text()
        for file in (tmp_path / "config-history").rglob("*"):
            if file.is_file():
                assert (
                    secret1.encode() not in file.read_bytes()
                    and secret2.encode() not in file.read_bytes()
                )
    with client() as c:
        c.post("/api/login", json={"key": key}).raise_for_status()
        r = c.get("/api/config")
        assert secret2 not in r.text and r.json()["credentials"][ref]["configured"]
    settings = load_bundle(tmp_path / "config-history").settings
    assert load_model_credentials(settings.credentials, settings.models["unit"]).api_key == secret2


@pytest.mark.parametrize(
    "protocol,base,path,auth",
    [
        ("openai-responses", "https://unit.invalid/v1", "/v1/responses", "authorization"),
        ("openai-compatible", "https://unit.invalid/v1", "/v1/chat/completions", "authorization"),
        ("anthropic", "https://unit.invalid", "/v1/messages", "x-api-key"),
        (
            "google-genai",
            "https://unit.invalid",
            "/v1beta/models/gemini-2.5-flash:generateContent",
            "x-goog-api-key",
        ),
    ],
)
async def test_provider_wire_requests(monkeypatch, protocol, base, path, auth):
    model = "gemini-2.5-flash" if protocol == "google-genai" else "test-model"
    spec = ModelSpec(
        name="unit",
        protocol=protocol,
        model=model,
        api_key_env="KEY",
        base_url_env="",
        base_url=base,
        timeout_seconds=3,
        price_input=1,
        price_output=1,
        api_version="v1beta" if protocol == "google-genai" else None,
    )
    called = []

    async def handle(r):
        called.append((r.url.path, dict(r.headers), json.loads(r.content)))
        if protocol == "openai-responses":
            body = dict(
                id="resp_1",
                object="response",
                created_at=1,
                status="completed",
                model=model,
                output=[
                    dict(
                        type="message",
                        role="assistant",
                        id="msg_1",
                        status="completed",
                        content=[dict(type="output_text", text="ok", annotations=[])],
                    )
                ],
                usage=dict(input_tokens=2, output_tokens=1, total_tokens=3),
            )
        elif protocol == "openai-compatible":
            body = dict(
                id="chat",
                object="chat.completion",
                created=1,
                model=model,
                choices=[
                    dict(
                        index=0, message=dict(role="assistant", content="ok"), finish_reason="stop"
                    )
                ],
                usage=dict(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            )
        elif protocol == "anthropic":
            body = dict(
                id="msg",
                type="message",
                role="assistant",
                model=model,
                content=[dict(type="text", text="ok")],
                stop_reason="end_turn",
                usage=dict(input_tokens=2, output_tokens=1),
            )
        else:
            body = dict(
                candidates=[
                    dict(content=dict(role="model", parts=[dict(text="ok")]), finishReason="STOP")
                ],
                usageMetadata=dict(promptTokenCount=2, candidatesTokenCount=1, totalTokenCount=3),
            )
        module = importlib.import_module(type(r).__module__.split(".")[0])
        return module.Response(200, json=body)

    for lib in [httpx] + (
        [importlib.import_module("httpx2")] if importlib.util.find_spec("httpx2") else []
    ):
        transport = lib.MockTransport(handle)
        monkeypatch.setattr(lib.AsyncClient, "_transport_for_url", lambda self, url, t=transport: t)
    pai = build_pai_model(spec, ModelCredentials(base, "test-value"))
    result = await Agent(pai).run("hello")
    assert (
        result.output == "ok" and result.usage.input_tokens == 2 and result.usage.output_tokens == 1
    )
    assert len(called) == 1 and called[0][0] == path and "test-value" in called[0][1][auth]
    assert "hello" in json.dumps(called[0][2])
