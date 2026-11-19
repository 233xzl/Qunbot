import asyncio
from dataclasses import replace
import json

import pytest

from fastapi.testclient import TestClient

from group_agent.admin.auth import reset_key
from group_agent.admin.cli import initialize
from group_agent.admin.server import create_app
from group_agent.config import ConfigError
from group_agent.config_service import ConfigService, load_bundle
from group_agent.runtime import RuntimeController


def fixture_instance(tmp_path):
    key = initialize(tmp_path)
    service = ConfigService(tmp_path / "config-history")
    return key, service, load_bundle(service.root).settings


def event(settings, mid="one", user="USER"):
    import time

    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": settings.bot_id,
        "group_id": "GROUP",
        "user_id": user,
        "message_id": mid,
        "time": int(time.time()),
        "message": [
            {"type": "at", "data": {"qq": settings.bot_id}},
            {"type": "text", "data": {"text": "你好"}},
        ],
    }


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_engine_lifecycle_lock_and_failed_startup(tmp_path, monkeypatch):
    _, _, settings = fixture_instance(tmp_path)
    controller = RuntimeController(settings, dry_models=True, offline=True, cleanup=False)
    assert (await controller.start())["state"] == "running"
    original = controller.task
    await controller.start()
    assert controller.task is original
    other = RuntimeController(settings, dry_models=True, offline=True)
    assert (await other.start())["state"] == "failed"
    assert "already" in other.error
    await controller.pause()
    assert controller.status()["state"] == "paused"
    await controller.stop()
    await controller.start()
    assert controller.status()["state"] == "paused"
    await controller.resume()
    await controller.stop()
    import group_agent.serve as serve

    closed = []
    close = serve.Store.close

    def closing(store):
        closed.append(True)
        close(store)

    monkeypatch.setattr(serve.Store, "close", closing)

    def fail(*args, **kwargs):
        raise ConfigError("bad model")

    monkeypatch.setattr(serve, "build_live_config", fail)
    assert (await controller.start())["state"] == "failed"
    assert controller.error == "bad model"
    assert closed == [True]


async def test_run_snapshot_and_model_retirement(tmp_path):
    _, _, settings = fixture_instance(tmp_path)
    c = RuntimeController(settings, dry_models=True, offline=True, cleanup=False)
    await c.start()
    try:
        orch = c.session.orchestrator
        old = orch._agent
        entered, release = asyncio.Event(), asyncio.Event()
        original_answer = old.answer

        async def answer(*args):
            entered.set()
            await release.wait()
            return await original_answer(*args)

        old.answer = answer
        c.session.holder["intake"].intake(event(settings))
        await asyncio.wait_for(entered.wait(), 5)
        run_id = next(iter(orch.active))
        new = replace(
            settings, prompts={"persona": "新人格"}, revision="new-persona", daily_cost_cny=8
        )
        result = await c.apply_config(new)
        assert result["state"] == "applied"
        assert orch._agent.config.prompt_text["answer"].startswith("新人格")
        assert orch._run_agents[run_id] is old
        assert orch._agent.gemini is old.gemini
        assert orch.usage.daily_cap_cny == 8
        accepted = orch._agent
        c.session.dry_models = False
        with pytest.raises(ConfigError):
            await c.apply_config(replace(new, revision="unavailable-credentials"))
        c.session.dry_models = True
        assert orch._agent is accepted
        assert c.settings.revision == "new-persona"
        # Explicit credential refresh builds a new client; the active run still holds old.
        retired = []
        original_close = old.gemini.aclose

        async def close():
            retired.append(True)
            await original_close()

        old.gemini.aclose = close
        await c.apply_config(new, reload_credentials=True)
        assert orch._agent.gemini is not old.gemini
        assert not retired
        release.set()
        await until(lambda: not orch.active and bool(retired))
        row = orch.store.con.execute("SELECT detail FROM runs WHERE run_id=?", (run_id,)).fetchone()
        assert json.loads(row[0])["config_revision"] == settings.revision
        assert retired == [True]
        assert orch._run_agents == {}
    finally:
        await c.stop()


async def test_revocation_cancels_and_preserves_budget(tmp_path):
    _, _, settings = fixture_instance(tmp_path)
    c = RuntimeController(settings, dry_models=True, offline=True, cleanup=False)
    await c.start()
    try:
        orch = c.session.orchestrator
        entered = asyncio.Event()

        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()

        orch._agent.answer = blocked
        c.session.holder["intake"].intake(event(settings))
        await asyncio.wait_for(entered.wait(), 5)
        orch.usage._daily = 2
        await c.apply_config(replace(settings, blocked_users=frozenset({"USER"}), daily_cost_cny=6))
        await until(lambda: not orch.active)
        assert orch.usage.daily_spent() == 2
        assert orch.store.con.execute("SELECT state FROM runs").fetchone()[0] == "cancelled"
    finally:
        await c.stop()


def test_admin_login_draft_apply_demo_and_rotation(tmp_path):
    key, service, _ = fixture_instance(tmp_path)
    app = create_app(tmp_path, demo=True, allowed_hosts={"testserver"})
    headers = {"X-Qunbot-Request": "1"}
    with TestClient(app) as client:
        assert client.get("/api/config").status_code == 401
        assert (
            client.post(
                "/api/login", json={"key": key}, headers={**headers, "Origin": "http://elsewhere"}
            ).status_code
            == 403
        )
        assert client.post("/api/login", json={"key": key}, headers=headers).status_code == 200
        assert client.get("/assets/app.js").headers["content-type"].startswith("text/javascript")
        data = client.get("/api/config").json()
        assert key not in json.dumps(data)
        assert (
            client.post("/api/runtime/start", json={}, headers=headers).json()["state"] == "running"
        )
        data["prompts"]["persona"] = "新的测试人格"
        draft = {
            "data": data["data"],
            "prompts": data["prompts"],
            "expected": data["draft_revision"],
        }
        saved = client.post("/api/config/draft", json=draft, headers=headers)
        assert saved.status_code == 200, saved.text
        assert client.post("/api/config/draft", json=draft, headers=headers).status_code == 400
        applied = client.post(
            "/api/config/apply",
            json={
                "expected": data["published_revision"],
                "draft_revision": saved.json()["draft_revision"],
            },
            headers=headers,
        )
        assert applied.json()["state"] == "applied", applied.text
        assert load_bundle(service.root).settings.prompts["persona"] == "新的测试人格"
        assert (
            client.post("/api/demo/request", json={"text": "你好"}, headers=headers).status_code
            == 200
        )
        assert (
            client.post("/api/runtime/stop", json={}, headers=headers).json()["state"] == "stopped"
        )
        assert client.get("/api/requests").status_code == 200
        assert client.get("/api/usage").status_code == 200
        assert client.post("/api/storage/backup", json={}, headers=headers).status_code == 200
        new_key = reset_key(tmp_path)
        assert client.get("/api/status").status_code == 401
        assert client.post("/api/login", json={"key": new_key}, headers=headers).status_code == 200
        client.post("/api/logout", json={}, headers=headers)
        assert client.get("/api/status").status_code == 401


async def test_mixed_maintenance_stays_pending(tmp_path):
    _, service, settings = fixture_instance(tmp_path)
    c = RuntimeController(settings, dry_models=True, offline=True, cleanup=False)
    await c.start()
    try:
        old = c.session.orchestrator._agent
        new = replace(settings, bot_id="NEW", prompts={"persona": "later"})
        result = await c.apply_config(new)
        assert result["state"] == "pending_maintenance"
        assert c.settings == settings
        assert c.session.orchestrator._agent is old
    finally:
        await c.stop()


async def test_disabling_live_send_blocks_owner_confirmation(tmp_path):
    _, _, settings = fixture_instance(tmp_path)
    settings = replace(settings, live_send=True)
    c = RuntimeController(settings, dry_models=True, offline=True, cleanup=False)
    await c.start()
    try:
        sender = c.session.holder["live"]
        assert sender.enabled()
        await c.apply_config(replace(settings, live_send=False))
        assert sender.refusal({"kind": "owner_control"}) == "sending_disabled"
        # Enabling an unavailable startup capability is an explicit maintenance operation.
        result = await c.apply_config(settings)
        assert result["state"] == "pending_maintenance"
    finally:
        await c.stop()


async def test_profiles_capture_before_queue_and_inherit_per_field(tmp_path):
    _, _, base = fixture_instance(tmp_path)
    profiles = {
        "A": {"enabled": True, "prompts": {"persona": "Alpha"}},
        "B": {"enabled": True, "model_roles": {"answer": "deepseek"}},
    }
    first = replace(base, group_profiles=profiles, revision="group-v1")
    c = RuntimeController(first, dry_models=True, offline=True, cleanup=False)
    assert (await c.start())["state"] == "running"
    try:
        orch = c.session.orchestrator
        a, b = orch.group_agents["A"], orch.group_agents["B"]
        assert a.config.prompt_text["answer"].startswith("Alpha")
        assert b.gemini.spec.name == "deepseek"
        assert b.gemini_decider.spec.name == orch._agent.gemini_decider.spec.name
        intake = c.session.holder["intake"]
        release = asyncio.Event()
        for selected in (a, b):

            async def wait_for_release(*args, answer=selected.answer):
                await release.wait()
                return await answer(*args)

            selected.answer = wait_for_release
        for gid in ("A", "B"):
            intake.intake({**event(first, gid, gid), "group_id": gid})
        await until(lambda: len(orch._run_agents) == 2)
        assert set(orch._run_agents.values()) == {a, b}
        updated = replace(
            first,
            revision="group-v2",
            prompts={"persona": "Global"},
            group_profiles={**profiles, "A": {"enabled": False}},
        )
        await c.apply_config(updated)
        assert set(orch._run_agents.values()) == {a, b}
        assert "A" not in orch.group_agents
        assert orch.group_agents["B"].config.prompt_text["answer"].startswith("Global")
        assert orch.group_agents["B"].gemini.spec.name == "deepseek"
        with pytest.raises(ConfigError):
            await c.apply_config(
                replace(
                    updated,
                    group_profiles={"B": {"enabled": True, "model_roles": {"answer": "missing"}}},
                )
            )
        assert c.settings.revision == "group-v2"
        release.set()
        await until(lambda: not orch.active)
        assert {
            json.loads(r[0])["config_revision"]
            for r in orch.store.con.execute("SELECT detail FROM runs")
        } == {"group-v1"}
    finally:
        await c.stop()


def test_admin_group_scope_budget_and_usage(tmp_path):
    from group_agent.models import CallRecord
    from group_agent.runflow import CURRENT_GROUP

    key, _, _ = fixture_instance(tmp_path)
    app = create_app(tmp_path, demo=True, allowed_hosts={"testserver"})
    with TestClient(app, headers={"X-Qunbot-Request": "1"}) as client:
        client.post("/api/login", json={"key": key}).raise_for_status()
        cfg = client.get("/api/config").json()
        cfg["data"]["budgets"]["daily_cost_cny"] = 18
        cfg["data"]["group_profiles"] = {"A": {"enabled": True, "daily_cost_cny": 5}}
        draft = client.post(
            "/api/config/draft",
            json={"data": cfg["data"], "prompts": {}, "expected": cfg["draft_revision"]},
        ).json()
        client.post(
            "/api/config/apply",
            json={"expected": cfg["published_revision"], "draft_revision": draft["draft_revision"]},
        ).raise_for_status()
        assert client.post("/api/runtime/start", json={}).json()["state"] == "running"

        def charge(gid, amount):
            token = CURRENT_GROUP.set(gid)
            try:
                app.state.controller.session.usage.settle(
                    0, CallRecord("search", "test", "http", cost_cny=amount)
                )
            finally:
                CURRENT_GROUP.reset(token)

        def request(gid):
            before = client.get("/api/usage").json()["requests"]
            client.post("/api/demo/request", json={"group_id": gid}).raise_for_status()

            async def finished():
                orch = app.state.controller.session.orchestrator
                await until(
                    lambda: (
                        orch.store.con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] > before
                        and not orch.active
                    )
                )

            client.portal.call(finished)
            row = client.get("/api/requests", params={"group_id": gid}).json()[0]
            assert row["group_id"] == gid
            return client.get("/api/requests/" + row["run_id"]).json()

        client.portal.call(charge, "A", 5)
        a = request("A")
        assert a["state"] == "rejected" and json.loads(a["detail"])["budget_scope"] == "group"
        b = request("B")
        assert b["state"] in ("succeeded", "recorded"), b
        client.portal.call(charge, "B", 13)
        client.portal.call(charge, None, 1)
        assert json.loads(request("B")["detail"])["budget_scope"] == "global"
        assert client.get("/api/groups").json() == ["A", "B"]
        for query, cost in (("?group_id=A", 5), ("?group_id=B", 13), ("?group_id=", 1), ("", 19)):
            usage = client.get("/api/usage" + query).json()
            assert usage["cost"] == sum(r["cost"] for r in usage["days"]) == cost
            assert sum(r["cost"] for r in usage["providers"]) == cost
        client.post("/api/runtime/stop", json={}).raise_for_status()
        assert client.post("/api/runtime/start", json={}).json()["state"] == "running"
        assert request("A")["state"] == request("B")["state"] == "rejected"
