"""Authenticated, same-origin local management service."""

from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager, closing
import json
import mimetypes
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import received_groups
from ..config import ConfigError, MODELS_SCALARS, _models
from ..config_service import ConfigService, load_bundle, materialize, schema
from ..credentials import prepare_provider_keys
from ..runtime import RuntimeController, application_plan
from . import monitor
from .auth import Auth

COOKIE = "qunbot_session"


def create_app(instance, *, demo=False, allowed_hosts=None, controller_options=None):
    instance = Path(instance).resolve()
    service = ConfigService(instance / "config-history")
    auth = Auth(instance)
    overrides = {}
    if demo:
        overrides = {
            "serve.db_path": str(instance / "demo" / "agent.sqlite3"),
            "knowledge.db_path": str(instance / "demo" / "knowledge.sqlite3"),
            "billing.db_path": str(instance / "demo" / "billing.sqlite3"),
            "collection.media_dir": str(instance / "demo" / "media"),
            "serve.live_send": False,
            "serve.owner_command_send": False,
            "serve.processing_reaction": False,
            "serve.self_sync": False,
            "agent.web_enabled": False,
        }

    def active():
        return load_bundle(service.root, overrides)

    controller = RuntimeController(
        active().settings, **{"dry_models": demo, "offline": demo, **(controller_options or {})}
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        await controller.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.controller, app.state.auth, app.state.service = controller, auth, service
    hosts = allowed_hosts or {"127.0.0.1:8765", "localhost:8765"}

    @app.middleware("http")
    async def authorize(request, call_next):
        host = request.headers.get("host", "")
        if host not in hosts:
            return JSONResponse({"error": "访问地址不匹配"}, status_code=400)
        if request.method not in {"GET", "HEAD"}:
            origin = request.headers.get("origin")
            if (origin and urlsplit(origin).netloc != host) or request.headers.get(
                "x-qunbot-request"
            ) != "1":
                return JSONResponse({"error": "请求来源不匹配"}, status_code=403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/login":
            if not auth.valid(request.cookies.get(COOKIE)):
                return JSONResponse({"error": "请先登录"}, status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.exception_handler(ConfigError)
    async def config_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    async def body(request):
        raw = await request.body()
        if len(raw) > 1024 * 1024:
            raise ConfigError("请求超过 1 MB")
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ConfigError("请求需要有效 JSON") from None
        if not isinstance(data, dict):
            raise ConfigError("请求需要 JSON 对象")
        return data

    @app.post("/api/login")
    async def login(request: Request):
        data = await body(request)
        token, status = auth.login(data.get("key"))
        response = JSONResponse(
            {"ok": bool(token), **({"error": "密钥无效或尝试过于频繁"} if not token else {})},
            status_code=status,
        )
        if token:
            response.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=43200)
        return response

    @app.post("/api/logout")
    async def logout(request: Request):
        auth.sessions.pop(request.cookies.get(COOKIE), None)
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE)
        return response

    @app.get("/api/status")
    async def status():
        current = service.status()
        return {
            **controller.status(),
            "demo": demo,
            "target_revision": current["revision"],
            "pending": current["revision"] != controller.settings.published_revision,
        }

    @app.post("/api/runtime/{action}")
    async def runtime_action(action: str):
        if action == "start":
            if controller.state in {"stopped", "failed"}:
                controller.settings = active().settings
            return await controller.start()
        if action in {"stop", "pause", "resume"}:
            return await getattr(controller, action)()
        raise ConfigError("未知运行操作")

    @app.get("/api/config")
    async def get_config():
        draft_path = service.root / "draft.toml"
        draft = load_bundle(draft_path) if draft_path.is_file() else load_bundle(service.root)
        from ..persona import CORRECTIONS, LEGACY_PERSONA, prompt_options
        from ..gemini_search import SEARCH_PROMPT

        prompts = dict(draft.settings.prompts)
        return {
            "data": materialize(draft),
            "retired_models": draft.retired_models,
            "prompts": prompts,
            "default_persona": LEGACY_PERSONA,
            "prompt_defaults": {
                **prompt_options(draft.settings),
                "persona": LEGACY_PERSONA,
                "evidence": CORRECTIONS,
                "search": SEARCH_PROMPT,
            },
            "draft_revision": draft.revision if draft_path.is_file() else None,
            "published_revision": service.status()["revision"],
            "schema": schema(),
            "overrides": overrides,
            "credentials": draft.settings.credentials.status(),
        }

    def group_catalog():
        with database() as con:
            rows = received_groups.directory(con, controller.settings.bot_id)
        return [row for row in rows if row["group_id"] not in controller.settings.blocked_groups]

    def validate_profiles(profiles):
        if not isinstance(profiles, Mapping):
            raise ConfigError("群个性化设置格式有误")
        current = active().settings.group_profiles
        changed = {gid for gid, profile in profiles.items() if current.get(gid) != profile}
        if changed and not changed <= {
            r["group_id"] for r in group_catalog() if r["membership"] == "present"
        }:
            raise ConfigError("请选择已收到消息且已确认在群的群聊")

    @app.get("/api/received-groups")
    def get_received_groups():
        return group_catalog()

    @app.post("/api/received-groups/{gid}/reply")
    async def switch_group(gid: str, request: Request):
        enabled = (await body(request)).get("enabled")
        if not isinstance(enabled, bool):
            raise ConfigError("请选择本群回复状态")
        async with controller.lock:
            session = controller.session
            if not session or controller.state != "running" or (enabled and session.pause.paused):
                raise ConfigError("请先在总览启动或恢复全局回复")
            orch = session.orchestrator
            con, bot = orch.store.con, controller.settings.bot_id
            row = con.execute(
                "SELECT membership FROM received_groups WHERE bot_id=? AND group_id=?", (bot, gid)
            ).fetchone()
            if not row or (
                enabled and (row[0] != "present" or gid in orch.access.snapshot.blocked_groups)
            ):
                raise ConfigError("该群目前不可启动")
            with con:
                if enabled:
                    con.execute(
                        "UPDATE received_groups SET enabled=1 WHERE bot_id=? AND group_id=?",
                        (bot, gid),
                    )
                else:
                    received_groups.stop_boundary(con, bot, gid)
                    orch.cancel_group(gid)
            return {"enabled": enabled}

    @app.post("/api/config/draft")
    async def save_draft(request: Request):
        data = await body(request)
        if not isinstance(data.get("data"), dict) or not isinstance(data.get("prompts", {}), dict):
            raise ConfigError("配置与提示词必须为对象")
        for table in ("models", "connections", "web", "model_roles", "credentials"):
            if not isinstance(data["data"].get(table, {}), dict):
                raise ConfigError("模型与供应商配置格式无效")
        validate_profiles(data["data"].get("group_profiles", {}))
        write_keys = prepare_provider_keys(instance, data["data"], data.get("provider_keys", []))
        editor = data["data"]
        previous = active().settings
        models = editor.get("models", {})
        web = editor.get("web", {})
        if isinstance(models, dict) and isinstance(web, dict):
            roles = editor.get("model_roles", {})
            if isinstance(roles, dict):
                for role, default in [("answer", "gemini"), ("fallback", "deepseek")]:
                    alias = roles.get(role, default)
                    if isinstance(alias, str) and alias in previous.models and alias not in models:
                        raise ConfigError(f"请先换绑 {role} 使用的模型 {alias}")
            if "search" not in web and (
                web.get("search_provider") == "gemini" or web.get("gemini_supplement")
            ):
                alias = web.get("gemini_credentials_model", "gemini")
                if isinstance(alias, str) and alias in previous.models and alias not in models:
                    raise ConfigError(f"请先换绑搜索使用的模型 {alias}")
        for name, conn in editor.get("connections", {}).items():
            if not isinstance(conn, dict):
                raise ConfigError("供应商配置格式无效")
            if conn.get("protocol") == "systemone":
                raise ConfigError("SystemOne 已退役，请选择其他协议")
            if (
                conn.get("protocol")
                and conn.get("api_key_env")
                and (conn.get("base_url") or conn.get("base_url_env") or conn.get("vertex"))
            ):
                _models(
                    {
                        "models": {
                            "provider": {
                                **conn,
                                "model": "probe",
                                "price_input": 0,
                                "price_output": 0,
                            }
                        }
                    }
                )
        for name, model in models.items():
            if (
                name not in MODELS_SCALARS
                and isinstance(model, dict)
                and model.get("protocol") == "systemone"
            ):
                raise ConfigError("SystemOne 已退役，请选择其他协议")
        draft = service.save_editor(
            data["data"], data.get("prompts", {}), data.get("expected"), before_write=write_keys
        )
        effective = load_bundle(draft.path, overrides)
        return {
            "draft_revision": draft.revision,
            "diff": service.diff(draft.path),
            "plan": application_plan(controller.settings, effective.settings),
        }

    async def apply_source(source, expected, digest=None, rollback_of=None):
        candidate = load_bundle(source, overrides)

        def publish():
            validate_profiles(candidate.settings.group_profiles)
            if digest and load_bundle(source).revision != digest:
                raise ConfigError("draft: changed; reload before applying")
            service.publish(source, expected, rollback_of=rollback_of)
            return active().settings

        return await controller.apply_config(
            candidate.settings,
            publish=publish,
            reload_credentials=bool(
                candidate.settings.credential_refs
                and any(k.startswith("QUNBOT_KEY_") for k in candidate.settings.credential_refs)
            ),
        )

    @app.post("/api/config/apply")
    async def apply(request: Request):
        data = await body(request)
        if not data.get("draft_revision"):
            raise ConfigError("请先保存草稿")
        return await apply_source(
            service.root / "draft.toml", data.get("expected"), data["draft_revision"]
        )

    @app.post("/api/config/rollback")
    async def rollback(request: Request):
        data = await body(request)
        source = service.version_path(data.get("revision"))
        load_bundle(source).settings.credentials.check_references()
        return await apply_source(source, data.get("expected"), rollback_of=data["revision"])

    @app.get("/api/config/versions")
    async def versions():
        paths = sorted(
            (service.root / "versions").glob("*/revision.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:50]
        return [json.loads(p.read_text("utf-8")) for p in paths]

    @app.post("/api/credentials/reload")
    async def reload_credentials():
        return await controller.apply_config(active().settings, reload_credentials=True)

    @app.post("/api/credentials")
    async def credentials(request: Request):
        data = await body(request)
        settings = active().settings
        settings.credentials.update(data.get("name"), data.get("action"), data.get("value"))
        if str(data.get("name", "")).startswith("SNOWLUMA_"):
            return {"state": "pending_maintenance"}
        try:
            return await controller.apply_config(settings, reload_credentials=True)
        except Exception as exc:
            return {"state": "pending_credentials", "error": type(exc).__name__}

    @contextmanager
    def database():
        path = controller.settings.serve_db
        if not path.is_file():
            yield None
            return
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    @app.get("/api/groups")
    def groups(named: bool = False):
        with database() as con:
            if con is None:
                return []
            ids = [
                r[0]
                for r in con.execute(
                    f"SELECT group_id FROM runs UNION SELECT group_id FROM {monitor.ledger(con)} ORDER BY 1"
                )
                if r[0] is not None
            ]
            if not named:
                return ids
            directory = monitor.names(con, controller.settings.bot_id)
            return [dict(id=g, name=monitor.group_title(g, directory)) for g in ids]

    @app.get("/api/requests")
    def requests(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=51),
        state: str = "",
        group_id: str | None = None,
        period: monitor.Period | None = None,
    ):
        with database() as con:
            return monitor.requests(
                con,
                offset=offset,
                limit=limit,
                state=state,
                gid=group_id,
                bounds=monitor.window(period, time.time()) if period else None,
            )

    @app.get("/api/requests/{run_id}")
    def request_detail(run_id: str):
        with database() as con:
            rows = monitor.requests(con, run_id=run_id)
            return rows[0] if rows else JSONResponse({"error": "请求不存在"}, status_code=404)

    @app.get("/api/usage")
    def usage(
        group_id: str | None = None,
        period: monitor.Period = "30d",
        dimension: monitor.Dimension = "total",
    ):
        with database() as con:
            return monitor.usage(
                con, controller.settings.bot_id, period, dimension, group_id, time.time()
            )

    def storage_paths():
        s = controller.settings
        return {"messages": s.serve_db, "knowledge": s.knowledge_db, "billing": s.billing.db_path}

    @app.get("/api/storage")
    async def storage():
        return [
            {
                "name": name,
                "path": str(path) if path else None,
                "bytes": path.stat().st_size if path and path.exists() else 0,
            }
            for name, path in storage_paths().items()
        ]

    @app.post("/api/storage/preview")
    async def cleanup_preview():
        if not controller.session:
            raise ConfigError("请先启动引擎以预览清理")
        return controller.session.cleanup(dry_run=True)

    @app.post("/api/storage/backup")
    async def backup():
        async with controller.lock:
            if controller.session or controller.state in {"starting", "draining"}:
                raise ConfigError("请先停止引擎再创建一致备份")
            folder = instance / "backups" / uuid4().hex
            folder.mkdir(parents=True)
            for name, path in storage_paths().items():
                if path and path.is_file():
                    with closing(
                        sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                    ) as source:
                        target = sqlite3.connect(folder / f"{name}.sqlite3")
                        try:
                            source.backup(target)
                        finally:
                            target.close()
            return {"path": str(folder), "kind": "database_snapshot"}

    @app.post("/api/demo/request")
    async def demo_request(request: Request):
        if not demo or not controller.session:
            raise ConfigError("模拟请求需要启动模拟引擎")
        data = await body(request)
        text = data.get("text", "你好")
        if not isinstance(text, str) or not 1 <= len(text) <= 4000:
            raise ConfigError("模拟输入需为 1–4000 字符")
        group = data.get("group_id", "DEMO")
        if not isinstance(group, str) or not group.strip() or len(group) > 64:
            raise ConfigError("模拟群号需为 1–64 字符")
        s = controller.settings
        event = {
            "post_type": "message",
            "message_type": "group",
            "self_id": s.bot_id,
            "group_id": group,
            "user_id": "DEMO_USER",
            "sender": {"nickname": "模拟成员"},
            "message_id": uuid4().hex,
            "time": int(time.time()),
            "message": [
                {"type": "at", "data": {"qq": s.bot_id}},
                {"type": "text", "data": {"text": text}},
            ],
        }
        result = controller.session.holder["intake"].intake(event)
        return {"status": result.status}

    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    static = Path(__file__).with_name("static")
    app.mount("/assets", StaticFiles(directory=static), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html")

    return app
