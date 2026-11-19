"""Detached admin service. Local commands address an instance by directory and generation."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from ..config import ConfigError
from ..config_service import atomic_write
from ..runtime import LockBusy, exclusive_lock


def read_state(root):
    try:
        return json.loads((root / "service.json").read_text("utf-8"))
    except FileNotFoundError:
        return {}


def save_state(root, state):
    atomic_write(root / "service.json", json.dumps(state))


def admin_lock(instance):
    return exclusive_lock(instance / "private" / "admin.lock", "此实例已有管理台正在运行")


def service_lock(root):
    return exclusive_lock(root / "service.lock", "本实例的后台管理服务正在运行")


def running(root):
    try:
        with service_lock(root):
            return False
    except LockBusy:
        return True


def address(state):
    host = state["host"]
    return f"http://{'[' + host + ']' if ':' in host else host}:{state['port']}"


def launch(instance, root, state):
    generation = uuid4().hex
    state.update(generation=generation, phase="starting")
    save_state(root, state)
    log = root / "service.log"
    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with log.open("ab") as output:
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", __name__, str(instance), generation],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            **options,
        )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise ConfigError(f"后台服务启动失败，请检查端口占用和日志 {log}")
        report = read_state(root)
        if report.get("generation") == generation and report.get("phase") == "ready":
            if report.get("agent") == "failed":
                raise ConfigError(
                    f"管理台已启动，Agent 恢复失败。请登录 {address(state)} 查看运行状态"
                )
            return f"后台服务已启动 {address(state)}（Agent {report['agent']}）"
        time.sleep(0.15)
    raise ConfigError(f"后台服务仍在启动，请查看 {log}")


def manage(args):
    instance = args.instance.resolve()
    if not (instance / "private" / "admin.json").is_file():
        raise ConfigError("请先用 admin init 初始化该实例")
    root = instance / "private" / "background"
    root.mkdir(mode=0o700, exist_ok=True)
    with exclusive_lock(root / "command.lock", "该实例正在执行另一条管理命令"):
        state = read_state(root)
        alive = running(root)
        if args.command == "status" or (args.command == "start" and alive):
            return f"后台服务运行中 {address(state)}" if alive else "后台服务已停止"
        if args.command in {"restart", "stop"} and alive:
            # Only the worker holding this generation's lock consumes the request.
            atomic_write(root / "stop", state["generation"])
            deadline = time.monotonic() + 150
            while running(root):
                if time.monotonic() >= deadline:
                    raise ConfigError("后台服务正在等待请求结束，请稍后检查状态")
                time.sleep(0.15)
            state = read_state(root)
            if state.get("phase") != "drained":
                raise ConfigError("服务退出时未完成状态保存，请检查后台日志")
        if args.command == "stop":
            return "后台服务已停止"
        state.setdefault("host", "127.0.0.1")
        state.setdefault("port", 8765)
        state.setdefault("demo", False)
        if args.command == "start":
            if args.host is not None:
                state["host"] = args.host
            if args.port is not None:
                state["port"] = args.port
            if args.demo is not None and args.demo != state["demo"]:
                state.update(demo=args.demo, resume=False)
        if not 1 <= state["port"] <= 65535:
            raise ConfigError("端口须在 1 到 65535 之间")
        with admin_lock(instance):
            pass  # Refuse to take over a foreground service. The worker holds this lock while serving.
        return launch(instance, root, state)


async def serve(instance, root, state):
    import uvicorn
    from .server import create_app

    host, port = state["host"], state["port"]
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError("管理台仅支持回环地址")
    hosts = {f"{name}:{port}" for name in ("127.0.0.1", "localhost", "[::1]")}
    app = create_app(instance, demo=state["demo"], allowed_hosts=hosts)
    controller = app.state.controller
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, access_log=False))
    stop = root / "stop"

    async def supervise():
        while not server.started:
            await asyncio.sleep(0.1)
        if state.get("resume"):
            await controller.start()  # The engine reloads its persisted pause gate.
        state.update(phase="ready", agent=controller.status()["state"])
        save_state(root, state)
        while not server.should_exit:
            if stop.exists() and stop.read_text("utf-8") == state["generation"]:
                state["resume"] = controller.state in {"starting", "running"}
                server.should_exit = True  # FastAPI lifespan drains the engine on shutdown.
                return
            await asyncio.sleep(0.25)

    monitor = asyncio.create_task(supervise())
    try:
        await server.serve()
        if monitor.done():
            monitor.result()
            state["phase"] = "drained"
            save_state(root, state)
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


if __name__ == "__main__":
    instance = Path(sys.argv[1])
    root = instance / "private" / "background"
    with service_lock(root), admin_lock(instance):
        state = read_state(root)
        if state["generation"] != sys.argv[2]:
            raise SystemExit("启动命令已过期")
        asyncio.run(serve(instance, root, state))
