"""Command-line management of a local Qunbot instance."""

import argparse
from pathlib import Path
import tomllib

from ..config import ConfigError
from ..config_service import ConfigService, load_bundle, materialize
from .auth import reset_key


def initialize(instance, source=None):
    instance = instance.resolve()
    service = ConfigService(instance / "config-history")
    if service.status()["revision"]:
        raise ConfigError("admin: instance already initialized; use reset-key to replace the key")
    if source:
        bundle = load_bundle(source)
        data = materialize(bundle)
        prompts = dict(bundle.settings.prompts)
    else:
        data = tomllib.loads(Path(__file__).with_name("example.toml").read_text("utf-8"))
        data["instance_root"] = str(instance)
        data.pop("prompts", None)
        names = {"SNOWLUMA_WS_URL", "SNOWLUMA_WS_TOKEN", data["web"]["api_key_env"]}
        for model in data["models"].values():
            if isinstance(model, dict):
                names.update(
                    model[k] for k in ("api_key_env", "base_url_env", "project_env") if model.get(k)
                )
        data["credentials"] = {
            name: {
                "source": "managed",
                "key": name,
                "path": str(instance / "private" / "credentials.env"),
            }
            for name in sorted(names)
        }
        prompts = {"persona": "你是一个群聊助手。用自然、简洁的中文回答，依据当前材料说明事实。"}
    draft = service.save_editor(data, prompts)
    service.publish(draft.path, None)
    return reset_key(instance)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qunbot 本地管理台")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "reset-key", "serve", "start", "restart", "stop", "status"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--instance", type=Path, required=True)
        if name == "init":
            cmd.add_argument("--settings", type=Path, help="导入已有配置，保留其数据路径")
        if name in {"serve", "start"}:
            cmd.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default=None)
            cmd.add_argument("--port", type=int)
            cmd.add_argument(
                "--demo",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="使用独立模拟库，不连接 QQ 或外部模型",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            key = initialize(args.instance, args.settings)
            print("管理密钥（仅本次显示，请保存）：" + key)
        elif args.command == "reset-key":
            print("新管理密钥（旧会话失效）：" + reset_key(args.instance))
        elif args.command in {"start", "restart", "stop", "status"}:
            from .background import manage

            print(manage(args))
        else:
            args.host = args.host or "127.0.0.1"
            args.port = 8765 if args.port is None else args.port
            import uvicorn
            from .background import admin_lock
            from .server import create_app

            if not 1 <= args.port <= 65535:
                parser.error("port must be between 1 and 65535")
            host = f"[{args.host}]" if ":" in args.host else args.host
            print(f"Qunbot 管理台：http://{host}:{args.port}（Agent 由页面启动）", flush=True)
            app = create_app(
                args.instance,
                demo=args.demo,
                allowed_hosts={
                    f"{host}:{args.port}",
                    f"localhost:{args.port}",
                    f"127.0.0.1:{args.port}",
                },
            )
            with admin_lock(args.instance.resolve()):
                uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))
    return 0
