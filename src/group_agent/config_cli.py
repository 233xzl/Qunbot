"""Offline configuration commands. No model calls or Agent startup."""

import argparse
import json
from pathlib import Path
import sys

from .config import ConfigError
from .config_service import ConfigService, effective, load_bundle, migrate, schema
from .persona import prompt_options


def main(argv):
    parser = argparse.ArgumentParser(prog="group-agent config")
    parser.add_argument(
        "action",
        choices=(
            "validate",
            "effective",
            "schema",
            "prompts",
            "diff",
            "publish",
            "rollback",
            "status",
            "migrate",
            "credential",
        ),
    )
    parser.add_argument("--settings", type=Path, default=Path("config/settings.example.toml"))
    parser.add_argument("--store", type=Path, default=Path("data/config"))
    parser.add_argument("--expect", help="current revision; omit only for an empty store")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, help="migration output; omitted means preview")
    parser.add_argument("--name", help="credential reference name")
    parser.add_argument("--operation", choices=("keep", "replace", "clear"), default="keep")
    args = parser.parse_args(argv)
    service = ConfigService(args.store)
    try:
        if args.action == "status":
            result = service.report()
        elif args.action == "schema":
            result = schema()
        elif args.action == "rollback":
            result = service.rollback(args.revision, args.expect)
        elif args.action == "publish":
            result = service.publish(args.settings, args.expect)
        elif args.action == "diff":
            result = service.diff(args.settings)
        else:
            bundle = load_bundle(args.settings)
            if args.action == "validate":
                result = {"valid": True, "revision": bundle.revision, "apply": "restart"}
            elif args.action == "effective":
                result = effective(bundle)
            elif args.action == "prompts":
                result = dict(prompt_options(bundle.settings))
            elif args.action == "migrate":
                result = migrate(bundle, args.output)
            else:
                store = bundle.settings.credentials
                if args.name:
                    value = (
                        sys.stdin.readline().rstrip("\r\n") if args.operation == "replace" else None
                    )
                    store.update(args.name, args.operation, value)
                result = store.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ConfigError, OSError) as exc:
        parser.error(
            str(exc)
            if isinstance(exc, ConfigError)
            else "configuration: filesystem operation failed"
        )
    return 0
