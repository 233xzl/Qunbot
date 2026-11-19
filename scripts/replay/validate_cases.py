"""Validate replay structure and optional draft annotations without printing chat."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


def annotated_case(case: dict, sidecar: dict | None, reviewed: dict | None = None) -> dict:
    merged = deepcopy(case)
    if sidecar is not None:
        if sidecar.get("schema_version") != "annotations-sidecar-1":
            raise ValueError("unsupported annotation sidecar")
        annotation = sidecar["cases"][case["case_id"]]
        merged["annotations"] = deepcopy(annotation)
    if reviewed is not None:
        # Owner-reviewed overrides (import_review.py); entries without annotations
        # (unsure, hints for undrafted cases) leave the draft in place.
        if reviewed.get("schema_version") != "annotations-reviewed-1":
            raise ValueError("unsupported reviewed sidecar")
        entry = reviewed["cases"].get(case["case_id"])
        if entry and entry.get("annotations") is not None:
            if entry["annotations"].get("status") != "reviewed":
                raise ValueError("reviewed override must have status reviewed")
            merged["annotations"] = deepcopy(entry["annotations"])
    return merged


def check_annotation_references(case: dict):
    annotations = case["annotations"]
    if annotations["status"] == "needs_review":
        return
    ids = {e["message_id"] for e in case["events"]}
    required = set(annotations["required_context_ids"])
    forbidden = set(annotations["forbidden_context_ids"])
    if not (required | forbidden) <= ids or required & forbidden:
        raise ValueError("invalid annotation context references")
    boundary = next(
        i for i, e in enumerate(case["events"]) if e["message_id"] == case["anchor"]["message_id"]
    )
    prior = {e["message_id"] for e in case["events"][:boundary]}
    if not required <= prior:
        raise ValueError("required history includes current or future events")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="case JSON or directory of real_/synthetic- case JSON"
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--reviewed", type=Path, help="owner-reviewed overrides (annotations.reviewed.json)"
    )
    args = parser.parse_args()
    try:
        schema = json.loads((ROOT / "tests/replay/schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        sidecar = (
            json.loads(args.annotations.read_text(encoding="utf-8")) if args.annotations else None
        )
        reviewed = json.loads(args.reviewed.read_text(encoding="utf-8")) if args.reviewed else None
        paths = (
            sorted([*args.path.glob("real_*.json"), *args.path.glob("synthetic-*.json")])
            if args.path.is_dir()
            else [args.path]
        )
        if not paths:
            raise ValueError("no cases")
        failed = 0
        for path in paths:
            try:
                case = annotated_case(
                    json.loads(path.read_text(encoding="utf-8")), sidecar, reviewed
                )
                if not validator.is_valid(case):
                    raise ValueError("schema validation failed")
                check_annotation_references(case)
            except (ValueError, KeyError, TypeError, StopIteration):
                failed += 1
                print(f"invalid case: {path.name} (payload omitted)")
    except (OSError, ValueError, TypeError):
        parser.error("cannot read schema, cases or annotations (details omitted)")
    source = "sidecar" if sidecar else "inline"
    if reviewed:
        source += "+reviewed"
    print(f"validated_cases={len(paths)} failures={failed} annotations={source}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
