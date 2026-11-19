"""Import an owner review export into <batch_dir>/annotations.reviewed.json.

The export comes from the offline review page (render_review.py). Every case id,
message id and value is validated against the batch before anything is written;
the draft sidecar is never modified. Only counts are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review as rr  # noqa: E402

REVIEWED_FORMAT = "annotations-reviewed-1"
MAX_COMMENT = 4000


class ReviewImportError(ValueError):
    def __init__(self, counts: Counter):
        super().__init__("invalid review export")
        self.counts = counts


def _ids(value) -> list | None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return value


def build_reviewed(export: dict, cases: list[dict], sidecar: dict | None, batch: str) -> dict:
    """Validate an export and return reviewed sidecar entries; raises ReviewImportError."""
    errors: Counter = Counter()
    if export.get("format") != rr.REVIEW_FORMAT:
        errors["bad_format"] += 1
    if export.get("batch") != batch:
        errors["batch_mismatch"] += 1
    entries = export.get("cases")
    if not isinstance(entries, dict):
        errors["bad_cases"] += 1
        raise ReviewImportError(errors)
    by_id = {c["case_id"]: c for c in cases}
    out: dict[str, dict] = {}
    stale = 0
    for cid, entry in entries.items():
        case = by_id.get(cid)
        if case is None:
            errors["unknown_case"] += 1
            continue
        if not isinstance(entry, dict) or entry.get("verdict") not in rr.VERDICTS:
            errors["bad_verdict"] += 1
            continue
        comment = entry.get("comment", "")
        if not isinstance(comment, str) or len(comment) > MAX_COMMENT:
            errors["bad_comment"] += 1
            continue
        draft = rr.annotation_for(case, sidecar)
        drafted = rr.is_drafted(draft)
        verdict = entry["verdict"]
        if verdict == "agree" and not drafted:
            errors["agree_without_draft"] += 1
            continue
        events = case["events"]
        order = [e["message_id"] for e in events]
        anchor_index = order.index(case["anchor"]["message_id"])
        before = set(order[:anchor_index])
        change = None
        if verdict == "modify":
            route = entry.get("route")
            clarify = entry.get("should_clarify")
            required = _ids(entry.get("required_context_ids", []))
            forbidden = _ids(entry.get("forbidden_context_ids", []))
            if route is not None and route not in rr.ROUTES:
                errors["bad_route"] += 1
                continue
            if clarify is not None and not isinstance(clarify, bool):
                errors["bad_should_clarify"] += 1
                continue
            if required is None or forbidden is None:
                errors["bad_context_list"] += 1
                continue
            known = set(order)
            if not set(required) | set(forbidden) <= known:
                errors["unknown_message"] += 1
                continue
            if not set(required) | set(forbidden) <= before:
                errors["context_not_before_anchor"] += 1
                continue
            if set(required) & set(forbidden):
                errors["context_conflict"] += 1
                continue
            change = {
                "route": route,
                "should_clarify": clarify,
                "required": [m for m in order if m in set(required)],
                "forbidden_before_anchor": [m for m in order if m in set(forbidden)],
            }
        if entry.get("draft_fingerprint") != rr.fingerprint(draft):
            stale += 1
            continue
        record = {
            "verdict": verdict,
            "comment": comment,
            "reviewed_at": entry.get("updated_at")
            if isinstance(entry.get("updated_at"), str)
            else None,
            "draft_fingerprint": entry.get("draft_fingerprint"),
            "annotations": None,
            "owner_hint": None,
        }
        if drafted and verdict in ("agree", "modify"):
            ann = deepcopy(draft)
            ann["status"] = "reviewed"
            if change is not None:
                if change["route"] is not None:
                    ann["expected_route"] = change["route"]
                if change["should_clarify"] is not None:
                    ann["should_clarify"] = change["should_clarify"]
                after = order[anchor_index + 1 :]
                ann["required_context_ids"] = change["required"]
                ann["forbidden_context_ids"] = change["forbidden_before_anchor"] + after
            record["annotations"] = ann
        elif change is not None:
            # No draft yet: keep the owner's input as a hint for the annotator.
            record["owner_hint"] = change
        out[cid] = record
    if errors:
        raise ReviewImportError(errors)
    return {"cases": out, "stale": stale}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, help="default: <batch>/annotations.draft.json")
    parser.add_argument("--out", type=Path, help="default: <batch>/annotations.reviewed.json")
    args = parser.parse_args()
    sidecar_path = args.annotations or args.batch / "annotations.draft.json"
    out_path = args.out or args.batch / "annotations.reviewed.json"
    if out_path.resolve().parent != args.batch.resolve():
        parser.error("--out must be inside the batch directory")
    try:
        cases = rr.load_cases(args.batch)
        rr.check_output_location(args.batch, cases)
        sidecar = rr.load_sidecar(sidecar_path)
        export = json.loads(args.export.read_text(encoding="utf-8"))
        if not isinstance(export, dict):
            raise ValueError("export must be an object")
    except (OSError, ValueError, KeyError, TypeError):
        parser.error("cannot read batch, sidecar or export (details omitted)")
    batch = args.batch.resolve().name
    try:
        result = build_reviewed(export, cases, sidecar, batch)
    except ReviewImportError as exc:
        detail = " ".join(f"{k}={v}" for k, v in sorted(exc.counts.items()))
        print(f"rejected: {detail}; nothing written")
        return 1
    existing = {}
    if out_path.is_file():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("schema_version") == REVIEWED_FORMAT:
            existing = previous.get("cases", {})
    merged = {**existing, **result["cases"]}
    payload = {
        "schema_version": REVIEWED_FORMAT,
        "batch": batch,
        "base_sidecar": sidecar_path.name if sidecar is not None else None,
        "imported_at": datetime.now(rr.LOCAL_TZ).isoformat(timespec="seconds"),
        "reviewer": "owner",
        "cases": dict(sorted(merged.items())),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verdicts = Counter(r["verdict"] for r in result["cases"].values())
    print(
        f"imported={len(result['cases'])} agree={verdicts['agree']} modify={verdicts['modify']} "
        f"unsure={verdicts['unsure']} reviewed_annotations="
        f"{sum(1 for r in result['cases'].values() if r['annotations'])} "
        f"hints_for_undrafted={sum(1 for r in result['cases'].values() if r['owner_hint'])} "
        f"stale_skipped={result['stale']} kept_previous={len(set(existing) - set(result['cases']))} "
        f"total={len(merged)}"
    )
    print(f"out={rr.display_path(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
