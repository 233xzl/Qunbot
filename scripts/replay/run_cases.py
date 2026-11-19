"""Run replay cases through the deterministic runner and score them (T1.9 / T1.10).

Usage:
    python scripts/replay/run_cases.py <dir|file> [--strategy A|B|C|D] [--json] [--traces]
        [--annotations <sidecar>] [--reviewed <file>] [--check]

Prints a short per-case table and the summary; `--json` prints scores and summary as
JSON, `--traces` adds the full run traces. No network, model or QQ calls; real cases'
outbound texts are omitted from traces.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from group_agent.replay import run_case, score_case, summarize, table  # noqa: E402
from validate_cases import annotated_case  # noqa: E402


def load(path: Path, sidecar: Path | None, reviewed: Path | None) -> list[dict]:
    side = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar else None
    rev = json.loads(reviewed.read_text(encoding="utf-8")) if reviewed else None
    paths = (
        sorted([*path.glob("real_*.json"), *path.glob("synthetic-*.json")])
        if path.is_dir()
        else [path]
    )
    cases = []
    for p in paths:
        case = json.loads(p.read_text(encoding="utf-8"))
        if side is not None and case["case_id"] not in side.get("cases", {}):
            continue
        cases.append(annotated_case(case, side, rev))
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--strategy", choices=["A", "B", "C", "D"], default="C")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--traces", action="store_true", help="with --json: include run traces")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 unless every case passes with all isolation counters at 0 (T3.10)",
    )
    args = parser.parse_args(argv)
    cases = load(args.path, args.annotations, args.reviewed)
    if not cases:
        parser.error("no cases")
    scores, traces = [], []
    for case in cases:
        result = run_case(case, args.strategy)
        scores.append(score_case(case, result))
        traces.append(result)
    summary = summarize(scores)
    if args.json:
        out = {"strategy": args.strategy, "summary": summary, "cases": scores}
        if args.traces:
            out["traces"] = traces
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(table(scores))
        print(json.dumps({"strategy": args.strategy, **summary}, ensure_ascii=False, indent=2))
    if args.check and (summary["passed"] != summary["cases"] or any(summary["isolation"].values())):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
