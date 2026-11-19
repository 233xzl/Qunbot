"""Replay frozen/current prompt profiles without editing runtime sources.

Example: python scripts/compare_prompt_profiles.py --profile before --force-fallback
    --name prompt-before-fallback --budget-cny 0.3 --ids <selection> --concurrency 1

All remaining arguments are passed to replay_shadow.main. Its read-only source backup,
budget accounting, image/tool handling and no-send boundary remain in effect. Forced
fallback is controlled fault injection, not a measurement of primary availability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from group_agent.models import ModelBreaker  # noqa: E402


class ForcedPrimaryBreaker(ModelBreaker):
    """Keep the real primary-unavailable branch open for this offline comparison."""

    def __init__(self, primary_key: str, delegate: ModelBreaker | None = None):
        super().__init__()
        self.primary_key, self.delegate = primary_key, delegate or ModelBreaker()
        self.injected_skips = 0

    def state(self, key):
        return "open_injected" if key == self.primary_key else self.delegate.state(key)

    def _skip(self):
        with self._lock:
            self.injected_skips += 1

    def available(self, key, count_skip=False):
        if key != self.primary_key:
            return self.delegate.available(key, count_skip=count_skip)
        if count_skip:
            self._skip()
        return False

    def acquire(self, key):
        if key != self.primary_key:
            return self.delegate.acquire(key)
        self._skip()
        return None

    def record(self, key, ticket, outcome):
        if key != self.primary_key:
            self.delegate.record(key, ticket, outcome)

    def summary(self):
        result = self.delegate.summary()
        with self._lock:
            result[self.primary_key] = {
                "state": "open_injected",
                "fault_injection": "controlled_primary_unavailable",
                "skipped": self.injected_skips,
                "opened": 0,
                "open_seconds": 0.0,
                "probes": 0,
            }
        return result


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load replay profile module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and relative imports need registration.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_profile(profile: str, snapshot_dir: Path):
    if profile == "before":
        source = snapshot_dir / "before-live_agent.py"
        prompts_path = snapshot_dir / "before-prompts.json"
        if not source.is_file() or not prompts_path.is_file():
            raise SystemExit("before requires complete frozen live_agent and prompt snapshots")
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        if not all(
            isinstance(prompts.get(k), str) and prompts[k].strip()
            for k in ("answer_instructions", "decision_instructions")
        ):
            raise SystemExit("before snapshot requires both nonempty prompt instructions")
        module = _load_file(
            "group_agent._prompt_profile_before_" + _sha(source.read_text(encoding="utf-8"))[:16],
            source,
        )
        if not hasattr(module, "LiveAgent"):
            raise SystemExit("before snapshot does not define the complete LiveAgent")
        module.ANSWER_INSTRUCTIONS = prompts["answer_instructions"]
        module.DECISION_INSTRUCTIONS = prompts["decision_instructions"]
        for name in ("TOOL_DECISION_INSTRUCTIONS", "REPLY_CONTRACT"):
            if name.lower() in prompts:
                setattr(module, name, prompts[name.lower()])
        web_source = snapshot_dir / "before-web.py"
        if web_source.is_file():
            web_module = _load_file(
                "group_agent._web_profile_before_"
                + _sha(web_source.read_text(encoding="utf-8"))[:16],
                web_source,
            )
            module.WebTool = web_module.WebTool
    else:
        from group_agent import live_agent as module

        source = Path(module.__file__)
    metadata = {
        "profile": profile,
        "answer_prompt_sha256": _sha(module.ANSWER_INSTRUCTIONS),
        "decision_prompt_sha256": _sha(module.DECISION_INSTRUCTIONS),
        "tool_decision_prompt_sha256": _sha(
            getattr(module, "TOOL_DECISION_INSTRUCTIONS", module.DECISION_INSTRUCTIONS)
        ),
        "answer_prompt_chars": len(module.ANSWER_INSTRUCTIONS),
        "reply_contract_sha256": _sha(getattr(module, "REPLY_CONTRACT", "")),
        "reply_contract_chars": len(getattr(module, "REPLY_CONTRACT", "")),
        "prompt_sha256": _sha(module.ANSWER_INSTRUCTIONS + "\0" + module.DECISION_INSTRUCTIONS),
        "live_agent_sha256": _sha(source.read_text(encoding="utf-8")),
        "web_sha256": _sha(Path(inspect.getfile(module.WebTool)).read_text(encoding="utf-8")),
    }
    return module.LiveAgent, metadata


def inject_config(config):
    if config.gemini.name == config.deepseek.name:
        raise SystemExit("forced fallback requires distinct primary and fallback role aliases")
    config.model_breaker = ForcedPrimaryBreaker(config.gemini.name, config.model_breaker)
    return config


def _load_replay():
    return _load_file("_prompt_profile_replay_shadow", ROOT / "scripts" / "replay_shadow.py")


def _write_metadata(out: Path, metadata: dict):
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prompt_profile"] = metadata
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pairs_path = out / "pairs.jsonl"
    if pairs_path.is_file():
        rows = [
            json.loads(line)
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            row["prompt_profile"] = metadata
        pairs_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", required=True, choices=("before", "after"))
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path, default=DATA_ROOT / "serve/replay/prompt-ab")
    args, passthrough = parser.parse_known_args(argv)
    replay_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    replay_parser.add_argument("--name", default="replay")
    replay_parser.add_argument("--out")
    replay_parser.add_argument("--work-db")
    replay_parser.add_argument("--budget-cny", type=float, required=True)
    outputs, _ = replay_parser.parse_known_args(passthrough)
    if not math.isfinite(outputs.budget_cny) or outputs.budget_cny <= 0:
        parser.error("--budget-cny must be a positive finite number")
    out = Path(outputs.out or DATA_ROOT / "serve/replay" / outputs.name).resolve()
    work = Path(outputs.work_db or DATA_ROOT / "serve" / f"replay-{outputs.name}.sqlite3").resolve()
    if not all(path.is_relative_to(DATA_ROOT.resolve()) for path in (out, work)):
        parser.error("replay output and work database must stay inside data")
    # Pass the validated absolute destinations even when the caller uses defaults.
    passthrough += ["--out", str(out), "--work-db", str(work)]
    agent, metadata = load_profile(args.profile, args.snapshot_dir)
    metadata.update(
        forced_fallback=args.force_fallback,
        fault_injection="controlled_primary_unavailable" if args.force_fallback else "none",
        label=f"{args.profile}: controlled primary-unavailable fault injection"
        if args.force_fallback
        else args.profile,
    )
    replay = _load_replay()
    original_agent, original_build = replay.LiveAgent, replay.build_live_config

    def build(*pos, **kw):
        config = original_build(*pos, **kw)
        return inject_config(config) if args.force_fallback else config

    replay.LiveAgent, replay.build_live_config = agent, build
    try:
        result = replay.main(passthrough)
    finally:
        replay.LiveAgent, replay.build_live_config = original_agent, original_build
    if result == 0:
        _write_metadata(out, metadata)
        print(json.dumps({"prompt_profile": metadata}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
