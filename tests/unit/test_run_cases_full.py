"""T3.10: the full deterministic replay (`scripts/replay/run_cases.py`) inside the suite.

Every synthetic case must pass under the default strategy C with all 7 isolation counters
at 0, through the same `RunFlow` run path the online orchestrator uses.
"""

import importlib.util
import json
from pathlib import Path

from group_agent.orchestrator import Orchestrator
from group_agent.replay import ISOLATION_KEYS
from group_agent.replay.runner import CaseReplay
from group_agent.runflow import RunFlow

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/replay/cases"
# P0a acceptance items T3.10 requires a passing replay case for (AC-20's P0a part is also
# covered by unit tests: collection switched off per group, retention).
P0A_ACCEPTANCE = (
    "AC-01", "AC-02", "AC-03", "AC-04", "AC-05", "AC-14", "AC-15", "AC-17",
    "AC-19", "AC-20", "AC-21",
)  # fmt: skip

spec = importlib.util.spec_from_file_location("run_cases_cli", ROOT / "scripts/replay/run_cases.py")
run_cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_cases)


def test_run_cases_all_pass_with_zero_isolation(capsys):
    code = run_cases.main([str(CASES), "--json", "--check"])
    out = json.loads(capsys.readouterr().out)
    summary = out["summary"]
    assert out["strategy"] == "C"
    assert summary["cases"] == len(list(CASES.glob("synthetic-*.json"))) >= 37
    assert summary["passed"] == summary["cases"]
    assert summary["isolation"] == {k: 0 for k in ISOLATION_KEYS}
    assert summary["forbidden_context_used"] == 0
    assert summary["required_context_recall_micro"] == 1.0
    for ac in P0A_ACCEPTANCE:
        assert summary["by_acceptance"].get(ac, {}).get("pass", 0) >= 1, ac
        assert "fail" not in summary["by_acceptance"][ac], ac
    assert code == 0
    assert all(c["pass"] for c in out["cases"])


def test_check_flag_fails_on_a_failing_case(tmp_path, capsys):
    case = json.loads((CASES / "synthetic-top-level-at.json").read_text(encoding="utf-8"))
    case["annotations"]["expected_route"] = "web"  # planted mismatch
    path = tmp_path / "synthetic-planted.json"
    path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    assert run_cases.main([str(path), "--check"]) == 1
    assert run_cases.main([str(path)]) == 0  # scoring only, as before
    capsys.readouterr()


def test_replay_and_online_share_the_runflow_run_path():
    # The run itself (scheduler, deadline, context, teaching gate, pre-publish check,
    # publish) lives only in RunFlow; hosts implement hooks, not their own run loop.
    shared = ("_run", "_work", "_build_context", "_publish", "launch", "cancel_run")
    for host in (CaseReplay, Orchestrator):
        assert issubclass(host, RunFlow)
        for name in shared:
            assert name not in host.__dict__, (host.__name__, name)
