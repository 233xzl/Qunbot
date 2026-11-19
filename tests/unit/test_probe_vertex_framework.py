"""Offline tests of scripts/probe_vertex_framework.py (--dry fakes, no network)."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel
import pytest

from group_agent.models import Ledger

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "probe_vertex_framework", ROOT / "scripts/probe_vertex_framework.py"
)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)

DRY_SECRETS = ("dry-run-api-key-0000", "dry-run-project-0000")


def argv(tmp_path, budget="5", *extra):
    # A missing settings file makes --dry use synthetic specs (settings.toml is local-only).
    return [
        "--dry",
        "--budget-cny",
        budget,
        "--settings",
        str(tmp_path / "missing.toml"),
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def load_results(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))


def test_dry_run_covers_every_case_and_passes(tmp_path, capsys):
    assert probe.main(argv(tmp_path)) == 0
    results = load_results(tmp_path)
    cases = {r["case"]: r for r in results["cases"]}
    assert list(cases) == list(probe.CASES)
    assert results["totals"] == {"passed": 7, "failed": 0, "skipped": 0}
    assert results["required_passed"] and not results["budget_stopped"] and results["dry"]
    assert all(r["usage"]["requests"] >= 1 and r["latency_s"] is not None for r in cases.values())
    assert all(r["mode"] == "vertex" and r["error"] is None for r in cases.values())
    # Generation cases name the Pydantic AI provider; search cases use the HTTP client.
    assert cases["answer_text"]["provider"] == "function"
    assert cases["answer_text"]["call_mode"] == "text"
    assert (
        cases["answer_structured"]["call_mode"]
        == cases["flash_structured"]["call_mode"]
        == "native"
    )
    assert cases["flash_search"]["sources"] == 2 and cases["flash_search"]["queries"] == 1
    assert cases["flash_url_context"]["content_chars"] > 0
    assert "flash_search" in capsys.readouterr().out


def test_tool_roundtrip_really_calls_the_tool(tmp_path):
    probe.main(argv(tmp_path))
    rec = next(r for r in load_results(tmp_path)["cases"] if r["case"] == "answer_tool_roundtrip")
    assert rec["tool_called"] is True and rec["tool_calls"] == 1
    assert rec["usage"]["requests"] == 2 and rec["passed"]
    assert probe.TOOL_CODE in rec["answer"]
    image = next(r for r in load_results(tmp_path)["cases"] if r["case"] == "answer_image")
    assert image["answer"] == "Red"


def test_tool_case_fails_when_the_model_skips_the_tool(tmp_path):
    from pydantic_ai.messages import ModelResponse, TextPart

    def no_tool(messages, info):
        return ModelResponse(parts=[TextPart(f"I think it is {probe.TOOL_CODE}")])

    code = probe.main(
        argv(tmp_path, "5", "--cases", "answer_tool_roundtrip"),
        model_builder=lambda spec, creds: FunctionModel(no_tool),
    )
    rec = load_results(tmp_path)["cases"][0]
    assert code == 1 and rec["tool_called"] is False and rec["check"] == "tool_not_called"


def test_ledger_stays_within_budget_and_settles(tmp_path):
    ledger = Ledger(5.0)
    args = probe.parse_args(argv(tmp_path))
    results = asyncio.run(probe.run_probe(args, ledger=ledger))
    assert 0 < ledger.spent <= 5.0 and ledger.pending == 0
    assert results["spent_cny"] == pytest.approx(ledger.spent, abs=1e-6)
    assert sum(r["cost_cny"] for r in results["cases"]) == pytest.approx(ledger.spent, abs=1e-5)
    # Every model request (the tool round trip makes 2) and search request is an entry.
    assert len(ledger.records) == 8
    tasks = [r.task for r in ledger.records]
    assert tasks.count("probe_search") == 1 and tasks.count("probe_url_context") == 1


def test_no_secret_looking_strings_in_outputs(tmp_path, capsys):
    probe.main(argv(tmp_path))
    text = (tmp_path / "out" / "results.json").read_text(encoding="utf-8") + capsys.readouterr().out
    for secret in DRY_SECRETS:
        assert secret not in text
    assert not re.search(r"AIza[0-9A-Za-z_-]{10,}|sk-[A-Za-z0-9_-]{20,}|projects/|https?://", text)


def test_tiny_budget_stops_remaining_cases_with_a_budget_record(tmp_path, capsys):
    ledger = Ledger(0.001)
    args = probe.parse_args(argv(tmp_path, "0.001"))
    results = asyncio.run(probe.run_probe(args, ledger=ledger))
    first, *rest = results["cases"]
    assert first["case"] == "answer_text" and first["error"] == "BudgetExceeded"
    assert results["budget_stopped"] and len(results["cases"]) == 7
    assert all(r["skipped"] and r["error"] == "budget_stop" and not r["passed"] for r in rest)
    assert results["totals"] == {"passed": 0, "failed": 1, "skipped": 6}
    assert ledger.spent <= 0.001 and ledger.pending == 0 and not results["required_passed"]
    assert probe.main(argv(tmp_path, "0.001")) == 1
    assert "budget stop" in capsys.readouterr().out


def test_provider_failures_record_only_kinds(tmp_path):
    leak = "https://aiplatform.example/v1?key=SECRET-KEY-VALUE projects/secret-project"

    def failing(messages, info):
        raise ModelHTTPError(400, "dry-model", leak)

    def failing_factory(timeout):
        class Boom(Exception):
            code = 403

        class Client:
            class aio:  # noqa: N801 - mimics client.aio
                class models:  # noqa: N801
                    @staticmethod
                    async def generate_content(**kwargs):
                        raise Boom(leak)

        return Client

    code = probe.main(
        argv(tmp_path, "5", "--cases", "answer_text", "flash_search", "flash_url_context"),
        model_builder=lambda spec, creds: FunctionModel(failing),
        client_factory=failing_factory,
    )
    text = (tmp_path / "out" / "results.json").read_text(encoding="utf-8")
    assert "SECRET" not in text and "secret-project" not in text and "aiplatform" not in text
    cases = {r["case"]: r for r in json.loads(text)["cases"]}
    assert code == 1
    assert cases["answer_text"]["error"] == "http_400" and not cases["answer_text"]["passed"]
    assert cases["flash_search"]["error"] == "http_403"
    assert (
        cases["flash_url_context"]["error"] == "http_403" and cases["flash_url_context"]["optional"]
    )


def test_optional_url_context_failure_keeps_exit_zero(tmp_path):
    def factory(timeout):
        client = probe.DryGenaiClient()
        original = client._generate

        async def generate(*, model, contents, config):
            if getattr(config.tools[0], "url_context", None) is not None:
                raise TimeoutError
            return await original(model=model, contents=contents, config=config)

        client.aio.models.generate_content = generate
        return client

    assert probe.main(argv(tmp_path), client_factory=factory) == 0
    cases = {r["case"]: r for r in load_results(tmp_path)["cases"]}
    assert (
        cases["flash_url_context"]["error"] == "timeout"
        and not cases["flash_url_context"]["passed"]
    )
    assert load_results(tmp_path)["totals"] == {"passed": 6, "failed": 1, "skipped": 0}


def test_case_subset_keeps_canonical_order(tmp_path):
    assert probe.main(argv(tmp_path, "5", "--cases", "flash_search", "answer_text")) == 0
    assert [r["case"] for r in load_results(tmp_path)["cases"]] == ["answer_text", "flash_search"]


def test_gateway_and_store_modules_are_not_imported():
    # Fresh interpreter: other tests in this session import the gateway themselves.
    code = (
        "import importlib.util, sys;"
        f"s = importlib.util.spec_from_file_location('p', r'{ROOT / 'scripts/probe_vertex_framework.py'}');"
        "m = importlib.util.module_from_spec(s); sys.modules['p'] = m; s.loader.exec_module(m);"
        "print([n for n in sys.modules if n.split('.')[-1] in ('gateway', 'live_send', 'outbox', 'serve')"
        " and n.startswith('group_agent')])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr[-300:]


def test_budget_is_required():
    with pytest.raises(SystemExit):
        probe.parse_args(["--dry"])
