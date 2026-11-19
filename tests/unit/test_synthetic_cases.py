"""Handwritten synthetic replay cases: schema, annotation bounds, privacy and CLI smoke run."""

import copy
import importlib.util
import json
from pathlib import Path
import re

import jsonschema
import pytest

from group_agent.app import main

ROOT = Path(__file__).resolve().parents[2]
CASES = sorted((ROOT / "tests/replay/cases").glob("synthetic-*.json"))
# `nudge_merged`: a bare urging run that ends without an answer (runflow, strategy C `urges`).
ROUTES = {None, "clarify", "local_knowledge", "web", "chat", "teaching", "nudge_merged"}
URL = re.compile(r"https?://([^/\s\"'?#]+)")

spec = importlib.util.spec_from_file_location(
    "replay_validation_synthetic", ROOT / "scripts/replay/validate_cases.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator():
    schema = json.loads((ROOT / "tests/replay/schema.json").read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def test_case_set_is_unique_and_documented():
    assert len(CASES) >= 30
    ids = [load(p)["case_id"] for p in CASES]
    assert len(ids) == len(set(ids))
    coverage = (ROOT / "docs/SYNTHETIC-CASES.md").read_text(encoding="utf-8")
    for path in CASES:
        assert path.name in coverage, path.name
        assert path.stem.replace("-", "_") == load(path)["case_id"]


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_is_valid_synthetic_and_fictional(path, validator):
    case = load(path)
    validator.validate(case)
    validation.check_annotation_references(case)
    assert case["source"] == "synthetic"
    assert case["environment"]["config_overrides"] == {}
    assert case["annotations"]["expected_route"] in ROUTES
    assert case["acceptance_ids"] and all(
        re.fullmatch(r"AC-\d\d", a) for a in case["acceptance_ids"]
    )
    # Every URL, including those inside card JSON strings, uses a reserved test domain.
    for host in URL.findall(json.dumps(case, ensure_ascii=False).replace("\\/", "/")):
        assert host.endswith((".invalid", ".example", "example.com")), host
    for event in case["events"]:
        for segment in event.get("message", []):
            if segment["type"] == "json":
                card = json.loads(segment["data"]["data"])
                assert {"app", "view", "prompt", "meta"} <= card.keys()


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_runs_through_offline_cli(path, capsys):
    assert main(["--replay", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["sent_to_qq"] == 0


def test_schema_rejects_card_segment_in_real_sample(validator):
    case = load(ROOT / "tests/replay/cases/synthetic-card-bilibili-quoted.json")
    real = copy.deepcopy(case)
    real["source"] = "real"
    real["provenance"].update(
        source_kind="snowluma_messages_db",
        read_mode="sqlite_uri_mode_ro_query_only",
        pseudonymization="hmac_sha256_domain_separated_96bit",
        storage_policy="local_raw_only",
    )
    assert not validator.is_valid(real)
    card = real["events"][0]["message"][0]
    real["events"][0]["message"] = [
        {"type": "redacted", "data": {"original_type": "json", "reason": "opaque_payload_removed"}}
    ]
    assert validator.is_valid(real)
    # A card nested inside a quoted message is rejected for real samples too.
    real["events"][-1]["message"][0]["data"]["message"] = [card]
    assert not validator.is_valid(real)
