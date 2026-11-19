import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "replay_validation", Path("scripts/replay/validate_cases.py")
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_sidecar_merge_does_not_modify_extraction():
    case = {"case_id": "synthetic_case", "annotations": {"status": "needs_review"}}
    sidecar = {
        "schema_version": "annotations-sidecar-1",
        "cases": {"synthetic_case": {"status": "draft"}},
    }
    merged = validation.annotated_case(case, sidecar)
    assert merged["annotations"]["status"] == "draft"
    assert case["annotations"]["status"] == "needs_review"


@pytest.mark.parametrize(
    "required,forbidden",
    [(["future"], []), (["q"], []), (["missing"], []), (["before"], ["before"])],
)
def test_rejects_future_missing_or_conflicting_context(required, forbidden):
    case = {
        "events": [{"message_id": v} for v in ("before", "q", "future")],
        "anchor": {"message_id": "q"},
        "annotations": {
            "status": "draft",
            "required_context_ids": required,
            "forbidden_context_ids": forbidden,
        },
    }
    with pytest.raises(ValueError):
        validation.check_annotation_references(case)
