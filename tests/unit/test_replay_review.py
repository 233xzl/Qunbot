"""Offline review renderer and importer, with synthetic fixtures only."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/replay"))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/replay" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = load("render_review_under_test", "render_review.py")
ir = load("import_review_under_test", "import_review.py")
validation = load("validate_cases_under_test", "validate_cases.py")

BOT = "u_bot"


def text(value):
    return {"type": "text", "data": {"text": value}}


def at(uid):
    return {"type": "at", "data": {"qq": uid}}


def reply(mid):
    return {"type": "reply", "data": {"id": mid}}


def event(mid, uid, segments, t=1_700_000_000, sent=False):
    return {
        "time": t,
        "self_id": BOT,
        "post_type": "message_sent" if sent else "message",
        "message_type": "group",
        "sub_type": "normal",
        "group_id": "g_x",
        "message_id": mid,
        "user_id": uid,
        "sender": {"user_id": uid},
        "message": segments,
    }


def make_case(case_id="synthetic-review", events=None, anchor="m_q"):
    events = events or [
        event("m_a", "u_alice", [text("first")], 1_700_000_000),
        event("m_b", "u_bob", [text("second")], 1_700_000_060),
        event("m_q", "u_alice", [reply("m_b"), at(BOT), text("question?")], 1_700_000_120),
        event("m_r", BOT, [reply("m_q"), text("answer")], 1_700_000_130, sent=True),
    ]
    return {
        "case_id": case_id,
        "source": "synthetic",
        "categories": ["配额:测试", "引用追问"],
        "environment": {"bot_id": BOT},
        "anchor": {"message_id": anchor},
        "events": events,
        "annotations": {"status": "needs_review"},
        "legacy_result": {
            "status": "linked",
            "responses": [],
            "bot_reply": {
                "message_id": "m_r",
                "rule": "reply_segment_to_anchor",
                "confidence": "high",
                "delay_seconds": 10.0,
                "window_seconds": 300,
                "intervening_anchor_count": 0,
            },
        },
    }


def draft(required=(), forbidden=("m_r",), confidence="medium", route="chat"):
    return {
        "status": "draft",
        "confidence": confidence,
        "should_reply": True,
        "required_context_ids": list(required),
        "forbidden_context_ids": list(forbidden),
        "expected_relations": [],
        "expected_route": route,
        "should_clarify": False,
        "expected_knowledge_ids": [],
        "answer_points": [],
        "notes": [],
    }


def renderer(case=None, annotation=None, media_root=Path("/nonexistent-media")):
    return rr.CaseRenderer(case or make_case(), annotation, media_root, "../media")


def test_chat_text_and_annotation_strings_are_escaped():
    hostile = '<script>alert(1)</script><img src=x onerror=alert(2)> "q" http://example.invalid/a'
    case = make_case(
        events=[
            event("m_a", "u_alice", [text(hostile)]),
            event("m_q", "u_bob", [reply("m_a"), at(BOT), text(hostile)], 1_700_000_100),
        ]
    )
    case["legacy_result"]["bot_reply"] = None
    ann = draft(required=["m_a"], forbidden=[])
    ann["answer_points"] = [hostile]
    r = renderer(case, ann)
    payload = r.payload([], False)
    blob = payload["header_html"] + payload["conv_html"] + payload["panel_html"]
    assert "<script" not in blob
    assert "<img src=x" not in blob
    assert "&lt;script&gt;" in blob
    assert "<a " not in blob and "href=" not in blob  # URLs stay plain text
    # The JS literal used for case files cannot terminate an inline script.
    assert "</" not in rr.js_json({"x": "</script>"})


def test_aliases_follow_first_appearance_and_are_stable():
    events = [
        event("m_1", "u_c", [text("x")]),
        event("m_2", "u_a", [at("u_z"), text("y")]),
        event("m_3", BOT, [text("z")], sent=True),
        event("m_4", "u_b", [text("w")]),
        event("m_5", "u_c", [text("v")]),
    ]
    first = rr.assign_aliases(events, BOT)
    assert first == {BOT: "机器人", "u_c": "甲", "u_a": "乙", "u_b": "丙", "u_z": "丁"}
    assert rr.assign_aliases(json.loads(json.dumps(events)), BOT) == first
    assert rr.alias_color("甲", first) == rr.alias_color("甲", rr.assign_aliases(events, BOT))
    assert rr.alias_name(30) == "成员31"


def test_collapse_keeps_anchor_window_focus_and_bot_reply():
    visible = rr.visible_indices(100, anchor_index=60, focus={10}, keep={70})
    assert set(range(52, 61)) <= visible  # 8 before + anchor
    assert {8, 9, 10, 11, 12} <= visible and 7 not in visible and 13 not in visible
    assert 70 in visible and 69 not in visible and 71 not in visible
    runs = rr.fold_runs(100, visible)
    assert runs[0] == ("fold", 0, 7)
    assert ("fold", 13, 51) in runs and ("fold", 71, 99) in runs
    covered = sum(end - start + 1 for _, start, end in runs)
    assert covered == 100


def test_rendered_conversation_folds_far_unmarked_messages():
    events = [event(f"m_{i:03d}", "u_a", [text(str(i))], 1_700_000_000 + i) for i in range(30)]
    events.append(event("m_q", "u_b", [at(BOT), text("?")], 1_700_000_100))
    case = make_case(events=events)
    case["legacy_result"]["bot_reply"] = None
    html = renderer(case, draft(required=["m_002"], forbidden=[])).conversation_html()
    # Visible: 0..4 (around the mark) and 22..30 (8 before the anchor); 5..21 fold.
    assert html.count('class="fold"') == 1 and 'data-count="17"' in html
    assert 'id="msg-m_002"' in html and 'data-draft="required"' in html


def test_missing_image_placeholder_and_available_image(tmp_path):
    sha = "a" * 64
    (tmp_path / f"{sha}.png").write_bytes(b"\x89PNG")
    ok = {
        "media_ref": {
            "status": "available",
            "reason": "fetched_via_onebot_get_image",
            "path": f"media/{sha}.png",
            "width": 2,
            "height": 2,
        },
        "local_metadata": {"sub_type": 0},
    }
    html = rr.image_html(ok, tmp_path, "../../media")
    assert f'src="../../media/{sha}.png"' in html and "thumb" in html
    gone = {**ok, "media_ref": {**ok["media_ref"], "path": f"media/{'b' * 64}.png"}}
    assert "原图不可用" in rr.image_html(gone, tmp_path, "../../media")
    assert "本地文件缺失" in rr.image_html(gone, tmp_path, "../../media")
    v10 = {"availability": "unavailable", "reason": "not_acquired_no_network"}
    assert "原图不可用" in rr.image_html(v10, tmp_path, "x")
    assert "首批未下载图片" in rr.image_html(v10, tmp_path, "x")
    bad = {**ok, "media_ref": {**ok["media_ref"], "path": "media/../../secret.png"}}
    assert "图片路径无效" in rr.image_html(bad, tmp_path, "x")


def test_priority_selection_is_deterministic():
    annotations = {
        f"case_{i:02d}": draft(confidence="low" if i % 7 == 0 else "high") for i in range(30)
    }
    annotations["case_undrafted"] = {"status": "needs_review"}
    first = rr.select_priority(annotations)
    shuffled = dict(reversed(list(annotations.items())))
    assert rr.select_priority(shuffled) == first
    assert len(first["random"]) == 10 and "case_undrafted" not in first["random"]
    assert first["low_confidence"] == ["case_00", "case_07", "case_14", "case_21", "case_28"]
    assert rr.select_priority({"only": draft()})["random"] == ["only"]


def test_reply_ancestry_follows_quotes_without_cycles():
    events = [
        event("m_1", "u_a", [reply("m_3"), text("x")]),
        event("m_2", "u_b", [reply("m_1"), text("y")]),
        event("m_3", "u_c", [reply("m_2"), at(BOT)]),
    ]
    assert rr.reply_ancestry(events, "m_3") == ["m_2", "m_1"]


def write_batch(tmp_path: Path, annotation: dict | None):
    batch = tmp_path / "batch-x"
    batch.mkdir()
    case = make_case()
    (batch / "synthetic-review.json").write_text(json.dumps(case), encoding="utf-8")
    sidecar = {
        "schema_version": "annotations-sidecar-1",
        "cases": {"synthetic-review": annotation or {"status": "needs_review"}},
    }
    (batch / "annotations.draft.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return batch, case, sidecar


def test_render_batch_writes_offline_site(tmp_path):
    batch, _, _ = write_batch(tmp_path, draft(required=["m_b"]))
    result = rr.render_batch(batch, batch / "annotations.draft.json", tmp_path / "media")
    assert result["cases"] == 1 and result["drafted"] == 1
    index = (batch / "review/index.html").read_text(encoding="utf-8")
    assert "如何复核" in index and "review-result.json" in index
    assert "http://" not in index.replace("http-equiv", "") and "https://" not in index
    assert (batch / "review/cases/synthetic-review.js").is_file()
    priority = json.loads((batch / "review/priority.json").read_text(encoding="utf-8"))
    assert priority["seed"] == 20261107 and priority["random"] == ["synthetic-review"]


def export_for(annotation, entry):
    entry.setdefault("draft_fingerprint", rr.fingerprint(annotation))
    return {"format": rr.REVIEW_FORMAT, "batch": "batch-x", "cases": {"synthetic-review": entry}}


@pytest.mark.parametrize(
    "entry,error",
    [
        ({"verdict": "agree"}, None),
        ({"verdict": "maybe"}, "bad_verdict"),
        (
            {
                "verdict": "modify",
                "route": "teleport",
                "required_context_ids": [],
                "forbidden_context_ids": [],
            },
            "bad_route",
        ),
        (
            {"verdict": "modify", "required_context_ids": ["m_nope"], "forbidden_context_ids": []},
            "unknown_message",
        ),
        (
            {"verdict": "modify", "required_context_ids": ["m_r"], "forbidden_context_ids": []},
            "context_not_before_anchor",
        ),
        (
            {
                "verdict": "modify",
                "required_context_ids": ["m_a"],
                "forbidden_context_ids": ["m_a"],
            },
            "context_conflict",
        ),
    ],
)
def test_import_validates_values_and_message_ids(tmp_path, entry, error):
    ann = draft()
    _, case, sidecar = write_batch(tmp_path, ann)
    export = export_for(ann, entry)
    if error is None:
        result = ir.build_reviewed(export, [case], sidecar, "batch-x")
        assert result["cases"]["synthetic-review"]["annotations"]["status"] == "reviewed"
        return
    with pytest.raises(ir.ReviewImportError) as exc:
        ir.build_reviewed(export, [case], sidecar, "batch-x")
    assert exc.value.counts[error] == 1


def test_import_rejects_unknown_case_and_wrong_batch(tmp_path):
    ann = draft()
    _, case, sidecar = write_batch(tmp_path, ann)
    export = export_for(ann, {"verdict": "agree"})
    export["cases"]["synthetic-missing"] = {"verdict": "agree"}
    export["batch"] = "batch-y"
    with pytest.raises(ir.ReviewImportError) as exc:
        ir.build_reviewed(export, [case], sidecar, "batch-x")
    assert exc.value.counts["unknown_case"] == 1 and exc.value.counts["batch_mismatch"] == 1


def test_import_modify_applies_overrides_and_validates(tmp_path):
    ann = draft(required=[], forbidden=["m_r"])
    _, case, sidecar = write_batch(tmp_path, ann)
    export = export_for(
        ann,
        {
            "verdict": "modify",
            "route": "clarify",
            "should_clarify": True,
            "required_context_ids": ["m_b"],
            "forbidden_context_ids": ["m_a"],
            "comment": "需要引用的消息",
        },
    )
    result = ir.build_reviewed(export, [case], sidecar, "batch-x")
    reviewed = {"schema_version": ir.REVIEWED_FORMAT, "cases": result["cases"]}
    merged = validation.annotated_case(case, sidecar, reviewed)
    got = merged["annotations"]
    assert got["expected_route"] == "clarify" and got["should_clarify"] is True
    assert got["required_context_ids"] == ["m_b"]
    assert got["forbidden_context_ids"] == ["m_a", "m_r"]
    validation.check_annotation_references(merged)
    assert sidecar["cases"]["synthetic-review"]["expected_route"] == "chat"  # draft untouched


def test_import_skips_stale_and_keeps_hints_for_undrafted(tmp_path):
    ann = draft()
    _, case, sidecar = write_batch(tmp_path, ann)
    stale = export_for(ann, {"verdict": "agree", "draft_fingerprint": "0" * 16})
    assert ir.build_reviewed(stale, [case], sidecar, "batch-x") == {"cases": {}, "stale": 1}
    (tmp_path / "u").mkdir()
    _, case2, sidecar2 = write_batch(tmp_path / "u", None)
    undrafted = sidecar2["cases"]["synthetic-review"]
    with pytest.raises(ir.ReviewImportError):
        ir.build_reviewed(export_for(undrafted, {"verdict": "agree"}), [case2], sidecar2, "batch-x")
    hint = export_for(
        undrafted,
        {
            "verdict": "modify",
            "route": "web",
            "required_context_ids": [],
            "forbidden_context_ids": [],
        },
    )
    record = ir.build_reviewed(hint, [case2], sidecar2, "batch-x")["cases"]["synthetic-review"]
    assert record["annotations"] is None and record["owner_hint"]["route"] == "web"
