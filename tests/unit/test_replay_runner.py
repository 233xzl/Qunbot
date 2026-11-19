"""Deterministic replay tier (T1.9): every synthetic case through the runner and scorer."""

import copy
import json
from pathlib import Path

import pytest

from group_agent.policy import RuntimeLimits
from group_agent.replay import ISOLATION_KEYS, run_case, score_case
from group_agent.replay.mocks import MockBook

ROOT = Path(__file__).resolve().parents[2]
CASES = sorted((ROOT / "tests/replay/cases").glob("synthetic-*.json"))

# Mocks that must stay unused because the behaviour under test is not calling them.
EXPECTED_UNUSED = {"synthetic_cost_80pct": ["tools[0]:web_search"]}
# Annotation checks known to fail on the deterministic tier, with the reason. Empty now:
# every synthetic case passes with strategy C (the production default since 2026-11-08,
# shared with the online orchestrator). Add entries instead of weakening checks.
KNOWN_FAILING: dict[str, str] = {}
# Cases that need strategy C's implicit relations; strategy B misses or over-selects there.
C_ONLY = {
    "synthetic_injection_forged_section": "author_burst picks the quoted card's companion text",
    "synthetic_two_questions_branch_followup": "B's asker window pulls in the other branch",
    "synthetic_new_request_after_answer": "B's asker window pulls in the earlier requests",
    "synthetic_nudge_merged": "only C has the urges rule and the nudge merge",
}


def load(name: str) -> dict:
    return json.loads((ROOT / "tests/replay/cases" / f"{name}.json").read_text(encoding="utf-8"))


_cache: dict[tuple[str, str], tuple[dict, dict]] = {}


def replay(name: str, strategy: str = "C") -> tuple[dict, dict]:
    key = (name, strategy)
    if key not in _cache:
        case = load(name)
        _cache[key] = (case, run_case(case, strategy))
    return _cache[key]


def run_of(result: dict, message_id: str) -> dict:
    return next(r for r in result["runs"] if r["message_id"] == message_id)


def core_ids(run: dict) -> list[str]:
    """Context ids without the `recent_chat` background slice (pinned separately)."""
    reasons = run["context_reasons"]
    return [c for c in run["context_ids"] if not reasons.get(c, "").startswith("recent_chat/")]


def outbound_to(result: dict, message_id: str) -> list[dict]:
    return [o for o in result["outbound"] if o["reply_to"] == message_id]


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_mock_names_follow_convention(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    ids = {str(e["message_id"]) for e in case["events"] if "message_id" in e}
    assert MockBook(case["mocks"], ids, case["anchor"]["message_id"]).unknown == []


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_passes_annotations_with_zero_isolation_errors(path, request):
    case, result = replay(path.stem)
    if case["case_id"] in KNOWN_FAILING:
        request.applymarker(pytest.mark.xfail(reason=KNOWN_FAILING[case["case_id"]], strict=True))
    score = score_case(case, result)
    assert score["isolation"] == {k: 0 for k in ISOLATION_KEYS}
    assert score["failed"] == [], score["checks"]
    assert result["sent_to_qq"] == 0
    assert result["unknown_mocks"] == [] and result["warnings"] == []
    assert result["unused_mocks"] == EXPECTED_UNUSED.get(case["case_id"], [])


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_strategy_b_baseline_passes_except_c_only_cases(path):
    case, result = replay(path.stem, "B")
    score = score_case(case, result)
    assert score["isolation"] == {k: 0 for k in ISOLATION_KEYS}
    if case["case_id"] in C_ONLY:
        assert score["failed"], C_ONLY[case["case_id"]]  # documents the difference
    else:
        assert score["failed"] == [], score["checks"]


@pytest.mark.parametrize("strategy", ["A", "D"])
def test_baseline_strategies_keep_non_context_isolation_at_zero(strategy):
    # A/D are comparison baselines: they may select forbidden context and (D) attach other
    # users' images, but filtering, sending and knowledge writes stay isolated.
    for path in CASES:
        case, result = replay(path.stem, strategy)
        iso = score_case(case, result)["isolation"]
        assert {k: v for k, v in iso.items() if k != "wrong_user_attachment" and v} == {}


def test_strategy_d_attaches_other_users_image_and_is_counted():
    case, result = replay("synthetic-concurrent-six-users", "D")
    assert score_case(case, result)["isolation"]["wrong_user_attachment"] > 0


# -- AC-01 admission ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name,status",
    [
        ("synthetic-nested-at", "collected"),
        ("synthetic-at-all-fake-text", "collected"),
        ("synthetic-private-message", "ignored_invalid"),
        ("synthetic-blacklisted-user", "ignored_blocked"),
        ("synthetic-bot-self-message", "collected"),
    ],
)
def test_non_triggers_create_no_anchor_run(name, status):
    case, result = replay(name)
    anchor = case["anchor"]["message_id"]
    assert all(r["message_id"] != anchor for r in result["runs"])
    # Unparseable events (private chat) carry no message id in the event log.
    assert [e["status"] for e in result["events"] if e["message_id"] in (anchor, None)] == [status]
    assert outbound_to(result, anchor) == []


# -- AC-02 / AC-19 concurrency and send rate ----------------------------------------


def test_concurrent_six_users_queue_and_rate_limit():
    _, result = replay("synthetic-concurrent-six-users")
    assert result["max_active_runs"] == 4
    for mid in ("m_q5", "m_q6"):
        run = run_of(result, mid)
        assert run["queued"] and run["started_at"] - run["arrived_at"] <= 30
    sends = sorted(o["sent_at"] for o in result["outbound"])
    assert len(sends) == 6
    assert all(b - a >= 2 for a, b in zip(sends, sends[1:]))  # D-10 min interval
    assert all(sum(1 for t in sends if s <= t < s + 60) <= 10 for s in sends)
    for o in result["outbound"]:
        run = next(r for r in result["runs"] if r["run_id"] == o["run_id"])
        assert (o["reply_to"], o["requester"]) == (run["message_id"], run["user_id"])
    assert [r["images"] for r in result["runs"] if r["images"]] == [
        [{"message_id": "m_q2", "user_id": "u_2"}]
    ]


# -- AC-03 / AC-04 blocking, cancellation, timeout, queue bounds --------------------


def test_tool_block_cancel_and_timeout_only_change_their_own_run():
    _, result = replay("synthetic-tool-block-timeout-cancel")
    a, b = run_of(result, "m_a"), run_of(result, "m_b")
    c, d = run_of(result, "m_c"), run_of(result, "m_d")
    assert b["ended_at"] < a["ended_at"] and b["ended_at"] - b["arrived_at"] < 1
    assert a["tool_calls"][0]["t_end"] - a["tool_calls"][0]["t_start"] == 10
    assert (c["state"], c["reason"], c["ended_at"]) == ("cancelled", "user_cancel:m_cancel", 10)
    assert (d["state"], d["reason"], d["ended_at"]) == ("timeout", "deadline", 120)
    assert outbound_to(result, "m_c") == outbound_to(result, "m_d") == []
    assert [o["state"] for o in outbound_to(result, "m_a")] == ["acked"]


def test_fast_request_wait_p95_below_one_second_over_100_rounds():
    # AC-03: A's tool blocks 10 s, B uses a 100 ms model; slots are free. 100 rounds with B
    # arriving at a different moment of A's block (0..9.9 s) each time.
    base = load("synthetic-tool-block-timeout-cancel")
    waits, finished_first = [], 0
    for n in range(100):
        case = copy.deepcopy(base)
        case["events"] = [e for e in case["events"] if e["message_id"] in ("m_a", "m_b")]
        case["mocks"]["faults"] = []
        a_tool = case["mocks"]["tools"][0]
        a_tool["delay_ms"] = 10000 + 100 * n  # A blocks for 10 s and more
        case["events"][1]["time"] = case["events"][0]["time"] + n % 10
        result = run_case(case)
        a, b = run_of(result, "m_a"), run_of(result, "m_b")
        waits.append(b["started_at"] - b["arrived_at"])
        finished_first += b["ended_at"] < a["ended_at"]
    waits.sort()
    assert finished_first == 100
    assert waits[94] < 1.0  # p95 scheduling wait


def test_single_user_flood_is_bounded_with_one_busy_notice():
    _, result = replay("synthetic-single-user-flood")
    states = {r["message_id"]: r["state"] for r in result["runs"]}
    assert states["m_f7"] == states["m_f8"] == "rejected_busy"
    notices = [o for o in result["outbound"] if o["kind"] == "busy_notice"]
    assert [(o["reply_to"], o["requester"]) for o in notices] == [("m_f7", "u_1")]
    queued = [r for r in result["runs"] if r["queued"]]
    assert {r["message_id"] for r in queued} == {"m_f3", "m_f4", "m_f5", "m_f6"}
    assert all(r["started_at"] - r["arrived_at"] <= 30 for r in queued)
    x = run_of(result, "m_x")
    assert not x["queued"] and x["started_at"] == x["arrived_at"]


@pytest.mark.xfail(strict=True, reason="RUN-11 per-user fair scheduling is P0b; P0a is FIFO")
def test_p0b_fair_scheduling_when_global_slots_are_full():
    # With all global slots busy, another user's request should get a free slot ahead of
    # the flooding user's remaining backlog.
    result = run_case(load("synthetic-single-user-flood"), "B", RuntimeLimits(global_active=2))
    assert run_of(result, "m_x")["started_at"] < run_of(result, "m_f5")["started_at"]


# -- AC-05 independent branches of one user ------------------------------------------


def test_followup_quoting_one_answer_takes_that_branch_only():
    _, result = replay("synthetic-two-questions-branch-followup")
    q1, q2, q = run_of(result, "m_q1"), run_of(result, "m_q2"), run_of(result, "m_q")
    assert core_ids(q) == ["m_q1", "m_a1"]
    assert q["context_reasons"]["m_a1"].startswith("reply_to/")
    assert q["context_reasons"]["m_q1"].startswith("answers/")
    # Each question is its own run with its own reply mapping.
    assert {(o["reply_to"], o["requester"]) for o in result["outbound"]} == {
        ("m_q1", "u_1"),
        ("m_q2", "u_1"),
        ("m_q", "u_1"),
    }
    assert q1["run_id"] != q2["run_id"] and not {"m_a1", "m_a2"} & set(core_ids(q2))


def test_new_request_after_answer_and_pending_request_stay_separate():
    # Incident 2026-11-09 pattern: the asker fires two new, unrelated requests right after
    # an answer while another member also asks; every run keeps its own question.
    _, result = replay("synthetic-new-request-after-answer")
    for mid in ("m_q2", "m_q3"):
        assert core_ids(run_of(result, mid)) == []
    assert {(o["reply_to"], o["requester"]) for o in result["outbound"]} == {
        ("m_q1", "u_1"),
        ("m_o1", "u_2"),
        ("m_q2", "u_1"),
        ("m_q3", "u_1"),
    }
    runs = {r["run_id"]: r for r in result["runs"]}
    assert all(runs[o["run_id"]]["message_id"] == o["reply_to"] for o in result["outbound"])


# -- AC-09 scope ---------------------------------------------------------------------


def test_cross_group_runs_reply_in_their_own_group():
    _, result = replay("synthetic-cross-group-private-scope")
    assert {(o["group_id"], o["reply_to"]) for o in result["outbound"]} == {
        ("g_2", "m_g2_1"),
        ("g_1", "m_g1_q"),
    }
    assert core_ids(run_of(result, "m_g1_q")) == []


def test_public_knowledge_reused_and_foreign_group_entry_filtered():
    _, result = replay("synthetic-public-knowledge-embedding-offline")
    run = run_of(result, "m_q")
    assert run["knowledge"] == {"kb_syn_public_yuni": 1}
    assert run["knowledge_filtered"] == ["kb_syn_g2_yuni"]
    assert "embedding_offline" in run["degraded"]
    assert run["tool_calls"][0]["args"]["mode"] == "keyword_only"


# -- AC-10 teaching -------------------------------------------------------------------


def test_duplicate_teaching_keeps_one_record_and_revision():
    _, result = replay("synthetic-teach-duplicate")
    assert result["knowledge_final"] == {
        "kb_syn_kaihei": {"scope": "group:g_1", "revision": 1, "active": True, "term": "开黑时间"}
    }
    assert [e["outcome"] for e in result["knowledge_log"]] == ["committed", "duplicate"]
    assert result["triggered"] == ["m_t1", "m_t2"]  # the redelivered m_t1 is deduplicated
    (t1,), (t2,) = outbound_to(result, "m_t1"), outbound_to(result, "m_t2")
    assert t2["sent_at"] < t1["sent_at"] == 45  # slow_ack:m_t1


def test_stale_revision_correction_is_a_visible_conflict():
    _, result = replay("synthetic-teach-stale-revision")
    teaching = run_of(result, "m_fix2")["teaching"]
    assert (teaching["outcome"], teaching["expected"], teaching["revision"]) == ("conflict", 1, 2)
    assert result["knowledge_final"]["kb_syn_moyu"]["revision"] == 2
    assert not any(e["applied"] for e in result["knowledge_log"])


def test_missing_scope_is_not_committed():
    _, result = replay("synthetic-teach-missing-scope")
    assert result["knowledge_final"] == {}
    assert run_of(result, "m_t")["teaching"]["outcome"] == "missing_scope"


# -- AC-11 degradation ----------------------------------------------------------------


def test_rate_limited_primary_falls_back():
    _, result = replay("synthetic-degrade-model-rate-limit")
    calls = run_of(result, "m_q")["model_calls"]
    assert [(c["name"], c["status"]) for c in calls] == [("primary", "error"), ("fallback", "ok")]
    assert outbound_to(result, "m_q")[0]["text"].startswith("また明日")


def test_vision_timeout_is_bounded_and_asks_for_material():
    _, result = replay("synthetic-degrade-vision-timeout")
    run = run_of(result, "m_img_q")
    vision = run["model_calls"][0]
    assert (vision["status"], vision["t_end"] - vision["t_start"]) == ("timeout", 25)
    assert run["material_gaps"] == ["image_unrecognized"] and run["clarify"]


def test_web_failures_and_card_fetch_timeout_are_recorded():
    _, result = replay("synthetic-degrade-web-failure")
    assert [c["status"] for c in run_of(result, "m_q")["tool_calls"]] == ["error", "error"]
    _, result = replay("synthetic-card-tuwen-fetch-failed")
    run = run_of(result, "m_q")
    assert [c["status"] for c in run["tool_calls"]] == ["timeout"]
    assert run["cards"][0]["source"] == "虚构日报" and run["state"] == "sent"


# -- AC-14 mid-run changes ------------------------------------------------------------


def test_knowledge_change_blocks_old_answer_and_regenerates():
    _, result = replay("synthetic-knowledge-corrected-midrun")
    run = run_of(result, "m_q")
    assert run["context_ids"] == []  # snapshot frozen; m_other/m_fix arrive later
    assert "regenerating" in [s for _, s in run["transitions"]]
    assert run["knowledge"] == {"kb_syn_quantum_tea": 2}
    (sent,) = outbound_to(result, "m_q")
    assert "没去取" in sent["text"]


def test_blacklist_during_generation_cancels_publish():
    _, result = replay("synthetic-blacklist-midrun")
    run = run_of(result, "m_q")
    assert (run["state"], run["reason"]) == ("cancelled", "requester_blocked")
    assert result["outbound"] == []


# -- AC-15 redelivery -----------------------------------------------------------------


def test_redelivery_creates_one_run_and_lost_ack_is_uncertain():
    _, result = replay("synthetic-redelivery-lost-ack")
    assert result["triggered"] == ["m_q"]
    assert [e["status"] for e in result["events"] if e["message_id"] == "m_q"] == [
        "trigger",
        "duplicate",
        "duplicate",
    ]
    (out,) = result["outbound"]
    assert (out["state"], out["error_kind"]) == ("uncertain", "ActionTimeout")


@pytest.mark.xfail(strict=True, reason="RUN-12 outbound reconciliation is P0b")
def test_p0b_uncertain_outbound_reconciled_from_platform_echo():
    # The bot's own echoed message replying to m_q proves delivery.
    _, result = replay("synthetic-redelivery-lost-ack")
    assert result["outbound"][0]["state"] == "acked"


# -- AC-17 injection ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,context",
    [
        ("synthetic-injection-in-history", ["m_inj"]),
        ("synthetic-forged-owner", ["m_fake"]),
        ("synthetic-web-injection", []),
        ("synthetic-injection-quoted-owner", ["m_own"]),
        ("synthetic-injection-forged-section", ["m_inj", "m_card"]),
        ("synthetic-injection-card-write-request", ["m_card"]),
    ],
)
def test_injected_material_never_writes(name, context):
    _, result = replay(name)
    run = run_of(result, "m_q")
    assert core_ids(run) == context and run["teaching"] is None
    assert run["dropped_requests"] == []
    assert result["knowledge_log"] == [] and result["knowledge_final"] == {}


# -- AC-20 / AC-13 recall and retention ------------------------------------------------


def test_recalled_and_expired_quotes_become_material_gaps():
    _, result = replay("synthetic-image-recall")
    run = run_of(result, "m_image_question")
    assert run["context_ids"] == [] and run["material_gaps"] == ["quote_recalled"]
    _, result = replay("synthetic-recall-and-expiry")
    run = run_of(result, "m_q")
    assert run["material_gaps"] == ["quote_expired"]
    assert not {"m_old", "m_rc"} & set(run["context_ids"])


# -- AC-21 cost cap -------------------------------------------------------------------


def test_cost_80pct_disables_web():
    _, result = replay("synthetic-cost-80pct")
    run = run_of(result, "m_q")
    assert (run["intended_route"], run["route"]) == ("web", "chat")
    assert run["tool_calls"] == [] and "web_disabled" in run["degraded"]


def test_tool_budgets_used_up_near_timeout_still_finish_within_deadline():
    _, result = replay("synthetic-tool-budget-near-timeout")
    run = run_of(result, "m_q")
    calls = [(c["name"], c["status"]) for c in run["tool_calls"]]
    assert calls[:4] == [("web_search", "ok")] * 3 + [("web_search", "budget_exhausted")]
    assert calls[4] == ("web_fetch", "timeout") and calls[-1] == ("web_fetch", "budget_exhausted")
    assert {s for n, s in calls[5:-1]} == {"no_time_left"}
    # Each call stops at min(25 s, remaining - 30 s reserve); the run ends inside 120 s.
    assert run["tool_calls"][4]["t_end"] - run["tool_calls"][4]["t_start"] == 18
    assert run["state"] == "sent" and run["ended_at"] - run["arrived_at"] <= 120 - 29
    assert [o["state"] for o in outbound_to(result, "m_q")] == ["acked"]


def test_cost_100pct_serves_only_owner_without_notice():
    _, result = replay("synthetic-cost-100pct")
    assert [e["status"] for e in result["events"]] == ["ignored_cost_cap", "trigger"]
    assert outbound_to(result, "m_u") == []
    assert [o["state"] for o in outbound_to(result, "m_o")] == ["acked"]


# -- AC-22 / AC-23 / AC-24 ------------------------------------------------------------


def test_group_info_tools_are_bound_to_the_request_group():
    _, result = replay("synthetic-group-info-roles")
    calls = run_of(result, "m_q")["tool_calls"]
    assert [(c["name"], c["args"], c["status"]) for c in calls] == [
        ("group_info", {"group_id": "g_1"}, "ok"),
        ("group_member_list", {"group_id": "g_1"}, "ok"),
    ]
    case = copy.deepcopy(load("synthetic-group-info-refusals"))
    case["mocks"]["tools"][0]["response"]["group_id"] = "g_other"
    calls = run_of(run_case(case), "m_q")["tool_calls"]
    assert calls[0]["status"] == "rejected_cross_group"


def test_sensitive_draft_replaced_by_neutral_answer():
    _, result = replay("synthetic-sensitive-review")
    run = run_of(result, "m_q")
    assert run["moderation"]["blocked"]
    assert outbound_to(result, "m_q")[0]["text"].startswith("这类时政解读")


def test_bilibili_card_expands_short_link_before_video_info():
    _, result = replay("synthetic-card-bilibili-quoted")
    run = run_of(result, "m_q")
    assert run["cards"][0]["kind"] == "miniapp" and core_ids(run) == ["m_card"]
    calls = [(c["name"], c["args"].get("url", "")) for c in run["tool_calls"]]
    assert calls[0][0] == "short_link_expand" and "b23." in calls[0][1]
    assert calls[1] == (
        "bilibili_video_info",
        "https://www.bilibili.example.invalid/video/BV1xxxxxxxxx",
    )


def test_login_wall_answers_from_card_without_headless():
    _, result = replay("synthetic-card-login-wall")
    run = run_of(result, "m_q")
    assert [(c["name"], c["status"]) for c in run["tool_calls"]] == [
        ("web_fetch", "ok"),
        ("headless_fetch", "unavailable_p1"),
    ]
    assert run["cards"][0]["title"] and run["state"] == "sent"


@pytest.mark.xfail(strict=True, reason="LINK-02 headless browser fallback is P1")
def test_p1_headless_fallback_for_login_wall():
    _, result = replay("synthetic-card-login-wall")
    assert run_of(result, "m_q")["tool_calls"][-1]["status"] == "ok"


# -- scorer catches planted errors ------------------------------------------------------


def test_isolation_counters_detect_planted_errors():
    case, clean = replay("synthetic-teach-duplicate")
    result = copy.deepcopy(clean)
    result["outbound"][0]["group_id"] = "g_other"
    result["outbound"][1]["text"] += " u_2"
    t2 = run_of(result, "m_t2")
    t2["context_ids"].append("m_missing_elsewhere")
    t2["images"].append({"message_id": "m_1", "user_id": "u_2"})
    result["knowledge_log"].append({**result["knowledge_log"][0], "actor": "u_2"})
    iso = score_case(case, result)["isolation"]
    assert iso["wrong_group_send"] == 1 and iso["account_id_leak"] == 1
    assert iso["out_of_scope_context"] == 1 and iso["wrong_user_attachment"] == 1
    assert iso["duplicate_teaching_commit"] == 2 and iso["injection_write"] == 1
