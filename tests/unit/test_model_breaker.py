"""D-20 model breaker: open after consecutive infrastructure failures, skip while open,
half-open probe; route decision and answer go straight to DeepSeek while Gemini is open."""

import asyncio
import json
import threading

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile
import pytest
from test_live_replay import GEMINI, USAGE, decision_model, live_config, make_case, run_live

from group_agent.models import (
    CallRecord,
    Ledger,
    LiveModel,
    ModelBreaker,
    ModelCallError,
    ModelSkipped,
    infrastructure_failure,
)

WEB = {"route": "web", "should_clarify": False, "reason": "要查", "search_terms": ["x"]}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def breaker(**kw):
    clock = Clock()
    logs: list[str] = []
    kw.setdefault("min_timeout_s", 0.0)
    return ModelBreaker(2, 300.0, clock=clock, log=logs.append, **kw), clock, logs


def record(status, error=None, timeout_s=15.0):
    return CallRecord("t", "gemini", "text", status=status, error=error, timeout_s=timeout_s)


# -- breaker state machine ---------------------------------------------------------------


def test_consecutive_failures_open_and_cooldown_skips():
    b, clock, logs = breaker()
    assert b.acquire("gemini") == "closed"
    b.record("gemini", "closed", "infra")
    assert b.state("gemini") == "closed"  # one failure is not enough
    b.record("gemini", "closed", "infra")
    assert b.state("gemini") == "open" and logs[-1].startswith("model breaker gemini: open")
    assert b.acquire("gemini") is None and not b.available("gemini")
    clock.t += 299
    assert b.acquire("gemini") is None
    summary = b.summary()["gemini"]
    assert summary["opened"] == 1 and summary["skipped"] == 2
    assert summary["open_seconds"] == pytest.approx(299)
    assert b.state("deepseek") == "closed"  # keys are independent


def test_success_resets_the_count():
    b, _, _ = breaker()
    b.record("gemini", "closed", "infra")
    b.record("gemini", "closed", "ok")
    b.record("gemini", "closed", "infra")
    assert b.state("gemini") == "closed"


def test_half_open_probe_success_closes():
    b, clock, logs = breaker()
    for _ in range(2):
        b.record("gemini", "closed", "infra")
    clock.t += 300
    assert b.state("gemini") == "half_open" and b.available("gemini")
    assert b.acquire("gemini") == "probe"
    assert b.acquire("gemini") is None  # one probe at a time
    b.record("gemini", "probe", "ok")
    assert b.state("gemini") == "closed" and b.acquire("gemini") == "closed"
    assert [line.split(": ")[1].split(" ")[0] for line in logs] == ["open", "half_open", "closed"]
    assert b.summary()["gemini"]["open_seconds"] == pytest.approx(300)


def test_half_open_probe_failure_reopens_and_cancel_frees_the_slot():
    b, clock, _ = breaker()
    for _ in range(2):
        b.record("gemini", "closed", "infra")
    clock.t += 300
    assert b.acquire("gemini") == "probe"
    b.record("gemini", "probe", "infra")
    assert b.state("gemini") == "open" and b.summary()["gemini"]["opened"] == 2
    clock.t += 300
    assert b.acquire("gemini") == "probe"
    b.record("gemini", "probe", "cancelled")  # run deadline: not a verdict
    assert b.state("gemini") == "half_open" and b.acquire("gemini") == "probe"
    b.record("gemini", "probe", "other")  # the model answered (bad output): reachable
    assert b.state("gemini") == "closed"


def test_only_infrastructure_failures_count():
    assert infrastructure_failure(record("timeout", "TimeoutError"))
    assert infrastructure_failure(record("error", "ModelHTTPError:503"))
    assert infrastructure_failure(record("error", "ModelHTTPError:429"))
    assert infrastructure_failure(record("error", "ConnectError"))
    assert infrastructure_failure(record("error", "ModelAPIError"))
    assert not infrastructure_failure(record("error", "ModelHTTPError:400"))
    assert not infrastructure_failure(record("error", "UnexpectedModelBehavior"))
    assert not infrastructure_failure(record("error", "ContentFilterError"))
    assert not infrastructure_failure(record("budget"))
    # A timeout of a short caller budget says nothing about the model.
    assert not infrastructure_failure(record("timeout", "TimeoutError", 3.0), min_timeout_s=8)
    b, _, _ = breaker()
    for _ in range(5):
        b.record("gemini", "closed", "other")
    assert b.state("gemini") == "closed"


# -- LiveModel integration -----------------------------------------------------------------


def failing(exc_factory, calls):
    def fn(messages, info):
        calls.append(1)
        raise exc_factory()

    return FunctionModel(fn)


def test_live_model_counts_failures_skips_without_a_request_and_probes():
    b, clock, _ = breaker()
    calls: list = []
    ledger = Ledger(1.0)
    model = LiveModel(
        GEMINI,
        lambda: failing(lambda: ModelHTTPError(503, "fake-gemini"), calls),
        ledger,
        breaker=b,
    )

    async def call():
        return await model.generate("answer", "指令", ["材料"], timeout=5)

    for _ in range(2):
        with pytest.raises(ModelCallError) as error:
            asyncio.run(call())
        assert error.value.record.breaker == "closed"
    with pytest.raises(ModelSkipped) as skipped:
        asyncio.run(call())
    assert skipped.value.record.status == "skipped" and len(calls) == 2
    assert len(ledger.records) == 2  # the skip made no request and is not in the ledger
    clock.t += 300
    model._factory = lambda: FunctionModel(
        lambda m, i: ModelResponse([TextPart("好")], usage=USAGE)
    )
    model._model = None
    text, rec = asyncio.run(call())
    assert text == "好" and rec.breaker == "probe" and b.state("gemini") == "closed"


def test_validation_failures_do_not_open_the_breaker():
    b, _, _ = breaker()
    calls: list = []

    def bad(messages, info):
        calls.append(1)
        return ModelResponse([TextPart("not json")], usage=USAGE)

    profile = ModelProfile(supports_json_schema_output=True)
    model = LiveModel(GEMINI, lambda: FunctionModel(bad, profile=profile), Ledger(1.0), breaker=b)
    from group_agent.live_agent import RouteDecision

    for _ in range(3):
        with pytest.raises(ModelCallError):
            asyncio.run(
                model.structured("decision", "指令", ["材料"], RouteDecision, "native", timeout=5)
            )
    assert b.state("gemini") == "closed" and len(calls) == 6  # 1 retry each


def test_breaker_is_shared_across_threads_and_tasks():
    b = ModelBreaker(2, 300.0, min_timeout_s=0.0)
    calls: list = []

    def worker():
        model = LiveModel(
            GEMINI, lambda: failing(lambda: ConnectionError(), calls), Ledger(1.0), breaker=b
        )
        with pytest.raises(ModelCallError):
            asyncio.run(model.generate("answer", "i", ["m"], timeout=5))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert b.state("gemini") == "open" and len(calls) == 2

    async def many():
        model = LiveModel(GEMINI, lambda: failing(RuntimeError, calls), Ledger(1.0), breaker=b)
        results = await asyncio.gather(
            *(model.generate("answer", "i", ["m"], timeout=5) for _ in range(4)),
            return_exceptions=True,
        )
        return results

    results = asyncio.run(many())
    assert all(isinstance(r, ModelSkipped) for r in results) and len(calls) == 2
    assert b.summary()["gemini"]["skipped"] == 4


# -- live agent: decision and answer -------------------------------------------------------


def gemini_fake(calls, *, decision=None, answer="Gemini 回答", fail=None, delay=0.0):
    async def fn(messages, info):
        calls.append(info.model_request_parameters.output_mode)
        if delay:
            await asyncio.sleep(delay)
        if fail is not None:
            raise fail()
        if info.model_request_parameters.output_mode == "native":
            body = json.dumps(decision or WEB, ensure_ascii=False)
            return ModelResponse(parts=[TextPart(body)], usage=USAGE)
        return ModelResponse(parts=[TextPart(answer)], usage=USAGE)

    return FunctionModel(fn, profile=ModelProfile(supports_json_schema_output=True))


def deepseek_fake(seen, decision=WEB, fail_first_answer=False):
    def fn(messages, info):
        content = messages[-1].parts[0].content
        has_image = isinstance(content, list) and any(
            getattr(p, "media_type", None) for p in content
        )
        if info.output_tools:
            seen.append(("decision", has_image))
            args = decision
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)], usage=USAGE)
        seen.append(("answer", has_image))
        if fail_first_answer and len([s for s in seen if s[0] == "answer"]) == 1:
            raise ModelHTTPError(502, "fake-deepseek")
        return ModelResponse(parts=[TextPart("DeepSeek 回答")], usage=USAGE)

    return FunctionModel(fn)


def opened_breaker():
    b = ModelBreaker(2, 300.0, min_timeout_s=0.0)
    for _ in range(2):
        b.record("gemini", b.acquire("gemini"), "infra")
    return b


def test_open_breaker_decision_and_answer_go_straight_to_deepseek(tmp_path):
    case = make_case(tmp_path)
    gemini_calls, seen = [], []
    config = live_config(
        tmp_path,
        lambda: gemini_fake(gemini_calls),
        lambda: deepseek_fake(seen),
        decider="gemini",
        model_breaker=opened_breaker(),
    )
    _, anchor, rec = run_live(case, config)
    assert gemini_calls == []  # no Gemini request at all
    assert anchor["route"] == "web"  # the web route survives (DeepSeek decided)
    assert rec["decision_attempts"][:2] == ["gemini:native:breaker_open", "deepseek:tool:ok"]
    assert rec["breaker"] == {"decision": "open", "answer": "open"}
    assert rec["breaker_skips"] == 2  # decision skip + answer skip
    # DeepSeek decides on text, answers with the image (it reads images).
    assert seen == [("decision", False), ("answer", True)]
    assert rec["answer"] == "DeepSeek 回答" and rec["answer_source"] == "deepseek"
    assert rec["answer_fallback_used"] is True
    assert [c["model"] for c in rec["calls"] if c["stage"] == "answer"] == ["deepseek"]
    # The fallbacks are visible in the run's degraded labels (persona-v2 summary counts).
    assert anchor["degraded"] == [
        "decision_gemini_skipped",
        "decision_fallback:deepseek",
        "answer_gemini_skipped",
        "answer_fallback:deepseek",
    ]


def test_open_breaker_image_answer_failure_falls_back_to_text_only(tmp_path):
    case = make_case(tmp_path)
    seen: list = []
    config = live_config(
        tmp_path,
        lambda: gemini_fake([]),
        lambda: deepseek_fake(seen, fail_first_answer=True),
        decider="gemini",
        model_breaker=opened_breaker(),
    )
    _, anchor, rec = run_live(case, config)
    assert anchor["state"] == "sent"
    assert [s for s in seen if s[0] == "answer"] == [("answer", True), ("answer", False)]
    assert rec["answer_source"] == "deepseek:no_images"
    assert "answer_fallback:deepseek:no_images" in anchor["degraded"]


def test_gemini_decision_timeout_falls_back_to_deepseek_at_once(tmp_path):
    case = make_case(tmp_path)
    gemini_calls, seen = [], []
    b = ModelBreaker(2, 300.0, min_timeout_s=0.0)
    config = live_config(
        tmp_path,
        lambda: gemini_fake(gemini_calls, delay=3.0),
        lambda: deepseek_fake(seen),
        decider="gemini",
        decision_timeout_s=0.2,
        model_breaker=b,
        answer_high_cap=1.2,
        answer_retry_cap=6.0,
    )
    _, anchor, rec = run_live(case, config)
    assert anchor["route"] == "web"  # not the rule fallback (chat)
    assert rec["decision_attempts"][:2] == ["gemini:native:timeout", "deepseek:tool:ok"]
    assert rec["breaker"]["decision"] == "closed"
    # The answer chain's Gemini timeouts feed the same breaker: HIGH fails (2nd failure),
    # so the LOW retry is skipped and DeepSeek answers.
    answer = [(c["model"], c["status"]) for c in rec["calls"] if c["stage"] == "answer"]
    assert answer == [("gemini", "timeout"), ("gemini", "skipped"), ("deepseek", "ok")]
    assert b.state("gemini") == "open" and b.summary()["gemini"]["opened"] == 1
    assert rec["answer_source"] == "deepseek"


def test_rule_fallback_only_when_deepseek_fails_too(tmp_path):
    case = make_case(tmp_path)
    ds_calls: list = []
    config = live_config(
        tmp_path,
        lambda: gemini_fake([], fail=lambda: ModelHTTPError(503, "g")),
        lambda: failing(lambda: ModelHTTPError(503, "d"), ds_calls),
        decider="gemini",
        decision_only=True,
    )
    _, anchor, rec = run_live(case, config)
    assert rec["decision_attempts"] == [
        "gemini:native:error",
        "deepseek:tool:error",
        "deepseek:prompted:error",
    ]
    assert rec["decision"] == {"route": "chat", "fallback": "rule"} and len(ds_calls) == 2
    assert "breaker" not in rec  # no breaker configured (replay default)
    # Without the DeepSeek fallback the rule decides right after Gemini.
    (tmp_path / "b").mkdir()
    config = live_config(
        tmp_path / "b",
        lambda: gemini_fake([], fail=lambda: ModelHTTPError(503, "g")),
        lambda: decision_model(),
        decider="gemini",
        decision_fallback=False,
        decision_only=True,
    )
    _, _, rec = run_live(make_case(tmp_path / "b"), config)
    assert rec["decision_attempts"] == ["gemini:native:error"]
    assert rec["decision"] == {"route": "chat", "fallback": "rule"}
