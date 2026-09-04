"""The orchestrator: does it run the plan, and does it survive things failing?

The knowledge base and the order tool are real here. Only Groq is faked, so
these tests exercise the actual wiring between the components.
"""

import pytest

from app.agent.orchestrator import (
    ASK_FOR_ORDER_ID,
    _confidence,
    _fallback_plan,
    answer_question,
)
from app.agent.generator import UNVERIFIED_FALLBACK
from app.llm.errors import LLMAPIError, LLMConfigurationError
from app.schemas.agent import Evidence, Plan
from app.schemas.kb import RetrievalResult
from app.schemas.orders import LookupOutcome, OrderLookupResult
from tests.fakes import FakeLLMClient, plan_reply

ANSWER_TEXT = "Here is what I found for you."


def client_for(plan_json: str, answer: str = ANSWER_TEXT) -> FakeLLMClient:
    """A fake that replies to the planner call, then the generator call."""
    return FakeLLMClient([plan_json, answer])


# The plan is executed


def test_kb_plan_searches_the_knowledge_base():
    client = client_for(plan_reply(needs_kb=True, kb_query="refund policy"))

    result = answer_question("What is ShipFlow's refund policy?", client)

    assert result.trace.kb_used is True
    assert result.trace.kb_calls == 1
    assert result.trace.tool_calls == 0
    assert result.trace.llm_calls == 2


def test_order_plan_calls_the_tool_and_skips_retrieval():
    client = client_for(plan_reply(needs_order_tool=True, order_id="ORD-1001"))

    result = answer_question("What is the status of ORD-1001?", client)

    assert result.trace.order_tool_used is True
    assert result.trace.tool_calls == 1
    assert result.trace.kb_used is False
    assert result.trace.kb_calls == 0


def test_combined_plan_uses_both_sources():
    client = client_for(
        plan_reply(
            needs_kb=True, needs_order_tool=True, order_id="ORD-1001", kb_query="refund policy"
        )
    )

    result = answer_question("Refund policy, and where is ORD-1001?", client)

    assert result.trace.kb_calls == 1
    assert result.trace.tool_calls == 1
    assert "refund-policy.md" in result.trace.sources
    assert "order:ORD-1001" in result.trace.sources


def test_plan_needing_nothing_gathers_nothing():
    client = client_for(plan_reply(intent="greeting"))

    result = answer_question("Hello there!", client)

    assert result.trace.kb_used is False
    assert result.trace.order_tool_used is False
    assert result.trace.tool_calls == 0
    assert result.trace.confidence == 0.0


# Confidence


def test_confidence_of_a_strong_kb_match_is_the_similarity_score():
    client = client_for(plan_reply(needs_kb=True, kb_query="how long do refunds take"))

    result = answer_question("How long do refunds take?", client)

    assert 0.3 < result.trace.confidence <= 1.0
    assert result.trace.confidence == pytest.approx(round(result.trace.kb_confidence, 2))


def test_confidence_of_a_found_order_is_full():
    client = client_for(plan_reply(needs_order_tool=True, order_id="ORD-1001"))

    result = answer_question("Where is ORD-1001?", client)

    assert result.trace.confidence == 1.0


def test_confidence_is_zero_when_the_order_does_not_exist():
    client = client_for(plan_reply(needs_order_tool=True, order_id="ORD-9999"))

    result = answer_question("Where is ORD-9999?", client)

    assert result.trace.confidence == 0.0
    assert result.trace.order_outcome == "not_found"


def test_confidence_averages_the_sources_a_plan_asked_for():
    plan = Plan(needs_kb=True, needs_order_tool=True, order_id="ORD-1001", kb_query="x")
    evidence = Evidence(
        kb=RetrievalResult(query="x", matches=[], threshold=0.3, has_evidence=False),
        order=OrderLookupResult(outcome=LookupOutcome.FOUND, order_id="ORD-1001", message="ok"),
    )

    # One source delivered, one did not: half credit.
    assert _confidence(plan, evidence) == 0.5


def test_confidence_never_leaves_the_zero_to_one_range():
    client = client_for(plan_reply(needs_kb=True, kb_query="refund policy"))

    result = answer_question("What is the refund policy?", client)

    assert 0.0 <= result.trace.confidence <= 1.0


# Failures degrade rather than crash


def test_malformed_planner_output_falls_back_to_a_usable_plan():
    client = FakeLLMClient(["not json at all", ANSWER_TEXT])

    result = answer_question("What is the refund policy?", client)

    assert "planner_malformed" in result.trace.degraded
    assert result.trace.kb_used is True
    assert result.answer.answer == ANSWER_TEXT


def test_fallback_plan_never_invents_an_order_id():
    assert _fallback_plan("Where is my order?").order_id is None
    assert _fallback_plan("Where is ORD-1002?").order_id == "ORD-1002"


def test_order_service_failure_is_recorded_and_survived():
    client = client_for(plan_reply(needs_order_tool=True, order_id="FAIL-001"))

    result = answer_question("Where is FAIL-001?", client)

    assert "order_service_error" in result.trace.degraded
    assert result.trace.order_outcome == "service_error"
    assert result.trace.confidence == 0.0
    assert result.answer.answer == ANSWER_TEXT


def test_knowledge_base_failure_does_not_stop_the_order_half(monkeypatch):
    def explode():
        raise RuntimeError("index unavailable")

    monkeypatch.setattr("app.agent.orchestrator.get_retriever", explode)
    client = client_for(
        plan_reply(needs_kb=True, needs_order_tool=True, order_id="ORD-1001", kb_query="refunds")
    )

    result = answer_question("Refund policy and where is ORD-1001?", client)

    assert "kb_unavailable" in result.trace.degraded
    assert result.trace.tool_calls == 1
    assert result.trace.confidence == 0.5


def test_empty_question_never_reaches_the_model():
    client = FakeLLMClient([])

    result = answer_question("   ", client)

    assert client.calls == []
    assert result.answer.answer == UNVERIFIED_FALLBACK
    assert "empty_question" in result.trace.degraded


def test_llm_api_failure_is_raised_for_the_caller_to_map():
    client = FakeLLMClient(error=LLMAPIError("Groq returned 503", status_code=503))

    with pytest.raises(LLMAPIError):
        answer_question("What is the refund policy?", client)


def test_missing_api_key_surfaces_as_a_configuration_error(monkeypatch):
    monkeypatch.setattr(
        "app.agent.orchestrator.get_llm_client",
        lambda: (_ for _ in ()).throw(LLMConfigurationError("GROQ_API_KEY is not set")),
    )

    with pytest.raises(LLMConfigurationError):
        answer_question("What is the refund policy?")


# Ambiguous requests


def test_order_question_without_a_reference_asks_for_one():
    client = FakeLLMClient([plan_reply(needs_order_tool=True, order_id=None, intent="order status")])

    result = answer_question("Tell me about my order.", client)

    assert result.answer.answer == ASK_FOR_ORDER_ID
    assert result.trace.tool_calls == 0
    assert "order_id_missing" in result.trace.degraded


def test_ambiguous_order_question_never_calls_the_tool():
    client = FakeLLMClient(
        [plan_reply(needs_order_tool=True, order_id="ORD-1001", intent="order status")]
    )

    result = answer_question("Tell me about my order.", client)

    assert result.trace.order_tool_used is False
    assert result.trace.tool_calls == 0
    # The reply asks for a reference; it reports nothing about any order.
    assert result.answer.answer == ASK_FOR_ORDER_ID
    assert result.answer.sources == []


# The trace carries nothing sensitive


def test_trace_contains_no_prompts_or_secrets():
    client = client_for(plan_reply(needs_kb=True, kb_query="refund policy"))

    result = answer_question("What is the refund policy?", client)
    dumped = result.trace.model_dump_json().lower()

    assert "gsk_" not in dumped
    assert "system_prompt" not in dumped
    assert "you are" not in dumped
