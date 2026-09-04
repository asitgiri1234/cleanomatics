"""The planner: does it ask for the right evidence, and only evidence it can trust?

The Groq call is faked, so these tests describe how the planner handles what a
model returns — including the ways a model gets it wrong.
"""

import pytest

from app.agent.planner import find_order_ids, plan_question
from app.llm.errors import LLMResponseError
from app.schemas.agent import Plan
from tests.fakes import FakeLLMClient, plan_reply


# A. Knowledge-base only

def test_kb_only_question():
    client = FakeLLMClient(
        plan_reply(needs_kb=True, kb_query="refund policy", intent="refund policy")
    )

    plan = plan_question("What is ShipFlow's refund policy?", client)

    assert plan.needs_kb is True
    assert plan.needs_order_tool is False
    assert plan.order_id is None
    assert plan.kb_query


# B. Order tool only

def test_order_only_question():
    client = FakeLLMClient(
        plan_reply(needs_order_tool=True, order_id="ORD-1001", intent="order status")
    )

    plan = plan_question("What is the status of ORD-1001?", client)

    assert plan.needs_kb is False
    assert plan.needs_order_tool is True
    assert plan.order_id == "ORD-1001"
    assert plan.kb_query is None


# C. Both

def test_question_needing_both_sources():
    client = FakeLLMClient(
        plan_reply(
            needs_kb=True,
            needs_order_tool=True,
            order_id="ORD-1001",
            kb_query="refund policy",
            intent="refund and order status",
        )
    )

    plan = plan_question(
        "What is ShipFlow's refund policy and what is the status of ORD-1001?", client
    )

    assert plan.needs_kb is True
    assert plan.needs_order_tool is True
    assert plan.order_id == "ORD-1001"
    assert plan.kb_query


# D. Neither

def test_casual_conversation_needs_nothing():
    client = FakeLLMClient(plan_reply(intent="greeting"))

    plan = plan_question("Hello there!", client)

    assert plan.needs_kb is False
    assert plan.needs_order_tool is False
    assert plan.order_id is None


def test_empty_question_skips_the_model_entirely():
    client = FakeLLMClient([])

    plan = plan_question("   ", client)

    assert plan.needs_kb is False
    assert plan.needs_order_tool is False
    assert client.calls == []


# The planner must not invent an order ID

def test_invented_order_id_is_discarded():
    """The model returns an ID the customer never wrote; it must be dropped."""
    client = FakeLLMClient(
        plan_reply(needs_order_tool=True, order_id="ORD-1001", intent="order status")
    )

    plan = plan_question("Where is my order?", client)

    assert plan.order_id is None
    assert plan.needs_order_tool is False


def test_hallucinated_id_is_dropped_even_when_a_different_one_was_given():
    client = FakeLLMClient(
        plan_reply(needs_order_tool=True, order_id="ORD-9999", intent="order status")
    )

    plan = plan_question("What about ORD-1002?", client)

    assert plan.order_id == "ORD-1002"


def test_customer_formatting_of_a_real_reference_is_accepted():
    client = FakeLLMClient(
        plan_reply(needs_order_tool=True, order_id="ORD-1001", intent="order status")
    )

    plan = plan_question("any news on ord 1001", client)

    assert plan.order_id == "ORD-1001"
    assert plan.needs_order_tool is True


def test_reference_the_planner_missed_is_recovered():
    """The customer gave an ID and the model ignored it — use it anyway."""
    client = FakeLLMClient(plan_reply(needs_kb=True, kb_query="delivery"))

    plan = plan_question("Is ORD-1004 going to arrive on time?", client)

    assert plan.order_id == "ORD-1004"
    assert plan.needs_order_tool is True


def test_find_order_ids_matches_realistic_references():
    assert find_order_ids("status of ORD-1001 please") == ["ORD-1001"]
    assert find_order_ids("ord-1001 and ORD-1002") == ["ORD-1001", "ORD-1002"]
    assert find_order_ids("no reference here") == []


# Invalid planner decisions are repaired rather than executed

def test_order_lookup_without_a_reference_is_switched_off():
    client = FakeLLMClient(plan_reply(needs_order_tool=True, order_id=None))

    plan = plan_question("Where is my parcel?", client)

    assert plan.needs_order_tool is False


def test_kb_search_without_a_query_falls_back_to_the_question():
    client = FakeLLMClient(plan_reply(needs_kb=True, kb_query=None))

    plan = plan_question("How do refunds work?", client)

    assert plan.kb_query == "How do refunds work?"


def test_kb_query_is_cleared_when_the_kb_is_not_needed():
    client = FakeLLMClient(plan_reply(needs_kb=False, kb_query="refunds"))

    plan = plan_question("Hello", client)

    assert plan.kb_query is None


@pytest.mark.parametrize("placeholder", ["null", "none", "N/A", "", "  "])
def test_string_placeholders_are_treated_as_absent(placeholder):
    plan = Plan.model_validate(
        {
            "needs_kb": False,
            "needs_order_tool": False,
            "order_id": placeholder,
            "kb_query": placeholder,
            "intent": "test",
        }
    )

    assert plan.order_id is None
    assert plan.kb_query is None


# Malformed output

def test_non_json_reply_is_reported():
    client = FakeLLMClient("Sure! I think you need the refund policy.")

    with pytest.raises(LLMResponseError):
        plan_question("What is the refund policy?", client)


def test_json_missing_required_fields_is_reported():
    client = FakeLLMClient('{"intent": "refund policy"}')

    with pytest.raises(LLMResponseError) as caught:
        plan_question("What is the refund policy?", client)

    assert caught.value.raw is not None


def test_json_wrapped_in_a_code_fence_is_recovered():
    client = FakeLLMClient(f"```json\n{plan_reply(needs_kb=True, kb_query='refunds')}\n```")

    plan = plan_question("What is the refund policy?", client)

    assert plan.needs_kb is True


def test_json_with_surrounding_prose_is_recovered():
    client = FakeLLMClient(
        f"Here is the plan: {plan_reply(needs_kb=True, kb_query='refunds')} Hope that helps."
    )

    plan = plan_question("What is the refund policy?", client)

    assert plan.needs_kb is True


def test_planner_asks_for_json_and_is_deterministic():
    client = FakeLLMClient(plan_reply(needs_kb=True, kb_query="refunds"))

    plan_question("What is the refund policy?", client)

    assert client.calls[0]["json_mode"] is True
    assert client.calls[0]["temperature"] == 0.0


def test_planner_never_produces_customer_facing_text():
    client = FakeLLMClient(
        plan_reply(needs_kb=True, kb_query="refunds", intent="refund policy")
    )

    plan = plan_question("What is the refund policy?", client)

    assert not hasattr(plan, "answer")
    assert set(plan.model_dump()) == {
        "needs_kb",
        "needs_order_tool",
        "order_id",
        "kb_query",
        "intent",
    }


# The kb_query must be able to stand on its own


def test_prompt_requires_self_contained_search_phrases():
    """A vague kb_query is the one planner mistake retrieval cannot recover from.

    "tell me more about this company" reached the retriever as "company
    information", which scores 0.33 and falls under the threshold, so a
    question the FAQ answers came back refused. Naming the subject fixes it:
    "what does ShipFlow do" scores 0.83 on the right section.
    """
    from app.llm.prompts import PLANNER_SYSTEM_PROMPT

    lowered = PLANNER_SYSTEM_PROMPT.lower()
    assert "stand on its own" in lowered
    assert "this company" in lowered
    assert "resolve" in lowered


def test_vague_kb_query_is_still_passed_through_unchanged():
    """The planner is asked to write good queries; nothing silently rewrites them.

    Retrieval quality is the planner's responsibility. If a poor query does get
    through, it reaches the retriever as written and the threshold catches it —
    the failure is a refusal, never a wrong answer.
    """
    client = FakeLLMClient(plan_reply(needs_kb=True, kb_query="company information"))

    plan = plan_question("tell me more about this company", client)

    assert plan.kb_query == "company information"


# Order references as customers actually write them


@pytest.mark.parametrize(
    "written,expected",
    [
        ("status of ORD-1001", "ORD-1001"),
        ("status of ORD 1003", "ORD-1003"),
        ("status of ord 1003", "ORD-1003"),
        ("status of ord1002", "ORD-1002"),
        ("status of ORD_1004", "ORD-1004"),
        ("what about FAIL 001", "FAIL-001"),
    ],
)
def test_loose_order_references_are_recognised(written, expected):
    assert find_order_ids(written) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "I ordered in 2024",
        "we have up to 500 orders",
        "call me on 555 0100",
        "my plan is 29 dollars",
        "no reference at all",
    ],
)
def test_ordinary_prose_is_not_mistaken_for_an_order(text):
    """A looser pattern must not turn "in 2024" into order IN-2024."""
    assert find_order_ids(text) == []


def test_a_reference_written_twice_is_found_once():
    assert find_order_ids("ORD-1001 — I mean ord 1001") == ["ORD-1001"]


def test_spaced_reference_is_recovered_when_the_planner_misses_it():
    client = FakeLLMClient(plan_reply(needs_kb=True, kb_query="order status"))

    plan = plan_question("what is the status of ORD 1003", client)

    assert plan.order_id == "ORD-1003"
    assert plan.needs_order_tool is True
