"""The six required scenarios, end to end through the API.

Deterministic: the real retriever, the real order tool, and the real
orchestrator run, with only Groq faked. Each scenario checks the four things
that matter — the answer, the sources, the confidence, and which components
were actually used.

The planner replies are what the live model returns for these questions; the
live versions of the same six run in `test_groq_integration.py`.
"""

import pytest
from fastapi.testclient import TestClient

from app.agent.orchestrator import ASK_FOR_ORDER_ID
from app.main import app
from tests.fakes import FakeLLMClient, plan_reply


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_llm(monkeypatch):
    def install(replies):
        fake = FakeLLMClient(replies)
        monkeypatch.setattr("app.agent.orchestrator.get_llm_client", lambda: fake)
        return fake

    return install


def ask(client, message: str) -> dict:
    response = client.post("/chat", json={"message": message})
    assert response.status_code == 200, response.text
    return response.json()


# Scenario 1 — knowledge base only


def test_scenario_1_kb_only(client, fake_llm):
    fake = fake_llm(
        [
            plan_reply(needs_kb=True, kb_query="refund policy", intent="refund policy"),
            "Refunds are approved within 3-5 business days once the return passes inspection.",
        ]
    )

    body = ask(client, "What is ShipFlow's refund policy?")

    assert body["trace"]["kb_used"] is True
    assert body["trace"]["kb_calls"] == 1
    assert body["tool_calls"] == 0
    assert body["trace"]["order_tool_used"] is False
    assert "refund-policy.md" in body["sources"]
    assert body["confidence"] > 0.3
    # The generator was given real policy text and nothing else.
    assert "refund" in fake.calls[1]["user_prompt"].lower()


# Scenario 2 — order tool only


def test_scenario_2_order_tool_only(client, fake_llm):
    fake = fake_llm(
        [
            plan_reply(needs_order_tool=True, order_id="ORD-1001", intent="order status"),
            "Your order shipped on 3 September with FastPost and is due on 8 September.",
        ]
    )

    body = ask(client, "What is the status of ORD-1001?")

    assert body["tool_calls"] == 1
    assert body["trace"]["order_tool_used"] is True
    assert body["trace"]["order_outcome"] == "found"
    assert body["trace"]["kb_used"] is False
    assert body["trace"]["kb_calls"] == 0
    assert body["sources"] == ["order:ORD-1001"]
    assert body["confidence"] == 1.0
    # The real order record reached the generator.
    prompt = fake.calls[1]["user_prompt"]
    assert "FastPost" in prompt and "2026-09-08" in prompt


# Scenario 3 — both sources


def test_scenario_3_kb_and_order_tool(client, fake_llm):
    fake = fake_llm(
        [
            plan_reply(
                needs_kb=True,
                needs_order_tool=True,
                order_id="ORD-1001",
                kb_query="refund policy",
                intent="refund policy and order status",
            ),
            "Your order is on its way with FastPost, and refunds take 3-5 business days.",
        ]
    )

    body = ask(client, "What is your refund policy and what is the status of ORD-1001?")

    assert body["trace"]["kb_calls"] == 1
    assert body["tool_calls"] == 1
    assert "refund-policy.md" in body["sources"]
    assert "order:ORD-1001" in body["sources"]
    assert body["confidence"] > 0.5
    # Both kinds of evidence were in the one prompt.
    prompt = fake.calls[1]["user_prompt"]
    assert "FastPost" in prompt
    assert "refund" in prompt.lower()


# Scenario 4 — unsupported question


def test_scenario_4_question_the_documents_do_not_cover(client, fake_llm):
    """A question with no answer anywhere in the knowledge base."""
    fake = fake_llm(
        [
            plan_reply(needs_kb=True, kb_query="warehouse staff shift patterns", intent="off topic"),
            "I couldn't verify that, so I'd rather not guess. Please contact ShipFlow support.",
        ]
    )

    body = ask(client, "What shift patterns do your warehouse staff work?")

    assert body["sources"] == [] or body["confidence"] < 0.45
    # The generator was told the documents do not cover this.
    assert "Nothing in ShipFlow" in fake.calls[1]["user_prompt"]
    assert body["trace"]["grounded"] is False


def test_scenario_4_low_confidence_is_reported_honestly(client, fake_llm):
    fake_llm(
        [
            plan_reply(needs_kb=True, kb_query="drone delivery policy", intent="shipping"),
            "I couldn't confirm anything about that.",
        ]
    )

    body = ask(client, "Do you deliver by drone?")

    # Whatever came back, the score is reported rather than rounded up.
    assert body["confidence"] == pytest.approx(round(body["trace"]["kb_confidence"], 2))
    assert body["confidence"] < 0.6


# Scenario 5 — tool failure


def test_scenario_5_order_tool_failure(client, fake_llm):
    fake = fake_llm(
        [
            plan_reply(needs_order_tool=True, order_id="FAIL-001", intent="order status"),
            "I can't check that order right now. Please try again shortly.",
        ]
    )

    body = ask(client, "What is the status of FAIL-001?")

    assert body["trace"]["order_outcome"] == "service_error"
    assert "order_service_error" in body["trace"]["degraded"]
    assert body["confidence"] == 0.0
    assert body["sources"] == []
    # No order was described, because none was retrieved.
    prompt = fake.calls[1]["user_prompt"]
    assert "Do not describe any order" in prompt
    assert "FastPost" not in prompt


def test_scenario_5_unknown_order_is_not_fabricated(client, fake_llm):
    fake = fake_llm(
        [
            plan_reply(needs_order_tool=True, order_id="ORD-9999", intent="order status"),
            "I couldn't find an order with that reference.",
        ]
    )

    body = ask(client, "What is the status of ORD-9999?")

    assert body["trace"]["order_outcome"] == "not_found"
    assert body["confidence"] == 0.0
    assert body["sources"] == []
    prompt = fake.calls[1]["user_prompt"]
    assert "No order exists with the reference ORD-9999" in prompt
    assert "FastPost" not in prompt


# Scenario 6 — ambiguous request


def test_scenario_6_ambiguous_order_request(client, fake_llm):
    fake_llm([plan_reply(needs_order_tool=True, order_id=None, intent="order status")])

    body = ask(client, "Tell me about my order.")

    assert body["answer"] == ASK_FOR_ORDER_ID
    assert "ORD-1001" in body["answer"]  # given as an example of the format
    assert body["tool_calls"] == 0
    assert body["trace"]["order_tool_used"] is False
    assert body["confidence"] == 0.0
    assert "order_id_missing" in body["trace"]["degraded"]


def test_scenario_6_invented_order_id_is_never_looked_up(client, fake_llm):
    """The model hallucinates a reference; the tool must not be called with it."""
    fake_llm([plan_reply(needs_order_tool=True, order_id="ORD-1002", intent="order status")])

    body = ask(client, "Tell me about my order.")

    assert body["tool_calls"] == 0
    assert "ORD-1002" not in body["answer"]
    assert body["sources"] == []


def test_scenario_6_asks_for_the_reference_even_when_the_kb_was_searched(client, fake_llm):
    """The live planner sets needs_kb on this question, and the ask must survive.

    Searching the documents for "order status" is reasonable, but it must not
    displace the one thing the customer actually has to be asked for.
    """
    fake = fake_llm(
        [
            plan_reply(
                needs_kb=True,
                needs_order_tool=False,
                order_id=None,
                kb_query="order status",
                intent="order status",
            ),
            "I couldn't find details for your order. Could you share the reference?",
        ]
    )

    body = ask(client, "Tell me about my order.")

    assert body["tool_calls"] == 0
    assert "order_id_missing" in body["trace"]["degraded"]
    assert "Ask the customer for their order reference" in fake.calls[1]["user_prompt"]
