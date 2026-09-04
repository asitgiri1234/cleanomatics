"""Integration tests that call the real Groq API.

Skipped unless a key is configured, so the ordinary suite stays fast and
offline. To run them:

    python -m pytest tests/test_groq_integration.py -m integration -v

The rest of the suite fakes the model. These check the assumptions the fakes
rest on — that the model returns JSON when asked, that it plans the four shapes
correctly, and that it declines when it is given nothing to work with.
"""

import pytest

from app.agent.generator import generate_answer
from app.agent.planner import plan_question
from app.config import get_settings
from app.schemas.agent import Evidence
from app.schemas.kb import Chunk, RetrievalResult, RetrievedChunk

pytestmark = pytest.mark.integration

# Live models write with typographic punctuation - en-dashes, narrow no-break
# spaces, curly apostrophes. Comparing against those literally makes a test
# fail on formatting while the answer itself is right, so text is flattened to
# plain ASCII before anything is asserted about it.
UNICODE_LOOKALIKES = {
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
    "‘": "'", "’": "'", "“": '"', "”": '"',
}


def flatten(text: str) -> str:
    """Lowercase `text` with typographic punctuation replaced by ASCII."""
    for fancy, plain in UNICODE_LOOKALIKES.items():
        text = text.replace(fancy, plain)
    return text.lower()


# Ways a model can decline. Any of these means it refused rather than answered.
REFUSALS = ("unable to", "couldn't", "could not", "cannot confirm", "can't confirm",
            "don't have", "do not have", "not able to", "no information")


def declined(text: str) -> bool:
    """True when the reply is a refusal rather than an assertion of fact."""
    return any(phrase in flatten(text) for phrase in REFUSALS)


needs_key = pytest.mark.skipif(
    not get_settings().groq_api_key.strip(),
    reason="GROQ_API_KEY is not set; skipping live Groq tests",
)


@needs_key
def test_planner_recognises_a_kb_only_question():
    plan = plan_question("What is ShipFlow's refund policy?")

    assert plan.needs_kb is True
    assert plan.needs_order_tool is False
    assert plan.order_id is None


@needs_key
def test_planner_recognises_an_order_only_question():
    plan = plan_question("What is the status of ORD-1001?")

    assert plan.needs_order_tool is True
    assert plan.order_id == "ORD-1001"


@needs_key
def test_planner_recognises_a_question_needing_both():
    plan = plan_question(
        "What is ShipFlow's refund policy and what is the status of ORD-1001?"
    )

    assert plan.needs_kb is True
    assert plan.needs_order_tool is True
    assert plan.order_id == "ORD-1001"


@needs_key
def test_planner_does_not_invent_an_order_id():
    plan = plan_question("Where is my order? It's been a week.")

    assert plan.order_id is None
    assert plan.needs_order_tool is False


@needs_key
def test_generator_answers_from_the_evidence_it_was_given():
    chunk = Chunk(
        chunk_id="refund-policy.md#2",
        source="refund-policy.md",
        title="Refund Policy",
        heading="Timing",
        text=(
            "Refund Policy — Timing\n\n"
            "Refunds are approved within 3-5 business days of the return passing "
            "inspection."
        ),
    )
    evidence = Evidence(
        kb=RetrievalResult(
            query="refund timing",
            matches=[RetrievedChunk(chunk=chunk, score=0.7)],
            threshold=0.3,
            has_evidence=True,
        )
    )

    answer = generate_answer("How long do refunds take to approve?", evidence)

    assert "3-5" in flatten(answer.answer) or "three to five" in flatten(answer.answer)
    assert answer.grounded is True
    assert "refund-policy.md" in answer.sources


@needs_key
def test_generator_declines_when_it_has_no_evidence():
    answer = generate_answer("What is the capital of France?", Evidence())

    assert "paris" not in flatten(answer.answer)
    assert declined(answer.answer)
    assert answer.grounded is False


# The six required scenarios, against the real model and the real API.


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


def _ask(api_client, message: str) -> dict:
    response = api_client.post("/chat", json={"message": message})
    assert response.status_code == 200, response.text
    return response.json()


@needs_key
def test_live_scenario_1_kb_only(api_client):
    body = _ask(api_client, "What is ShipFlow's refund policy?")

    assert body["trace"]["kb_used"] is True
    assert body["tool_calls"] == 0
    assert body["sources"]
    assert body["confidence"] > 0.35


@needs_key
def test_live_scenario_2_order_only(api_client):
    body = _ask(api_client, "What is the status of ORD-1001?")

    assert body["tool_calls"] == 1
    assert body["trace"]["order_outcome"] == "found"
    assert body["confidence"] == 1.0
    assert "shipped" in flatten(body["answer"]) or "fastpost" in flatten(body["answer"])


@needs_key
def test_live_scenario_3_both(api_client):
    body = _ask(
        api_client, "What is your refund policy and what is the status of ORD-1001?"
    )

    assert body["trace"]["kb_used"] is True
    assert body["tool_calls"] == 1
    assert body["confidence"] > 0.5


@needs_key
def test_live_scenario_4_unsupported(api_client):
    body = _ask(api_client, "What shift patterns do your warehouse staff work?")

    assert body["confidence"] < 0.5
    # Nothing about shift patterns exists, so the reply must decline rather
    # than state one. Naming the topic while refusing is fine; asserting a
    # fact about it is not.
    assert declined(body["answer"])


@needs_key
def test_live_scenario_5_tool_failure(api_client):
    body = _ask(api_client, "What is the status of FAIL-001?")

    assert body["trace"]["order_outcome"] == "service_error"
    assert body["confidence"] == 0.0
    assert body["sources"] == []
    # No order details may be invented for an order that could not be read.
    assert "fastpost" not in flatten(body["answer"])


@needs_key
def test_live_scenario_6_ambiguous(api_client):
    body = _ask(api_client, "Tell me about my order.")

    assert body["tool_calls"] == 0
    # No specific order may be described, and none of the seeded orders named.
    for reference in ["ORD-1002", "ORD-1003", "ORD-1004"]:
        assert reference not in body["answer"]
