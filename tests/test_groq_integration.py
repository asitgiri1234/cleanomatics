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

    assert "3-5" in answer.answer or "three to five" in answer.answer.lower()
    assert answer.grounded is True
    assert "refund-policy.md" in answer.sources


@needs_key
def test_generator_declines_when_it_has_no_evidence():
    answer = generate_answer("What is the capital of France?", Evidence())

    assert "paris" not in answer.answer.lower()
    assert answer.grounded is False
