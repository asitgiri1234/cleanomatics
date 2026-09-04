"""The answer generator: is it given the evidence, and only the evidence?

The model is faked, so what these tests check is the contract around it — what
goes into the prompt, what comes back out, and that the generator never fetches
anything itself.
"""

import pytest

from app.agent.generator import (
    UNVERIFIED_FALLBACK,
    build_evidence_prompt,
    collect_sources,
    generate_answer,
)
from app.llm.errors import LLMAPIError
from app.schemas.agent import Evidence
from app.schemas.kb import Chunk, RetrievalResult, RetrievedChunk
from app.schemas.orders import LookupOutcome, Order, OrderLookupResult
from tests.fakes import FakeLLMClient

REFUND_CHUNK = Chunk(
    chunk_id="refund-policy.md#2",
    source="refund-policy.md",
    title="Refund Policy",
    heading="Timing",
    text="Refund Policy — Timing\n\nRefunds are approved within 3-5 business days.",
)


def kb_evidence(has_evidence: bool = True) -> RetrievalResult:
    matches = [RetrievedChunk(chunk=REFUND_CHUNK, score=0.71)] if has_evidence else []
    return RetrievalResult(
        query="refund timing", matches=matches, threshold=0.3, has_evidence=has_evidence
    )


def order_evidence() -> OrderLookupResult:
    return OrderLookupResult(
        outcome=LookupOutcome.FOUND,
        order_id="ORD-1001",
        order=Order(
            order_id="ORD-1001",
            status="Shipped",
            carrier="FastPost",
            estimated_delivery="2026-09-08",
        ),
        message="Order ORD-1001 found.",
    )


# The supplied evidence reaches the model


def test_kb_evidence_is_put_in_the_prompt():
    client = FakeLLMClient("Refunds take 3-5 business days to approve.")

    generate_answer("How long do refunds take?", Evidence(kb=kb_evidence()), client)

    prompt = client.last_user_prompt
    assert "3-5 business days" in prompt
    assert "refund-policy.md" in prompt
    assert "How long do refunds take?" in prompt


def test_order_evidence_is_put_in_the_prompt():
    client = FakeLLMClient("Your order is on its way with FastPost.")

    generate_answer("Where is ORD-1001?", Evidence(order=order_evidence()), client)

    prompt = client.last_user_prompt
    assert "ORD-1001" in prompt
    assert "FastPost" in prompt
    assert "2026-09-08" in prompt


def test_both_kinds_of_evidence_appear_together():
    client = FakeLLMClient("Your order shipped, and refunds take 3-5 business days.")

    generate_answer(
        "Is my order late and can I get a refund?",
        Evidence(kb=kb_evidence(), order=order_evidence()),
        client,
    )

    prompt = client.last_user_prompt
    assert "3-5 business days" in prompt
    assert "FastPost" in prompt


def test_answer_text_is_returned_unchanged():
    client = FakeLLMClient("Refunds are approved within 3-5 business days.")

    answer = generate_answer("How long?", Evidence(kb=kb_evidence()), client)

    assert answer.answer == "Refunds are approved within 3-5 business days."
    assert answer.grounded is True


# Facts that were never supplied are never supplied


def test_prompt_contains_no_facts_beyond_the_evidence():
    """Whatever the model is told must trace back to the evidence given."""
    client = FakeLLMClient("ok")

    generate_answer("How long do refunds take?", Evidence(kb=kb_evidence()), client)

    prompt = client.last_user_prompt
    # Real policy numbers from other documents must not leak in.
    for unrelated in ["$6.95", "30 days", "ORD-1001", "FastPost", "$29"]:
        assert unrelated not in prompt


def test_missing_kb_evidence_is_stated_not_hidden():
    client = FakeLLMClient("I couldn't confirm that.")

    answer = generate_answer(
        "What is your policy on gift wrapping?",
        Evidence(kb=kb_evidence(has_evidence=False)),
        client,
    )

    prompt = client.last_user_prompt
    assert "Nothing in ShipFlow" in prompt
    assert "documentation covers this" in prompt
    assert answer.grounded is False


def test_no_evidence_at_all_instructs_a_refusal():
    client = FakeLLMClient("I couldn't verify that, please contact support.")

    answer = generate_answer("What is the capital of France?", Evidence(), client)

    assert "No verified information was found" in client.last_user_prompt
    assert answer.grounded is False
    assert answer.sources == []


def test_unknown_order_is_described_as_not_found():
    client = FakeLLMClient("I couldn't find that order reference.")

    result = OrderLookupResult(
        outcome=LookupOutcome.NOT_FOUND, order_id="ORD-9999", message="No order found."
    )
    answer = generate_answer("Where is ORD-9999?", Evidence(order=result), client)

    prompt = client.last_user_prompt
    assert "No order exists with the reference ORD-9999" in prompt
    assert "Do not describe any order" in prompt
    assert answer.grounded is False


def test_order_service_failure_is_described_honestly():
    client = FakeLLMClient("I can't check that order right now.")

    result = OrderLookupResult(
        outcome=LookupOutcome.SERVICE_ERROR, order_id="FAIL-001", message="unavailable"
    )
    generate_answer("Where is FAIL-001?", Evidence(order=result), client)

    prompt = client.last_user_prompt
    assert "could not be reached" in prompt
    assert "Do not describe any order" in prompt


def test_missing_order_reference_asks_for_one():
    client = FakeLLMClient("Could you share your order reference?")

    result = OrderLookupResult(
        outcome=LookupOutcome.INVALID_REQUEST, message="No order ID was supplied."
    )
    generate_answer("Where is my order?", Evidence(order=result), client)

    assert "Ask the customer for their order reference" in client.last_user_prompt


# Internal machinery stays internal


def test_similarity_scores_are_never_shown_to_the_model():
    client = FakeLLMClient("ok")

    generate_answer("How long do refunds take?", Evidence(kb=kb_evidence()), client)

    prompt = client.last_user_prompt
    assert "0.71" not in prompt
    assert "score" not in prompt.lower()
    assert "threshold" not in prompt.lower()


def test_system_prompt_forbids_exposing_internals():
    client = FakeLLMClient("ok")

    generate_answer("How long do refunds take?", Evidence(kb=kb_evidence()), client)

    system = client.last_system_prompt.lower()
    assert "never invent" in system
    assert "confidence scores" in system
    assert "supplied evidence only" in system


# The generator does not go looking for evidence itself


def test_generator_has_no_access_to_the_kb_or_the_order_tool():
    import app.agent.generator as module

    assert not hasattr(module, "get_retriever")
    assert not hasattr(module, "get_order_status")
    assert not hasattr(module, "lookup_order")


# Sources and failure handling


def test_sources_name_the_documents_and_the_order():
    evidence = Evidence(kb=kb_evidence(), order=order_evidence())

    assert collect_sources(evidence) == ["refund-policy.md", "order:ORD-1001"]


def test_sources_exclude_unusable_evidence():
    evidence = Evidence(
        kb=kb_evidence(has_evidence=False),
        order=OrderLookupResult(
            outcome=LookupOutcome.NOT_FOUND, order_id="ORD-9999", message="none"
        ),
    )

    assert collect_sources(evidence) == []


def test_api_failure_with_evidence_is_raised_not_hidden():
    client = FakeLLMClient(error=LLMAPIError("Groq returned 503", status_code=503))

    with pytest.raises(LLMAPIError):
        generate_answer("How long do refunds take?", Evidence(kb=kb_evidence()), client)


def test_api_failure_with_nothing_to_say_falls_back_to_a_safe_refusal():
    client = FakeLLMClient(error=LLMAPIError("Groq returned 503", status_code=503))

    answer = generate_answer("What is the capital of France?", Evidence(), client)

    assert answer.answer == UNVERIFIED_FALLBACK
    assert answer.grounded is False


def test_prompt_ends_by_asking_for_the_reply():
    prompt = build_evidence_prompt("How long?", Evidence(kb=kb_evidence()))

    assert prompt.strip().endswith("Write the reply to the customer now.")
