"""Semantic retrieval end to end: ranking, metadata, scores, and the threshold.

These tests run the real embedding model against the real knowledge base, so
they check the behaviour that actually matters — that a differently-worded
question still lands on the right document, and that an unrelated one is
refused rather than answered with the closest available chunk.
"""

import pytest

# Questions phrased the way a customer would, and the document that should win.
DIRECT_QUERIES = [
    ("How long does standard shipping take?", "shipping-and-delivery.md"),
    ("What is your return window?", "returns.md"),
    ("How long do refunds take to arrive?", "refund-policy.md"),
    ("How much does the Growth plan cost?", "subscription-plans.md"),
    ("What happens if my payment fails?", "billing-and-payments.md"),
    ("How do I add a user to my team?", "account-management.md"),
    ("What does the On Hold order status mean?", "order-management.md"),
]

# The same needs, worded without the vocabulary the documents use.
PARAPHRASED_QUERIES = [
    ("my parcel is taking ages to turn up", "shipping-and-delivery.md"),
    ("send an item back", "returns.md"),
    ("when will i see the money back on my card", "refund-policy.md"),
    ("i want to stop paying for this service", "subscription-plans.md"),
    ("someone else should be in charge of the account now", "account-management.md"),
]

# Nothing in the knowledge base speaks to these.
IRRELEVANT_QUERIES = [
    "What is the capital of France?",
    "Write me a poem about the sea",
    "How do I bake sourdough bread?",
    "What is the square root of 144?",
]


def test_index_is_built(retriever):
    assert len(retriever.chunks) > 20
    assert len(retriever.store) == len(retriever.chunks)


@pytest.mark.parametrize("query,expected_source", DIRECT_QUERIES)
def test_direct_question_retrieves_the_right_document(retriever, query, expected_source):
    result = retriever.search(query)

    assert result.has_evidence
    assert result.matches[0].chunk.source == expected_source


@pytest.mark.parametrize("query,expected_source", PARAPHRASED_QUERIES)
def test_paraphrased_question_still_finds_the_document(retriever, query, expected_source):
    result = retriever.search(query)

    assert result.has_evidence
    assert expected_source in result.sources


@pytest.mark.parametrize("query", IRRELEVANT_QUERIES)
def test_irrelevant_question_reports_no_evidence(retriever, query):
    result = retriever.search(query)

    assert not result.has_evidence
    assert result.matches == []
    assert result.best_score == 0.0


def test_results_carry_source_metadata(retriever):
    result = retriever.search("How long does standard shipping take?")

    for match in result.matches:
        chunk = match.chunk
        assert chunk.source.endswith(".md")
        assert chunk.chunk_id.startswith(chunk.source + "#")
        assert chunk.title
        assert chunk.text


def test_scores_are_ordered_and_within_bounds(retriever):
    result = retriever.search("What is your return window?")

    scores = [match.score for match in result.matches]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= score <= 1.0 for score in scores)


def test_relevant_questions_score_well_above_irrelevant_ones(retriever):
    relevant = retriever.search("How long does standard shipping take?", threshold=0.0)
    irrelevant = retriever.search("What is the capital of France?", threshold=0.0)

    assert relevant.best_score > 0.45
    assert irrelevant.best_score < 0.20
    assert relevant.best_score - irrelevant.best_score > 0.25


def test_top_k_limits_the_number_of_matches(retriever):
    result = retriever.search("refund policy", top_k=2, threshold=0.0)
    assert len(result.matches) <= 2


def test_threshold_filters_weak_matches(retriever):
    query = "How long does standard shipping take?"

    permissive = retriever.search(query, threshold=0.0)
    strict = retriever.search(query, threshold=0.99)

    assert permissive.has_evidence
    assert not strict.has_evidence
    assert strict.threshold == 0.99


def test_every_returned_match_clears_the_threshold(retriever):
    result = retriever.search("cancel my subscription", threshold=0.35)

    assert all(match.score >= 0.35 for match in result.matches)


def test_configured_threshold_is_used_by_default(retriever, settings):
    result = retriever.search("How do I cancel my plan?")
    assert result.threshold == settings.similarity_threshold


def test_empty_query_returns_no_evidence(retriever):
    for query in ["", "   "]:
        result = retriever.search(query)
        assert not result.has_evidence
        assert result.matches == []


def test_heavily_colloquial_phrasing_is_a_known_weak_spot(retriever):
    """Chatty phrasing retrieves poorly; the planner's rewrite is what fixes it.

    "i changed my mind, can i send this back" scores about 0.31 against every
    document and its closest match is the wrong one. Normalised to what the
    planner would emit, the same need scores about 0.63 on the right document.
    This is why retrieval is driven by `plan.kb_query` and not by the raw
    message — and it is recorded here so the behaviour is known rather than
    discovered.
    """
    raw = retriever.search("i changed my mind, can i send this back", threshold=0.0, top_k=1)
    planned = retriever.search("return an item", threshold=0.0, top_k=1)

    assert raw.best_score < planned.best_score
    assert planned.matches[0].chunk.source == "returns.md"
    assert not retriever.search("i changed my mind, can i send this back").has_evidence


def test_sources_are_distinct_and_best_first(retriever):
    result = retriever.search("returning a faulty item", top_k=4, threshold=0.0)

    assert len(result.sources) == len(set(result.sources))
    assert result.sources[0] == result.matches[0].chunk.source
