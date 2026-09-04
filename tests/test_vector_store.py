"""The in-memory vector store: ranking, scores, and bounds."""

import numpy as np
import pytest

from app.kb.vector_store import InMemoryVectorStore
from app.schemas.kb import Chunk


def make_chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc.md#{index}",
        source="doc.md",
        title="Doc",
        heading=f"Section {index}",
        text=f"Body {index}",
    )


def unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    vectors = np.vstack([unit([1, 0, 0]), unit([0, 1, 0]), unit([1, 1, 0])])
    store.add([make_chunk(0), make_chunk(1), make_chunk(2)], vectors)
    return store


def test_empty_store_returns_nothing():
    assert InMemoryVectorStore().search(unit([1, 0, 0]), top_k=3) == []
    assert len(InMemoryVectorStore()) == 0


def test_store_reports_its_size(store):
    assert len(store) == 3
    assert len(store.chunks) == 3


def test_results_are_ordered_by_similarity(store):
    matches = store.search(unit([1, 0, 0]), top_k=3)

    scores = [match.score for match in matches]
    assert scores == sorted(scores, reverse=True)
    assert matches[0].chunk.chunk_id == "doc.md#0"


def test_identical_vector_scores_about_one(store):
    best = store.search(unit([1, 0, 0]), top_k=1)[0]
    assert best.score == pytest.approx(1.0, abs=1e-3)


def test_orthogonal_vector_scores_about_zero(store):
    matches = {match.chunk.chunk_id: match.score for match in store.search(unit([1, 0, 0]), 3)}
    assert matches["doc.md#1"] == pytest.approx(0.0, abs=1e-3)


def test_top_k_limits_the_number_of_results(store):
    assert len(store.search(unit([1, 0, 0]), top_k=2)) == 2
    assert store.search(unit([1, 0, 0]), top_k=0) == []


def test_top_k_larger_than_the_store_is_safe(store):
    assert len(store.search(unit([1, 0, 0]), top_k=50)) == 3


def test_mismatched_chunks_and_vectors_are_rejected():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add([make_chunk(0)], np.vstack([unit([1, 0, 0]), unit([0, 1, 0])]))


def test_adding_in_batches_accumulates():
    store = InMemoryVectorStore()
    store.add([make_chunk(0)], np.vstack([unit([1, 0, 0])]))
    store.add([make_chunk(1)], np.vstack([unit([0, 1, 0])]))
    assert len(store) == 2
    assert len(store.search(unit([1, 1, 0]), top_k=5)) == 2
