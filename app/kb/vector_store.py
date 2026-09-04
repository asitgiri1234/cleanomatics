"""A small in-memory vector store.

The knowledge base is a few hundred chunks, so the whole index is one numpy
array and a search is a single matrix multiply. That is fast enough that a
database would add operational weight without buying anything.
"""

import numpy as np

from app.schemas.kb import Chunk, RetrievedChunk


class InMemoryVectorStore:
    """Holds chunks alongside their embeddings and searches by cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        """The stored chunks, in insertion order."""
        return list(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Store chunks and their embeddings.

        Vectors are expected to be normalised, which lets search use a dot
        product instead of dividing by norms on every query.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors; they must match"
            )
        if not chunks:
            return

        self._chunks.extend(chunks)
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Return the `top_k` closest chunks, highest similarity first.

        No threshold is applied here — the store reports what it found and how
        close it was, and the retriever decides what counts as good enough.
        """
        if self._vectors is None or top_k <= 0:
            return []

        scores = self._vectors @ query_vector
        count = min(top_k, len(self._chunks))

        # argpartition finds the top `count` without sorting all of them, then
        # only those few are sorted.
        top_indices = np.argpartition(-scores, count - 1)[:count]
        ranked = sorted(top_indices, key=lambda index: -scores[index])

        return [
            RetrievedChunk(chunk=self._chunks[index], score=round(float(scores[index]), 4))
            for index in ranked
        ]
