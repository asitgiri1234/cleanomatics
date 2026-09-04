"""Semantic search over the ShipFlow knowledge base.

Ties the pieces together: load documents, chunk them, embed the chunks once at
startup, then answer queries against that index.

The threshold matters as much as the ranking. A vector search always returns
its nearest neighbour, however far away it is, so without a floor the retriever
would hand back the least-bad chunk for a question the knowledge base cannot
answer. `has_evidence` on the result is how a caller tells "here is what the
documents say" apart from "the documents do not cover this".
"""

from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.kb.chunker import chunk_documents
from app.kb.embeddings import embed_query, embed_texts
from app.kb.loader import load_documents
from app.kb.vector_store import InMemoryVectorStore
from app.schemas.kb import Chunk, RetrievalResult


class KnowledgeBaseRetriever:
    """Loads the knowledge base into memory and searches it semantically."""

    def __init__(self, kb_dir: Path | None = None, model_name: str | None = None) -> None:
        settings = get_settings()
        self.kb_dir = kb_dir or settings.kb_path
        self.model_name = model_name or settings.embedding_model
        self.store = InMemoryVectorStore()
        self._build()

    def _build(self) -> None:
        """Load, chunk, and embed the knowledge base."""
        documents = load_documents(self.kb_dir)
        chunks = chunk_documents(documents)
        if chunks:
            vectors = embed_texts([chunk.text for chunk in chunks], self.model_name)
            self.store.add(chunks, vectors)

    @property
    def chunks(self) -> list[Chunk]:
        """Every indexed chunk."""
        return self.store.chunks

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RetrievalResult:
        """Find the chunks that best answer `query`.

        Only matches at or above the threshold are returned. When nothing
        clears it, the result comes back empty with `has_evidence` False rather
        than with a weak match that reads as if it were an answer.
        """
        settings = get_settings()
        limit = top_k if top_k is not None else settings.top_k
        floor = threshold if threshold is not None else settings.similarity_threshold

        if not query.strip():
            return RetrievalResult(
                query=query, matches=[], threshold=floor, has_evidence=False
            )

        candidates = self.store.search(embed_query(query, self.model_name), limit)
        matches = [match for match in candidates if match.score >= floor]

        return RetrievalResult(
            query=query,
            matches=matches,
            threshold=floor,
            has_evidence=bool(matches),
        )


@lru_cache
def get_retriever() -> KnowledgeBaseRetriever:
    """Return the shared retriever, building the index on first use."""
    return KnowledgeBaseRetriever()
