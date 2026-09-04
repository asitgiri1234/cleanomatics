"""Models for knowledge-base documents, chunks, and retrieval results."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """One knowledge-base file, cleaned and ready to chunk."""

    source: str = Field(description="Filename the text came from, e.g. 'returns.md'")
    title: str = Field(description="The document's top-level heading")
    text: str = Field(description="Cleaned document body")


class Chunk(BaseModel):
    """A retrievable slice of a document, carrying its own context."""

    chunk_id: str = Field(description="Stable identifier, e.g. 'returns.md#3'")
    source: str = Field(description="Filename the chunk came from")
    title: str = Field(description="Title of the document the chunk came from")
    heading: str = Field(default="", description="Section heading the chunk sits under")
    text: str = Field(description="Chunk text, prefixed with its document and section context")


class RetrievedChunk(BaseModel):
    """A chunk together with how well it matched a query."""

    chunk: Chunk
    score: float = Field(description="Cosine similarity to the query, roughly -1 to 1")


class RetrievalResult(BaseModel):
    """The outcome of one search over the knowledge base."""

    query: str
    matches: list[RetrievedChunk] = Field(default_factory=list)
    threshold: float = Field(description="Minimum score a match had to reach")
    has_evidence: bool = Field(
        description="False when nothing scored above the threshold"
    )

    @property
    def best_score(self) -> float:
        """Score of the strongest match, or 0.0 when there were none."""
        return self.matches[0].score if self.matches else 0.0

    @property
    def sources(self) -> list[str]:
        """Distinct source filenames behind the matches, best match first."""
        seen: list[str] = []
        for match in self.matches:
            if match.chunk.source not in seen:
                seen.append(match.chunk.source)
        return seen
