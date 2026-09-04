"""Splits documents into retrievable chunks.

The knowledge base is written as Markdown with `## ` sections, and those
sections are already the unit a reader would quote — "Return window", "Failed
payments". So sections are the natural chunk boundary, and only sections too
long to embed well get split further.

Every chunk is prefixed with its document title and section heading. A chunk
retrieved on its own then still says what it is about, instead of arriving as
a paragraph of loose sentences.
"""

from app.config import get_settings
from app.schemas.kb import Chunk, Document


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a document into `(heading, body)` pairs on `## ` headings.

    Text before the first `## ` becomes an untitled preamble section.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):
            sections.append((heading, "\n".join(body).strip()))
            heading = line[3:].strip()
            body = []
        elif not line.startswith("# "):
            body.append(line)

    sections.append((heading, "\n".join(body).strip()))
    return [(head, content) for head, content in sections if content]


def _split_long_body(body: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Break an oversized section on paragraph boundaries, with overlap.

    The overlap carries the tail of one piece into the next, so a fact that
    straddles a boundary is complete in at least one of them.
    """
    if len(body) <= max_chars:
        return [body]

    paragraphs = [para for para in body.split("\n\n") if para.strip()]
    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            pieces.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph

    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    document: Document,
    max_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[Chunk]:
    """Turn one document into context-carrying chunks."""
    settings = get_settings()
    limit = max_chars if max_chars is not None else settings.chunk_max_chars
    overlap = overlap_chars if overlap_chars is not None else settings.chunk_overlap_chars

    chunks: list[Chunk] = []
    for heading, body in _split_sections(document.text):
        for piece in _split_long_body(body, limit, overlap):
            context = f"{document.title} — {heading}" if heading else document.title
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source}#{len(chunks)}",
                    source=document.source,
                    title=document.title,
                    heading=heading,
                    text=f"{context}\n\n{piece}",
                )
            )
    return chunks


def chunk_documents(
    documents: list[Document],
    max_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[Chunk]:
    """Chunk a list of documents, preserving their order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars, overlap_chars))
    return chunks
