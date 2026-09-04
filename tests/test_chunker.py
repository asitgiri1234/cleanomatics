"""Chunking: documents are split into context-carrying, retrievable pieces."""

from app.config import get_settings
from app.kb.chunker import chunk_document, chunk_documents
from app.schemas.kb import Document

SAMPLE = Document(
    source="sample.md",
    title="Sample Policy",
    text=(
        "# Sample Policy\n\n"
        "Intro paragraph.\n\n"
        "## First Section\n\n"
        "First section body.\n\n"
        "## Second Section\n\n"
        "Second section body."
    ),
)


def test_chunks_are_produced_for_the_whole_knowledge_base(chunks):
    assert len(chunks) > 20


def test_every_chunk_keeps_its_source_and_identifier(chunks):
    for chunk in chunks:
        assert chunk.source.endswith(".md")
        assert chunk.chunk_id.startswith(chunk.source + "#")
        assert chunk.title


def test_chunk_ids_are_unique(chunks):
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_carry_document_and_section_context(chunks):
    for chunk in chunks:
        assert chunk.title in chunk.text
        if chunk.heading:
            assert chunk.heading in chunk.text


def test_chunks_respect_the_configured_size(chunks, settings):
    # The context prefix is added on top of the body limit, so allow for it.
    ceiling = settings.chunk_max_chars + settings.chunk_overlap_chars + 200
    for chunk in chunks:
        assert len(chunk.text) <= ceiling


def test_sections_become_separate_chunks():
    chunks = chunk_document(SAMPLE)
    headings = [chunk.heading for chunk in chunks]
    assert headings == ["", "First Section", "Second Section"]


def test_long_sections_are_split_with_overlap():
    paragraphs = "\n\n".join(f"Paragraph number {index}. " * 12 for index in range(8))
    document = Document(
        source="long.md",
        title="Long Doc",
        text=f"# Long Doc\n\n## Big Section\n\n{paragraphs}",
    )

    chunks = chunk_document(document, max_chars=400, overlap_chars=100)

    assert len(chunks) > 1
    assert all(chunk.heading == "Big Section" for chunk in chunks)
    assert all(len(chunk.text) < 800 for chunk in chunks)


def test_empty_sections_are_dropped():
    document = Document(
        source="sparse.md",
        title="Sparse",
        text="# Sparse\n\n## Empty Section\n\n## Real Section\n\nContent here.",
    )

    chunks = chunk_document(document)

    assert [chunk.heading for chunk in chunks] == ["Real Section"]


def test_chunking_many_documents_preserves_order():
    other = Document(source="other.md", title="Other", text="# Other\n\n## S\n\nBody.")
    chunks = chunk_documents([SAMPLE, other])
    sources = [chunk.source for chunk in chunks]
    assert sources == sorted(sources, key=lambda name: 0 if name == "sample.md" else 1)


def test_defaults_come_from_configuration():
    settings = get_settings()
    assert settings.chunk_max_chars > 0
    assert 0 <= settings.chunk_overlap_chars < settings.chunk_max_chars
