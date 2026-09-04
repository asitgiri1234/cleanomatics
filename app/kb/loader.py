"""Reads knowledge-base documents from disk and cleans them for retrieval."""

from pathlib import Path

from app.config import get_settings
from app.schemas.kb import Document

MARKDOWN_SUFFIXES = (".md", ".markdown", ".txt")


def clean_text(raw: str) -> str:
    """Normalise whitespace so chunking sees predictable text.

    Line endings become `\n`, trailing spaces go, and runs of blank lines
    collapse to one — paragraph breaks are meaningful to the chunker, extra
    blank lines are not.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned: list[str] = []
    for line in lines:
        if not line and cleaned and not cleaned[-1]:
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def extract_title(text: str, fallback: str) -> str:
    """Return the document's `# ` heading, or a title made from the filename."""
    for line in text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return fallback.replace("-", " ").replace("_", " ").title()


def load_document(path: Path) -> Document:
    """Load and clean a single knowledge-base file."""
    text = clean_text(path.read_text(encoding="utf-8"))
    return Document(
        source=path.name,
        title=extract_title(text, path.stem),
        text=text,
    )


def load_documents(kb_dir: Path | None = None) -> list[Document]:
    """Load every knowledge-base file, sorted by filename for stable ordering.

    Empty files are skipped: they produce no chunks and would only add noise.
    """
    directory = kb_dir or get_settings().kb_path
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    documents = [
        load_document(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in MARKDOWN_SUFFIXES
    ]
    return [document for document in documents if document.text]
