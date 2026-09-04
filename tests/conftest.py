"""Shared test fixtures.

The retriever is expensive to build — it loads an embedding model and embeds
every chunk — so it is built once per test session and shared.
"""

import pytest

from app.config import get_settings
from app.kb.chunker import chunk_documents
from app.kb.loader import load_documents
from app.kb.retriever import KnowledgeBaseRetriever


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def documents():
    return load_documents()


@pytest.fixture(scope="session")
def chunks(documents):
    return chunk_documents(documents)


@pytest.fixture(scope="session")
def retriever():
    return KnowledgeBaseRetriever()
