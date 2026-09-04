"""Document loading: the knowledge base is found, read, and cleaned."""

import pytest

from app.kb.loader import clean_text, extract_title, load_documents

EXPECTED_SOURCES = {
    "account-management.md",
    "billing-and-payments.md",
    "order-management.md",
    "refund-policy.md",
    "returns.md",
    "shipflow-faq.md",
    "shipping-and-delivery.md",
    "subscription-plans.md",
}


def test_loads_every_knowledge_base_file(documents):
    assert {document.source for document in documents} == EXPECTED_SOURCES


def test_documents_keep_their_source_filename(documents):
    for document in documents:
        assert document.source.endswith(".md")
        assert "/" not in document.source


def test_documents_have_titles_and_text(documents):
    for document in documents:
        assert document.title
        assert len(document.text) > 200


def test_title_comes_from_the_top_level_heading(documents):
    by_source = {document.source: document for document in documents}
    assert by_source["returns.md"].title == "Returns"
    assert by_source["refund-policy.md"].title == "Refund Policy"


def test_clean_text_normalises_line_endings_and_blank_runs():
    cleaned = clean_text("# Title\r\n\r\n\r\n\r\nBody text   \r\n")
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "# Title\n\nBody text"


def test_extract_title_falls_back_to_the_filename():
    assert extract_title("no heading here", "shipping-and-delivery") == "Shipping And Delivery"


def test_missing_directory_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")


def test_non_markdown_files_are_ignored(tmp_path):
    (tmp_path / "policy.md").write_text("# Policy\n\nSome text.", encoding="utf-8")
    (tmp_path / "notes.json").write_text('{"ignored": true}', encoding="utf-8")
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.source for document in documents] == ["policy.md"]
