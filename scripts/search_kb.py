"""Search the knowledge base from the command line.

A standalone way to see what the retriever does, with no Groq key and no API
server involved.

    python scripts/search_kb.py "how long do refunds take"
    python scripts/search_kb.py --threshold 0.2 --top-k 5 "return window"
    python scripts/search_kb.py --demo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.kb.retriever import get_retriever  # noqa: E402
from app.schemas.kb import RetrievalResult  # noqa: E402

DEMO_QUERIES = [
    "How long does standard shipping take?",
    "I want my money back for something I sent back",
    "What is the capital of France?",
]


def show(result: RetrievalResult) -> None:
    """Print one search result."""
    print(f"\nQuery: {result.query}")
    print(f"Threshold: {result.threshold}")

    if not result.has_evidence:
        print("  No relevant knowledge-base evidence found.")
        return

    for rank, match in enumerate(result.matches, start=1):
        chunk = match.chunk
        heading = chunk.heading or "(intro)"
        print(f"  {rank}. {match.score:.4f}  {chunk.source}  [{heading}]  {chunk.chunk_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the ShipFlow knowledge base.")
    parser.add_argument("query", nargs="*", help="The question to search for")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to return")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum similarity")
    parser.add_argument("--demo", action="store_true", help="Run a few example queries")
    parser.add_argument("--show-text", action="store_true", help="Print matched chunk text")
    args = parser.parse_args()

    queries = DEMO_QUERIES if args.demo else [" ".join(args.query)]
    if not any(query.strip() for query in queries):
        parser.error("Give a query, or use --demo")

    settings = get_settings()
    print(f"Knowledge base: {settings.kb_path}")
    print(f"Embedding model: {settings.embedding_model}")

    retriever = get_retriever()
    print(f"Indexed chunks: {len(retriever.chunks)}")

    for query in queries:
        result = retriever.search(query, top_k=args.top_k, threshold=args.threshold)
        show(result)
        if args.show_text:
            for match in result.matches:
                print(f"\n--- {match.chunk.chunk_id} ---\n{match.chunk.text}\n")


if __name__ == "__main__":
    main()
