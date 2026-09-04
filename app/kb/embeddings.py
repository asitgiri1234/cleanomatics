"""Turns text into vectors, locally.

Embeddings come from a sentence-transformers model running on this machine —
no embedding API is called. Groq is used later for chat completions only.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def get_model(model_name: str | None = None) -> SentenceTransformer:
    """Load the embedding model once and reuse it.

    The first call downloads the weights (roughly 90 MB for all-MiniLM-L6-v2)
    and is slow; later calls are instant.
    """
    name = model_name or get_settings().embedding_model
    return SentenceTransformer(name)


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Embed a list of texts into an `(n, dim)` array of unit vectors.

    The vectors are normalised, so a dot product between any two of them is
    their cosine similarity.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = get_model(model_name).encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def embed_query(query: str, model_name: str | None = None) -> np.ndarray:
    """Embed a single query into one unit vector of shape `(dim,)`."""
    return embed_texts([query], model_name)[0]
