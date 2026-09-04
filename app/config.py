"""Application configuration, loaded from the environment.

Every tunable number in the retrieval pipeline lives here rather than being
written into the code that uses it, so behaviour can be changed from `.env`
without touching a module.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Settings read from the environment, with `.env` as a fallback."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq — not used until the LLM step.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Knowledge base
    kb_dir: Path = Path("knowledge_base")

    # Retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4
    similarity_threshold: float = 0.30

    # Chunking
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 150

    # API
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    @property
    def kb_path(self) -> Path:
        """The knowledge-base directory as an absolute path.

        A relative `KB_DIR` is resolved against the project root, so the app
        behaves the same whatever directory it is started from.
        """
        if self.kb_dir.is_absolute():
            return self.kb_dir
        return PROJECT_ROOT / self.kb_dir


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton."""
    return Settings()
