"""List the Groq models this API key can use.

Groq retires models periodically, so the value in `.env` can go stale. This
prints what is actually available right now.

    python scripts/list_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groq import Groq  # noqa: E402

from app.config import get_settings  # noqa: E402

# Models that transcribe, guard, or synthesise speech rather than chat.
NON_CHAT_HINTS = ("whisper", "prompt-guard", "orpheus", "tts")


def main() -> None:
    settings = get_settings()
    if not settings.groq_api_key.strip():
        print("GROQ_API_KEY is not set. Add it to .env first.")
        raise SystemExit(1)

    models = sorted(Groq(api_key=settings.groq_api_key).models.list().data, key=lambda m: m.id)
    chat_models = [m.id for m in models if not any(h in m.id.lower() for h in NON_CHAT_HINTS)]

    print("Chat models available to this key:\n")
    for model_id in chat_models:
        marker = "  <- currently configured" if model_id == settings.groq_model else ""
        print(f"  {model_id}{marker}")

    if settings.groq_model not in chat_models:
        print(f"\nWARNING: GROQ_MODEL is '{settings.groq_model}', which is not in that list.")
        print("Update GROQ_MODEL in your .env file.")


if __name__ == "__main__":
    main()
