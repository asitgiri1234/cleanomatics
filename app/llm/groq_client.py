"""The application's LLM client.

Everything Groq-specific is confined to this module. The planner and the answer
generator talk to the `LLMClient` protocol, so swapping the provider means
writing one new class here and changing nothing else.
"""

import json
from typing import Any, Protocol, runtime_checkable

import groq
from groq import Groq

from app.config import get_settings
from app.llm.errors import (
    LLMAPIError,
    LLMConfigurationError,
    LLMResponseError,
    LLMTimeoutError,
)


@runtime_checkable
class LLMClient(Protocol):
    """What the rest of the application needs from a language model."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Return the model's reply as text."""
        ...


class GroqLLMClient:
    """`LLMClient` backed by the Groq API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        key = api_key if api_key is not None else settings.groq_api_key

        if not key or not key.strip():
            raise LLMConfigurationError(
                "GROQ_API_KEY is not set. Add it to your .env file or export it "
                "in the environment before starting the application."
            )

        self.model = model or settings.groq_model
        self.timeout = timeout if timeout is not None else settings.groq_timeout_seconds
        self._client = Groq(
            api_key=key,
            timeout=self.timeout,
            max_retries=max_retries if max_retries is not None else settings.groq_max_retries,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Send one prompt pair and return the reply text.

        `json_mode` asks Groq to constrain the output to a JSON object. That is
        what makes the planner's output structured rather than parsed out of
        prose — the model cannot wrap it in an explanation or a code fence.
        """
        settings = get_settings()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": settings.llm_max_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**request)
        except groq.APITimeoutError as error:
            raise LLMTimeoutError(
                f"Groq request timed out after {self.timeout}s"
            ) from error
        except groq.AuthenticationError as error:
            raise LLMConfigurationError(
                "Groq rejected the API key. Check GROQ_API_KEY is current and correct."
            ) from error
        except groq.NotFoundError as error:
            # Groq retires models, and the failure otherwise reads as a generic
            # 404 that looks like a bug in this code rather than stale config.
            raise LLMConfigurationError(
                f"Groq does not serve the model '{self.model}'. It may have been "
                "retired. Set GROQ_MODEL in your .env to a current model — run "
                "`python scripts/list_models.py` to see what your key can use."
            ) from error
        except groq.APIStatusError as error:
            raise LLMAPIError(
                f"Groq returned {error.status_code}: {error.message}",
                status_code=error.status_code,
            ) from error
        except groq.APIConnectionError as error:
            raise LLMAPIError(f"Could not reach Groq: {error}") from error
        except groq.GroqError as error:
            raise LLMAPIError(f"Groq request failed: {error}") from error

        if not response.choices:
            raise LLMResponseError("Groq returned no choices")

        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise LLMResponseError("Groq returned an empty message")

        return content.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model reply that is supposed to be a JSON object.

    JSON mode makes bare objects the norm, but a model can still fence the
    output or add a sentence around it, so the outermost braces are recovered
    before giving up.
    """
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError("Model reply was not JSON", raw=text) from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise LLMResponseError(f"Model reply was not valid JSON: {error}", raw=text) from error

    if not isinstance(parsed, dict):
        raise LLMResponseError(
            f"Expected a JSON object, got {type(parsed).__name__}", raw=text
        )
    return parsed


def get_llm_client() -> LLMClient:
    """Build the configured LLM client.

    Deliberately not cached: a client built before the key was set should not
    be kept, and the error message should reappear on the next attempt.
    """
    return GroqLLMClient()
