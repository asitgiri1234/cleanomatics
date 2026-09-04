"""Errors raised by the LLM layer.

Each one names a distinct failure the orchestrator will want to handle
differently: a missing key is a deployment problem, a rate limit is worth
retrying, malformed output means the model misbehaved on an otherwise healthy
connection. Nothing here is swallowed — callers are given something specific
enough to act on.
"""


class LLMError(Exception):
    """Base class for every LLM failure."""


class LLMConfigurationError(LLMError):
    """The client cannot be built — usually a missing API key."""


class LLMAPIError(LLMError):
    """The Groq API was called and did not return a usable response."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class LLMTimeoutError(LLMAPIError):
    """The request did not complete inside the configured timeout."""


class LLMResponseError(LLMError):
    """The model replied, but not with what was asked for.

    Raised for output that is not valid JSON, or is valid JSON that does not
    fit the expected schema. `raw` keeps the original text so the problem can
    be seen rather than guessed at.
    """

    def __init__(self, message: str, raw: str | None = None):
        self.raw = raw
        super().__init__(message)
