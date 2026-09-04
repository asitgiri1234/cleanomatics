"""Request and response models for the chat endpoint."""

from pydantic import BaseModel, Field, field_validator

from app.schemas.trace import Trace


class ChatRequest(BaseModel):
    """A customer question."""

    model_config = {"extra": "forbid"}

    message: str = Field(
        min_length=1,
        max_length=2000,
        description="The customer's question",
        examples=["Where is my order ORD-1001?"],
    )

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """A message of only whitespace is empty, whatever its length."""
        if not value.strip():
            raise ValueError("message must not be empty")
        return value


class ChatResponse(BaseModel):
    """The assistant's reply.

    The first four fields are the response contract. `trace` is additional
    observability — it shows which sources were consulted and how many calls
    were made, and carries nothing sensitive.
    """

    answer: str = Field(description="The reply to show the customer")
    sources: list[str] = Field(
        default_factory=list,
        description="Policy documents and order references the answer drew on",
    )
    tool_calls: int = Field(default=0, description="External tool calls made")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Evidence support, 0-1")
    trace: Trace | None = Field(default=None, description="What the assistant did")


class ErrorDetail(BaseModel):
    """A machine-readable description of a failure."""

    code: str = Field(description="Stable error code, e.g. 'invalid_request'")
    message: str = Field(description="What went wrong, safe to show a caller")


class ErrorResponse(BaseModel):
    """The body returned for any non-2xx response."""

    error: ErrorDetail
