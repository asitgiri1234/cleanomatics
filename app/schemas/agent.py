"""Internal models: the planner's output, gathered evidence, the final answer."""

from pydantic import BaseModel, Field, field_validator

from app.schemas.kb import RetrievalResult
from app.schemas.orders import OrderLookupResult


class Plan(BaseModel):
    """What the planner decided a question needs.

    This is a shopping list, not an answer. It says which sources to consult;
    it never contains anything to say to the customer.
    """

    needs_kb: bool = Field(description="Search the policy documents")
    needs_order_tool: bool = Field(description="Look up a specific order")
    order_id: str | None = Field(default=None, description="Order reference from the question")
    kb_query: str | None = Field(default=None, description="What to search the documents for")
    intent: str = Field(default="unknown", description="Short label for the question")

    @field_validator("order_id", "kb_query", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        """Treat the model's stand-ins for "nothing" as nothing.

        Models routinely return "", "null", or "none" as strings where the
        schema asks for null, and each of those would otherwise become a
        search for the word "none".
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"null", "none", "n/a"}:
                return None
            return stripped
        return value

    @field_validator("order_id")
    @classmethod
    def _uppercase_reference(cls, value: str | None) -> str | None:
        return value.upper() if value else value


class Evidence(BaseModel):
    """Everything verified that was gathered for one question.

    Both fields are optional because the plan decides what gets collected. The
    answer generator receives this and nothing else — it does not go looking.
    """

    kb: RetrievalResult | None = None
    order: OrderLookupResult | None = None

    @property
    def has_any(self) -> bool:
        """True when at least one source produced something usable."""
        kb_ok = self.kb is not None and self.kb.has_evidence
        order_ok = self.order is not None and self.order.found
        return kb_ok or order_ok


class Answer(BaseModel):
    """The customer-facing reply, with what it was based on."""

    answer: str
    sources: list[str] = Field(
        default_factory=list, description="Documents and tools the answer drew on"
    )
    grounded: bool = Field(
        default=True, description="False when there was no evidence to answer from"
    )
