"""Observability for one question.

A `Trace` records what the assistant actually did — which sources it consulted,
how many calls it made, how well supported the answer was, and anything that
went wrong along the way. It is how the architecture can be demonstrated from
the outside instead of described.

Nothing here is customer-facing copy. It carries no prompts, no chunk text, no
API keys, and no stack traces — only names, counts, and codes.
"""

from pydantic import BaseModel, Field


class Trace(BaseModel):
    """What happened while answering one question."""

    intent: str = Field(default="unknown", description="The planner's label for the question")

    kb_used: bool = Field(default=False, description="Was the knowledge base searched")
    kb_calls: int = Field(default=0, description="Number of retrieval calls made")
    kb_confidence: float = Field(default=0.0, description="Best similarity score found")
    kb_threshold: float = Field(default=0.0, description="Score a match had to clear")

    order_tool_used: bool = Field(default=False, description="Was the order tool called")
    tool_calls: int = Field(default=0, description="Number of external tool calls")
    order_outcome: str | None = Field(default=None, description="Result of the order lookup")

    llm_calls: int = Field(default=0, description="Groq calls made, planner plus generator")

    sources: list[str] = Field(default_factory=list, description="Evidence behind the answer")
    confidence: float = Field(default=0.0, description="Overall support for the answer")
    grounded: bool = Field(default=False, description="Was there evidence to answer from")

    degraded: list[str] = Field(
        default_factory=list,
        description="Codes for anything that failed or fell back, e.g. 'planner_malformed'",
    )
