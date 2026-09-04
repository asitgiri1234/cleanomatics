"""A stand-in LLM client, so tests do not depend on the network.

`FakeLLMClient` satisfies the same `LLMClient` protocol as the Groq client and
returns whatever it was handed. It records the prompts it saw, which is how the
tests check that the answer generator was actually given the evidence and that
the planner asked for JSON.
"""

import json

from app.llm.errors import LLMError


class FakeLLMClient:
    """Returns canned replies and remembers every call."""

    def __init__(self, replies: list[str] | str | None = None, error: Exception | None = None):
        if isinstance(replies, str):
            replies = [replies]
        self.replies = list(replies or [])
        self.error = error
        self.calls: list[dict] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "json_mode": json_mode,
            }
        )
        if self.error is not None:
            raise self.error
        if not self.replies:
            raise LLMError("FakeLLMClient ran out of replies")
        return self.replies.pop(0)

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1]["user_prompt"]

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1]["system_prompt"]


def plan_reply(
    needs_kb: bool = False,
    needs_order_tool: bool = False,
    order_id: str | None = None,
    kb_query: str | None = None,
    intent: str = "test",
) -> str:
    """Build the JSON a well-behaved planner would return."""
    return json.dumps(
        {
            "needs_kb": needs_kb,
            "needs_order_tool": needs_order_tool,
            "order_id": order_id,
            "kb_query": kb_query,
            "intent": intent,
        }
    )
