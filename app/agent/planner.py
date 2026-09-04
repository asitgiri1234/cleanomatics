"""LLM call #1 — the planner.

Reads the customer's question and decides which sources are needed to answer
it. It never answers the question; its output is a `Plan`, which the next stage
executes.

The model's decision is not taken on trust. Everything it returns is validated,
and one thing is checked against the question itself: an order reference that
does not appear in what the customer wrote is discarded. A hallucinated order
ID would otherwise send the tool looking up a real order belonging to someone
else, which is the worst failure this system could have.
"""

import re

from app.llm.errors import LLMResponseError
from app.llm.groq_client import LLMClient, get_llm_client, parse_json_object
from app.llm.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from app.schemas.agent import Plan

ORDER_ID_PATTERN = re.compile(r"\b([A-Z]{2,6}-\d{3,})\b", re.IGNORECASE)

# The planner should be near-deterministic: it is classifying, not writing.
PLANNER_TEMPERATURE = 0.0


def find_order_ids(text: str) -> list[str]:
    """Return every order-shaped reference in the text, uppercased."""
    return [match.group(1).upper() for match in ORDER_ID_PATTERN.finditer(text)]


def _strip_invented_order_id(plan: Plan, question: str) -> Plan:
    """Drop an order reference the customer never wrote.

    Comparison ignores spaces and hyphens, so "ord 1001" in the question still
    validates a returned "ORD-1001" — the customer's formatting varies, but the
    digits must be theirs.
    """
    if plan.order_id is None:
        return plan

    def squash(value: str) -> str:
        return re.sub(r"[\s\-_]", "", value).upper()

    if squash(plan.order_id) not in squash(question):
        plan.order_id = None
        plan.needs_order_tool = False

    return plan


def _repair_plan(plan: Plan, question: str) -> Plan:
    """Make an internally inconsistent plan safe to execute.

    Models produce combinations that cannot be carried out — asking for an
    order lookup with no reference, or a knowledge-base search with nothing to
    search for. Rather than failing the request, the plan is corrected to
    something coherent and the caller gets a plan it can always run.
    """
    plan = _strip_invented_order_id(plan, question)

    # An order lookup with no reference cannot be performed.
    if plan.needs_order_tool and not plan.order_id:
        plan.needs_order_tool = False

    # A reference the planner missed but the customer clearly wrote.
    if not plan.needs_order_tool and not plan.order_id:
        found = find_order_ids(question)
        if found:
            plan.order_id = found[0]
            plan.needs_order_tool = True

    # A search needs something to search for; fall back to the question.
    if plan.needs_kb and not plan.kb_query:
        plan.kb_query = question.strip()

    # A query without a search is dead weight downstream.
    if not plan.needs_kb:
        plan.kb_query = None

    return plan


def plan_question(question: str, client: LLMClient | None = None) -> Plan:
    """Decide what evidence `question` needs.

    Raises `LLMResponseError` if the model's reply cannot be read as a plan,
    and the errors in `app.llm.errors` for configuration and API problems.
    """
    if not question or not question.strip():
        return Plan(
            needs_kb=False,
            needs_order_tool=False,
            order_id=None,
            kb_query=None,
            intent="empty question",
        )

    llm = client or get_llm_client()
    reply = llm.complete(
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=build_planner_prompt(question),
        temperature=PLANNER_TEMPERATURE,
        json_mode=True,
    )

    payload = parse_json_object(reply)

    try:
        plan = Plan.model_validate(payload)
    except Exception as error:
        raise LLMResponseError(
            f"Planner output did not match the expected plan: {error}", raw=reply
        ) from error

    return _repair_plan(plan, question)
