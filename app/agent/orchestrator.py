"""The orchestrator: plan, gather, answer.

This is the layer that joins the pieces. It calls the planner, executes
whatever the plan asked for, and hands the gathered evidence to the generator.

It owns two decisions that no other component can make:

**What happens when a step fails.** A planner that returns nonsense, a
knowledge base that will not load, an order service that is down — each is
caught here and turned into a degraded but honest answer, rather than a stack
trace. The failure is recorded in the trace so it is visible; it is never
hidden from the operator, only from the customer.

**How confident the answer is.** Confidence here means "how well supported",
computed from the evidence that came back for the sources the plan asked for.
A question needing two sources that only got one is less well supported than a
question needing one that got it, and the number says so.
"""

import logging

from app.agent.generator import UNVERIFIED_FALLBACK, generate_answer
from app.agent.planner import find_order_ids, plan_question
from app.kb.retriever import get_retriever
from app.llm.errors import LLMError, LLMResponseError
from app.llm.groq_client import LLMClient, get_llm_client
from app.schemas.agent import Answer, Evidence, Plan
from app.schemas.orders import LookupOutcome, OrderLookupResult
from app.schemas.trace import Trace
from app.tools.orders import lookup_order

logger = logging.getLogger(__name__)

# An order record is a direct answer from the system of record, so a successful
# lookup is treated as fully supporting whatever it is asked about. Retrieval
# confidence, by contrast, is the measured similarity score.
ORDER_FOUND_CONFIDENCE = 1.0

ASK_FOR_ORDER_ID = (
    "I'd be glad to check on that. Could you give me your order reference? "
    "It looks like ORD-1001 and is on your confirmation email."
)


class ChatResult:
    """One answered question: the reply, plus what produced it."""

    def __init__(self, answer: Answer, trace: Trace):
        self.answer = answer
        self.trace = trace


def _fallback_plan(question: str) -> Plan:
    """A plan built without the model, for when the planner fails.

    Deliberately generous: search the documents for whatever was asked, and
    look up an order only if the customer plainly wrote a reference. It cannot
    match the model's judgement, but it keeps the assistant answering, and it
    cannot invent an order ID because it only ever copies one out of the text.
    """
    found = find_order_ids(question)
    return Plan(
        needs_kb=True,
        needs_order_tool=bool(found),
        order_id=found[0] if found else None,
        kb_query=question.strip(),
        intent="fallback",
    )


def _make_plan(question: str, client: LLMClient, trace: Trace) -> Plan:
    """Ask the planner what this question needs, tolerating a bad answer."""
    try:
        plan = plan_question(question, client)
        trace.llm_calls += 1
        return plan
    except LLMResponseError:
        # The model replied, just not usefully. Degrade rather than fail.
        logger.warning("Planner returned malformed output; using the fallback plan")
        trace.llm_calls += 1
        trace.degraded.append("planner_malformed")
        return _fallback_plan(question)


def _gather(plan: Plan, trace: Trace) -> Evidence:
    """Run the plan and collect the evidence it asked for."""
    evidence = Evidence()

    if plan.needs_kb and plan.kb_query:
        try:
            evidence.kb = get_retriever().search(plan.kb_query)
            trace.kb_used = True
            trace.kb_calls += 1
            trace.kb_confidence = evidence.kb.best_score
            trace.kb_threshold = evidence.kb.threshold
        except Exception:
            # A missing knowledge base or a model that will not load. The order
            # half of the answer can still go ahead.
            logger.exception("Knowledge base retrieval failed")
            trace.degraded.append("kb_unavailable")

    if plan.needs_order_tool:
        # lookup_order never raises: a service failure comes back as an outcome.
        evidence.order = lookup_order(plan.order_id)
        trace.order_tool_used = True
        trace.tool_calls += 1
        trace.order_outcome = evidence.order.outcome.value
        if evidence.order.outcome is LookupOutcome.SERVICE_ERROR:
            trace.degraded.append("order_service_error")

    return evidence


def _confidence(plan: Plan, evidence: Evidence) -> float:
    """How well the evidence supports an answer, from 0.0 to 1.0.

    Averaged across the sources the plan asked for, so a question that needed
    both the documents and an order, and only got one, scores lower than one
    that got everything it needed. A plan that asked for nothing — a greeting,
    an off-topic question — scores 0.0, which is honest: no evidence was
    gathered, so nothing is supported by any.
    """
    parts: list[float] = []

    if plan.needs_kb:
        parts.append(evidence.kb.best_score if evidence.kb and evidence.kb.has_evidence else 0.0)

    if plan.needs_order_tool:
        found = evidence.order is not None and evidence.order.found
        parts.append(ORDER_FOUND_CONFIDENCE if found else 0.0)

    if not parts:
        return 0.0

    return round(max(0.0, sum(parts) / len(parts)), 2)


def _needs_an_order_reference(question: str, plan: Plan) -> bool:
    """True when the customer asked about an order but gave no reference.

    Answered directly rather than by asking the model, because there is nothing
    to reason about: without a reference there is nothing to look up, and the
    only useful reply is to ask for one.
    """
    if plan.order_id or find_order_ids(question):
        return False
    return "order" in plan.intent.lower() or "tracking" in plan.intent.lower()


def answer_question(question: str, client: LLMClient | None = None) -> ChatResult:
    """Answer one customer question end to end.

    Raises the errors in `app.llm.errors` for problems the caller must know
    about — a missing API key, a Groq outage. Everything else degrades into an
    honest answer with the reason recorded in the trace.
    """
    trace = Trace()

    if not question or not question.strip():
        trace.degraded.append("empty_question")
        return ChatResult(
            Answer(answer=UNVERIFIED_FALLBACK, sources=[], grounded=False), trace
        )

    llm = client or get_llm_client()

    plan = _make_plan(question, llm, trace)
    trace.intent = plan.intent

    # An order question with no reference: ask for it instead of guessing, and
    # spend no LLM call doing so.
    if _needs_an_order_reference(question, plan) and not plan.needs_kb:
        trace.degraded.append("order_id_missing")
        return ChatResult(
            Answer(answer=ASK_FOR_ORDER_ID, sources=[], grounded=False), trace
        )

    evidence = _gather(plan, trace)

    # An order question with no reference, where the documents were searched
    # too. The tool was not called and must not be, but the customer still
    # needs to be asked for the reference — so the lookup's "no ID supplied"
    # outcome is supplied directly. The generator already knows to ask.
    if evidence.order is None and _needs_an_order_reference(question, plan):
        trace.degraded.append("order_id_missing")
        evidence.order = OrderLookupResult(
            outcome=LookupOutcome.INVALID_REQUEST,
            message="No order ID was supplied, so no order could be looked up.",
        )

    try:
        answer = generate_answer(question, evidence, llm)
        trace.llm_calls += 1
    except LLMError:
        logger.exception("Answer generation failed")
        trace.llm_calls += 1
        raise

    trace.sources = answer.sources
    trace.grounded = answer.grounded
    trace.confidence = _confidence(plan, evidence)

    logger.info(
        "answered intent=%s kb=%s tools=%d llm=%d confidence=%.2f degraded=%s",
        trace.intent,
        trace.kb_used,
        trace.tool_calls,
        trace.llm_calls,
        trace.confidence,
        trace.degraded or "-",
    )

    return ChatResult(answer, trace)
