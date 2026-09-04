"""LLM call #2 — the answer generator.

Takes the customer's question and the evidence already gathered for it, and
writes the reply.

It has no access to the knowledge base or the order tool. That is the point:
this call cannot go and find anything, so everything in its answer had to come
from evidence Python handed it. The separation is what makes "every fact is
grounded" a property of the design rather than a hope about the prompt.
"""

from app.llm.errors import LLMError
from app.llm.groq_client import LLMClient, get_llm_client
from app.llm.prompts import ANSWER_SYSTEM_PROMPT, NO_EVIDENCE_NOTICE
from app.schemas.agent import Answer, Evidence
from app.schemas.kb import RetrievalResult
from app.schemas.orders import LookupOutcome, OrderLookupResult

ANSWER_TEMPERATURE = 0.3

UNVERIFIED_FALLBACK = (
    "I couldn't verify that information, so I'd rather not guess. "
    "Please contact ShipFlow support and they'll be able to confirm it for you."
)


def _format_kb_evidence(result: RetrievalResult) -> str:
    """Render retrieved chunks for the prompt.

    Scores are deliberately left out. They are useful for deciding whether to
    pass evidence along at all, but a model shown a number tends to talk about
    it, and the customer should never see one.
    """
    if not result.has_evidence:
        return ""

    parts = ["POLICY DOCUMENTS", ""]
    for index, match in enumerate(result.matches, start=1):
        parts.append(f"[{index}] From {match.chunk.source}:")
        parts.append(match.chunk.text)
        parts.append("")
    return "\n".join(parts).strip()


def _format_order_evidence(result: OrderLookupResult) -> str:
    """Render the order lookup for the prompt.

    Every outcome is described, not just success. A model told "this order was
    not found" writes something honest; a model told nothing fills the silence.
    """
    if result.outcome is LookupOutcome.FOUND and result.order is not None:
        fields = result.order.model_dump(exclude_none=True)
        lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in fields.items()]
        return "ORDER RECORD\n\n" + "\n".join(lines)

    if result.outcome is LookupOutcome.NOT_FOUND:
        return (
            "ORDER LOOKUP\n\n"
            f"No order exists with the reference {result.order_id}. "
            "Tell the customer that reference was not found and ask them to check it. "
            "Do not describe any order."
        )

    if result.outcome is LookupOutcome.SERVICE_ERROR:
        return (
            "ORDER LOOKUP\n\n"
            "The order system could not be reached, so this order's details are "
            "unavailable right now. Tell the customer you cannot check the order at "
            "the moment and to try again shortly. Do not describe any order."
        )

    return (
        "ORDER LOOKUP\n\n"
        "No order reference was supplied, so no order could be looked up. "
        "Ask the customer for their order reference. Do not describe any order."
    )


def build_evidence_prompt(question: str, evidence: Evidence) -> str:
    """Assemble the user-side prompt for the answer call."""
    sections = [f"CUSTOMER QUESTION\n\n{question.strip()}"]

    if evidence.kb is not None:
        kb_text = _format_kb_evidence(evidence.kb)
        if kb_text:
            sections.append(kb_text)
        else:
            sections.append(
                "POLICY DOCUMENTS\n\nNothing in ShipFlow's documentation covers this. "
                "Do not answer the policy part from general knowledge."
            )

    if evidence.order is not None:
        sections.append(_format_order_evidence(evidence.order))

    if not evidence.has_any:
        sections.append(NO_EVIDENCE_NOTICE)

    sections.append("Write the reply to the customer now.")
    return "\n\n---\n\n".join(sections)


def collect_sources(evidence: Evidence) -> list[str]:
    """Name what the answer was allowed to draw on."""
    sources: list[str] = []
    if evidence.kb is not None and evidence.kb.has_evidence:
        sources.extend(evidence.kb.sources)
    if evidence.order is not None and evidence.order.found and evidence.order.order_id:
        sources.append(f"order:{evidence.order.order_id}")
    return sources


def generate_answer(
    question: str,
    evidence: Evidence,
    client: LLMClient | None = None,
) -> Answer:
    """Write the customer-facing reply from `evidence`.

    With no usable evidence the model is still asked to reply, so the refusal
    is worded naturally and in context. If that call fails, a fixed refusal is
    returned rather than an error — declining to answer is always safe.
    """
    grounded = evidence.has_any
    llm = client or get_llm_client()

    try:
        text = llm.complete(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=build_evidence_prompt(question, evidence),
            temperature=ANSWER_TEMPERATURE,
            json_mode=False,
        )
    except LLMError:
        if grounded:
            raise
        return Answer(answer=UNVERIFIED_FALLBACK, sources=[], grounded=False)

    return Answer(answer=text, sources=collect_sources(evidence), grounded=grounded)
