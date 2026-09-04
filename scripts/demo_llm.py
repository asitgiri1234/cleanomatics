"""Exercise the two LLM calls and the order tool by hand.

Needs GROQ_API_KEY. Nothing is orchestrated here — the planner and the
generator are called separately, and this script stands in for the wiring that
the next phase will build properly.

    python scripts/demo_llm.py "What is ShipFlow's refund policy?"
    python scripts/demo_llm.py --plan-only "Where is ORD-1001?"
    python scripts/demo_llm.py --demo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.generator import generate_answer  # noqa: E402
from app.agent.planner import plan_question  # noqa: E402
from app.kb.retriever import get_retriever  # noqa: E402
from app.llm.errors import LLMError  # noqa: E402
from app.schemas.agent import Evidence, Plan  # noqa: E402
from app.tools.orders import lookup_order  # noqa: E402

DEMO_QUESTIONS = [
    "What is ShipFlow's refund policy?",
    "What is the status of ORD-1001?",
    "What is ShipFlow's refund policy and what is the status of ORD-1001?",
    "Hello there!",
]


def gather(plan: Plan) -> Evidence:
    """Run whatever the plan asked for.

    A placeholder for the orchestrator: it does the obvious thing and nothing
    more — no retries, no fallbacks, no partial-failure policy.
    """
    evidence = Evidence()
    if plan.needs_kb and plan.kb_query:
        evidence.kb = get_retriever().search(plan.kb_query)
    if plan.needs_order_tool:
        evidence.order = lookup_order(plan.order_id)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the planner and answer generator.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--plan-only", action="store_true", help="Stop after planning")
    parser.add_argument("--demo", action="store_true", help="Run the example questions")
    args = parser.parse_args()

    questions = DEMO_QUESTIONS if args.demo else [" ".join(args.question)]
    if not any(question.strip() for question in questions):
        parser.error("Give a question, or use --demo")

    for question in questions:
        print("=" * 70)
        print(f"QUESTION: {question}")
        try:
            plan = plan_question(question)
        except LLMError as error:
            print(f"  PLANNER FAILED: {type(error).__name__}: {error}")
            continue

        print(f"  PLAN: {plan.model_dump_json()}")
        if args.plan_only:
            continue

        evidence = gather(plan)
        if evidence.kb is not None:
            print(f"  KB: {evidence.kb.sources or 'no relevant evidence'}")
        if evidence.order is not None:
            print(f"  ORDER: {evidence.order.outcome.value}")

        try:
            answer = generate_answer(question, evidence)
        except LLMError as error:
            print(f"  GENERATOR FAILED: {type(error).__name__}: {error}")
            continue

        print(f"\n  ANSWER: {answer.answer}")
        print(f"  SOURCES: {answer.sources or 'none'}  GROUNDED: {answer.grounded}\n")


if __name__ == "__main__":
    main()
