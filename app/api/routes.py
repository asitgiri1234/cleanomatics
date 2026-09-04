"""The chat endpoint.

The route does no agent work. It validates the request, hands the question to
the orchestrator, and shapes the result into a response — so the reasoning
lives in one place and the API stays a thin edge.
"""

import logging

from fastapi import APIRouter, status

from app.agent.orchestrator import answer_question
from app.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness check, plus whether the assistant is configured to answer.

    Reports only whether a key is present, never the key itself.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "groq_configured": bool(settings.groq_api_key.strip()),
        "model": settings.groq_model,
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["chat"],
    responses={
        422: {"model": ErrorResponse, "description": "The request was not valid"},
        502: {"model": ErrorResponse, "description": "The language model failed"},
        503: {"model": ErrorResponse, "description": "The assistant is not configured"},
        504: {"model": ErrorResponse, "description": "The language model timed out"},
    },
)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a customer question.

    Failures inside the pipeline — a knowledge base that will not load, an
    order service that is down, a planner that returns nonsense — do not fail
    the request. They produce an honest answer with the reason recorded in the
    trace. Only errors that make answering impossible reach the error handlers.
    """
    result = answer_question(request.message)

    return ChatResponse(
        answer=result.answer.answer,
        sources=result.answer.sources,
        tool_calls=result.trace.tool_calls,
        confidence=result.trace.confidence,
        trace=result.trace,
    )
