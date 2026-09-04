"""FastAPI application entry point.

Wires the router to the app and installs the exception handlers that keep
internal detail inside the process. Every error leaves here as
`{"error": {"code", "message"}}` — never a stack trace, a prompt, a filename,
or an API key.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.llm.errors import (
    LLMAPIError,
    LLMConfigurationError,
    LLMError,
    LLMTimeoutError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    """Build the one error shape this API returns."""
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the retriever at startup.

    Loading the embedding model and indexing the documents takes a couple of
    seconds. Doing it here means the first customer does not pay for it, and a
    broken knowledge base shows up at boot rather than mid-conversation.
    """
    try:
        from app.kb.retriever import get_retriever

        retriever = get_retriever()
        logger.info("Knowledge base ready: %d chunks indexed", len(retriever.chunks))
    except Exception:
        # Not fatal: the order tool still works, and /chat degrades honestly.
        logger.exception("Knowledge base failed to load; retrieval will be unavailable")

    yield


app = FastAPI(
    title="Cleanomatics Support Assistant",
    description="Grounded customer support for ShipFlow.",
    version="1.0.0",
    lifespan=lifespan,
)

# The chat UI is one static HTML file served by this same app, so the page
# and the API share an origin and the browser asks no CORS questions. The
# middleware below only matters if the page is served from somewhere else.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def chat_ui() -> FileResponse:
        """Serve the chat interface at the root."""
        return FileResponse(STATIC_DIR / "index.html")
else:  # pragma: no cover - only if the UI file has been removed
    logger.warning("No static directory at %s; the chat UI will not be served", STATIC_DIR)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Turn FastAPI's validation report into this API's error shape.

    The field name and reason are useful to a caller fixing their request; the
    rest of the report is not, so only the first problem is summarised.
    """
    errors = exc.errors()
    if errors:
        location = ".".join(str(part) for part in errors[0].get("loc", []) if part != "body")
        reason = errors[0].get("msg", "is not valid")
        detail = f"'{location}' {reason}" if location else reason
    else:
        detail = "The request body was not valid."

    return _error(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "invalid_request",
        f"The request was not valid: {detail}. Send {{\"message\": \"your question\"}}.",
    )


@app.exception_handler(LLMConfigurationError)
async def handle_configuration_error(request: Request, exc: LLMConfigurationError):
    """A deployment problem, not the caller's fault.

    The exception's own text is safe to return: it names the setting, never
    its value.
    """
    logger.error("Configuration error: %s", exc)
    return _error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "assistant_unavailable",
        "The assistant is not configured correctly and cannot answer right now.",
    )


@app.exception_handler(LLMTimeoutError)
async def handle_timeout_error(request: Request, exc: LLMTimeoutError):
    logger.warning("LLM timeout: %s", exc)
    return _error(
        status.HTTP_504_GATEWAY_TIMEOUT,
        "assistant_timeout",
        "The assistant took too long to respond. Please try again.",
    )


@app.exception_handler(LLMAPIError)
async def handle_api_error(request: Request, exc: LLMAPIError):
    logger.error("LLM API error: %s", exc)
    return _error(
        status.HTTP_502_BAD_GATEWAY,
        "assistant_error",
        "The assistant could not complete your request. Please try again shortly.",
    )


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError):
    """Anything from the LLM layer not caught above."""
    logger.error("LLM error: %s", exc)
    return _error(
        status.HTTP_502_BAD_GATEWAY,
        "assistant_error",
        "The assistant could not complete your request. Please try again shortly.",
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """The last line of defence.

    Logged in full for the operator, summarised to one sentence for the caller.
    Nothing about the exception reaches the response.
    """
    logger.exception("Unhandled error answering request")
    return _error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "Something went wrong. Please try again.",
    )
