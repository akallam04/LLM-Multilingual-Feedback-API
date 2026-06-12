"""FastAPI application -- language feedback API and demo UI."""

import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

from app.config import get_settings  # noqa: E402  (env must load first)
from app.feedback import get_feedback  # noqa: E402
from app.models import FeedbackRequest, FeedbackResponse  # noqa: E402
from app.providers import (  # noqa: E402
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Language Feedback API",
    description=(
        "Analyzes learner-written sentences and returns structured language "
        "feedback: a minimal correction, categorized errors with explanations "
        "in the learner's native language, and a CEFR difficulty estimate."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)

_PROVIDER_ERROR_STATUS = {
    ProviderAuthError: 503,
    ProviderRateLimitError: 503,
    ProviderTimeoutError: 504,
    ProviderUnavailableError: 502,
}


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    status = 502
    for error_class, mapped_status in _PROVIDER_ERROR_STATUS.items():
        if isinstance(exc, error_class):
            status = mapped_status
            break
    return JSONResponse(status_code=status, content={"detail": exc.message})


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.0f}"
    return response


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.provider,
        "model": settings.model,
        "api_key_configured": settings.api_key_configured,
        "demo_mode": settings.provider == "mock",
    }


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    return await get_feedback(request)
