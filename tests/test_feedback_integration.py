"""Integration tests -- make real LLM API calls.

These run against whichever provider is configured (OpenAI or Anthropic)
and are skipped automatically when no API key is available or when the
mock provider is selected. Run with:

    pytest tests/test_feedback_integration.py -v
"""

import os

import pytest

from app.feedback import get_feedback
from app.models import FeedbackRequest


def _real_provider_available() -> bool:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit == "mock":
        return False
    if explicit == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if explicit == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


pytestmark = pytest.mark.skipif(
    not _real_provider_available(),
    reason="No LLM API key configured -- skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


@pytest.mark.asyncio
async def test_spanish_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_correct_german():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_french_gender_errors():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_japanese_particle():
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)
