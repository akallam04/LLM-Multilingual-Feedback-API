"""Feedback generation: provider call, output normalization, response validation.

A raw LLM payload is never trusted. Whatever the provider returns is
normalized (near-miss enum values mapped to allowed ones, consistency rules
enforced, safe fallbacks filled in) and then validated against the
constrained Pydantic response model before it reaches the client.
"""

import logging

from app.models import FeedbackRequest, FeedbackResponse
from app.providers import ProviderResponseError, get_provider

logger = logging.getLogger("feedback")

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

ERROR_TYPE_ALIASES = {
    "verb_conjugation": "conjugation",
    "tense": "conjugation",
    "agreement": "grammar",
    "word choice": "word_choice",
    "word-choice": "word_choice",
    "word order": "word_order",
    "missing word": "missing_word",
    "extra word": "extra_word",
    "gender agreement": "gender_agreement",
    "number agreement": "number_agreement",
    "tone": "tone_register",
    "register": "tone_register",
}

DIFFICULTY_ALIASES = {
    "a-1": "A1",
    "a-2": "A2",
    "b-1": "B1",
    "b-2": "B2",
    "c-1": "C1",
    "c-2": "C2",
    "a1": "A1",
    "a2": "A2",
    "b1": "B1",
    "b2": "B2",
    "c1": "C1",
    "c2": "C2",
}


def normalize_error_type(value: str) -> str:
    normalized = value.strip().lower().replace("__", "_")
    normalized = normalized.replace(" ", "_").replace("-", "_")
    if normalized in VALID_ERROR_TYPES:
        return normalized
    if normalized in ERROR_TYPE_ALIASES:
        return ERROR_TYPE_ALIASES[normalized]
    return "other"


def normalize_difficulty(value: str) -> str:
    normalized = value.strip()
    upper = normalized.upper()
    if upper in VALID_DIFFICULTIES:
        return upper
    lower = normalized.lower()
    if lower in DIFFICULTY_ALIASES:
        return DIFFICULTY_ALIASES[lower]
    return "A1"


def normalize_feedback_payload(data: dict, request: FeedbackRequest) -> dict:
    corrected_sentence = str(data.get("corrected_sentence", request.sentence)).strip()
    if not corrected_sentence:
        corrected_sentence = request.sentence

    errors = data.get("errors", [])
    if not isinstance(errors, list):
        errors = []

    normalized_errors = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        original = str(error.get("original", "")).strip()
        correction = str(error.get("correction", "")).strip()
        explanation = str(error.get("explanation", "")).strip()
        error_type = normalize_error_type(str(error.get("error_type", "other")))

        if not explanation:
            explanation = (
                f"This part should be corrected in {request.target_language}. "
                f"The explanation should be given in {request.native_language}."
            )

        normalized_errors.append(
            {
                "original": original,
                "correction": correction,
                "error_type": error_type,
                "explanation": explanation,
            }
        )

    difficulty = normalize_difficulty(str(data.get("difficulty", "A1")))
    is_correct = bool(data.get("is_correct", False))

    if is_correct:
        corrected_sentence = request.sentence
        normalized_errors = []

    if not normalized_errors and corrected_sentence == request.sentence:
        is_correct = True

    if normalized_errors:
        is_correct = False

    return {
        "corrected_sentence": corrected_sentence,
        "is_correct": is_correct,
        "errors": normalized_errors,
        "difficulty": difficulty,
    }


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    provider = get_provider()

    try:
        data = await provider.complete(request)
    except ProviderResponseError:
        # One retry: malformed output from a small model is usually transient.
        logger.warning("Provider returned unusable JSON, retrying once")
        data = await provider.complete(request)

    normalized_data = normalize_feedback_payload(data, request)
    return FeedbackResponse(**normalized_data)
