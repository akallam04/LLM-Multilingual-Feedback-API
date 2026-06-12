"""Unit tests for feedback normalization -- run without an API key."""

import pytest

from app.feedback import get_feedback
from app.models import FeedbackRequest
from app.providers import ProviderResponseError

SPANISH_REQUEST = FeedbackRequest(
    sentence="Yo soy fue al mercado ayer.",
    target_language="Spanish",
    native_language="English",
)


@pytest.mark.asyncio
async def test_feedback_with_errors(stub_provider_factory):
    stub_provider_factory(
        {
            "corrected_sentence": "Yo fui al mercado ayer.",
            "is_correct": False,
            "errors": [
                {
                    "original": "soy fue",
                    "correction": "fui",
                    "error_type": "conjugation",
                    "explanation": "You mixed two verb forms.",
                }
            ],
            "difficulty": "A2",
        }
    )

    result = await get_feedback(SPANISH_REQUEST)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence(stub_provider_factory):
    request = FeedbackRequest(
        sentence="Ich habe gestern einen interessanten Film gesehen.",
        target_language="German",
        native_language="English",
    )
    stub_provider_factory(
        {
            "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
            "is_correct": True,
            "errors": [],
            "difficulty": "B1",
        }
    )

    result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_feedback_multiple_errors(stub_provider_factory):
    stub_provider_factory(
        {
            "corrected_sentence": "Le chat noir est sur la table.",
            "is_correct": False,
            "errors": [
                {
                    "original": "La chat",
                    "correction": "Le chat",
                    "error_type": "gender_agreement",
                    "explanation": "'Chat' is masculine.",
                },
                {
                    "original": "le table",
                    "correction": "la table",
                    "error_type": "gender_agreement",
                    "explanation": "'Table' is feminine.",
                },
            ],
            "difficulty": "A1",
        }
    )

    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_invalid_error_type_normalizes_to_other(stub_provider_factory):
    stub_provider_factory(
        {
            "corrected_sentence": "Ella fue a la tienda ayer.",
            "is_correct": False,
            "errors": [
                {
                    "original": "va",
                    "correction": "fue",
                    "error_type": "verb_tense_issue",
                    "explanation": "Use the past tense here.",
                }
            ],
            "difficulty": "A2",
        }
    )

    result = await get_feedback(
        FeedbackRequest(
            sentence="Ella va a la tienda ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )

    assert result.errors[0].error_type == "other"


@pytest.mark.asyncio
async def test_difficulty_normalizes_from_lowercase(stub_provider_factory):
    stub_provider_factory(
        {
            "corrected_sentence": "Bonjour, je vais bien.",
            "is_correct": True,
            "errors": [],
            "difficulty": "b1",
        }
    )

    result = await get_feedback(
        FeedbackRequest(
            sentence="Bonjour, je vais bien.",
            target_language="French",
            native_language="English",
        )
    )

    assert result.difficulty == "B1"


@pytest.mark.asyncio
async def test_is_correct_true_forces_original_sentence_and_empty_errors(
    stub_provider_factory,
):
    stub_provider_factory(
        {
            "corrected_sentence": "This should be ignored",
            "is_correct": True,
            "errors": [
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": "grammar",
                    "explanation": "Should be removed.",
                }
            ],
            "difficulty": "A1",
        }
    )

    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich bin müde.",
            target_language="German",
            native_language="English",
        )
    )

    assert result.is_correct is True
    assert result.corrected_sentence == "Ich bin müde."
    assert result.errors == []


@pytest.mark.asyncio
async def test_non_list_errors_becomes_empty_list(stub_provider_factory):
    stub_provider_factory(
        {
            "corrected_sentence": "Ciao, come stai?",
            "is_correct": False,
            "errors": "not-a-list",
            "difficulty": "A1",
        }
    )

    result = await get_feedback(
        FeedbackRequest(
            sentence="Ciao, come stai?",
            target_language="Italian",
            native_language="English",
        )
    )

    assert result.errors == []
    assert result.is_correct is True


@pytest.mark.asyncio
async def test_missing_explanation_gets_fallback_text(stub_provider_factory):
    stub_provider_factory(
        {
            "corrected_sentence": "Le chat est noir.",
            "is_correct": False,
            "errors": [
                {
                    "original": "La chat",
                    "correction": "Le chat",
                    "error_type": "gender_agreement",
                    "explanation": "",
                }
            ],
            "difficulty": "A1",
        }
    )

    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat est noir.",
            target_language="French",
            native_language="English",
        )
    )

    assert len(result.errors) == 1
    assert len(result.errors[0].explanation) > 0


@pytest.mark.asyncio
async def test_unusable_json_retries_once_then_succeeds(stub_provider_factory):
    good_payload = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "Mixed verb forms.",
            }
        ],
        "difficulty": "A2",
    }
    stub = stub_provider_factory(
        ProviderResponseError("The language model returned malformed JSON."),
        good_payload,
    )

    result = await get_feedback(SPANISH_REQUEST)

    assert stub.calls == 2
    assert result.corrected_sentence == "Yo fui al mercado ayer."


@pytest.mark.asyncio
async def test_unusable_json_twice_raises(stub_provider_factory):
    stub = stub_provider_factory(
        ProviderResponseError("malformed"),
        ProviderResponseError("malformed again"),
    )

    with pytest.raises(ProviderResponseError):
        await get_feedback(SPANISH_REQUEST)

    assert stub.calls == 2
