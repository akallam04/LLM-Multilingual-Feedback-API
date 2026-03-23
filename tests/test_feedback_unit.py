"""Unit tests -- run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest


def _mock_completion(response_data: dict) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.mark.asyncio
async def test_feedback_with_errors():
    mock_response = {
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

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence():
    mock_response = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_feedback_multiple_errors():
    mock_response = {
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

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)

@pytest.mark.asyncio
async def test_invalid_error_type_normalizes_to_other():
    mock_response = {
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

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Ella va a la tienda ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.errors[0].error_type == "other"


@pytest.mark.asyncio
async def test_difficulty_normalizes_from_lowercase():
    mock_response = {
        "corrected_sentence": "Bonjour, je vais bien.",
        "is_correct": True,
        "errors": [],
        "difficulty": "b1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Bonjour, je vais bien.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.difficulty == "B1"


@pytest.mark.asyncio
async def test_is_correct_true_forces_original_sentence_and_empty_errors():
    mock_response = {
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

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Ich bin müde.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.corrected_sentence == "Ich bin müde."
    assert result.errors == []


@pytest.mark.asyncio
async def test_non_list_errors_becomes_empty_list():
    mock_response = {
        "corrected_sentence": "Ciao, come stai?",
        "is_correct": False,
        "errors": "not-a-list",
        "difficulty": "A1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Ciao, come stai?",
            target_language="Italian",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.errors == []
    assert result.is_correct is True


@pytest.mark.asyncio
async def test_missing_explanation_gets_fallback_text():
    mock_response = {
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

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="La chat est noir.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert len(result.errors) == 1
    assert len(result.errors[0].explanation) > 0