"""Provider layer tests: JSON parsing, provider resolution, error mapping, mock mode."""

from unittest.mock import AsyncMock

import httpx
import pytest

from app.models import FeedbackRequest
from app.providers import (
    MOCK_RESPONSES,
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
    get_provider,
    parse_json_payload,
)

REQUEST = FeedbackRequest(
    sentence="Yo soy fue al mercado ayer.",
    target_language="Spanish",
    native_language="English",
)


class TestParseJsonPayload:
    def test_plain_json(self):
        assert parse_json_payload('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        assert parse_json_payload(text) == {"a": 1}

    def test_json_with_surrounding_prose(self):
        text = 'Here is the feedback:\n{"a": 1}\nHope that helps!'
        assert parse_json_payload(text) == {"a": 1}

    def test_empty_response_raises(self):
        with pytest.raises(ProviderResponseError):
            parse_json_payload("")

    def test_none_raises(self):
        with pytest.raises(ProviderResponseError):
            parse_json_payload(None)

    def test_no_json_object_raises(self):
        with pytest.raises(ProviderResponseError):
            parse_json_payload("I cannot help with that.")

    def test_malformed_json_raises(self):
        with pytest.raises(ProviderResponseError):
            parse_json_payload('{"a": unquoted}')

    def test_non_object_json_raises(self):
        with pytest.raises(ProviderResponseError):
            parse_json_payload("[1, 2, 3]")


class TestProviderResolution:
    def test_explicit_mock(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert isinstance(get_provider(), MockProvider)

    def test_openai_auto_detected_from_key(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert isinstance(get_provider(), OpenAIProvider)

    def test_anthropic_auto_detected_from_key(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
        assert isinstance(get_provider(), AnthropicProvider)

    def test_no_keys_falls_back_to_mock(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert isinstance(get_provider(), MockProvider)

    def test_invalid_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        with pytest.raises(ValueError):
            get_provider()

    def test_provider_is_cached(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert get_provider() is get_provider()


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_known_sentence_returns_curated_feedback(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        provider = get_provider()
        result = await provider.complete(REQUEST)
        assert result["is_correct"] is False
        assert result["errors"][0]["error_type"] == "conjugation"

    @pytest.mark.asyncio
    async def test_unknown_sentence_echoes_as_correct(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        provider = get_provider()
        request = FeedbackRequest(
            sentence="Una frase nueva que nadie ha visto.",
            target_language="Spanish",
            native_language="English",
        )
        result = await provider.complete(request)
        assert result["is_correct"] is True
        assert result["corrected_sentence"] == request.sentence
        assert result["errors"] == []
        assert result["difficulty"] in {"A1", "A2", "B1", "B2"}

    @pytest.mark.asyncio
    async def test_canned_responses_are_isolated_copies(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        provider = get_provider()
        first = await provider.complete(REQUEST)
        first["errors"].clear()
        second = await provider.complete(REQUEST)
        assert len(second["errors"]) == 1

    def test_all_canned_responses_match_response_shape(self):
        for sentence, payload in MOCK_RESPONSES.items():
            assert isinstance(payload["corrected_sentence"], str), sentence
            assert isinstance(payload["is_correct"], bool), sentence
            assert isinstance(payload["errors"], list), sentence
            assert payload["difficulty"] in {"A1", "A2", "B1", "B2", "C1", "C2"}


def _openai_provider(monkeypatch) -> OpenAIProvider:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    provider = get_provider()
    assert isinstance(provider, OpenAIProvider)
    return provider


def _openai_status_error(status_code: int):
    import openai

    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    classes = {
        401: openai.AuthenticationError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
        400: openai.BadRequestError,
    }
    return classes[status_code]("simulated error", response=response, body=None)


class TestOpenAIErrorMapping:
    @pytest.mark.asyncio
    async def test_auth_error(self, monkeypatch):
        provider = _openai_provider(monkeypatch)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=_openai_status_error(401)
        )
        with pytest.raises(ProviderAuthError):
            await provider.complete(REQUEST)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, monkeypatch):
        provider = _openai_provider(monkeypatch)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=_openai_status_error(429)
        )
        with pytest.raises(ProviderRateLimitError):
            await provider.complete(REQUEST)

    @pytest.mark.asyncio
    async def test_server_error_maps_to_unavailable(self, monkeypatch):
        provider = _openai_provider(monkeypatch)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=_openai_status_error(500)
        )
        with pytest.raises(ProviderUnavailableError):
            await provider.complete(REQUEST)

    @pytest.mark.asyncio
    async def test_client_error_maps_to_provider_error(self, monkeypatch):
        provider = _openai_provider(monkeypatch)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=_openai_status_error(400)
        )
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(REQUEST)
        assert not isinstance(exc_info.value, ProviderAuthError)


def _anthropic_status_error(status_code: int):
    import anthropic

    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    classes = {
        401: anthropic.AuthenticationError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
    }
    return classes[status_code]("simulated error", response=response, body=None)


class TestAnthropicErrorMapping:
    @pytest.mark.asyncio
    async def test_auth_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
        provider = get_provider()
        assert isinstance(provider, AnthropicProvider)
        provider._client.messages.create = AsyncMock(
            side_effect=_anthropic_status_error(401)
        )
        with pytest.raises(ProviderAuthError):
            await provider.complete(REQUEST)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
        provider = get_provider()
        provider._client.messages.create = AsyncMock(
            side_effect=_anthropic_status_error(429)
        )
        with pytest.raises(ProviderRateLimitError):
            await provider.complete(REQUEST)
