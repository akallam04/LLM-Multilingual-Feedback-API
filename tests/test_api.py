"""API-level tests against the FastAPI app, using the keyless mock provider."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_reports_provider(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["provider"] == "mock"
        assert body["demo_mode"] is True
        assert body["api_key_configured"] is True


class TestIndex:
    def test_root_serves_demo_ui(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Language Feedback API" in response.text


class TestFeedbackEndpoint:
    def test_known_example_returns_structured_feedback(self, client):
        response = client.post(
            "/feedback",
            json={
                "sentence": "La chat noir est sur le table.",
                "target_language": "French",
                "native_language": "English",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["is_correct"] is False
        assert body["corrected_sentence"] == "Le chat noir est sur la table."
        assert len(body["errors"]) == 2
        assert body["difficulty"] == "A1"

    def test_response_includes_timing_header(self, client):
        response = client.post(
            "/feedback",
            json={
                "sentence": "Hola, como estas?",
                "target_language": "Spanish",
                "native_language": "English",
            },
        )
        assert response.status_code == 200
        assert "x-process-time-ms" in response.headers

    def test_empty_sentence_rejected(self, client):
        response = client.post(
            "/feedback",
            json={
                "sentence": "",
                "target_language": "Spanish",
                "native_language": "English",
            },
        )
        assert response.status_code == 422

    def test_oversized_sentence_rejected(self, client):
        response = client.post(
            "/feedback",
            json={
                "sentence": "palabra " * 200,
                "target_language": "Spanish",
                "native_language": "English",
            },
        )
        assert response.status_code == 422

    def test_missing_fields_rejected(self, client):
        response = client.post("/feedback", json={"sentence": "Hola"})
        assert response.status_code == 422


class TestProviderErrorMapping:
    """Provider failures surface as clean JSON errors with the right status."""

    @pytest.mark.parametrize(
        ("error", "expected_status"),
        [
            (ProviderAuthError("bad key"), 503),
            (ProviderRateLimitError("rate limited"), 503),
            (ProviderTimeoutError("timed out"), 504),
            (ProviderUnavailableError("upstream down"), 502),
        ],
    )
    def test_provider_errors_map_to_status_codes(
        self, client, monkeypatch, error, expected_status
    ):
        async def failing_get_feedback(request):
            raise error

        monkeypatch.setattr("app.main.get_feedback", failing_get_feedback)
        response = client.post(
            "/feedback",
            json={
                "sentence": "Hola amigo",
                "target_language": "Spanish",
                "native_language": "English",
            },
        )
        assert response.status_code == expected_status
        assert response.json()["detail"] == error.message
