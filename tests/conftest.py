"""Shared test fixtures."""

import pytest

from app.providers import FeedbackProvider, reset_provider


class StubProvider(FeedbackProvider):
    """Test double: returns queued payloads (or raises queued exceptions) in order."""

    name = "stub"

    def __init__(self, payloads):
        self.model = "stub-model"
        self.timeout_seconds = 1.0
        self.max_retries = 0
        self.payloads = list(payloads)
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    """Each test resolves provider and settings from a clean slate."""
    reset_provider()
    yield
    reset_provider()


@pytest.fixture
def stub_provider_factory(monkeypatch):
    """Patch get_feedback's provider lookup with a stub returning given payloads."""

    def _install(*payloads) -> StubProvider:
        stub = StubProvider(payloads)
        monkeypatch.setattr("app.feedback.get_provider", lambda: stub)
        return stub

    return _install
