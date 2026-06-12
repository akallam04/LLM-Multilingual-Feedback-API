"""Runtime configuration -- resolved from environment variables.

Provider resolution order:
1. LLM_PROVIDER if explicitly set (openai | anthropic | mock)
2. openai if OPENAI_API_KEY is set
3. anthropic if ANTHROPIC_API_KEY is set
4. mock (keyless demo mode with canned responses)
"""

import os
from dataclasses import dataclass
from functools import lru_cache

VALID_PROVIDERS = {"openai", "anthropic", "mock"}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "mock": "mock",
}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    timeout_seconds: float
    max_retries: int

    @property
    def api_key_configured(self) -> bool:
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        if self.provider == "anthropic":
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        return True  # mock needs no key


def _resolve_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in VALID_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got '{explicit}'"
            )
        return explicit
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = _resolve_provider()
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    return Settings(
        provider=provider,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def reset_settings() -> None:
    """Clear the cached settings. Used by tests and provider reset."""
    get_settings.cache_clear()
