"""LLM provider abstraction.

Three interchangeable providers sit behind one interface:
- OpenAIProvider     -- chat completions with JSON mode
- AnthropicProvider  -- messages with structured outputs (schema-enforced JSON)
- MockProvider       -- deterministic canned responses, needs no API key

Each provider returns a parsed dict; normalization and validation happen
downstream in app.feedback so every provider goes through the same guardrails.
"""

import copy
import json
import logging
from abc import ABC, abstractmethod

from app.config import Settings, get_settings, reset_settings
from app.models import FeedbackRequest
from app.prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger("feedback.providers")


class ProviderError(Exception):
    """Base class for provider failures. The message is safe to show to clients."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ProviderAuthError(ProviderError):
    """API key missing, invalid, or lacking permissions."""


class ProviderRateLimitError(ProviderError):
    """Upstream provider rate limit hit."""


class ProviderTimeoutError(ProviderError):
    """Upstream provider did not respond within the configured timeout."""


class ProviderUnavailableError(ProviderError):
    """Upstream provider unreachable or returned a server error."""


class ProviderResponseError(ProviderError):
    """Provider responded, but the payload was not usable JSON."""


def parse_json_payload(text: str | None) -> dict:
    """Extract a JSON object from raw model output.

    Tolerates markdown code fences and surrounding prose, since smaller
    models occasionally wrap JSON despite instructions not to.
    """
    if not text or not text.strip():
        raise ProviderResponseError("The language model returned an empty response.")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ProviderResponseError("The language model response contained no JSON object.")

    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("The language model returned malformed JSON.") from exc

    if not isinstance(data, dict):
        raise ProviderResponseError("The language model returned JSON that is not an object.")
    return data


class FeedbackProvider(ABC):
    """One LLM backend capable of producing raw feedback payloads."""

    name: str = "base"

    def __init__(self, settings: Settings):
        self.model = settings.model
        self.timeout_seconds = settings.timeout_seconds
        self.max_retries = settings.max_retries

    @abstractmethod
    async def complete(self, request: FeedbackRequest) -> dict:
        """Return the raw (pre-normalization) feedback payload as a dict."""


class OpenAIProvider(FeedbackProvider):
    name = "openai"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from openai import AsyncOpenAI

        # One shared client per process so connections are pooled.
        self._client = AsyncOpenAI(
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    async def complete(self, request: FeedbackRequest) -> dict:
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(request)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except openai.AuthenticationError as exc:
            raise ProviderAuthError(
                "OpenAI rejected the API key. Check OPENAI_API_KEY in your environment."
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise ProviderAuthError(
                "The OpenAI API key does not have access to this model."
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimitError(
                "OpenAI rate limit reached. Try again shortly."
            ) from exc
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(
                f"OpenAI did not respond within {self.timeout_seconds:.0f}s."
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableError("Could not reach the OpenAI API.") from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailableError(
                    "OpenAI returned a server error. Try again shortly."
                ) from exc
            raise ProviderError(f"OpenAI rejected the request: {exc.message}") from exc

        return parse_json_payload(response.choices[0].message.content)


# JSON schema enforced server-side by Anthropic structured outputs.
# Mirrors schema/response.schema.json minus constraints that structured
# outputs does not support (string length bounds).
ANTHROPIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_sentence": {"type": "string"},
        "is_correct": {"type": "boolean"},
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "correction": {"type": "string"},
                    "error_type": {
                        "type": "string",
                        "enum": [
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
                        ],
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["original", "correction", "error_type", "explanation"],
                "additionalProperties": False,
            },
        },
        "difficulty": {"type": "string", "enum": ["A1", "A2", "B1", "B2", "C1", "C2"]},
    },
    "required": ["corrected_sentence", "is_correct", "errors", "difficulty"],
    "additionalProperties": False,
}


class AnthropicProvider(FeedbackProvider):
    name = "anthropic"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    async def complete(self, request: FeedbackRequest) -> dict:
        import anthropic

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(request)}],
                output_config={
                    "format": {"type": "json_schema", "schema": ANTHROPIC_RESPONSE_SCHEMA}
                },
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(
                "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in your environment."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderAuthError(
                "The Anthropic API key does not have access to this model."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(
                "Anthropic rate limit reached. Try again shortly."
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(
                f"Anthropic did not respond within {self.timeout_seconds:.0f}s."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailableError("Could not reach the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailableError(
                    "Anthropic returned a server error. Try again shortly."
                ) from exc
            raise ProviderError(f"Anthropic rejected the request: {exc.message}") from exc

        text = next(
            (block.text for block in response.content if block.type == "text"), None
        )
        return parse_json_payload(text)


# Canned responses for keyless demo mode. Keyed by the exact sentence.
MOCK_RESPONSES: dict[str, dict] = {
    "Yo soy fue al mercado ayer.": {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": (
                    "You mixed two verb forms. 'Soy' is the present tense of 'ser' (to be), "
                    "and 'fue' is the past tense of 'ir' (to go). Since you're talking about "
                    "going to the market yesterday, you only need 'fui' (I went)."
                ),
            }
        ],
        "difficulty": "A2",
    },
    "La chat noir est sur le table.": {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": (
                    "'Chat' (cat) is masculine in French, so it takes the masculine "
                    "article 'le', not the feminine 'la'."
                ),
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": (
                    "'Table' is feminine in French, so it takes the feminine article "
                    "'la', not the masculine 'le'."
                ),
            },
        ],
        "difficulty": "A1",
    },
    "私は東京を住んでいます。": {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": (
                    "The verb 住む (to live) takes the particle に to indicate location "
                    "of residence, not を. Think of に as marking where you exist/live."
                ),
            }
        ],
        "difficulty": "A2",
    },
    "Ich habe gestern einen interessanten Film gesehen.": {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    },
    "Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.": {
        "corrected_sentence": (
            "Eu quero comprar um presente para minha irmã, mas não sei do que ela gosta."
        ),
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": (
                    "'Present/gift' in Portuguese is spelled 'presente' with an 's', not a 'z'."
                ),
            },
            {
                "original": "o que ela gosta",
                "correction": "do que ela gosta",
                "error_type": "grammar",
                "explanation": (
                    "The verb 'gostar' (to like) requires the preposition 'de'. So 'what "
                    "she likes' is 'do que ela gosta' (de + o que)."
                ),
            },
        ],
        "difficulty": "B1",
    },
    "Io ho mangiato una panino.": {
        "corrected_sentence": "Io ho mangiato un panino.",
        "is_correct": False,
        "errors": [
            {
                "original": "una panino",
                "correction": "un panino",
                "error_type": "gender_agreement",
                "explanation": (
                    "'Panino' (sandwich) is masculine in Italian, so it takes the "
                    "masculine article 'un', not the feminine 'una'."
                ),
            }
        ],
        "difficulty": "A1",
    },
    "वह एक अच्छा लड़की है।": {
        "corrected_sentence": "वह एक अच्छी लड़की है।",
        "is_correct": False,
        "errors": [
            {
                "original": "अच्छा लड़की",
                "correction": "अच्छी लड़की",
                "error_type": "gender_agreement",
                "explanation": (
                    "'लड़की' (girl) is feminine in Hindi, so the adjective must take the "
                    "feminine form 'अच्छी', not the masculine 'अच्छा'."
                ),
            }
        ],
        "difficulty": "A1",
    },
    "저는 학교에 가요.": {
        "corrected_sentence": "저는 학교에 가요.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    },
}


class MockProvider(FeedbackProvider):
    """Keyless demo provider with deterministic canned responses.

    Known demo sentences return real curated feedback. Anything else is
    echoed back as correct with a difficulty guess based on length. The
    /health endpoint and the web UI both flag when this provider is active
    so canned output is never mistaken for real analysis.
    """

    name = "mock"

    async def complete(self, request: FeedbackRequest) -> dict:
        canned = MOCK_RESPONSES.get(request.sentence.strip())
        if canned is not None:
            return copy.deepcopy(canned)

        word_count = len(request.sentence.split())
        if word_count <= 4:
            difficulty = "A1"
        elif word_count <= 8:
            difficulty = "A2"
        elif word_count <= 14:
            difficulty = "B1"
        else:
            difficulty = "B2"
        return {
            "corrected_sentence": request.sentence,
            "is_correct": True,
            "errors": [],
            "difficulty": difficulty,
        }


_PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}

_provider: FeedbackProvider | None = None


def get_provider() -> FeedbackProvider:
    """Return the process-wide provider, building it on first use."""
    global _provider
    if _provider is None:
        settings = get_settings()
        _provider = _PROVIDER_CLASSES[settings.provider](settings)
        if settings.provider == "mock":
            logger.warning(
                "No API key configured. Running in mock demo mode with canned "
                "responses. Set OPENAI_API_KEY or ANTHROPIC_API_KEY for real feedback."
            )
        else:
            logger.info(
                "Using provider=%s model=%s", settings.provider, settings.model
            )
    return _provider


def reset_provider() -> None:
    """Drop the cached provider and settings. Used by tests."""
    global _provider
    _provider = None
    reset_settings()
