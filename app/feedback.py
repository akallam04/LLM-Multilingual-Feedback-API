import json

from openai import AsyncOpenAI

from app.models import FeedbackRequest, FeedbackResponse

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

SYSTEM_PROMPT = """You are a precise language-learning feedback assistant.

You analyze one learner-written sentence in a target language and return structured feedback.

Requirements:
1. Make only minimal corrections. Preserve the learner's meaning and style.
2. If the sentence is already correct, return:
   - is_correct = true
   - corrected_sentence exactly equal to the original sentence
   - errors = []
3. If the sentence has errors, return:
   - is_correct = false
   - a minimally corrected sentence
   - one or more errors
4. Each error must include:
   - original
   - correction
   - error_type
   - explanation
5. explanation must be written in the learner's native language, not the target language, unless both languages are the same.
6. Explanations must be concise, friendly, and educational.
7. Use only these exact error_type values:
   grammar, spelling, word_choice, punctuation, word_order, missing_word, extra_word, conjugation, gender_agreement, number_agreement, tone_register, other
8. Use only these exact difficulty values:
   A1, A2, B1, B2, C1, C2
9. Difficulty is based on sentence complexity, not correctness.
10. Return only valid JSON. No markdown. No extra text.

Return JSON with exactly this shape:
{
  "corrected_sentence": "string",
  "is_correct": true,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "grammar",
      "explanation": "string"
    }
  ],
  "difficulty": "A1"
}
"""


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
    client = AsyncOpenAI()

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}\n\n"
        f"Important: Write every explanation in {request.native_language}."
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    normalized_data = normalize_feedback_payload(data, request)
    return FeedbackResponse(**normalized_data)