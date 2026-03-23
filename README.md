# LLM-Powered Multilingual Feedback API

A FastAPI service that analyzes learner-written sentences and returns structured language feedback for language learning applications.

Given a sentence, a target language, and the learner’s native language, the API returns:
- a minimally corrected sentence
- a structured list of errors
- a correctness flag
- a CEFR difficulty estimate

This project was built for the Pangea Chat Gen AI Intern Task and was designed to balance accuracy, reliability, speed, and production feasibility.

## Overview

The goal of this project is not just to generate corrections, but to return schema-safe, learner-friendly feedback that could realistically power a language-learning product.

The implementation focuses on four priorities:
- accurate minimal corrections
- reliable structured JSON output
- low-latency, low-cost inference
- robust handling of imperfect model output

## Features

- `POST /feedback` endpoint for multilingual sentence feedback
- `GET /health` endpoint for health checks
- OpenAI-based structured feedback generation
- response normalization for invalid or inconsistent model output
- schema-safe response validation with Pydantic
- Dockerized local and containerized execution
- unit, schema, and integration test coverage

## Architecture

The service is intentionally small and simple:

- `app/main.py`  
  FastAPI app and route definitions

- `app/models.py`  
  Request and response models with constrained feedback fields

- `app/feedback.py`  
  Prompt definition, OpenAI call, output normalization, and response validation

- `tests/`  
  Unit, schema, and integration tests

This keeps the project easy to understand while still separating API structure, models, and feedback-generation logic.

## Prompt Strategy

The prompt is designed to produce structured learner feedback rather than free-form grammar commentary.

Key prompt goals:
- preserve the learner’s original meaning and style
- make minimal corrections instead of rewriting the sentence
- classify errors using only the allowed schema categories
- write explanations in the learner’s native language
- assign CEFR difficulty based on sentence complexity, not correctness
- return JSON only

The prompt also explicitly defines the correct behavior for already-correct sentences:
- `is_correct = true`
- `errors = []`
- `corrected_sentence` must exactly match the original input

## Reliability and Validation

A raw LLM response is not treated as trusted output.

After generation, the response is normalized and validated before returning it to the client. This includes:
- normalizing near-match error types to allowed schema values
- normalizing CEFR difficulty values to valid labels
- enforcing consistency rules for `is_correct`, `errors`, and `corrected_sentence`
- filling safe fallback values for malformed or incomplete output
- validating the final payload against constrained Pydantic models

This post-processing layer improves schema compliance and makes the service more robust to imperfect model output.

## Model Choice

The current implementation uses `gpt-4o-mini`.

I chose this model as a practical tradeoff between:
- latency
- cost
- multilingual capability
- structured output reliability

For this task, production feasibility matters alongside raw output quality. A lightweight model with a stronger prompt and a validation layer is a better engineering tradeoff than using a larger model without guardrails.

I would only switch to a larger model if testing showed clear accuracy gains on multilingual edge cases that justified the added cost and latency.

## Testing

The test suite covers:
- standard incorrect sentence handling
- correct sentence behavior
- multiple simultaneous errors
- non-Latin script input
- schema compliance
- normalization of invalid error types
- normalization of invalid difficulty values
- consistency enforcement when `is_correct = true`
- fallback handling for malformed `errors`
- fallback handling for missing explanations

The project includes:
- unit tests with mocked OpenAI responses
- schema tests against the provided JSON schemas
- integration tests that make real API calls when an API key is available

## Run Locally

```bash
git clone https://github.com/akallam04/intern-task-2026.git
cd intern-task-2026

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add OPENAI_API_KEY to .env

uvicorn app.main:app --reload
