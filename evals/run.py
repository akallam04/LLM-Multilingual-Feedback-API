"""Eval harness: measures feedback quality against a labeled multilingual dataset.

Usage:
    python -m evals.run                # uses the configured provider
    python -m evals.run --concurrency 4 --output evals/results.md

Metrics:
- schema validity: response validates against schema/response.schema.json
- is_correct accuracy: model agrees with the label on whether the sentence has errors
- correction match: corrected sentence is one of the labeled acceptable corrections
- error type match: at least one predicted error category is in the labeled
  acceptable set (only scored on cases that contain errors)
- latency: p50 / p95 per request as observed by the harness

Run against a real provider (OPENAI_API_KEY or ANTHROPIC_API_KEY set). With the
mock provider the harness still runs, but the numbers only exercise the canned
demo responses and say nothing about real model quality.
"""

import argparse
import asyncio
import json
import statistics
import time
import unicodedata
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import jsonschema  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.feedback import get_feedback  # noqa: E402
from app.models import FeedbackRequest  # noqa: E402

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "evals" / "dataset.json"
RESPONSE_SCHEMA_PATH = ROOT / "schema" / "response.schema.json"


def normalize_sentence(text: str) -> str:
    """Normalization for fair comparison: NFC, plain apostrophes, collapsed spaces."""
    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("’", "'").replace("ʼ", "'")
    return " ".join(text.split())


async def run_case(case: dict, schema: dict, semaphore: asyncio.Semaphore) -> dict:
    request = FeedbackRequest(**case["request"])
    expected = case["expected"]

    async with semaphore:
        started = time.perf_counter()
        try:
            response = await get_feedback(request)
        except Exception as exc:  # noqa: BLE001 - evals must report, not crash
            return {
                "id": case["id"],
                "language": case["language"],
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        latency_ms = (time.perf_counter() - started) * 1000

    payload = response.model_dump()
    try:
        jsonschema.validate(payload, schema)
        schema_valid = True
    except jsonschema.ValidationError:
        schema_valid = False

    is_correct_match = payload["is_correct"] == expected["is_correct"]

    predicted = normalize_sentence(payload["corrected_sentence"])
    acceptable = {normalize_sentence(s) for s in expected["acceptable_corrections"]}
    correction_match = predicted in acceptable

    has_expected_errors = not expected["is_correct"]
    error_type_match = None
    if has_expected_errors:
        predicted_types = {e["error_type"] for e in payload["errors"]}
        error_type_match = bool(predicted_types & set(expected["acceptable_error_types"]))

    return {
        "id": case["id"],
        "language": case["language"],
        "failed": False,
        "schema_valid": schema_valid,
        "is_correct_match": is_correct_match,
        "correction_match": correction_match,
        "error_type_match": error_type_match,
        "latency_ms": latency_ms,
        "predicted_correction": payload["corrected_sentence"],
        "predicted_is_correct": payload["is_correct"],
    }


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.0f}%"


def build_report(results: list[dict], elapsed_s: float) -> str:
    settings = get_settings()
    ok = [r for r in results if not r["failed"]]
    failed = [r for r in results if r["failed"]]
    with_error_labels = [r for r in ok if r["error_type_match"] is not None]
    latencies = sorted(r["latency_ms"] for r in ok)

    lines = []
    lines.append("# Eval Results")
    lines.append("")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append(f"- Provider: {settings.provider}")
    lines.append(f"- Model: {settings.model}")
    lines.append(f"- Cases: {len(results)} ({len(failed)} failed to complete)")
    lines.append(f"- Wall time: {elapsed_s:.1f}s")
    if settings.provider == "mock":
        lines.append("")
        lines.append(
            "> Warning: run with the mock provider. These numbers exercise the "
            "canned demo responses only and do not measure real model quality."
        )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    lines.append(f"| Schema validity | {percent(sum(r['schema_valid'] for r in ok), len(ok))} |")
    lines.append(
        f"| is_correct accuracy | {percent(sum(r['is_correct_match'] for r in ok), len(ok))} |"
    )
    lines.append(
        f"| Correction match | {percent(sum(r['correction_match'] for r in ok), len(ok))} |"
    )
    type_hits = sum(r["error_type_match"] for r in with_error_labels)
    lines.append(f"| Error type match | {percent(type_hits, len(with_error_labels))} |")
    if latencies:
        p50 = statistics.median(latencies)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        lines.append(f"| Latency p50 | {p50:.0f} ms |")
        lines.append(f"| Latency p95 | {p95:.0f} ms |")
    lines.append("")
    lines.append("## By language")
    lines.append("")
    lines.append("| Language | Cases | is_correct accuracy | Correction match |")
    lines.append("|---|---|---|---|")
    languages = sorted({r["language"] for r in ok})
    for language in languages:
        group = [r for r in ok if r["language"] == language]
        lines.append(
            f"| {language} | {len(group)} "
            f"| {percent(sum(r['is_correct_match'] for r in group), len(group))} "
            f"| {percent(sum(r['correction_match'] for r in group), len(group))} |"
        )

    misses = [r for r in ok if not (r["is_correct_match"] and r["correction_match"])]
    if misses:
        lines.append("")
        lines.append("## Missed cases")
        lines.append("")
        lines.append("| Case | is_correct ok | Correction ok | Model output |")
        lines.append("|---|---|---|---|")
        for r in misses:
            output = r["predicted_correction"].replace("|", "\\|")
            lines.append(
                f"| {r['id']} | {r['is_correct_match']} | {r['correction_match']} | {output} |"
            )
    if failed:
        lines.append("")
        lines.append("## Failed requests")
        lines.append("")
        for r in failed:
            lines.append(f"- {r['id']}: {r['error']}")
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the feedback quality evals.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--output", default=str(ROOT / "evals" / "results.md"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    schema = json.loads(RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))

    settings = get_settings()
    print(f"Running {len(cases)} cases against provider={settings.provider} model={settings.model}")
    if settings.provider == "mock":
        print(
            "WARNING: mock provider active. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "to measure real model quality."
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(*(run_case(case, schema, semaphore) for case in cases))
    elapsed_s = time.perf_counter() - started

    report = build_report(list(results), elapsed_s)
    Path(args.output).write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
