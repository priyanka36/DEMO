#!/usr/bin/env python3
"""Robust per-chunk evaluation runner for local Ollama models.

This script is intentionally more conservative than the one-shot evaluator:
- runs only a selected question range
- retries timeouts and transient errors
- saves progress after every question
- supports resume from a partially finished file
- warms the model before evaluation
- uses low temperature and short output length to reduce runtime variance
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

import evaluate_benchmark as eb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "model_run.json"
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def load_all_questions() -> List[dict]:
    geometry = eb.GEOMETRY_PATH
    affordance = eb.AFFORDANCE_PATH
    rows: List[dict] = []
    if geometry.exists():
        rows.extend(eb.load_questions(geometry))
    if affordance.exists():
        rows.extend(eb.load_questions(affordance))
    return rows


def coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
    if isinstance(value, dict):
        text = value.get("content") or value.get("text") or value.get("value")
        if isinstance(text, str):
            return text
    return None


def _extract_valid_short_label(candidate: str) -> str | None:
    cleaned = str(candidate).strip()
    if not cleaned:
        return None
    lower = cleaned.lower()
    for label in [
        "yes", "no", "left", "right", "above", "below",
        "closer", "farther", "changed", "unchanged",
        "object_only", "observer_only", "both", "neither",
        "y", "n", "true", "false",
        "to the left", "to the right", "became closer", "became farther",
    ]:
        if re.search(rf"\b{re.escape(label)}\b", lower):
            return label
    return None


def _is_valid_short_label(candidate: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(candidate).strip())
    if not cleaned:
        return False
    if re.search(r"\s", cleaned) and len(cleaned.split()) > 3:
        return False
    normalized = cleaned.lower().strip(". ,;:()[]{}\"")
    if normalized in {"yes", "no", "left", "right", "above", "below",
                       "closer", "farther", "changed", "unchanged",
                       "object_only", "observer_only", "both", "neither",
                       "y", "n", "true", "false"}:
        return True
    if normalized in {"to the left", "to the right", "became closer", "became farther"}:
        return True
    return _extract_valid_short_label(cleaned) is not None


def call_ollama(prompt: str, model: str, timeout: int = 180, max_tokens: int = 512) -> str:
    url = OLLAMA_BASE.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "think": False,
        },
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unsupported Ollama response: {data!r}")

    candidates: List[str] = []
    for key in ("response", "content"):
        text = coerce_text(data.get(key))
        if text:
            candidates.append(text)
    message = data.get("message")
    if isinstance(message, dict):
        text = coerce_text(message.get("content"))
        if text:
            candidates.append(text)

    thinking = data.get("thinking")
    if thinking is not None and not candidates:
        thinking_text = str(thinking)
        for marker in ("final answer:", "final answer is:", "answer:", "answer is:"):
            idx = thinking_text.lower().rfind(marker)
            if idx >= 0:
                suffix = thinking_text[idx + len(marker):].strip()
                if suffix:
                    candidates.append(suffix)
                    break
        if not candidates:
            for pattern in (
                r"final answer(?:\s+is)?\s*[:\-]?\s*([^\n]+)",
                r"answer(?:\s+is)?\s*[:\-]?\s*([^\n]+)",
            ):
                match = re.search(pattern, thinking_text, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    candidates.append(match.group(1).strip())
                    break

    for candidate in candidates:
        cleaned = str(candidate).strip()
        label = _extract_valid_short_label(cleaned)
        if label is not None:
            return label
        if _is_valid_short_label(cleaned):
            return cleaned

    if data.get("done_reason") == "length" or not str(data.get("response", "")).strip():
        raise RuntimeError(f"generation_truncated:{data.get('done_reason', 'unknown')}")

    raise RuntimeError(f"Unsupported Ollama response: {data!r}")


def warmup_model(model: str, timeout: int = 60) -> None:
    try:
        call_ollama("Say OK", model=model, timeout=timeout, max_tokens=8)
    except Exception:
        # Warmup is best-effort; a model that rejects warmup should still be evaluated.
        pass


def load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"results": []}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {"results": []}


def save_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    temp.replace(path)


def evaluate_chunk(rows: List[dict], model: str, output_path: Path, start_index: int, end_index: int, timeout: int, max_retries: int, max_tokens: int) -> Dict[str, Any]:
    report = load_report(output_path)
    report.setdefault("model", model)
    report.setdefault("provider", "ollama")
    report.setdefault("start_index", start_index)
    report.setdefault("end_index", end_index)
    report.setdefault("timeout", timeout)
    report.setdefault("max_retries", max_retries)
    report.setdefault("max_tokens", max_tokens)
    report.setdefault("results", [])

    completed = {int(item.get("index")) for item in report["results"] if isinstance(item.get("index"), int)}
    results = report["results"]

    for offset, row in enumerate(rows):
        index = start_index + offset
        if index in completed:
            continue

        prompt = eb.build_prompt(row)
        last_error: str | None = None
        prediction: str | None = None
        status = "ok"

        for attempt in range(1, max_retries + 3):
            token_budget = max_tokens if attempt == 1 else 1024
            try:
                prediction = call_ollama(prompt, model=model, timeout=timeout, max_tokens=token_budget)
                status = "ok"
                break
            except Exception as exc:  # pragma: no cover - network/model failures are expected here
                last_error = str(exc)
                if "generation_truncated" in str(exc):
                    if attempt == 1:
                        time.sleep(2)
                        continue
                    status = "generation_failure"
                    prediction = None
                    break
                if attempt <= max_retries:
                    time.sleep(min(8 * attempt, 30))
                    continue
                status = "generation_failure"
                prediction = None
                break

        if status == "generation_failure":
            result = {
                "index": index,
                "question_type": row.get("question_type", "unknown"),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "prediction": None,
                "correct": None,
                "error": last_error,
                "status": "generation_failure",
            }
        else:
            result = {
                "index": index,
                "question_type": row.get("question_type", "unknown"),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "prediction": prediction,
                "correct": bool(prediction and eb.score_prediction(prediction, row.get("answer", ""))),
                "error": None,
                "status": "ok",
            }
        results.append(result)
        report["results"] = results
        report["processed_count"] = len(results)
        valid = [item for item in results if item.get("status") == "ok"]
        generation_failures = [item for item in results if item.get("status") == "generation_failure"]
        report["total_questions"] = len(results)
        report["valid_responses"] = len(valid)
        report["generation_failures"] = len(generation_failures)
        report["valid_response_rate"] = (len(valid) / len(results)) if results else 0.0
        report["accuracy_over_all_questions"] = (
            sum(1 for item in results if item.get("correct") is True) / len(results)
            if results else 0.0
        )
        report["accuracy_over_valid_responses"] = (
            sum(1 for item in valid if item.get("correct") is True) / len(valid)
            if valid else 0.0
        )
        report["overall_accuracy"] = report["accuracy_over_all_questions"]
        save_report(report, output_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust chunked evaluation runner for local Ollama models.")
    parser.add_argument("--model", required=True, help="Ollama model name, for example phi4-mini:3.8b or llama3.2:3b")
    parser.add_argument("--start", type=int, default=0, help="Starting question index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Ending question index (inclusive). Defaults to last row.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the incremental report")
    parser.add_argument("--timeout", type=int, default=180, help="Per-question timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count for transient failures")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generated tokens per call; length-truncated responses are retried with 1024 tokens")
    parser.add_argument("--warmup", action="store_true", help="Send a tiny warmup prompt before running the chunk")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing result file")
    args = parser.parse_args()

    rows = load_all_questions()
    if not rows:
        raise RuntimeError("No benchmark rows found in benchmark/geometry or benchmark/affordance.")

    start = max(0, args.start)
    end = len(rows) - 1 if args.end is None else min(len(rows) - 1, args.end)
    if end < start:
        raise ValueError(f"Invalid range: start={start} end={end}")

    selected = rows[start:end + 1]

    if args.warmup:
        warmup_model(args.model, timeout=min(args.timeout, 60))

    if args.resume:
        report = load_report(args.output)
        if report.get("results"):
            completed = {int(item.get("index")) for item in report["results"] if isinstance(item.get("index"), int)}
            selected = [row for index, row in enumerate(rows[start:end + 1], start=start) if index not in completed]
            if not selected:
                print(f"Nothing to do; {args.output} already contains all requested rows.")
                return

    final_report = evaluate_chunk(
        rows=selected,
        model=args.model,
        output_path=args.output,
        start_index=start,
        end_index=end,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
    )

    print(f"Finished {args.model} on rows {start}..{end}.")
    print(f"Saved report to {args.output}")
    print(f"Overall accuracy: {final_report.get('overall_accuracy', 0.0):.3f}")


if __name__ == "__main__":
    main()
