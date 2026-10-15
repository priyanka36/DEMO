#!/usr/bin/env python3
"""Evaluate benchmark questions with an LLM or dry-run prompt generator.

This script loads the geometry and affordance benchmark CSV files, formats each
question into a closed-form prompt, and then either:
- calls an LLM through OpenAI or Ollama, or
- prints the prompt in dry-run mode for manual inspection.

It reports per-question-type accuracy against the benchmark answer key.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = ROOT / "benchmark" / "geometry" / "questions.csv"
AFFORDANCE_PATH = ROOT / "benchmark" / "affordance" / "questions.csv"
DEFAULT_OUTPUT = ROOT / "results" / "llm_eval_report.json"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def parse_answer(value: str) -> str:
    return normalize_text(value)


def load_questions(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def build_prompt(row: dict) -> str:
    question = row.get("question", "")
    question_type = row.get("question_type", "unknown")
    question_text = (
        f"Question type: {question_type}\n"
        f"Question: {question}\n\n"
        "Return only a single short final answer, such as yes/no, left/right, above/below, "
        "closer/farther, changed/unchanged. Do not explain your reasoning."
    )
    return question_text


def allowed_answer_set(row: dict) -> List[str]:
    answer = str(row.get("answer", "")).strip()
    return [answer]


def call_openai(prompt: str, model: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    if hasattr(response, "output"):
        chunks = []
        for item in response.output:
            if hasattr(item, "content"):
                for part in item.content:
                    if hasattr(part, "text"):
                        chunks.append(part.text)
        if chunks:
            return "\n".join(chunks)
    raise RuntimeError(f"Unsupported OpenAI response object: {type(response)}")


def call_ollama(prompt: str, model: str) -> str:
    if requests is None:
        raise RuntimeError("requests package is not installed.")
    url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"think": False},
    }
    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unsupported Ollama response: {data}")

    def coerce_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            chunks = []
            for item in value:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value")
                    if isinstance(text, str):
                        chunks.append(text)
            if chunks:
                return "".join(chunks)
        return None

    # Common Ollama response shapes: direct text, chat message, or reasoning-only model payloads.
    candidates = []
    for key in ("response", "content"):
        text = coerce_text(data.get(key))
        if text:
            candidates.append(text)

    message = data.get("message")
    if isinstance(message, dict):
        text = coerce_text(message.get("content"))
        if text:
            candidates.append(text)

    if not candidates and isinstance(data.get("thinking"), str):
        thinking = data["thinking"]
        patterns = [
            r"final\s+answer(?:\s+is)?\s*[:\-]?\s*(.+)",
            r"answer(?:\s+is)?\s*[:\-]?\s*(.+)",
            r"(?:^|\n)\s*(?:final|answer)\s*[:\-]?\s*(.+)",
        ]
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, thinking, flags=re.IGNORECASE | re.DOTALL):
                text = match.group(1).strip()
                text = re.sub(r"\s+", " ", text)
                if text:
                    matches.append(text)
        if matches:
            candidates.extend(matches)
        if not candidates:
            # Some reasoning-heavy models produce a trace without a final answer.
            # Use a non-crashing fallback so the evaluation loop keeps running.
            candidates.append("unknown")

    for text in candidates:
        text = str(text).strip()
        if text:
            return text

    raise RuntimeError(f"Unsupported Ollama response: {data}")


def score_prediction(answer: str, expected: str) -> bool:
    a = parse_answer(answer)
    e = parse_answer(expected)
    if a == e:
        return True

    # a few practical equivalences for close-ended benchmark labels
    equivalences = {
        "yes": {"yes", "y", "true"},
        "no": {"no", "n", "false"},
        "left": {"left", "to the left"},
        "right": {"right", "to the right"},
        "above": {"above", "up"},
        "below": {"below", "down"},
        "closer": {"closer", "became closer"},
        "farther": {"farther", "became farther"},
        "changed": {"changed", "relation changed"},
        "unchanged": {"unchanged", "relation unchanged"},
    }
    for canonical, variants in equivalences.items():
        if a == canonical and e in variants:
            return True
        if e == canonical and a in variants:
            return True
    return False


def evaluate_rows(rows: Iterable[dict], provider: str, model: str, limit: int | None = None) -> Tuple[List[dict], Dict[str, float]]:
    rows = list(rows)
    if limit is not None:
        rows = rows[:limit]

    predictions = []
    for row in rows:
        prompt = build_prompt(row)
        if provider == "dry-run":
            prediction = "[DRY_RUN]"
        elif provider == "openai":
            prediction = call_openai(prompt, model)
        elif provider == "ollama":
            prediction = call_ollama(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        predictions.append({
            "question": row.get("question", ""),
            "question_type": row.get("question_type", "unknown"),
            "answer": row.get("answer", ""),
            "prediction": prediction,
            "correct": score_prediction(prediction, row.get("answer", "")),
            "prompt": prompt,
        })

    by_type: Dict[str, List[bool]] = defaultdict(list)
    for item in predictions:
        by_type[item["question_type"]].append(item["correct"])

    metrics: Dict[str, float] = {}
    for qtype, values in sorted(by_type.items()):
        metrics[qtype] = sum(values) / len(values) if values else 0.0

    return predictions, metrics


def write_report(path: Path, predictions: List[dict], metrics: Dict[str, float], provider: str, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "provider": provider,
        "model": model,
        "num_questions": len(predictions),
        "accuracy_by_type": metrics,
        "overall_accuracy": sum(1 for p in predictions if p["correct"]) / len(predictions) if predictions else 0.0,
        "predictions": predictions,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the HD-EPIC benchmark with an LLM.")
    parser.add_argument("--geometry", type=Path, default=GEOMETRY_PATH, help="Path to the geometry benchmark CSV.")
    parser.add_argument("--affordance", type=Path, default=AFFORDANCE_PATH, help="Path to the affordance benchmark CSV.")
    parser.add_argument("--provider", choices=["dry-run", "openai", "ollama"], default="dry-run", help="LLM provider to call.")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gpt-4o-mini"), help="LLM model name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on rows to evaluate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the JSON evaluation report.")
    args = parser.parse_args()

    rows = load_questions(args.geometry) + load_questions(args.affordance)
    predictions, metrics = evaluate_rows(rows, args.provider, args.model, args.limit)
    write_report(args.output, predictions, metrics, args.provider, args.model)

    print(f"Evaluated {len(predictions)} benchmark questions with provider={args.provider} model={args.model}")
    print("Accuracy by type:")
    for qtype, score in metrics.items():
        print(f"  {qtype}: {score:.3f}")
    print(f"Overall accuracy: {sum(1 for p in predictions if p['correct']) / len(predictions):.3f}")


if __name__ == "__main__":
    main()
