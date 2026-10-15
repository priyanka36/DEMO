#!/usr/bin/env python3
"""Run a small video-grounded VLM benchmark suite for HD-EPIC."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

from vlm_video_eval import evaluate_rows, load_questions, resolve_video_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "vlm_models.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "vlm_suite"


def sanitize_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace("-", "_")


def load_config(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    return [entry["name"] for entry in config.get("models", [])]


def is_model_available(model: str) -> bool:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False

    names = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        names.add(parts[0])
    return model in names


def ensure_model(model: str, pull_missing: bool) -> bool:
    if is_model_available(model):
        return True
    if not pull_missing:
        return False
    print(f"Pulling missing model: {model}")
    subprocess.run(["ollama", "pull", model], check=True)
    return is_model_available(model)


def select_video_rows(rows: Iterable[dict], start: int, end: int, valid_only: bool) -> List[dict]:
    all_rows = list(rows)
    selected = all_rows[start : end + 1]
    if not valid_only:
        return selected

    valid_in_window = [
        row for row in selected if resolve_video_path(str(row.get("video_id") or "")) is not None
    ]
    if valid_in_window:
        return valid_in_window

    fallback = [
        row for idx, row in enumerate(all_rows)
        if idx >= start and resolve_video_path(str(row.get("video_id") or "")) is not None
    ]
    if fallback:
        return fallback

    return []


def run_suite(models: List[str], start: int, end: int, limit: int | None, valid_only: bool, output_dir: Path, pull_missing: bool, num_frames: int | None = None) -> Dict[str, Any]:
    rows = load_questions()
    if not rows:
        raise RuntimeError("No benchmark rows found.")

    selected = select_video_rows(rows, start, end, valid_only)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError("No rows remain after applying the index and valid-video filters.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    for model in models:
        if not ensure_model(model, pull_missing=pull_missing):
            summary.append({
                "model": model,
                "status": "missing",
                "num_questions": len(selected),
                "valid_responses": 0,
                "accuracy_over_all_questions": 0.0,
                "accuracy_over_valid_responses": 0.0,
                "note": "Model is not installed and pull_missing was not enabled.",
            })
            continue

        out_path = output_dir / f"{sanitize_model_name(model)}.json"
        report = evaluate_rows(selected, model=model, output_path=out_path, num_frames=num_frames)
        summary.append({
            "model": model,
            "status": "ok",
            "num_questions": report["num_questions"],
            "valid_responses": report["valid_responses"],
            "accuracy_over_all_questions": report["accuracy_over_all_questions"],
            "accuracy_over_valid_responses": report["accuracy_over_valid_responses"],
            "output": str(out_path),
        })

    suite_report = {
        "suite_name": "hd_epic_video_vlm_baselines",
        "valid_only": valid_only,
        "start": start,
        "end": end,
        "limit": limit,
        "num_selected_rows": len(selected),
        "models": summary,
    }
    summary_path = output_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")
    return suite_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small video-grounded VLM suite on HD-EPIC benchmark rows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON config with model names.")
    parser.add_argument("--model", action="append", default=None, help="Override config with one or more model names; can be repeated.")
    parser.add_argument("--start", type=int, default=0, help="Starting benchmark row index.")
    parser.add_argument("--end", type=int, default=9, help="Ending benchmark row index (inclusive).")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of rows evaluated per model.")
    parser.add_argument("--valid-only", action="store_true", help="Only evaluate rows whose video_id resolves to a real MP4.")
    parser.add_argument("--num-frames", type=int, default=None, help="Optional override for sampled frames per question; if omitted, the adaptive question-aware policy is used.")
    parser.add_argument("--pull-missing", action="store_true", help="Automatically run `ollama pull` for missing models before evaluation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for per-model JSON reports and suite summary.")
    args = parser.parse_args()

    models = args.model or load_config(args.config)
    if not models:
        raise RuntimeError("No models were supplied; add --model or configure the JSON file.")

    report = run_suite(
        models=models,
        start=args.start,
        end=args.end,
        limit=args.limit,
        valid_only=args.valid_only,
        output_dir=args.output_dir,
        pull_missing=args.pull_missing,
        num_frames=args.num_frames,
    )

    print(json.dumps(report["models"], indent=2))


if __name__ == "__main__":
    main()
