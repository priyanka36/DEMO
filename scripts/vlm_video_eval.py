#!/usr/bin/env python3
"""Pilot VLM evaluation on HD-EPIC using sampled video frames.

This script intentionally keeps the video input grounded in the actual HD-EPIC clips:
- resolves each benchmark row to the corresponding raw MP4
- samples a small number of frames around the relevant time points
- sends the frames plus the question to a VLM over Ollama
- stores a JSON report with predicted answers and correctness

This is the first real multimodal baseline for the benchmark.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_ROOT = ROOT / "data" / "raw" / "HD-EPIC" / "Videos"
GEOMETRY_PATH = ROOT / "benchmark" / "geometry" / "questions.csv"
AFFORDANCE_PATH = ROOT / "benchmark" / "affordance" / "questions.csv"
DEFAULT_OUTPUT = ROOT / "results" / "vlm_video_eval.json"
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def parse_answer(value: str) -> str:
    return normalize_text(value)


def load_questions() -> List[dict]:
    rows: List[dict] = []
    for path in [GEOMETRY_PATH, AFFORDANCE_PATH]:
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as fh:
                rows.extend(list(csv.DictReader(fh)))
    return rows


def score_prediction(prediction: str, answer: str) -> bool:
    p = parse_answer(prediction)
    a = parse_answer(answer)
    if p == a:
        return True

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
        "unchanged": {"unchanged", "no change"},
    }
    for canonical, variants in equivalences.items():
        if p == canonical and a in variants:
            return True
        if a == canonical and p in variants:
            return True
    return False


def resolve_video_path(video_id: str) -> Path | None:
    if not video_id:
        return None
    # Prefer the exact video name prefix, and otherwise fall back to a recursive match.
    candidates = [
        Path(f"{video_id}.mp4"),
        Path(f"{video_id}.MP4"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for path in sorted(VIDEOS_ROOT.rglob("*.mp4")):
        if path.name.startswith(video_id):
            return path
        if path.stem == video_id:
            return path
    return None


def linspace(start: float, end: float, count: int) -> List[float]:
    if count <= 1:
        return [float(start)]
    step = (float(end) - float(start)) / (count - 1)
    return [float(start) + i * step for i in range(count)]


def adaptive_frame_count(row: dict) -> int:
    qtype = str(row.get("question_type", "")).strip().lower()
    start = float(row.get("time_start") or 0.0)
    end = float(row.get("time_end") or start)
    span = max(0.0, end - start)

    if qtype == "static_spatial":
        return 3

    if qtype == "visibility_occlusion":
        if span > 60.0:
            return 16
        if span > 10.0:
            return 12
        return 8

    if qtype in {"object_motion", "relation_change", "historical_current_update"}:
        if span > 60.0:
            return 8
        if span > 10.0:
            return 8
        if span > 2.0:
            return 6
        return 6

    if span > 60.0:
        return 12
    if span > 10.0:
        return 8
    return 6


def sample_times_for_row(row: dict, num_frames: int | None = None) -> List[float]:
    qtype = str(row.get("question_type", "")).strip().lower()
    start = float(row.get("time_start") or 0.0)
    end = float(row.get("time_end") or start)
    if num_frames is None:
        num_frames = 3
    num_frames = max(1, int(num_frames))

    if qtype == "static_spatial":
        if end <= start:
            end = start
        half_span = 0.75
        left = start - half_span
        right = start + half_span
        ts = linspace(left, right, num_frames)
        return [max(0.0, t) for t in ts]

    if qtype == "visibility_occlusion":
        if end <= start:
            end = start + 1.0
        if end <= start:
            end = start
        return linspace(start, end, num_frames)

    if qtype in {"object_motion", "relation_change", "historical_current_update"}:
        if end <= start:
            end = start + 1.0
        return linspace(start, end, num_frames)

    if end <= start:
        end = start + 1.0
    return linspace(start, end, num_frames)


def extract_frame(video_path: Path, t: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(max(0.0, t)),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        "-vf",
        "scale=384:384:force_original_aspect_ratio=decrease,pad=384:384:(ow-iw)/2:(oh-ih)/2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_sampled_frames(video_path: Path, row: dict, max_frames: int | None = None) -> List[Path]:
    times = sample_times_for_row(row, num_frames=max_frames)
    if not times:
        return []

    with tempfile.TemporaryDirectory(prefix="vqa_frames_") as tmpdir:
        outdir = Path(tmpdir)
        files: List[Path] = []
        for idx, t in enumerate(times):
            file_path = outdir / f"frame_{idx:02d}.jpg"
            extract_frame(video_path, float(t), file_path)
            files.append(file_path)
        return files


def build_prompt(row: dict) -> str:
    question = row.get("question", "")
    question_type = row.get("question_type", "unknown")
    return (
        f"Question type: {question_type}\n"
        f"Question: {question}\n\n"
        "Answer with exactly one short label from this set: yes, no, left, right, above, below, closer, farther, changed, unchanged. "
        "Do not explain or add extra text."
    )


def encode_image(path: Path) -> str:
    with path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def call_ollama_vlm(model: str, prompt: str, image_paths: List[Path], timeout: int = 300) -> str:
    if not image_paths:
        raise RuntimeError("No image frames were extracted for the sample.")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [encode_image(path) for path in image_paths],
        "options": {"temperature": 0, "num_predict": 32},
    }
    response = requests.post(OLLAMA_BASE.rstrip("/") + "/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    resp_text = str(data.get("response") or "")
    return resp_text.strip()


def evaluate_rows(rows: Iterable[dict], model: str, output_path: Path, limit: int | None = None, num_frames: int | None = None) -> Dict[str, Any]:
    rows = list(rows)
    if limit is not None:
        rows = rows[:limit]

    results: List[dict] = []
    for idx, row in enumerate(rows):
        video_id = str(row.get("video_id") or "").strip()
        video_path = resolve_video_path(video_id)
        if video_path is None:
            results.append({
                "index": idx,
                "video_id": video_id,
                "question_type": row.get("question_type", "unknown"),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "prediction": None,
                "correct": None,
                "status": "missing_video",
                "error": "No matching MP4 found",
            })
            continue

        sample_times = sample_times_for_row(row, num_frames=num_frames)
        with tempfile.TemporaryDirectory(prefix="vqa_eval_") as tmpdir:
            tmp_path = Path(tmpdir)
            files: List[Path] = []
            for file_idx, t in enumerate(sample_times):
                out_file = tmp_path / f"frame_{file_idx:02d}.jpg"
                extract_frame(video_path, float(t), out_file)
                files.append(out_file)

            prompt = build_prompt(row)
            try:
                prediction = call_ollama_vlm(model=model, prompt=prompt, image_paths=files, timeout=300)
                prediction = re.sub(r"\s+", " ", prediction).strip()
                correct = bool(prediction) and score_prediction(prediction, str(row.get("answer", "")))
                status = "ok"
                error = None
            except Exception as exc:  # pragma: no cover - model errors are reported in results.
                prediction = None
                correct = None
                status = "generation_failure"
                error = str(exc)

        result = {
            "index": idx,
            "video_id": video_id,
            "question_type": row.get("question_type", "unknown"),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "prediction": prediction,
            "correct": correct,
            "status": status,
            "sample_times": sample_times,
            "error": error,
            "n_frames": len(files) if "files" in locals() else 0,
        }
        results.append(result)

    total = len(results)
    valid = [r for r in results if r.get("status") == "ok"]
    correct = sum(1 for r in valid if r.get("correct") is True)
    report = {
        "model": model,
        "provider": "ollama_vlm",
        "num_questions": total,
        "valid_responses": len(valid),
        "total_correct": correct,
        "accuracy_over_all_questions": (correct / total) if total else 0.0,
        "accuracy_over_valid_responses": (correct / len(valid)) if valid else 0.0,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot VLM evaluation using sampled frames from HD-EPIC videos.")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Ollama VLM model name.")
    parser.add_argument("--start", type=int, default=0, help="Starting benchmark row index.")
    parser.add_argument("--end", type=int, default=9, help="Ending benchmark row index (inclusive).")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for number of rows.")
    parser.add_argument("--num-frames", type=int, default=3, help="Optional override for the number of sampled frames per question. The default is set to the local-model-safe value for current qwen2.5vl inference.")
    parser.add_argument("--valid-only", action="store_true", help="Evaluate only rows whose video_id resolves to a real MP4 in HD-EPIC.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the JSON report.")
    args = parser.parse_args()

    rows = load_questions()
    if not rows:
        raise RuntimeError("No benchmark rows found in benchmark/geometry or benchmark/affordance.")
    start = max(0, args.start)
    end = min(len(rows) - 1, args.end)
    selected = rows[start:end + 1]
    if args.valid_only:
        selected = [row for row in selected if resolve_video_path(str(row.get("video_id") or "")) is not None]
    if args.limit is not None:
        selected = selected[:args.limit]

    if not selected:
        raise RuntimeError("No video-backed rows remain after applying the filters.")

    report = evaluate_rows(selected, model=args.model, output_path=args.output, num_frames=args.num_frames)
    print(json.dumps({
        "model": args.model,
        "num_questions": report["num_questions"],
        "valid_responses": report["valid_responses"],
        "accuracy_over_all_questions": report["accuracy_over_all_questions"],
        "accuracy_over_valid_responses": report["accuracy_over_valid_responses"],
    }, indent=2))


if __name__ == "__main__":
    main()
