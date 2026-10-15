#!/usr/bin/env python3
"""Build an affordance-aware extension for the HD-EPIC real benchmark pilot.

This script augments the existing geometry-first benchmark with a richer set of
functional and affordance-aware questions using only the real object-track CSV.
The goal is to stay compact, interpretable, and grounded in real motion metadata.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path("/home/priyanka/VQA")
TRACKS_PATH = ROOT / "benchmark" / "geometry" / "all_object_4d_tracks.csv"
OUT_DIR = ROOT / "benchmark" / "affordance"

FIXTURE_LABELS = {
    "sink": ["sink"],
    "counter": ["counter"],
    "stove": ["stove", "hob"],
    "cutting_board": ["board", "chopping"],
    "storage": ["cupboard", "drawer", "cabinet"],
}

OBJECT_TO_AFFORDANCES = {
    "mug": ["sink", "counter", "storage"],
    "cup": ["sink", "counter", "storage"],
    "bowl": ["counter", "sink", "storage"],
    "plate": ["counter", "sink", "storage"],
    "knife": ["counter", "cutting_board", "sink"],
    "pan": ["stove", "sink", "counter"],
    "pot": ["stove", "sink", "counter"],
    "spoon": ["sink", "counter"],
    "fork": ["sink", "counter"],
    "cutting_board": ["counter", "cutting_board"],
    "box": ["storage", "counter"],
    "bottle": ["counter", "sink"],
}


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_tracks(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def distance3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def fixture_centroid(points: Iterable[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    pts = list(points)
    if not pts:
        return (0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def classify_object(obj_name: str) -> str:
    text = obj_name.lower()
    for label in OBJECT_TO_AFFORDANCES:
        if label in text:
            return label
    return "generic"


def normalize_fixture(fixture: str) -> str:
    f = fixture.lower()
    for key, variants in FIXTURE_LABELS.items():
        if any(v in f for v in variants):
            return key
    return "other"


def set_object_affordances(obj_name: str) -> List[str]:
    key = classify_object(obj_name)
    return OBJECT_TO_AFFORDANCES.get(key, ["counter", "sink"])


def choose_top_objects(video_rows: List[dict], max_count: int = 3) -> List[str]:
    by_obj: Dict[str, List[dict]] = defaultdict(list)
    for row in video_rows:
        by_obj[row["object_id"]].append(row)
    return [k for k, _ in sorted(by_obj.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_count]]


def object_center(rows: List[dict]) -> Tuple[float, float, float]:
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    zs = [float(r["z"]) for r in rows]
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def object_displacement(rows: List[dict]) -> float:
    if len(rows) < 2:
        return 0.0
    start = (float(rows[0]["x"]), float(rows[0]["y"]), float(rows[0]["z"]))
    end = (float(rows[-1]["x"]), float(rows[-1]["y"]), float(rows[-1]["z"]))
    return distance3d(start, end)


def relation_label_from_points(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> str:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "aligned"
    if abs(dx) > abs(dy):
        return "left" if dx > 0 else "right"
    return "above" if dy < 0 else "below"


def relation_pair_label(rows_a: List[dict], rows_b: List[dict], use_last: bool = False) -> str:
    a_row = rows_a[-1] if use_last else rows_a[0]
    b_row = rows_b[-1] if use_last else rows_b[0]
    a_pt = (float(a_row["x"]), float(a_row["y"]), float(a_row["z"]))
    b_pt = (float(b_row["x"]), float(b_row["y"]), float(b_row["z"]))
    return relation_label_from_points(a_pt, b_pt)


def nearest_fixture(center: Tuple[float, float, float], fixture_centers: Dict[str, Tuple[float, float, float]], compatible: List[str]) -> str | None:
    candidates = [(name, distance3d(center, fixture_centers[name])) for name in compatible if name in fixture_centers]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def object_is_occluded(rows: List[dict], gap_threshold: float = 8.0) -> bool:
    if len(rows) < 2:
        return False
    times = [float(r["timestamp"]) for r in rows]
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return any(g > gap_threshold for g in gaps)


def make_question(question_type: str, difficulty: str, question: str, answer: str, video_id: str, target: str, reference: str = "", evidence: dict | None = None) -> dict:
    return {
        "video_id": video_id,
        "question_type": question_type,
        "difficulty": difficulty,
        "target_object": target,
        "reference_object": reference,
        "question": question,
        "answer": answer,
        "evidence": evidence or {},
    }


def build_affordance_questions(video_id: str, rows: List[dict]) -> List[dict]:
    by_obj: Dict[str, List[dict]] = defaultdict(list)
    fixture_points: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        by_obj[row["object_id"]].append(row)
        fixture = normalize_fixture(row["fixture"])
        if fixture != "other":
            fixture_points[fixture].append((float(row["x"]), float(row["y"]), float(row["z"])))

    fixture_centers = {name: fixture_centroid(vals) for name, vals in fixture_points.items()}
    selected_objects = choose_top_objects(rows, max_count=3)
    qrows: List[dict] = []

    for obj_id in selected_objects:
        obj_rows = by_obj[obj_id]
        obj_name = obj_rows[0]["object_name"]
        category = classify_object(obj_name)
        compatible = set_object_affordances(obj_name)
        center = object_center(obj_rows)
        latest_fixture = normalize_fixture(obj_rows[-1]["fixture"])
        start_fixture = normalize_fixture(obj_rows[0]["fixture"])
        disp = object_displacement(obj_rows)
        nearest_compatible = nearest_fixture(center, fixture_centers, compatible)

        if nearest_compatible:
            qrows.append(
                make_question(
                    "affordance_spatial",
                    "medium",
                    f"Which fixture is most compatible with the {obj_name}'s current location?",
                    nearest_compatible,
                    video_id,
                    obj_name,
                    evidence={
                        "current_fixture": latest_fixture,
                        "compatible_fixtures": compatible,
                        "nearest_compatible_fixture": nearest_compatible,
                        "position": [round(center[0], 3), round(center[1], 3), round(center[2], 3)],
                    },
                )
            )

        if category in {"knife", "mug", "cup", "pan", "pot", "bowl", "plate"}:
            plausible = "yes" if nearest_compatible in {"sink", "counter", "stove", "cutting_board"} else "no"
            qrows.append(
                make_question(
                    "affordance_spatial",
                    "easy",
                    f"Is the {obj_name} positioned in a place where it could plausibly be used for its typical task?",
                    plausible,
                    video_id,
                    obj_name,
                    evidence={
                        "category": category,
                        "current_fixture": latest_fixture,
                        "compatible_fixtures": compatible,
                        "nearest_compatible_fixture": nearest_compatible,
                    },
                )
            )

        if latest_fixture != start_fixture and nearest_compatible:
            qrows.append(
                make_question(
                    "historical_functional_state",
                    "medium",
                    f"Did the functional context of {obj_name} change from {start_fixture} to {latest_fixture}?",
                    "yes",
                    video_id,
                    obj_name,
                    evidence={
                        "start_fixture": start_fixture,
                        "latest_fixture": latest_fixture,
                        "nearest_compatible_fixture": nearest_compatible,
                    },
                )
            )

        if object_is_occluded(obj_rows):
            qrows.append(
                make_question(
                    "persistent_affordance_occlusion",
                    "medium",
                    f"The {obj_name} became temporarily occluded after movement. Is its affordance at the current fixture still reasonable?",
                    "yes" if nearest_compatible in {"sink", "counter", "stove", "cutting_board"} else "no",
                    video_id,
                    obj_name,
                    evidence={
                        "occluded": True,
                        "current_fixture": latest_fixture,
                        "nearest_compatible_fixture": nearest_compatible,
                    },
                )
            )

        if disp < 0.5 and nearest_compatible:
            qrows.append(
                make_question(
                    "ego_motion_affordance_stability",
                    "medium",
                    f"The object appears somewhat closer after motion, but did its functional relationship to the relevant fixture actually change?",
                    "no",
                    video_id,
                    obj_name,
                    evidence={
                        "displacement": round(disp, 3),
                        "nearest_compatible_fixture": nearest_compatible,
                        "current_fixture": latest_fixture,
                    },
                )
            )

        if disp > 0.5 and nearest_compatible:
            qrows.append(
                make_question(
                    "counterfactual_reasoning",
                    "hard",
                    f"If the observer moved but {obj_name} stayed fixed, should the world-level affordance relation to its fixture meaningfully change?",
                    "no",
                    video_id,
                    obj_name,
                    evidence={
                        "displacement": round(disp, 3),
                        "current_fixture": latest_fixture,
                        "nearest_compatible_fixture": nearest_compatible,
                    },
                )
            )

    # pairwise questions
    for a_id, b_id in combinations(selected_objects, 2):
        a_rows = by_obj[a_id]
        b_rows = by_obj[b_id]
        a_name = a_rows[0]["object_name"]
        b_name = b_rows[0]["object_name"]
        start_rel = relation_pair_label(a_rows, b_rows, use_last=False)
        end_rel = relation_pair_label(a_rows, b_rows, use_last=True)
        disp_a = object_displacement(a_rows)
        disp_b = object_displacement(b_rows)

        if start_rel != end_rel:
            qrows.append(
                make_question(
                    "interaction_consequence",
                    "medium",
                    f"After {a_name} moved, which previously valid spatial relation is no longer true with {b_name}?",
                    f"{a_name} was {start_rel} {b_name}",
                    video_id,
                    a_name,
                    reference=b_name,
                    evidence={
                        "start_relation": start_rel,
                        "end_relation": end_rel,
                        "a_displacement": round(disp_a, 3),
                        "b_displacement": round(disp_b, 3),
                    },
                )
            )

        if disp_a > 0.5 and disp_b < 0.2:
            revised = [f"{a_name}-{b_name}"]
            for other_id in [oid for oid in selected_objects if oid not in {a_id, b_id}]:
                other_rows = by_obj[other_id]
                other_name = other_rows[0]["object_name"]
                other_rel = relation_pair_label(a_rows, other_rows, use_last=False)
                other_end_rel = relation_pair_label(a_rows, other_rows, use_last=True)
                if other_rel == other_end_rel:
                    revised.append(f"{other_name} relation unchanged")
            qrows.append(
                make_question(
                    "selective_relation_revision",
                    "hard",
                    f"Object {a_name} moved. Which relations involving {a_name} should be updated, and which relations among the stationary objects should remain unchanged?",
                    "update A-B relations for moved object; leave non-moving object relations unchanged",
                    video_id,
                    a_name,
                    reference=b_name,
                    evidence={
                        "moved_object": a_name,
                        "stationary_object": b_name,
                        "start_relation": start_rel,
                        "end_relation": end_rel,
                    },
                )
            )

        if disp_a < 0.5 and disp_b < 0.5:
            qrows.append(
                make_question(
                    "interaction_readiness",
                    "easy",
                    f"Are {a_name} and {b_name} spatially arranged so that an interaction between them is plausible?",
                    "yes",
                    video_id,
                    a_name,
                    reference=b_name,
                    evidence={
                        "a_displacement": round(disp_a, 3),
                        "b_displacement": round(disp_b, 3),
                        "start_relation": start_rel,
                        "end_relation": end_rel,
                    },
                )
            )

    # keep clean and compact
    dedup = []
    seen = set()
    for q in qrows:
        key = (q["video_id"], q["question_type"], q["question"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(q)
    return dedup[:12]


def build_multivideo_affordance() -> List[dict]:
    rows = load_tracks(TRACKS_PATH)
    by_video: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)

    all_questions: List[dict] = []
    for video_id in sorted(by_video):
        questions = build_affordance_questions(video_id, by_video[video_id])
        all_questions.extend(questions)
    return all_questions


def write_results(rows: List[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "questions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_id", "question_type", "difficulty", "target_object", "reference_object", "question", "answer", "evidence"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "video_id": row["video_id"],
                "question_type": row["question_type"],
                "difficulty": row["difficulty"],
                "target_object": row["target_object"],
                "reference_object": row["reference_object"],
                "question": row["question"],
                "answer": row["answer"],
                "evidence": json.dumps(row["evidence"], sort_keys=True),
            })

    with (OUT_DIR / "template_bank.json").open("w", encoding="utf-8") as fh:
        json.dump([
            {
                "question_type": "affordance_spatial",
                "template": "Which fixture is most compatible with the {object}'s current location?",
                "description": "Uses object category and fixture compatibility to interpret current spatial affordance.",
            },
            {
                "question_type": "interaction_readiness",
                "template": "Are {object_a} and {object_b} arranged in a way that supports a plausible interaction?",
                "description": "Detects whether two relevant items are in a functionally sensible arrangement for interaction.",
            },
            {
                "question_type": "historical_functional_state",
                "template": "Did the functional context of {object} change from {start_fixture} to {latest_fixture}?",
                "description": "Captures change in object function over time using fixture transitions.",
            },
            {
                "question_type": "interaction_consequence",
                "template": "After {object_a} moved, which previously valid spatial relation is no longer true with {object_b}?",
                "description": "Tests whether a prior spatial relation becomes invalid after manipulation.",
            },
            {
                "question_type": "selective_relation_revision",
                "template": "Object {object_a} moved. Which relations involving {object_a} should be updated, and which relations among the stationary objects should remain unchanged?",
                "description": "Selective update rule for changed vs unchanged relations.",
            },
            {
                "question_type": "ego_motion_affordance_stability",
                "template": "The object appears somewhat closer after motion, but did its functional relationship to the relevant fixture actually change?",
                "description": "Distinguishes object movement from genuine functional affordance change.",
            },
            {
                "question_type": "persistent_affordance_occlusion",
                "template": "The {object} became temporarily occluded after movement. Is its affordance at the current fixture still reasonable?",
                "description": "Tests functional memory and affordance persistence under temporary occlusion.",
            },
            {
                "question_type": "counterfactual_reasoning",
                "template": "If the observer moved but {object} stayed fixed, should the world-level affordance relation to its fixture meaningfully change?",
                "description": "Tests whether object-level affordance is stable under observer-only motion.",
            },
        ], fh, indent=2)

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "num_videos": len({row["video_id"] for row in rows}),
            "num_questions": len(rows),
            "question_types": {k: sum(1 for row in rows if row["question_type"] == k) for k in sorted({row["question_type"] for row in rows})},
        }, fh, indent=2)


def main() -> None:
    rows = build_multivideo_affordance()
    write_results(rows)
    print(f"Generated {len(rows)} affordance questions across {len({r['video_id'] for r in rows})} videos.")
    print("Question types:", {k: sum(1 for r in rows if r["question_type"] == k) for k in sorted({r["question_type"] for r in rows})})


if __name__ == "__main__":
    main()
