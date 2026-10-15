#!/usr/bin/env python3
"""Build a small multi-video real-data HD-EPIC pilot.

This script generates a compact but real benchmark from the official
scene-and-object movement annotations. It uses a short list of real video IDs,
extracts object trajectories from the annotation metadata, and emits a consistent
set of around 11 QA pairs per video, giving roughly 220 questions for 20 videos.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

FPS = 30.0
DEFAULT_VIDEO_IDS = [
    "P01-20240202-110250",
    "P01-20240202-161354",
    "P01-20240202-161948",
    "P01-20240202-171220",
    "P01-20240202-175627",
    "P01-20240202-195538",
    "P01-20240203-093333",
    "P01-20240203-121517",
    "P01-20240203-123350",
    "P01-20240203-130505",
    "P01-20240203-132119",
    "P01-20240203-135502",
    "P01-20240203-150506",
    "P01-20240203-152323",
    "P01-20240203-152956",
    "P01-20240203-161757",
    "P01-20240203-184045",
    "P01-20240203-184214",
    "P01-20240204-095114",
    "P01-20240204-120411",
]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def relative_direction(a: Tuple[float, float], b: Tuple[float, float]) -> str:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "aligned"
    if abs(dx) > abs(dy):
        return "left" if dx > 0 else "right"
    return "above" if dy < 0 else "below"


def nearest_position(points: List[dict], time: float) -> dict:
    return min(points, key=lambda item: abs(item["timestamp"] - time))


def safe_distance(a: dict, b: dict) -> float:
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2) ** 0.5


def add_question(rows: List[dict], qtype: str, difficulty: str, question: str, answer: str,
                 target_object: str, reference_object: str = "", time_start: float = 0.0,
                 time_end: float = 0.0, evidence: dict | None = None) -> None:
    rows.append({
        "question_type": qtype,
        "difficulty": difficulty,
        "target_object": target_object,
        "reference_object": reference_object,
        "time_start": time_start,
        "time_end": time_end,
        "question": question,
        "answer": answer,
        "evidence": evidence or {},
    })


def extract_real_tracks(video_id: str) -> Dict[str, List[dict]]:
    assoc = load_json(Path("/tmp/hd-epic-annotations/scene-and-object-movements/assoc_info.json"))
    mask_info = load_json(Path("/tmp/hd-epic-annotations/scene-and-object-movements/mask_info.json"))
    real_tracks: Dict[str, List[dict]] = {}
    for info in assoc.get(video_id, {}).values():
        name = info.get("name")
        if not name:
            continue
        obj_id = sanitize_name(name)
        points: List[dict] = []
        for track in info.get("tracks", []):
            for mask_id in track.get("masks", []):
                meta = mask_info.get(video_id, {}).get(mask_id)
                if meta is None:
                    continue
                coord = meta.get("3d_location")
                if coord is None:
                    continue
                frame_no = float(meta.get("frame_number", 0))
                points.append({
                    "object_id": obj_id,
                    "object_name": name,
                    "timestamp": frame_no / FPS,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                    "visible": 1,
                    "fixture": meta.get("fixture", "unknown"),
                })
        if points:
            real_tracks[obj_id] = sorted(points, key=lambda item: item["timestamp"])
    return real_tracks


def select_objects(real_tracks: Dict[str, List[dict]], max_count: int = 3) -> List[str]:
    objs = sorted(real_tracks, key=lambda key: len(real_tracks[key]), reverse=True)
    return objs[:max_count]


def build_questions_for_video(video_id: str, real_tracks: Dict[str, List[dict]]) -> List[dict]:
    if len(real_tracks) < 2:
        return []

    objs = select_objects(real_tracks, 3)
    obj_a, obj_b, obj_c = objs[0], objs[1], objs[2] if len(objs) > 2 else objs[0]

    rows: List[dict] = []
    t1 = real_tracks[obj_a][0]["timestamp"]
    t2 = real_tracks[obj_a][-1]["timestamp"]
    t_ref = (t1 + t2) / 2.0 if t2 > t1 else t1

    start_a = real_tracks[obj_a][0]
    end_a = real_tracks[obj_a][-1]
    start_b = real_tracks[obj_b][0]
    end_b = real_tracks[obj_b][-1]
    start_c = real_tracks[obj_c][0]
    end_c = real_tracks[obj_c][-1]

    pos_a_t1 = nearest_position(real_tracks[obj_a], t_ref)
    pos_b_t1 = nearest_position(real_tracks[obj_b], t_ref)
    direction = relative_direction((pos_a_t1["x"], pos_a_t1["y"]), (pos_b_t1["x"], pos_b_t1["y"]))
    add_question(
        rows,
        "static_spatial",
        "easy",
        f"At time t={t_ref:.1f}s, is {obj_a} to the left or right of {obj_b}?",
        "left" if direction == "left" else "right",
        obj_a,
        obj_b,
        t_ref,
        t_ref,
        {
            "target_world_position": [pos_a_t1["x"], pos_a_t1["y"], pos_a_t1["z"]],
            "reference_world_position": [pos_b_t1["x"], pos_b_t1["y"], pos_b_t1["z"]],
        },
    )

    pos_a_t2 = nearest_position(real_tracks[obj_a], t1)
    pos_c_t2 = nearest_position(real_tracks[obj_c], t1)
    direction_2 = relative_direction((pos_a_t2["x"], pos_a_t2["y"]), (pos_c_t2["x"], pos_c_t2["y"]))
    add_question(
        rows,
        "static_spatial",
        "easy",
        f"At time t={t1:.1f}s, is {obj_a} above or below {obj_c}?",
        "above" if direction_2 == "above" else "below",
        obj_a,
        obj_c,
        t1,
        t1,
        {
            "target_world_position": [pos_a_t2["x"], pos_a_t2["y"], pos_a_t2["z"]],
            "reference_world_position": [pos_c_t2["x"], pos_c_t2["y"], pos_c_t2["z"]],
        },
    )

    d_a = safe_distance(start_a, end_a)
    add_question(
        rows,
        "object_motion",
        "medium",
        f"Did {obj_a} move substantially between t={start_a['timestamp']:.1f}s and t={end_a['timestamp']:.1f}s?",
        "yes" if d_a > 0.25 else "no",
        obj_a,
        "",
        start_a["timestamp"],
        end_a["timestamp"],
        {"displacement_m": round(d_a, 3)},
    )

    d_b = safe_distance(start_b, end_b)
    add_question(
        rows,
        "object_motion",
        "medium",
        f"Did {obj_b} move substantially between t={start_b['timestamp']:.1f}s and t={end_b['timestamp']:.1f}s?",
        "yes" if d_b > 0.25 else "no",
        obj_b,
        "",
        start_b["timestamp"],
        end_b["timestamp"],
        {"displacement_m": round(d_b, 3)},
    )

    d_c = safe_distance(start_c, end_c)
    add_question(
        rows,
        "object_motion",
        "medium",
        f"Did {obj_c} move substantially between t={start_c['timestamp']:.1f}s and t={end_c['timestamp']:.1f}s?",
        "yes" if d_c > 0.25 else "no",
        obj_c,
        "",
        start_c["timestamp"],
        end_c["timestamp"],
        {"displacement_m": round(d_c, 3)},
    )

    start_dir = relative_direction((real_tracks[obj_a][0]["x"], real_tracks[obj_a][0]["y"]), (real_tracks[obj_b][0]["x"], real_tracks[obj_b][0]["y"]))
    end_dir = relative_direction((real_tracks[obj_a][-1]["x"], real_tracks[obj_a][-1]["y"]), (real_tracks[obj_b][-1]["x"], real_tracks[obj_b][-1]["y"]))
    add_question(
        rows,
        "relation_change",
        "medium",
        f"How did the position of {obj_a} relative to {obj_b} change from the beginning to the end of its motion?",
        "unchanged" if start_dir == end_dir else "changed",
        obj_a,
        obj_b,
        start_a["timestamp"],
        end_a["timestamp"],
        {"start_relation": start_dir, "end_relation": end_dir},
    )

    add_question(
        rows,
        "relation_change",
        "medium",
        f"Did {obj_a} become closer to or farther from {obj_b} across the motion interval?",
        "closer" if safe_distance(start_a, start_b) > safe_distance(end_a, end_b) else "farther",
        obj_a,
        obj_b,
        start_a["timestamp"],
        end_a["timestamp"],
        {
            "start_distance_m": round(safe_distance(start_a, start_b), 3),
            "end_distance_m": round(safe_distance(end_a, end_b), 3),
        },
    )

    for target in [obj_a, obj_b, obj_c]:
        points = real_tracks[target]
        add_question(
            rows,
            "visibility_occlusion",
            "easy",
            f"Was {target} visible during the interval from t={points[0]['timestamp']:.1f}s to t={points[-1]['timestamp']:.1f}s?",
            "yes",
            target,
            "",
            points[0]["timestamp"],
            points[-1]["timestamp"],
            {"samples": len(points)},
        )

    add_question(
        rows,
        "historical_current_update",
        "medium",
        f"At the start of the {obj_a} motion, how was {obj_b} positioned relative to {obj_a}, and how was it positioned by the end?",
        f"{start_dir} then {end_dir}",
        obj_a,
        obj_b,
        start_a["timestamp"],
        end_a["timestamp"],
        {"start_relation": start_dir, "end_relation": end_dir},
    )

    while len(rows) < 11:
        add_question(
            rows,
            "object_motion",
            "medium",
            f"Did {obj_a} remain stationary across the tracked interval in {video_id}?",
            "no" if d_a > 0.05 else "yes",
            obj_a,
            "",
            start_a["timestamp"],
            end_a["timestamp"],
            {"status": "motion_check"},
        )
        break

    return rows


def write_object_table(rows_by_video: Dict[str, List[dict]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_id", "object_id", "object_name", "timestamp", "x", "y", "z", "visible", "fixture"])
        writer.writeheader()
        for video_id in sorted(rows_by_video):
            for row in rows_by_video[video_id]:
                writer.writerow({
                    "video_id": video_id,
                    "object_id": row["object_id"],
                    "object_name": row["object_name"],
                    "timestamp": f"{row['timestamp']:.3f}",
                    "x": f"{row['x']:.6f}",
                    "y": f"{row['y']:.6f}",
                    "z": f"{row['z']:.6f}",
                    "visible": row["visible"],
                    "fixture": row["fixture"],
                })


def write_questions(rows: List[dict], qpath: Path, cpath: Path) -> None:
    qpath.parent.mkdir(parents=True, exist_ok=True)
    with qpath.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    with cpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "video_id", "question_type", "difficulty", "target_object", "reference_object",
            "time_start", "time_end", "question", "answer", "evidence"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "video_id": row["video_id"],
                "question_type": row["question_type"],
                "difficulty": row["difficulty"],
                "target_object": row["target_object"],
                "reference_object": row["reference_object"],
                "time_start": row["time_start"],
                "time_end": row["time_end"],
                "question": row["question"],
                "answer": row["answer"],
                "evidence": json.dumps(row["evidence"], sort_keys=True),
            })


def build_multivideo(video_ids: Iterable[str]) -> Tuple[List[dict], Dict[str, List[dict]], Dict[str, int]]:
    all_rows: List[dict] = []
    all_tracks: Dict[str, List[dict]] = {}
    counts: Dict[str, int] = {}

    for video_id in video_ids:
        tracks = extract_real_tracks(video_id)
        if not tracks:
            continue
        for obj_key in sorted(tracks):
            for row in tracks[obj_key]:
                all_tracks.setdefault(video_id, []).append({
                    "object_id": obj_key,
                    "object_name": row["object_name"],
                    "timestamp": row["timestamp"],
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                    "visible": row["visible"],
                    "fixture": row["fixture"],
                })
        video_questions = build_questions_for_video(video_id, tracks)
        for row in video_questions:
            row["video_id"] = video_id
        all_rows.extend(video_questions)
        counts[video_id] = len(video_questions)
    return all_rows, all_tracks, counts


def write_summary(counts: Dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "questions_generated"])
        for video_id in sorted(counts):
            writer.writerow([video_id, counts[video_id]])


def main() -> None:
    video_ids = DEFAULT_VIDEO_IDS
    all_rows, all_tracks, counts = build_multivideo(video_ids)

    root = Path("/home/priyanka/VQA")
    multi_root = root / "benchmark" / "geometry"
    multi_root.mkdir(parents=True, exist_ok=True)
    write_questions(all_rows, multi_root / "questions.json", multi_root / "questions.csv")
    write_object_table(all_tracks, multi_root / "all_object_4d_tracks.csv")
    write_summary(counts, root / "results" / "multi_video_question_summary.csv")
    (multi_root / "video_ids.txt").write_text("\n".join(video_ids) + "\n", encoding="utf-8")

    print(f"Generated {len(all_rows)} questions across {len(counts)} videos.")
    print("Per-video counts:", dict(sorted(counts.items())))
    print("Total object rows:", sum(len(v) for v in all_tracks.values()))


if __name__ == "__main__":
    main()
