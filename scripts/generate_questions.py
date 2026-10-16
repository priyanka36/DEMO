#!/usr/bin/env python3
"""Generate research-oriented 3D spatiotemporal questions from HD-EPIC object tracks.

This generator targets controlled temporal reasoning questions rather than generic
open-ended VQA. It supports six benchmark categories:

1. static_spatial
2. object_motion
3. relation_change
4. visibility_occlusion
5. historical_current_update
6. ego_motion_vs_object_motion

Each question is grounded in actual 3D coordinates and, when available, camera
poses over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


QUESTION_TYPES = {
    "static_spatial",
    "object_motion",
    "relation_change",
    "visibility_occlusion",
    "historical_current_update",
    "ego_motion_vs_object_motion",
}


def load_tracks(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames = []
        for csv_path in sorted(path.glob("*.csv")):
            frames.append(pd.read_csv(csv_path))
        if not frames:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)

    required = {"video_id", "object_id", "timestamp", "x", "y", "z", "visible"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Track file missing required columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["visible"] = pd.to_numeric(df["visible"], errors="coerce").fillna(1).astype(int)
    return df


def load_camera_poses(path: Path | None) -> Dict[str, pd.DataFrame]:
    if path is None or not path.exists():
        return {}

    if path.is_dir():
        frames = []
        for csv_path in sorted(path.glob("*.csv")):
            frames.append(pd.read_csv(csv_path))
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)

    if {"video_id", "timestamp", "x", "y", "z"}.issubset(df.columns):
        return {vid: group.sort_values("timestamp").reset_index(drop=True) for vid, group in df.groupby("video_id")}
    if {"timestamp", "x", "y", "z"}.issubset(df.columns):
        return {"__global__": df.sort_values("timestamp").reset_index(drop=True)}
    raise ValueError(
        "Camera pose file must contain video_id,timestamp,x,y,z or timestamp,x,y,z."
    )


def obj_positions(df: pd.DataFrame, obj_id: str) -> List[dict]:
    out = df.loc[df["object_id"] == obj_id, ["timestamp", "x", "y", "z", "visible"]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out.to_dict("records")


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def relative_relation(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> str:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    mag = {"x": abs(dx), "y": abs(dy), "z": abs(dz)}
    primary = max(mag, key=mag.get)
    if primary == "x":
        return "left" if dx < 0 else "right"
    if primary == "y":
        return "above" if dy < 0 else "below"
    return "front" if dz < 0 else "behind"


def interpolate_position(points: List[dict], t: float) -> dict | None:
    if not points:
        return None
    pos = [p for p in points if p["timestamp"] == t]
    if pos:
        return pos[0]
    if len(points) == 1:
        return points[0]
    points_sorted = sorted(points, key=lambda p: p["timestamp"])
    for i in range(len(points_sorted) - 1):
        a = points_sorted[i]
        b = points_sorted[i + 1]
        if a["timestamp"] <= t <= b["timestamp"]:
            if math.isclose(b["timestamp"], a["timestamp"]):
                return a
            alpha = (t - a["timestamp"]) / (b["timestamp"] - a["timestamp"])
            return {
                "timestamp": t,
                "x": a["x"] + alpha * (b["x"] - a["x"]),
                "y": a["y"] + alpha * (b["y"] - a["y"]),
                "z": a["z"] + alpha * (b["z"] - a["z"]),
                "visible": 1,
            }
    return points_sorted[-1]


def make_question(question_type: str, difficulty: str, target_object: str, reference_object: str, question: str, answer: str, time_start: float, time_end: float, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "video_id": "",
        "question_type": question_type,
        "difficulty": difficulty,
        "target_object": target_object,
        "reference_object": reference_object,
        "time_start": float(time_start),
        "time_end": float(time_end),
        "question": question,
        "answer": answer,
        "evidence": evidence,
    }


def find_common_times(obj_a: List[dict], obj_b: List[dict]) -> List[float]:
    a_times = {p["timestamp"] for p in obj_a if p.get("visible", 1)}
    b_times = {p["timestamp"] for p in obj_b if p.get("visible", 1)}
    if a_times and b_times:
        times = sorted(a_times | b_times)
        if len(times) >= 2:
            return times
    return sorted(a_times & b_times)


def compute_static_spatial_questions(video_id: str, obj_a: str, obj_b: str, obj_a_tracks: List[dict], obj_b_tracks: List[dict]) -> List[dict]:
    times = find_common_times(obj_a_tracks, obj_b_tracks)
    if not times:
        times = sorted({p["timestamp"] for p in obj_a_tracks} | {p["timestamp"] for p in obj_b_tracks})
    if not times:
        return []
    questions: List[dict] = []
    t = times[len(times) // 2]
    pa = interpolate_position(obj_a_tracks, t)
    pb = interpolate_position(obj_b_tracks, t)
    if pa is None or pb is None:
        return []
    rel = relative_relation((pa["x"], pa["y"], pa["z"]), (pb["x"], pb["y"], pb["z"]))
    relation_map = {
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
        "front": "in front of",
        "behind": "behind",
    }
    qtext = f"At time t={t:.1f}s, is {obj_a} {relation_map.get(rel, 'relative to')} {obj_b}?"
    if rel in {"left", "right"}:
        qtext = f"At time t={t:.1f}s, is {obj_a} to the left or right of {obj_b}?"
        answer = "left" if rel == "left" else "right"
    elif rel in {"above", "below"}:
        qtext = f"At time t={t:.1f}s, is {obj_a} above or below {obj_b}?"
        answer = rel
    elif rel in {"front", "behind"}:
        qtext = f"At time t={t:.1f}s, is {obj_a} in front of or behind {obj_b}?"
        answer = "in front of" if rel == "front" else "behind"
    else:
        qtext = f"At time t={t:.1f}s, how is {obj_a} positioned relative to {obj_b}?"
        answer = rel

    questions.append(
        make_question(
            "static_spatial",
            "easy",
            obj_a,
            obj_b,
            qtext,
            answer,
            t,
            t,
            {
                "target_world_position": [pa["x"], pa["y"], pa["z"]],
                "reference_world_position": [pb["x"], pb["y"], pb["z"]],
                "relative_relation": rel,
            },
        )
    )

    d = distance((pa["x"], pa["y"], pa["z"]), (pb["x"], pb["y"], pb["z"]))
    questions.append(
        make_question(
            "static_spatial",
            "easy",
            obj_a,
            obj_b,
            f"At time t={t:.1f}s, what is the distance between {obj_a} and {obj_b}?",
            f"{d:.3f} m",
            t,
            t,
            {
                "distance_m": round(d, 3),
                "target_world_position": [pa["x"], pa["y"], pa["z"]],
                "reference_world_position": [pb["x"], pb["y"], pb["z"]],
            },
        )
    )
    return questions


def compute_object_motion_questions(video_id: str, obj_id: str, points: List[dict]) -> List[dict]:
    if len(points) < 2:
        return []
    start = points[0]
    end = points[-1]
    d = distance((start["x"], start["y"], start["z"]), (end["x"], end["y"], end["z"]))
    answer = "yes" if d > 0.15 else "no"
    q = make_question(
        "object_motion",
        "medium",
        obj_id,
        "",
        f"Did {obj_id} move substantially between t={start['timestamp']:.1f}s and t={end['timestamp']:.1f}s?",
        answer,
        start["timestamp"],
        end["timestamp"],
        {"displacement_m": round(d, 3), "start_position": [start["x"], start["y"], start["z"]], "end_position": [end["x"], end["y"], end["z"]]},
    )
    return [q]


def compute_relation_change_questions(video_id: str, obj_a: str, obj_b: str, a_tracks: List[dict], b_tracks: List[dict]) -> List[dict]:
    times = find_common_times(a_tracks, b_tracks)
    if len(times) < 2:
        times = sorted({p["timestamp"] for p in a_tracks} | {p["timestamp"] for p in b_tracks})
    if len(times) < 2:
        return []
    t1 = times[0]
    t2 = times[-1]
    pa1 = interpolate_position(a_tracks, t1)
    pb1 = interpolate_position(b_tracks, t1)
    pa2 = interpolate_position(a_tracks, t2)
    pb2 = interpolate_position(b_tracks, t2)
    if pa1 is None or pb1 is None or pa2 is None or pb2 is None:
        return []
    r1 = relative_relation((pa1["x"], pa1["y"], pa1["z"]), (pb1["x"], pb1["y"], pb1["z"]))
    r2 = relative_relation((pa2["x"], pa2["y"], pa2["z"]), (pb2["x"], pb2["y"], pb2["z"]))
    d1 = distance((pa1["x"], pa1["y"], pa1["z"]), (pb1["x"], pb1["y"], pb1["z"]))
    d2 = distance((pa2["x"], pa2["y"], pa2["z"]), (pb2["x"], pb2["y"], pb2["z"]))

    if r1 == r2:
        rel_answer = "unchanged"
    else:
        rel_answer = f"{r1}_to_{r2}"

    questions: List[dict] = []
    questions.append(
        make_question(
            "relation_change",
            "medium",
            obj_a,
            obj_b,
            f"How did the position of {obj_a} relative to {obj_b} change between t={t1:.1f}s and t={t2:.1f}s?",
            rel_answer,
            t1,
            t2,
            {
                "start_relation": r1,
                "end_relation": r2,
                "start_distance_m": round(d1, 3),
                "end_distance_m": round(d2, 3),
            },
        )
    )

    if d2 < d1 - 1e-3:
        close_answer = "closer"
    elif d2 > d1 + 1e-3:
        close_answer = "farther"
    else:
        close_answer = "unchanged"
    questions.append(
        make_question(
            "relation_change",
            "medium",
            obj_a,
            obj_b,
            f"Did {obj_a} become closer to or farther from {obj_b} between t={t1:.1f}s and t={t2:.1f}s?",
            close_answer,
            t1,
            t2,
            {
                "start_distance_m": round(d1, 3),
                "end_distance_m": round(d2, 3),
                "start_relation": r1,
                "end_relation": r2,
            },
        )
    )
    return questions


def compute_visibility_questions(video_id: str, obj_id: str, points: List[dict]) -> List[dict]:
    if not points:
        return []
    visible_points = [p for p in points if int(p.get("visible", 1)) == 1]
    if not visible_points:
        answer = "no"
    else:
        answer = "yes"
    start_t = points[0]["timestamp"]
    end_t = points[-1]["timestamp"]
    return [
        make_question(
            "visibility_occlusion",
            "easy",
            obj_id,
            "",
            f"Was {obj_id} visible during the interval from t={start_t:.1f}s to t={end_t:.1f}s?",
            answer,
            start_t,
            end_t,
            {"visible_samples": len(visible_points), "total_samples": len(points)},
        )
    ]


def compute_historical_current_update_questions(video_id: str, obj_a: str, obj_b: str, a_tracks: List[dict], b_tracks: List[dict]) -> List[dict]:
    times = find_common_times(a_tracks, b_tracks)
    if len(times) < 2:
        times = sorted({p["timestamp"] for p in a_tracks} | {p["timestamp"] for p in b_tracks})
    if len(times) < 2:
        return []
    t1 = min(times)
    t2 = max(times)
    pa1 = interpolate_position(a_tracks, t1)
    pb1 = interpolate_position(b_tracks, t1)
    pa2 = interpolate_position(a_tracks, t2)
    pb2 = interpolate_position(b_tracks, t2)
    if pa1 is None or pb1 is None or pa2 is None or pb2 is None:
        return []
    r1 = relative_relation((pa1["x"], pa1["y"], pa1["z"]), (pb1["x"], pb1["y"], pb1["z"]))
    r2 = relative_relation((pa2["x"], pa2["y"], pa2["z"]), (pb2["x"], pb2["y"], pb2["z"]))
    return [
        make_question(
            "historical_current_update",
            "medium",
            obj_a,
            obj_b,
            f"At the start of the motion, how was {obj_b} positioned relative to {obj_a}, and how was it positioned by the end?",
            f"{r1} then {r2}",
            t1,
            t2,
            {"start_relation": r1, "end_relation": r2},
        )
    ]


def compute_ego_motion_vs_object_motion_questions(video_id: str, obj_id: str, obj_tracks: List[dict], camera_poses: pd.DataFrame) -> List[dict]:
    if camera_poses.empty or len(obj_tracks) < 2:
        return []
    cp = camera_poses.loc[camera_poses["video_id"] == video_id] if "video_id" in camera_poses.columns else camera_poses
    if cp.empty:
        return []
    cp = cp.sort_values("timestamp").reset_index(drop=True)
    start_t = min(obj_tracks[0]["timestamp"], cp["timestamp"].min())
    end_t = max(obj_tracks[-1]["timestamp"], cp["timestamp"].max())
    start_obj = interpolate_position(obj_tracks, start_t)
    end_obj = interpolate_position(obj_tracks, end_t)
    start_cam = interpolate_position(cp.to_dict("records"), start_t) if isinstance(cp, pd.DataFrame) else None
    end_cam = interpolate_position(cp.to_dict("records"), end_t) if isinstance(cp, pd.DataFrame) else None
    if start_obj is None or end_obj is None or start_cam is None or end_cam is None:
        return []

    obj_displacement = distance((start_obj["x"], start_obj["y"], start_obj["z"]), (end_obj["x"], end_obj["y"], end_obj["z"]))
    cam_displacement = distance((start_cam["x"], start_cam["y"], start_cam["z"]), (end_cam["x"], end_cam["y"], end_cam["z"]))
    if obj_displacement > 0.2 and cam_displacement > 0.2:
        answer = "both"
    elif obj_displacement > 0.2:
        answer = "object_motion"
    elif cam_displacement > 0.2:
        answer = "camera_motion"
    else:
        answer = "neither"

    return [
        make_question(
            "ego_motion_vs_object_motion",
            "hard",
            obj_id,
            "camera",
            f"Between t={start_t:.1f}s and t={end_t:.1f}s, was the apparent motion mainly caused by camera motion, object motion, both, or neither?",
            answer,
            start_t,
            end_t,
            {
                "object_displacement_m": round(obj_displacement, 3),
                "camera_displacement_m": round(cam_displacement, 3),
                "camera_start": [start_cam["x"], start_cam["y"], start_cam["z"]],
                "camera_end": [end_cam["x"], end_cam["y"], end_cam["z"]],
            },
        )
    ]


def select_object_pairs(objects: List[str], max_pairs: int | None = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            pairs.append((objects[i], objects[j]))
    if max_pairs is not None:
        return pairs[:max_pairs]
    return pairs


def limit_question_rows(rows: List[dict], max_questions_per_type: int | None = None) -> List[dict]:
    if max_questions_per_type is None:
        return rows
    counts: Dict[str, int] = defaultdict(int)
    kept: List[dict] = []
    for row in rows:
        qtype = row["question_type"]
        if counts[qtype] >= max_questions_per_type:
            continue
        kept.append(row)
        counts[qtype] += 1
    return kept


def generate_rows(tracks_df: pd.DataFrame, camera_poses: Dict[str, pd.DataFrame], video_id: str | None = None, max_pairs_per_video: int | None = None) -> List[dict]:
    rows: List[dict] = []
    videos = [video_id] if video_id else sorted(tracks_df["video_id"].unique())

    for vid in videos:
        video_tracks = tracks_df.loc[tracks_df["video_id"] == vid].copy()
        objects = sorted(video_tracks["object_id"].unique().tolist())
        if len(objects) < 2:
            continue

        # object motion and visibility questions for each object
        for obj in objects:
            points = obj_positions(video_tracks, obj)
            for q in compute_object_motion_questions(vid, obj, points):
                q["video_id"] = vid
                rows.append(q)
            for q in compute_visibility_questions(vid, obj, points):
                q["video_id"] = vid
                rows.append(q)

        # pairwise relation and temporal questions
        for a, b in select_object_pairs(objects, max_pairs=max_pairs_per_video):
            a_tracks = obj_positions(video_tracks, a)
            b_tracks = obj_positions(video_tracks, b)
            if not a_tracks or not b_tracks:
                continue
            for q in compute_static_spatial_questions(vid, a, b, a_tracks, b_tracks):
                q["video_id"] = vid
                rows.append(q)
            for q in compute_relation_change_questions(vid, a, b, a_tracks, b_tracks):
                q["video_id"] = vid
                rows.append(q)
            for q in compute_historical_current_update_questions(vid, a, b, a_tracks, b_tracks):
                q["video_id"] = vid
                rows.append(q)

        # ego-motion vs object-motion question using camera poses if available
        camera_df = camera_poses.get(vid)
        if camera_df is not None and not camera_df.empty:
            for obj in objects:
                points = obj_positions(video_tracks, obj)
                for q in compute_ego_motion_vs_object_motion_questions(vid, obj, points, camera_df):
                    q["video_id"] = vid
                    rows.append(q)

    # de-duplicate if generated by mistake
    seen = set()
    unique_rows: List[dict] = []
    for row in rows:
        key = (row["video_id"], row["question_type"], row["target_object"], row["reference_object"], row["time_start"], row["time_end"], row["question"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def write_jsonl(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "question_type",
        "difficulty",
        "target_object",
        "reference_object",
        "time_start",
        "time_end",
        "question",
        "answer",
        "evidence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "video_id": row.get("video_id", ""),
                "question_type": row.get("question_type", ""),
                "difficulty": row.get("difficulty", ""),
                "target_object": row.get("target_object", ""),
                "reference_object": row.get("reference_object", ""),
                "time_start": row.get("time_start", 0.0),
                "time_end": row.get("time_end", 0.0),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "evidence": json.dumps(row.get("evidence", {}), sort_keys=True),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate spatiotemporal benchmark questions from object 4D tracks.")
    parser.add_argument("--tracks", type=Path, required=True, help="CSV or directory containing per-video track CSVs with video_id, object_id, timestamp, x, y, z, visible columns.")
    parser.add_argument("--camera-poses", type=Path, default=None, help="Optional CSV or directory with video_id,timestamp,x,y,z for camera trajectory.")
    parser.add_argument("--video-id", type=str, default=None, help="Optional video ID to process; otherwise all videos are used.")
    parser.add_argument("--max-pairs-per-video", type=int, default=None, help="Optional cap on the number of object-object pairs to process per video. Default: all pairs.")
    parser.add_argument("--max-questions-per-type", type=int, default=None, help="Optional cap per question type to keep the benchmark balanced.")
    parser.add_argument("--output-json", type=Path, default=Path("benchmark/generated_questions.json"), help="Path for generated JSON output.")
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark/generated_questions.csv"), help="Path for generated CSV output.")
    args = parser.parse_args()

    tracks_df = load_tracks(args.tracks)
    camera_poses = load_camera_poses(args.camera_poses)
    rows = generate_rows(
        tracks_df,
        camera_poses,
        video_id=args.video_id,
        max_pairs_per_video=args.max_pairs_per_video,
    )
    rows = limit_question_rows(rows, args.max_questions_per_type)
    for row in rows:
        row["video_id"] = row.get("video_id", args.video_id or "")
        if row["question_type"] not in QUESTION_TYPES:
            row["question_type"] = "static_spatial"

    write_jsonl(rows, args.output_json)
    write_csv(rows, args.output_csv)
    print(f"Generated {len(rows)} questions.")
    by_type = defaultdict(int)
    for row in rows:
        by_type[row["question_type"]] += 1
    print("Question types:", dict(sorted(by_type.items())))


if __name__ == "__main__":
    main()
