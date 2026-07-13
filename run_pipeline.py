"""End-to-end video analysis pipeline for demo and user videos.

This script extracts pose landmarks from both videos, computes motion metrics,
synchronizes the knee-angle sequences, and generates coaching feedback for the
selected snow sport technique.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, cast

from extract_demo import extract_pose_from_video
from feedback_engine import (
    SPORT_CURRICULUM,
    build_selection,
    build_selection_summary,
    build_ksia_feedback_items,
    generate_llm_feedback_report,
    selection_from_legacy_event,
)
from geometry_utils import (
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    calculate_joint_angle,
    calculate_stance_ratio,
    normalize_landmarks_by_shoulder_width,
)
from sync_engine import synchronize_angle_sequences


LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_KNEE_INDEX = 25
RIGHT_KNEE_INDEX = 26
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pose extraction, synchronization, and LLM-backed feedback "
            "generation for a demo video and a user video."
        )
    )
    parser.add_argument("demo_video", help="Path to the demo video file")
    parser.add_argument("user_video", help="Path to the user video file")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory for generated JSON and summary files",
    )
    parser.add_argument(
        "--sport",
        choices=sorted(SPORT_CURRICULUM.keys()),
        help="Choose the sport explicitly instead of prompting interactively",
    )
    parser.add_argument(
        "--level",
        help="Choose the level explicitly instead of prompting interactively",
    )
    parser.add_argument(
        "--technique",
        help="Choose the technique explicitly instead of prompting interactively",
    )
    parser.add_argument(
        "--event",
        help=(
            "Legacy event name such as '스키_레벨1_스노우플라우턴(보겐)'. "
            "If supplied, it overrides sport/level/technique."
        ),
    )
    parser.add_argument(
        "--threshold-degrees",
        type=float,
        default=15.0,
        help="Angle gap threshold used by the sync/anomaly step",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame during pose extraction",
    )
    return parser.parse_args()


def _prompt_choice(label: str, options: list[str]) -> str:
    if not options:
        raise ValueError(f"No options available for {label}")

    if not sys.stdin.isatty():
        raise ValueError(f"Interactive selection is unavailable for {label}; pass an explicit argument")

    print(f"Select {label}:")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")

    while True:
        raw_value = input(f"Enter {label} number: ").strip()
        try:
            choice_index = int(raw_value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if 1 <= choice_index <= len(options):
            return options[choice_index - 1]

        print("Choice out of range.")


def _resolve_selection(args: argparse.Namespace):
    if args.event:
        return selection_from_legacy_event(args.event)

    sport = args.sport or _prompt_choice("sport", sorted(SPORT_CURRICULUM.keys()))
    level = args.level or _prompt_choice("level", list(SPORT_CURRICULUM[sport].keys()))
    technique = args.technique or _prompt_choice(
        "technique",
        list(SPORT_CURRICULUM[sport][level]),
    )
    return build_selection(sport, level, technique)


def _video_pose_output_path(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}_pose.json"


def _load_pose_json(pose_json_path: Path) -> dict[int, list[dict[str, float | None]]]:
    with pose_json_path.open("r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    parsed: dict[int, list[dict[str, float | None]]] = {}
    for frame_key, landmarks in raw_data.items():
        parsed[int(frame_key)] = landmarks
    return dict(sorted(parsed.items(), key=lambda item: item[0]))


def _landmark_at(landmarks: list[dict[str, float | None]], index: int) -> dict[str, float | None]:
    if index >= len(landmarks):
        return {"x": None, "y": None, "z": None}
    return landmarks[index]


def _line_angle_degrees(left: dict[str, float | None], right: dict[str, float | None]) -> float | None:
    if left["x"] is None or left["y"] is None or right["x"] is None or right["y"] is None:
        return None
    return math.degrees(math.atan2(float(right["y"]) - float(left["y"]), float(right["x"]) - float(left["x"])))


def _analyze_pose_sequence(
    pose_frames: dict[int, list[dict[str, float | None]]],
) -> dict[str, list[float | None]]:
    stance_ratios: list[float | None] = []
    outside_knee_angles: list[float | None] = []
    shoulder_slopes: list[float | None] = []
    knee_slopes: list[float | None] = []

    for _, landmarks in pose_frames.items():
        normalized_landmarks, _ = normalize_landmarks_by_shoulder_width(landmarks)
        stance_ratios.append(
            calculate_stance_ratio(
                cast(list[dict[str, float | None]], normalized_landmarks)
            )
        )

        left_knee_angle = calculate_joint_angle(
            _landmark_at(landmarks, LEFT_HIP_INDEX),
            _landmark_at(landmarks, LEFT_KNEE_INDEX),
            _landmark_at(landmarks, LEFT_ANKLE_INDEX),
        )
        right_knee_angle = calculate_joint_angle(
            _landmark_at(landmarks, RIGHT_HIP_INDEX),
            _landmark_at(landmarks, RIGHT_KNEE_INDEX),
            _landmark_at(landmarks, RIGHT_ANKLE_INDEX),
        )

        valid_knee_angles = [angle for angle in (left_knee_angle, right_knee_angle) if angle is not None]
        outside_knee_angles.append(min(valid_knee_angles) if valid_knee_angles else None)

        shoulder_slopes.append(
            _line_angle_degrees(
                _landmark_at(landmarks, LEFT_SHOULDER_INDEX),
                _landmark_at(landmarks, RIGHT_SHOULDER_INDEX),
            )
        )
        knee_slopes.append(
            _line_angle_degrees(
                _landmark_at(landmarks, LEFT_KNEE_INDEX),
                _landmark_at(landmarks, RIGHT_KNEE_INDEX),
            )
        )

    return {
        "stance_ratios": stance_ratios,
        "outside_knee_angles": outside_knee_angles,
        "shoulder_slopes": shoulder_slopes,
        "knee_slopes": knee_slopes,
    }


def _valid_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _summary_metrics(values: list[float | None]) -> tuple[float | None, float | None]:
    numeric_values = _valid_values(values)
    if not numeric_values:
        return None, None

    if len(numeric_values) == 1:
        return 0.0, 0.0

    value_range = max(numeric_values) - min(numeric_values)
    mean_value = statistics.fmean(numeric_values)
    if mean_value == 0:
        return value_range, 0.0

    return value_range, statistics.pstdev(numeric_values) / abs(mean_value)


def _select_apex_frame(
    synced_path: list[dict[str, int]],
    demo_angles: list[float | None],
    user_angles: list[float | None],
) -> tuple[int | None, int | None, float | None, float | None]:
    best_pair: tuple[int | None, int | None, float | None, float | None] = (None, None, None, None)
    best_user_angle: float | None = None

    for pair in synced_path:
        demo_frame = pair["demo_frame"]
        user_frame = pair["user_frame"]
        if demo_frame >= len(demo_angles) or user_frame >= len(user_angles):
            continue

        demo_angle = demo_angles[demo_frame]
        user_angle = user_angles[user_frame]
        if demo_angle is None or user_angle is None:
            continue

        if best_user_angle is None or user_angle < best_user_angle:
            best_user_angle = user_angle
            best_pair = (demo_frame, user_frame, demo_angle, user_angle)

    return best_pair


def _build_analysis_result(
    demo_sequence: dict[str, list[float | None]],
    user_sequence: dict[str, list[float | None]],
    sync_result: Any,
) -> dict[str, float | bool]:
    stance_range, stance_cv = _summary_metrics(user_sequence["stance_ratios"])
    demo_apex_frame, user_apex_frame, demo_apex_angle, user_apex_angle = _select_apex_frame(
        sync_result.path,
        demo_sequence["outside_knee_angles"],
        user_sequence["outside_knee_angles"],
    )

    analysis_result: dict[str, float | bool] = {}
    if stance_range is not None:
        analysis_result["stance_ratio_range"] = stance_range
    if stance_cv is not None:
        analysis_result["stance_ratio_cv"] = stance_cv

    if demo_apex_frame is not None and user_apex_frame is not None:
        if demo_apex_angle is not None and user_apex_angle is not None:
            analysis_result["demo_apex_outside_knee_angle"] = float(demo_apex_angle)
            analysis_result["user_apex_outside_knee_angle"] = float(user_apex_angle)
            analysis_result["apex_outside_knee_angle_gap"] = abs(
                float(user_apex_angle) - float(demo_apex_angle)
            )

    shoulder_slopes = _valid_values(user_sequence["shoulder_slopes"])
    knee_slopes = _valid_values(user_sequence["knee_slopes"])
    if shoulder_slopes and knee_slopes:
        shoulder_slope = statistics.fmean(shoulder_slopes)
        knee_slope = statistics.fmean(knee_slopes)
        analysis_result["shoulder_line_slope"] = shoulder_slope
        analysis_result["knee_line_slope"] = knee_slope
        analysis_result["angulation_difference"] = abs(shoulder_slope - knee_slope)

    return analysis_result


def _build_sync_summary(sync_result: Any) -> dict[str, Any]:
    return {
        "distance": float(sync_result.distance),
        "matched_frames": len(sync_result.path),
        "anomaly_count": len(sync_result.anomaly_frames),
    }


def _run_extraction(video_path: Path, output_path: Path, frame_step: int) -> None:
    status = extract_pose_from_video(video_path, output_path, frame_step=frame_step)
    if status != 0:
        raise RuntimeError(f"Pose extraction failed for {video_path}")


def main() -> int:
    args = parse_args()
    try:
        selection = _resolve_selection(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    demo_video = Path(args.demo_video)
    user_video = Path(args.user_video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not demo_video.exists():
        print(f"Error: demo video does not exist: {demo_video}", file=sys.stderr)
        return 1
    if not user_video.exists():
        print(f"Error: user video does not exist: {user_video}", file=sys.stderr)
        return 1

    demo_pose_json = _video_pose_output_path(demo_video, output_dir)
    user_pose_json = _video_pose_output_path(user_video, output_dir)

    from multiprocessing import Process
    procs = []
    procs.append(Process(target=_run_extraction, args=(demo_video, demo_pose_json, args.frame_step)))
    procs.append(Process(target=_run_extraction, args=(user_video, user_pose_json, args.frame_step)))
    print(f"Extracting demo pose data to {demo_pose_json}...", flush=True)
    print(f"Extracting user pose data to {user_pose_json}...", flush=True)

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    demo_frames = _load_pose_json(demo_pose_json)
    user_frames = _load_pose_json(user_pose_json)

    demo_sequence = _analyze_pose_sequence(demo_frames)
    user_sequence = _analyze_pose_sequence(user_frames)

    sync_result = synchronize_angle_sequences(
        demo_sequence["outside_knee_angles"],
        user_sequence["outside_knee_angles"],
        threshold_degrees=args.threshold_degrees,
    )

    analysis_result = _build_analysis_result(demo_sequence, user_sequence, sync_result)
    sync_summary = _build_sync_summary(sync_result)
    rule_items = build_ksia_feedback_items(selection, analysis_result)
    feedback_text = generate_llm_feedback_report(selection, analysis_result, sync_summary, rule_items)
    summary_path = output_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "selection": build_selection_summary(selection),
                "distance": sync_result.distance,
                "path": sync_result.path,
                "anomaly_frames": sync_result.anomaly_frames,
                "analysis_result": analysis_result,
                "sync_summary": sync_summary,
                "rule_items": rule_items,
                "llm_feedback": feedback_text,
            },
            file_handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved analysis summary to {summary_path}")
    print(f"Matched frame pairs: {len(sync_result.path)}")
    print(f"Anomaly frames: {len(sync_result.anomaly_frames)}")
    print(f"Selected technique: {selection.label}")
    print()
    print(feedback_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())