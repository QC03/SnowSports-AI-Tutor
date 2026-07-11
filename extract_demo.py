"""Extract MediaPipe Pose landmarks from a video file.

This script reads a video frame by frame, runs MediaPipe Pose with
model_complexity=2, and stores the 33 pose landmarks as JSON keyed by
frame number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

POSE_LANDMARK_COUNT = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MediaPipe Pose (33 landmarks) coordinates from a video and "
            "save them to JSON keyed by frame number."
        )
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output JSON file. Defaults to <video_stem>_pose.json",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame and store None for skipped frames",
    )
    return parser.parse_args()


def build_output_path(video_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return video_path.with_name(f"{video_path.stem}_pose.json")


def serialize_landmarks(landmarks: Any) -> list[dict[str, float | None]]:
    if landmarks is None:
        return [
            {"x": None, "y": None, "z": None}
            for _ in range(POSE_LANDMARK_COUNT)
        ]

    serialized: list[dict[str, float | None]] = []
    for landmark in landmarks.landmark:
        serialized.append(
            {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
            }
        )

    if len(serialized) < POSE_LANDMARK_COUNT:
        serialized.extend(
            {"x": None, "y": None, "z": None}
            for _ in range(POSE_LANDMARK_COUNT - len(serialized))
        )

    return serialized[:POSE_LANDMARK_COUNT]


def save_json(output_path: Path, frame_data: dict[str, list[dict[str, float | None]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(frame_data, file_handle, ensure_ascii=False, indent=2)


def extract_pose_from_video(video_path: Path, output_path: Path, frame_step: int = 1) -> int:
    try:
        import mediapipe as mp
    except ImportError as e:
        print(
            f"Error: MediaPipe import failed. {e}",
            file=sys.stderr,
        )
        return 1

    frame_data: dict[str, list[dict[str, float | None]]] = {}
    capture: cv2.VideoCapture | None = None
    pose: Any = None
    frame_index = 0
    status = 0

    if frame_step < 1:
        print("Error: frame_step must be at least 1", file=sys.stderr)
        return 1

    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video with {total_frames} frames...", flush=True)

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        while True:
            success, frame = capture.read()
            if not success:
                print(f"Finished reading video. Total frames processed: {frame_index}", flush=True)
                break

            if frame_index % frame_step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                frame_data[str(frame_index)] = serialize_landmarks(results.pose_landmarks)
            else:
                frame_data[str(frame_index)] = serialize_landmarks(None)
            
            # Progress indicator
            if (frame_index + 1) % 100 == 0:
                print(f"Processed {frame_index + 1} frames...", flush=True)
            
            frame_index += 1

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        status = 1
    finally:
        if pose is not None:
            pose.close()
        if capture is not None:
            capture.release()
        try:
            print(f"Saving {len(frame_data)} frames to JSON...", flush=True)
            save_json(output_path, frame_data)
            print(f"Successfully saved to {output_path}", flush=True)
        except Exception as save_exc:
            print(f"Failed to save JSON: {save_exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            status = 1

    return status


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path)
    output_path = build_output_path(video_path, args.output)

    if not video_path.exists():
        print(f"Error: input video does not exist: {video_path}", file=sys.stderr)
        return 1

    return extract_pose_from_video(video_path, output_path, frame_step=args.frame_step)


if __name__ == "__main__":
    raise SystemExit(main())
