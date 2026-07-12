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


import sys
from pathlib import Path
from typing import Any
import cv2

import sys
from pathlib import Path
from typing import Any
import cv2

def extract_pose_from_video(video_path: Path, output_path: Path, frame_step: int = 1) -> int:
    try:
        import mediapipe as mp
    except ImportError as e:
        print(f"Error: MediaPipe import failed. {e}", file=sys.stderr)
        return 1

    frame_data: dict[str, list[dict[str, float | None]]] = {}
    capture: cv2.VideoCapture | None = None
    pose: Any = None
    tracker: cv2.Tracker | None = None  # OpenCV 추적기 객체
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

        # 1. 첫 번째 프레임을 읽어와 사용자가 추적할 영역(ROI)을 선택하게 합니다.
        success, first_frame = capture.read()
        if not success:
            raise RuntimeError("영상에서 첫 번째 프레임을 읽을 수 없습니다.")

        h, w, _ = first_frame.shape
        if h > w:
            target_h = w  
            start_y = (h - target_h) // 2
            end_y = start_y + target_h
            first_frame = first_frame[start_y:end_y, 0:w]

        # 안내창 출력 후 마우스 드래그로 박스 지정 (드래그 후 'Space' 또는 'Enter' 입력)
        print("\n[안내] 팝업창이 뜨면 추적할 대상을 마우스로 드래그한 뒤 'Space' 또는 'Enter'를 누르세요.\n")
        roi = cv2.selectROI('Select Target Object', first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('Select Target Object')

        # 유효한 박스가 지정되었다면 OpenCV CSRT 추적기 초기화
        if roi[2] > 0 and roi[3] > 0:
            # OpenCV 4.5.1 이상 버전 표준 생성 방식
            tracker = cv2.TrackerCSRT.create()
            tracker.init(first_frame, roi)
            print("객체 추적기가 성공적으로 초기화되었습니다.", flush=True)
        else:
            print("영역이 지정되지 않아 전체 화면 기준으로 분석을 시작합니다.", flush=True)

        # MediaPipe Pose 초기화
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        cv2.namedWindow('MediaPipe Pose Video', cv2.WINDOW_NORMAL)
        
        # 첫 프레임 처리 과정 포함을 위해 파일 포인터를 처음으로 되돌림
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            success, frame = capture.read()
            if not success:
                print(f"Finished reading video. Total frames processed: {frame_index}", flush=True)
                break

            h, w, _ = frame.shape
            if h > w:
                target_h = w
                start_y = (h - target_h) // 2
                end_y = start_y + target_h
                frame = frame[start_y:end_y, 0:w]

            if frame_index % frame_step == 0:
                # 2. 추적기가 활성화되어 있다면 현재 프레임에서 박스 위치를 강제로 갱신
                bbox_cropped = None
                if tracker is not None:
                    track_success, bbox = tracker.update(frame)
                    if track_success:
                        # 추적된 좌표 추출 (x, y, width, height)
                        x, y, w, h = [int(v) for v in bbox]
                        
                        # 화면 밖에 나가지 않도록 예외 처리 후 이미지 크롭(Crop)
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        
                        if (x2 - x1) > 10 and (y2 - y1) > 10:
                            # 3. 박싱한 영역 내부만 잘라내어 MediaPipe의 분석 연산 범위로 한정
                            bbox_cropped = frame[y1:y2, x1:x2]
                            
                            # 시각화를 위해 원본 화면에 내가 지정한 추적 박스 그리기 (파란색)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, "Tracking Object", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 4. MediaPipe 분석 대상 결정 (박스 내부 vs 전체 화면)
                analysis_target = bbox_cropped if bbox_cropped is not None else frame
                
                rgb_frame = cv2.cvtColor(analysis_target, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                # 5. 크롭된 영역에서 나온 결과를 다시 원본 화면 좌표계로 변환하여 그리기
                if results.pose_landmarks:
                    if bbox_cropped is not None:
                        # 크롭 좌표계의 랜드마크들을 원본 이미지 상의 절대 좌표로 맵핑
                        for landmark in results.pose_landmarks.landmark:
                            # 크롭 이미지 내부 비율 좌표 -> 원본 이미지 픽셀 좌표 변환
                            pixel_x = int(landmark.x * bbox_cropped.shape[1]) + x1
                            pixel_y = int(landmark.y * bbox_cropped.shape[0]) + y1
                            
                            # MediaPipe 내부 규격을 유지하기 위해 원본 비율 좌표로 다시 환산
                            landmark.x = pixel_x / frame.shape[1]
                            landmark.y = pixel_y / frame.shape[0]

                    mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
                
                cv2.imshow('MediaPipe Pose Video', frame)
                frame_data[str(frame_index)] = serialize_landmarks(results.pose_landmarks)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User interrupted the process. Saving progress...", flush=True)
                    break
            else:
                frame_data[str(frame_index)] = serialize_landmarks(None)
            
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
            cv2.destroyAllWindows()
            cv2.waitKey(1) 
            
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
