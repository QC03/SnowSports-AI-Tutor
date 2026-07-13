from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict

from fastdtw import fastdtw # type: ignore


class FramePair(TypedDict):
    demo_frame: int
    user_frame: int


class AnomalyFrame(TypedDict):
    demo_frame: int
    user_frame: int
    demo_angle: float
    user_angle: float
    angle_diff: float


@dataclass(frozen=True)
class SyncResult:
    distance: float
    path: list[FramePair]
    anomaly_frames: list[AnomalyFrame]


def _prepare_angle_sequence(angles: Sequence[float | None]) -> list[tuple[int, float]]:
    prepared_sequence: list[tuple[int, float]] = []
    for frame_index, angle in enumerate(angles):
        if angle is None:
            continue
        prepared_sequence.append((frame_index, float(angle)))
    return prepared_sequence


def _frame_pairs_from_path(
    demo_sequence: Sequence[tuple[int, float]],
    user_sequence: Sequence[tuple[int, float]],
    path: Sequence[tuple[int, int]],
) -> tuple[list[FramePair], list[tuple[int, int, float, float]]]:
    frame_pairs: list[FramePair] = []
    matched_angles: list[tuple[int, int, float, float]] = []

    for demo_index, user_index in path:
        demo_frame, demo_angle = demo_sequence[demo_index]
        user_frame, user_angle = user_sequence[user_index]
        frame_pairs.append({"demo_frame": demo_frame, "user_frame": user_frame})
        matched_angles.append((demo_frame, user_frame, demo_angle, user_angle))

    return frame_pairs, matched_angles


def _extract_anomaly_frames(
    matched_angles: Sequence[tuple[int, int, float, float]],
    threshold_degrees: float,
) -> list[AnomalyFrame]:
    anomaly_frames: list[AnomalyFrame] = []

    for demo_frame, user_frame, demo_angle, user_angle in matched_angles:
        angle_diff = abs(demo_angle - user_angle)
        if angle_diff < threshold_degrees:
            continue

        anomaly_frames.append(
            {
                "demo_frame": demo_frame,
                "user_frame": user_frame,
                "demo_angle": demo_angle,
                "user_angle": user_angle,
                "angle_diff": angle_diff,
            }
        )

    return anomaly_frames


def synchronize_angle_sequences(
    demo_angles: Sequence[float | None],
    user_angles: Sequence[float | None],
    threshold_degrees: float = 15.0,
) -> SyncResult:
    """정렬된 데모 각도 시퀀스와 사용자 각도 시퀀스를 FastDTW를 사용하여 동기화합니다.

    Args:
        demo_angles: 데모의 knee-angle 값이 프레임 순서대로 정렬된 시퀀스.
        user_angles: 사용자 knee-angle 값이 프레임 순서대로 정렬된 시퀀스.
        threshold_degrees: 절대값 각도 차이가 이 임계값보다 큰 경우, 해당 프레임 쌍을 이상치로 간주합니다.

    Returns:
        A ``SyncResult`` 객체, 여기에는 동기화 거리, 프레임 쌍 경로, 이상치 프레임 목록이 포함됩니다.

    Raises:
        ValueError: demo_angles 또는 user_angles에 유효한 값이 없는 경우.
    """

    prepared_demo = _prepare_angle_sequence(demo_angles)
    prepared_user = _prepare_angle_sequence(user_angles)

    if not prepared_demo:
        raise ValueError("demo_angles에는 최소 1개의 유효한 값이 필요합니다.")
    if not prepared_user:
        raise ValueError("user_angles에는 최소 1개의 유효한 값이 필요합니다.")

    distance, path = fastdtw(
        [angle for _, angle in prepared_demo],
        [angle for _, angle in prepared_user],
        dist=lambda left, right: abs(float(left) - float(right)),
    )

    frame_pairs, matched_angles = _frame_pairs_from_path(
        prepared_demo,
        prepared_user,
        path,
    )
    anomaly_frames = _extract_anomaly_frames(matched_angles, threshold_degrees)

    return SyncResult(
        distance=float(distance),
        path=frame_pairs,
        anomaly_frames=anomaly_frames,
    )


def sync_demo_and_user_angles(
    demo_angles: Sequence[float | None],
    user_angles: Sequence[float | None],
    threshold_degrees: float = 15.0,
) -> tuple[list[FramePair], list[AnomalyFrame]]:
    """데모 각도 시퀀스와 사용자 각도 시퀀스를 동기화하고 이상치 프레임을 반환합니다."""

    result = synchronize_angle_sequences(
        demo_angles=demo_angles,
        user_angles=user_angles,
        threshold_degrees=threshold_degrees,
    )
    return result.path, result.anomaly_frames