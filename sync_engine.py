"""Synchronize demo and user angle sequences with FastDTW.

The input sequences are expected to be per-frame knee-angle values that were
derived from pose landmarks, for example using helpers from
``geometry_utils.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict

from fastdtw import fastdtw


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
    """Align two angle sequences and flag large deviations.

    Args:
        demo_angles: Demo knee-angle values ordered by frame.
        user_angles: User knee-angle values ordered by frame.
        threshold_degrees: Absolute angle difference required to mark a frame
            pair as anomalous.

    Returns:
        A ``SyncResult`` containing the FastDTW distance, the matched frame
        path, and the anomaly frame list.

    Raises:
        ValueError: If either sequence has no valid angle samples after missing
            values are removed.
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
    """Convenience wrapper that returns just the path and anomaly frames."""

    result = synchronize_angle_sequences(
        demo_angles=demo_angles,
        user_angles=user_angles,
        threshold_degrees=threshold_degrees,
    )
    return result.path, result.anomaly_frames