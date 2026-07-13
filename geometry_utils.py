from __future__ import annotations

import math
from typing import Mapping, Sequence, TypedDict


class LandmarkDict(TypedDict):
    x: float | None
    y: float | None
    z: float | None


PointLike = Mapping[str, float | None] | Sequence[float | None]

POSE_LANDMARK_COUNT = 33
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_ANKLE_INDEX = 27
RIGHT_ANKLE_INDEX = 28


def _as_point(landmark: PointLike) -> tuple[float | None, float | None, float | None]:
    if isinstance(landmark, Mapping):
        return (
            landmark.get("x"),
            landmark.get("y"),
            landmark.get("z"),
        )

    values = list(landmark)
    if len(values) < 3:
        values.extend([None] * (3 - len(values)))
    return values[0], values[1], values[2]


def _euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def _build_landmark(x: float | None, y: float | None, z: float | None) -> LandmarkDict:
    return {"x": x, "y": y, "z": z}


def normalize_landmarks_by_shoulder_width(
    landmarks: Sequence[PointLike],
    left_shoulder_index: int = LEFT_SHOULDER_INDEX,
    right_shoulder_index: int = RIGHT_SHOULDER_INDEX,
) -> tuple[list[LandmarkDict], float | None]:
    """가운데 어깨 너비를 기준으로 포즈 랜드마크를 정규화합니다.

    반환 값은 정규화된 랜드마크 목록과 어깨 너비입니다.
    어깨 너비를 계산할 수 없는 경우,
    모든 정규화된 좌표는 ``None``으로 반환되며 스케일도 ``None``입니다.
    """

    if len(landmarks) < POSE_LANDMARK_COUNT:
        padded_landmarks: list[PointLike] = list(landmarks) + [
            {"x": None, "y": None, "z": None}
            for _ in range(POSE_LANDMARK_COUNT - len(landmarks))
        ]
    else:
        padded_landmarks = list(landmarks[:POSE_LANDMARK_COUNT])

    left_shoulder = _as_point(padded_landmarks[left_shoulder_index])
    right_shoulder = _as_point(padded_landmarks[right_shoulder_index])

    if (
        left_shoulder[0] is None
        or left_shoulder[1] is None
        or right_shoulder[0] is None
        or right_shoulder[1] is None
    ):
        return [
            _build_landmark(None, None, None)
            for _ in range(POSE_LANDMARK_COUNT)
        ], None

    shoulder_width = _euclidean_distance(
        (float(left_shoulder[0]), float(left_shoulder[1])),
        (float(right_shoulder[0]), float(right_shoulder[1])),
    )
    if shoulder_width == 0:
        return [
            _build_landmark(None, None, None)
            for _ in range(POSE_LANDMARK_COUNT)
        ], None

    shoulder_midpoint_x = (float(left_shoulder[0]) + float(right_shoulder[0])) / 2.0
    shoulder_midpoint_y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2.0
    if left_shoulder[2] is not None and right_shoulder[2] is not None:
        shoulder_midpoint_z = (float(left_shoulder[2]) + float(right_shoulder[2])) / 2.0
    else:
        shoulder_midpoint_z = 0.0

    normalized_landmarks: list[LandmarkDict] = []
    for landmark in padded_landmarks:
        x, y, z = _as_point(landmark)
        if x is None or y is None:
            normalized_landmarks.append(_build_landmark(None, None, None))
            continue

        normalized_x = (float(x) - shoulder_midpoint_x) / shoulder_width
        normalized_y = (float(y) - shoulder_midpoint_y) / shoulder_width
        normalized_z = None
        if z is not None:
            normalized_z = (float(z) - shoulder_midpoint_z) / shoulder_width

        normalized_landmarks.append(
            _build_landmark(normalized_x, normalized_y, normalized_z)
        )

    return normalized_landmarks, shoulder_width


def calculate_stance_ratio(
    normalized_landmarks: Sequence[PointLike],
    left_ankle_index: int = LEFT_ANKLE_INDEX,
    right_ankle_index: int = RIGHT_ANKLE_INDEX,
) -> float | None:
    """반환 값은 왼쪽 발목과 오른쪽 발목 사이의 거리입니다."""

    if len(normalized_landmarks) <= max(left_ankle_index, right_ankle_index):
        return None

    left_ankle = _as_point(normalized_landmarks[left_ankle_index])
    right_ankle = _as_point(normalized_landmarks[right_ankle_index])

    if left_ankle[0] is None or left_ankle[1] is None or right_ankle[0] is None or right_ankle[1] is None:
        return None

    left_values = [float(value) for value in left_ankle if value is not None]
    right_values = [float(value) for value in right_ankle if value is not None]
    common_dimensions = min(len(left_values), len(right_values))
    if common_dimensions < 2:
        return None

    return _euclidean_distance(left_values[:common_dimensions], right_values[:common_dimensions])


def calculate_joint_angle(
    point_a: PointLike,
    point_b: PointLike,
    point_c: PointLike,
) -> float | None:
    """반환 값은 점 A, B, C가 이루는 각도입니다. 점 B는 꼭짓점입니다. 각도는 0°에서 180° 사이입니다."""

    a = _as_point(point_a)
    b = _as_point(point_b)
    c = _as_point(point_c)

    if a[0] is None or a[1] is None or b[0] is None or b[1] is None or c[0] is None or c[1] is None:
        return None

    a_values = [float(value) for value in a if value is not None]
    b_values = [float(value) for value in b if value is not None]
    c_values = [float(value) for value in c if value is not None]
    common_dimensions = min(len(a_values), len(b_values), len(c_values))
    if common_dimensions < 2:
        return None

    vector_ba = [a_values[index] - b_values[index] for index in range(common_dimensions)]
    vector_bc = [c_values[index] - b_values[index] for index in range(common_dimensions)]

    norm_ba = math.sqrt(sum(component * component for component in vector_ba))
    norm_bc = math.sqrt(sum(component * component for component in vector_bc))
    if norm_ba == 0 or norm_bc == 0:
        return None

    dot_product = sum(left * right for left, right in zip(vector_ba, vector_bc))
    cosine_value = dot_product / (norm_ba * norm_bc)
    cosine_value = max(-1.0, min(1.0, cosine_value))
    return math.degrees(math.acos(cosine_value))
