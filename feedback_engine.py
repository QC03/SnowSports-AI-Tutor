"""Build feedback text from motion analysis results.

This module stores the snow sports curriculum, turns a compact analysis
summary into a structured report, and optionally asks an OpenAI-compatible LLM
to turn that summary into Korean coaching feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Sequence
from urllib import error, request


from dotenv import load_dotenv
load_dotenv()

SPORT_CURRICULUM: dict[str, dict[str, list[str]]] = {
    "스키": {
        "레벨1": [
            "스노우플라우",
            "스노우플라우턴",
            "슈템턴",
            "페러렐롱턴",
            "페러렐숏턴",
        ],
        "레벨2": ["다이나믹_롱턴", "다이나믹_미들턴", "다이나믹_숏턴", "모굴"],
        "레벨3": ["카빙_롱턴", "카빙_미들턴", "카빙_숏턴", "모굴"],
    },
    "스노보드": {
        "레벨1": ["사이드슬립", "팬듈럼", "비기너턴", "너비스턴"],
        "레벨2": ["인터미디어트_슬라이딩턴", "인터미디어트_카빙턴"],
        "레벨3": ["어드밴스드_슬라이딩턴", "어드밴스드_카빙턴", "모굴"],
    },
}

LEGACY_EVENT_ALIASES: dict[str, tuple[str, str, str]] = {
    "스키_레벨1_스노우플라우턴(보겐)": ("스키", "레벨1", "스노우플라우턴"),
    "스키_레벨2_페러렐롱턴": ("스키", "레벨2", "페러렐롱턴"),
}

DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

STANCE_VARIATION_RANGE_THRESHOLD = 0.15
STANCE_VARIATION_CV_THRESHOLD = 0.10
KNEE_EXTENSION_MARGIN_DEGREES = 10.0
ANGULATION_MISMATCH_THRESHOLD_DEGREES = 8.0


@dataclass(frozen=True)
class FeedbackReport:
    """Structured feedback items plus the rendered markdown body."""

    selection_label: str
    items: list[str]

    def render(self) -> str:
        lines = [f"- 종목: {self.selection_label}"]
        if self.items:
            lines.extend(f"- {item}" for item in self.items)
        else:
            lines.append("- 현재 분석 결과에서는 KSIA 피드백 대상 문제가 확인되지 않았습니다.")
        return "\n".join(lines)


@dataclass(frozen=True)
class Selection:
    """Chosen sport, level, and technique."""

    sport: str
    level: str
    technique: str

    @property
    def key(self) -> str:
        return f"{self.sport}::{self.level}::{self.technique}"

    @property
    def label(self) -> str:
        return f"{self.sport} {self.level} {self.technique}"


def list_sports() -> list[str]:
    return list(SPORT_CURRICULUM.keys())


def list_levels(sport: str) -> list[str]:
    return list(SPORT_CURRICULUM.get(sport, {}))


def list_techniques(sport: str, level: str) -> list[str]:
    return list(SPORT_CURRICULUM.get(sport, {}).get(level, []))


def build_selection(sport: str, level: str, technique: str) -> Selection:
    if sport not in SPORT_CURRICULUM:
        raise ValueError(f"Unsupported sport: {sport}")
    if level not in SPORT_CURRICULUM[sport]:
        raise ValueError(f"Unsupported level for {sport}: {level}")
    if technique not in SPORT_CURRICULUM[sport][level]:
        raise ValueError(f"Unsupported technique for {sport} {level}: {technique}")
    return Selection(sport=sport, level=level, technique=technique)


def selection_from_legacy_event(event_name: str) -> Selection:
    if event_name in LEGACY_EVENT_ALIASES:
        sport, level, technique = LEGACY_EVENT_ALIASES[event_name]
        return build_selection(sport, level, technique)

    for sport, levels in SPORT_CURRICULUM.items():
        for level, techniques in levels.items():
            if event_name == f"{sport}_{level}_{techniques[0]}":
                return build_selection(sport, level, techniques[0])
            if event_name == f"{sport} {level} {techniques[0]}":
                return build_selection(sport, level, techniques[0])

    raise ValueError(f"Unsupported event name: {event_name}")


def selection_from_any(
    sport: str | None = None,
    level: str | None = None,
    technique: str | None = None,
    event_name: str | None = None,
) -> Selection:
    if event_name:
        return selection_from_legacy_event(event_name)
    if sport is None or level is None or technique is None:
        raise ValueError("Sport, level, and technique are required when event_name is not provided")
    return build_selection(sport, level, technique)


def build_selection_summary(selection: Selection) -> dict[str, str]:
    return {
        "sport": selection.sport,
        "level": selection.level,
        "technique": selection.technique,
        "key": selection.key,
        "label": selection.label,
    }


def _render_numeric_summary(analysis_result: Mapping[str, Any] | None) -> list[str]:
    data = dict(analysis_result or {})
    lines: list[str] = []

    if "stance_ratio_range" in data or "stance_ratio_cv" in data:
        stance_range = data.get("stance_ratio_range")
        stance_cv = data.get("stance_ratio_cv")
        lines.append(f"- 자세 폭 변화: range={stance_range}, cv={stance_cv}")

    if "apex_outside_knee_angle_gap" in data:
        lines.append(f"- 외측 무릎 각도 차이: {data.get('apex_outside_knee_angle_gap')}°")

    if "angulation_difference" in data:
        lines.append(f"- 외경/상체 정렬 차이: {data.get('angulation_difference')}°")

    return lines


def build_llm_context(
    selection: Selection,
    analysis_result: Mapping[str, Any] | None = None,
    sync_result: Mapping[str, Any] | None = None,
    rule_items: Sequence[str] | None = None,
) -> str:
    lines = [
        f"선택 종목: {selection.label}",
        "비교 기준: 데모 영상과 사용자 영상을 MediaPipe Pose 기반으로 비교한 결과.",
    ]

    if sync_result:
        lines.append(f"동기화 거리: {sync_result.get('distance')}")
        lines.append(f"매칭 프레임 수: {sync_result.get('matched_frames')}")
        lines.append(f"이상 프레임 수: {sync_result.get('anomaly_count')}")

    lines.extend(_render_numeric_summary(analysis_result))

    if rule_items:
        lines.append("규칙 기반 감지 항목:")
        lines.extend(f"- {item}" for item in rule_items)

    return "\n".join(lines)


def _call_openai_compatible_chat(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print("")
        print("너는 스키와 스노보드 동작을 평가하는 한국어 코치다. ")
        print("주어진 비교 요약만 사용해서 간결하고 구체적인 피드백을 작성하고, ")
        print("없는 정보를 추측하지 마라.")

        print("아래 분석 요약을 바탕으로 3~5개의 불릿으로 피드백을 작성해라. ")
        print("구성은 총평, 잘한 점, 개선점, 다음 연습 포인트를 포함하되, ")
        print("데모 대비 차이가 드러나도록 써라.\n\n")
        print(f"{prompt}")
        return None

    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/")
    model = os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 스키와 스노보드 동작을 평가하는 한국어 코치다. "
                    "주어진 비교 요약만 사용해서 간결하고 구체적인 피드백을 작성하고, "
                    "없는 정보를 추측하지 마라."
                ),
            },
            {
                "role": "user",
                "content": (
                    "아래 분석 요약을 바탕으로 3~5개의 불릿으로 피드백을 작성해라. "
                    "구성은 총평, 잘한 점, 개선점, 다음 연습 포인트를 포함하되, "
                    "데모 대비 차이가 드러나도록 써라.\n\n"
                    f"{prompt}"
                ),
            },
        ],
        "temperature": 0.2,
    }

    api_url = f"{base_url}/chat/completions"
    req = request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, ValueError, KeyError) as e:
        print(f"Error occurred while fetching LLM response: {e}")
        return None
    
    print(f"LLM response: {data}")

    choices = data.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _get_first_present_value(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_stance_variation_irregular(data: Mapping[str, Any]) -> bool:
    explicit_flag = _as_bool(
        _get_first_present_value(
            data,
            (
                "stance_variation_irregular",
                "stance_ratio_irregular",
                "stance_ratio_unstable",
                "wide_a_frame_irregular",
            ),
        )
    )
    if explicit_flag is not None:
        return explicit_flag

    stance_range = _as_float(
        _get_first_present_value(data, ("stance_ratio_range", "stance_variation_range"))
    )
    if stance_range is not None and stance_range >= STANCE_VARIATION_RANGE_THRESHOLD:
        return True

    stance_cv = _as_float(
        _get_first_present_value(data, ("stance_ratio_cv", "stance_variation_cv"))
    )
    if stance_cv is not None and stance_cv >= STANCE_VARIATION_CV_THRESHOLD:
        return True

    return False


def _is_outside_knee_overextended(data: Mapping[str, Any]) -> bool:
    explicit_flag = _as_bool(
        _get_first_present_value(
            data,
            (
                "outside_knee_overextended",
                "outside_knee_too_extended",
                "apex_outside_knee_too_straight",
                "apex_knee_overextended",
            ),
        )
    )
    if explicit_flag is not None:
        return explicit_flag

    demo_angle = _as_float(
        _get_first_present_value(
            data,
            (
                "demo_outside_knee_angle_at_apex",
                "demo_apex_outside_knee_angle",
                "demo_apex_knee_angle",
            ),
        )
    )
    user_angle = _as_float(
        _get_first_present_value(
            data,
            (
                "user_outside_knee_angle_at_apex",
                "user_apex_outside_knee_angle",
                "user_apex_knee_angle",
            ),
        )
    )
    angle_gap = _as_float(
        _get_first_present_value(
            data,
            (
                "apex_outside_knee_angle_gap",
                "apex_knee_angle_gap",
                "outside_knee_angle_gap",
            ),
        )
    )

    if angle_gap is not None:
        return angle_gap >= KNEE_EXTENSION_MARGIN_DEGREES

    if demo_angle is not None and user_angle is not None:
        return (user_angle - demo_angle) >= KNEE_EXTENSION_MARGIN_DEGREES

    return False


def _is_angulation_misaligned(data: Mapping[str, Any]) -> bool:
    explicit_flag = _as_bool(
        _get_first_present_value(
            data,
            (
                "angulation_misaligned",
                "angulation_loss",
                "external_edge_misaligned",
                "external_canting_misaligned",
            ),
        )
    )
    if explicit_flag is not None:
        return explicit_flag

    shoulder_slope = _as_float(
        _get_first_present_value(
            data,
            (
                "shoulder_line_slope",
                "shoulder_slope",
                "shoulder_angle",
            ),
        )
    )
    knee_slope = _as_float(
        _get_first_present_value(
            data,
            (
                "knee_line_slope",
                "knee_slope",
                "knee_angle",
            ),
        )
    )
    angulation_gap = _as_float(
        _get_first_present_value(
            data,
            (
                "angulation_difference",
                "angulation_gap",
                "slope_difference",
            ),
        )
    )

    if angulation_gap is not None:
        return angulation_gap >= ANGULATION_MISMATCH_THRESHOLD_DEGREES

    if shoulder_slope is not None and knee_slope is not None:
        return abs(shoulder_slope - knee_slope) >= ANGULATION_MISMATCH_THRESHOLD_DEGREES

    return False


def build_ksia_feedback_items(
    selection: Selection,
    analysis_result: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return the KSIA feedback bullets that match the selected event."""

    data = dict(analysis_result or {})
    items: list[str] = []

    if selection.key == "스키::레벨1::스노우플라우턴":
        if _is_stance_variation_irregular(data):
            items.append(
                "❌ 턴 제어 중 A자 넓이가 불규칙하게 변합니다. 발바닥 전체에 일정한 압력을 유지하세요."
            )
        if _is_outside_knee_overextended(data):
            items.append(
                "❌ 턴 할 때 바깥쪽 무릎이 너무 펴져 있어 체중 이동이 부족합니다. 데몬처럼 무릎을 더 구부리세요."
            )
        return items

    if selection.key == "스키::레벨2::페러렐롱턴":
        if _is_angulation_misaligned(data):
            items.append(
                "❌ 외경(Angulation) 자세가 무너져 상체가 안쪽으로 쏠리고 있습니다. 어깨 수평을 데몬처럼 유지하세요."
            )
        return items

    return items


def generate_ksia_feedback_report(
    selection: Selection,
    analysis_result: Mapping[str, Any] | None = None,
) -> str:
    """Render a markdown list report from the selected event and analysis."""

    report = FeedbackReport(
        selection_label=selection.label,
        items=build_ksia_feedback_items(selection, analysis_result),
    )
    return report.render()


def generate_llm_feedback_report(
    selection: Selection,
    analysis_result: Mapping[str, Any] | None = None,
    sync_result: Mapping[str, Any] | None = None,
    rule_items: Sequence[str] | None = None,
) -> str:
    """Render LLM feedback, falling back to a deterministic report if needed."""

    context = build_llm_context(selection, analysis_result, sync_result, rule_items)
    llm_text = _call_openai_compatible_chat(context)
    if llm_text:
        return llm_text

    fallback_items = list(rule_items or build_ksia_feedback_items(selection, analysis_result))
    fallback_report = FeedbackReport(selection_label=selection.label, items=fallback_items)
    return fallback_report.render()


__all__ = [
    "FeedbackReport",
    "LEGACY_EVENT_ALIASES",
    "Selection",
    "SPORT_CURRICULUM",
    "build_llm_context",
    "build_selection",
    "build_selection_summary",
    "build_ksia_feedback_items",
    "generate_llm_feedback_report",
    "generate_ksia_feedback_report",
    "list_levels",
    "list_sports",
    "list_techniques",
    "selection_from_any",
    "selection_from_legacy_event",
]