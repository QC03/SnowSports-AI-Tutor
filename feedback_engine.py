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


@dataclass(frozen=True)
class Selection:
    """레벨, 종목, 기술 선택을 나타내는 데이터 클래스입니다."""

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
        raise ValueError(f"지원하지 않는 종목입니다: {sport}")
    if level not in SPORT_CURRICULUM[sport]:
        raise ValueError(f"지원하지 않는 레벨입니다 {sport}: {level}")
    if technique not in SPORT_CURRICULUM[sport][level]:
        raise ValueError(f"지원하지 않는 기술입니다 {sport} {level}: {technique}")
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

    raise ValueError(f"지원하지 않는 이벤트 이름입니다: {event_name}")


def selection_from_any(
    sport: str | None = None,
    level: str | None = None,
    technique: str | None = None,
    event_name: str | None = None,
) -> Selection:
    if event_name:
        return selection_from_legacy_event(event_name)
    if sport is None or level is None or technique is None:
        raise ValueError("sport, level, technique 는 모두 필요합니다.")
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

    return "\n".join(lines)


def _call_openai_compatible_chat(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("genai_api_key")
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

    payload = {
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
    
    try:
        '''
        from openai import OpenAI
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/")
        model = os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL)
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.responses.create(
            model=model,
            input=payload["messages"]
        )
        data = response.output_text if hasattr(response, 'output_text') else response.to_dict()
        return data if isinstance(data, str) else None
        '''
    
        from google import genai
        model = os.getenv("genai_model", "gemini-3.5-flash")
        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=model,
            input=f"{payload['messages'][0]['content']}\n{payload['messages'][1]['content']}"
        )
        return interaction.output_text # type: ignore

    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        return None
    
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

def generate_llm_feedback_report(
    selection: Selection,
    analysis_result: Mapping[str, Any] | None = None,
    sync_result: Mapping[str, Any] | None = None,
) -> str:
    """LLM을 사용하여 분석 결과를 기반으로 한국어 피드백 보고서를 생성합니다."""

    context = build_llm_context(selection, analysis_result, sync_result)
    return _call_openai_compatible_chat(context) or "LLM 피드백을 생성할 수 없습니다."


__all__ = [
    "LEGACY_EVENT_ALIASES",
    "Selection",
    "SPORT_CURRICULUM",
    "build_llm_context",
    "build_selection",
    "build_selection_summary",
    "generate_llm_feedback_report",
    "list_levels",
    "list_sports",
    "list_techniques",
    "selection_from_any",
    "selection_from_legacy_event",
]