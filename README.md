# SnowSports-AI-Tutor

스키와 스노보드 동작을 영상 기반으로 분석하기 위한 유틸리티 모음입니다. MediaPipe Pose로 관절 좌표를 추출하고, 정규화와 각도 계산을 통해 동작 차이를 비교한 뒤, KSIA 스타일의 규칙 기반 피드백을 생성하는 흐름으로 구성되어 있습니다.

## 핵심 흐름

1. 비디오에서 포즈 랜드마크를 추출합니다.
2. 어깨 너비를 기준으로 관절 좌표를 정규화합니다.
3. 무릎, 발목 같은 관절 각도와 자세 비율을 계산합니다.
4. Demo 동작과 사용자 동작의 시퀀스를 정렬합니다.
5. 사용자가 선택한 종목/레벨/기술 기준으로 피드백을 생성합니다.
6. 가능한 경우 OpenAI 호환 LLM으로 한국어 코칭 피드백을 작성합니다.

## 프로젝트 구성

- `extract_demo.py`: MediaPipe Pose를 사용해 비디오 프레임별 33개 랜드마크를 JSON으로 저장합니다.
- `geometry_utils.py`: 좌표 정규화, 자세 비율 계산, 관절 각도 계산 함수를 제공합니다.
- `sync_engine.py`: FastDTW로 Demo/사용자 각도 시퀀스를 정렬하고, 이상 프레임을 찾습니다.
- `feedback_engine.py`: 분석 결과를 KSIA 스타일의 한국어 피드백 문장으로 변환합니다.

## 설치

### 필수 환경

- Python 3.10 이상

### 의존성 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 0. Demo/사용자 영상 통합 실행

두 영상을 한 번에 분석하려면 `run_pipeline.py`를 사용합니다. 실행 시 스키/스노보드와 레벨, 기술을 직접 지정하거나, 인자를 생략하면 대화형으로 선택할 수 있습니다.

```bash
python run_pipeline.py <데몬_영상> <사용자_영상> --sport 스키 --level 레벨1 --technique 스노우플라우턴
```

예시:

```bash
python run_pipeline.py "C:\\Users\\tjdwn\\OneDrive\\Desktop\\Development\\Sample\\DemoSample1.mp4" "C:\\Users\\tjdwn\\OneDrive\\Desktop\\Development\\Sample\\MySample1.mp4"
```

긴 영상에서 빠르게 확인하고 싶다면 `--frame-step 10`처럼 샘플링 간격을 늘릴 수 있습니다.

LLM 피드백을 사용하려면 다음 환경변수를 설정합니다.

```bash
set OPENAI_API_KEY=...
set OPENAI_MODEL=gpt-5.4-mini
set OPENAI_BASE_URL=https://api.openai.com/v1
```

API 키가 없으면, 코드가 규칙 기반 피드백으로 자동 대체합니다.

### 1. 비디오에서 포즈 랜드마크 추출

```bash
python extract_demo.py <비디오_파일_경로>
```

예시:

```bash
python extract_demo.py videos/demo.mp4
```

기본 출력 파일은 입력 비디오와 같은 폴더에 `*_pose.json` 이름으로 저장됩니다.

커스텀 출력 경로를 지정하려면:

```bash
python extract_demo.py <비디오_파일_경로> -o <출력_JSON_경로>
```

예시:

```bash
python extract_demo.py videos/demo.mp4 -o outputs/demo_landmarks.json
```

### 2. 정규화 및 각도 계산

`geometry_utils.py`는 직접 실행하는 스크립트가 아니라, 다른 코드에서 import해서 사용하는 유틸리티 모듈입니다.

주요 함수:

- `normalize_landmarks_by_shoulder_width(...)`: 어깨 중심 기준 정규화
- `calculate_stance_ratio(...)`: 발목 간 거리로 자세 폭 계산
- `calculate_joint_angle(...)`: 세 점 A-B-C의 관절 각도 계산

예시:

```python
from geometry_utils import calculate_joint_angle

angle = calculate_joint_angle(left_hip, left_knee, left_ankle)
```

### 3. Demo/사용자 시퀀스 정렬

`sync_engine.py`도 import용 모듈입니다. Demo와 사용자 각도 시퀀스를 정렬하고, 각도 차이가 큰 프레임을 이상 프레임으로 표시합니다.

주요 함수:

- `synchronize_angle_sequences(...)`: FastDTW 기반 정렬 결과와 이상 프레임을 함께 반환
- `sync_demo_and_user_angles(...)`: 경량 래퍼로 경로와 이상 프레임만 반환

예시:

```python
from sync_engine import sync_demo_and_user_angles

path, anomaly_frames = sync_demo_and_user_angles(demo_angles, user_angles)
```

### 4. KSIA 피드백 생성

`feedback_engine.py`는 종목/레벨/기술 선택값과 분석 결과 딕셔너리를 입력받아 한국어 피드백 문장을 생성합니다. LLM 호출이 가능한 환경에서는 요약 결과를 기반으로 더 자연스러운 코칭 문장으로 변환합니다.

주요 함수:

- `build_ksia_feedback_items(...)`: 조건에 맞는 피드백 항목 목록 생성
- `generate_ksia_feedback_report(...)`: 마크다운 형태의 피드백 리포트 생성

예시:

```python
from feedback_engine import build_selection, generate_ksia_feedback_report

report = generate_ksia_feedback_report(
  build_selection("스키", "레벨1", "스노우플라우턴"),
    {
        "stance_ratio_range": 0.18,
        "outside_knee_overextended": True,
    },
)
print(report)
```

## 출력 형식

`extract_demo.py`가 생성하는 JSON은 프레임 번호를 키로 하고, 각 프레임마다 33개 관절의 좌표를 저장합니다.

```json
{
  "0": [
    {"x": 0.5, "y": 0.3, "z": -0.1},
    {"x": 0.51, "y": 0.31, "z": -0.11}
  ],
  "1": [...]
}
```

- 각 키는 프레임 번호입니다.
- 각 값은 33개 랜드마크의 `(x, y, z)` 좌표 목록입니다.
- 좌표는 MediaPipe 정규화 좌표이며, `z`는 깊이 축 값입니다.

## 참고 사항

- `feedback_engine.py`는 스키/스노보드의 레벨 1~3 기술 목록을 모두 포함합니다.
- `sync_engine.py`는 `None` 값을 제외하고 유효한 각도만으로 동기화합니다.
- `geometry_utils.py`는 기본적으로 어깨 11/12번 랜드마크를 기준으로 정규화합니다.
- `run_pipeline.py`는 `--event` 레거시 인자도 받아서 기존 방식과 호환됩니다.

## 개발 메모

이 저장소는 현재 단일 애플리케이션보다는 분석 파이프라인용 모듈 세트에 가깝습니다. 이후 필요하면 다음 구성을 추가하는 것이 좋습니다.

- 예제 입력/출력 데이터
- CLI 통합 실행 스크립트
- 간단한 테스트 코드
- Web UI 또는 API 서버
