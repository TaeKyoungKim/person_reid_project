# AI 기반 사람 행동 분석 및 모니터링 시스템

본 프로젝트는 거리의 CCTV나 웹캠에서 실시간 영상을 분석하여 사람의 수를 세고(Counting), 비정상적인 상황(쓰러짐, 배회)을 감지하는 지능형 모니터링 시스템입니다. YOLOv8 객체 탐지 알고리즘 및 YOLO-pose 관절 추정 알고리즘을 결합하여 가볍고 실시간 처리가 가능한 분석 파이프라인을 구축하였습니다. 

---

## 🚀 주요 기능
1. **인원수 카운팅**: 현재 프레임에 존재하는 사람들의 총 수를 실시간으로 계산하고 아이디별로 추적(Tracking)합니다.
2. **쓰러짐(Fall) 감지**: 추출된 사람의 골격(Keypoints) 각도 및 탐지 박스의 종횡비를 활용하여 길거리에 사람이 쓰러진 응급 상황을 즉시 감지합니다.
3. **배회(Loitering) 및 정지 감지**: 지정된 시간(초) 동안 일정 범위 이상 벗어나지 않고 멈춰 있는 사람(배회자/거동 불편자 등)을 색출하여 경고를 발생시킵니다.
4. **실시간 정보 제공 API**: FastAPI 서버를 내장하여 외부 프론트엔드나 웹 서버에서 현재 탐지 상태를 JSON 형태로 요청할 수 있습니다.

---

## 📂 디렉토리 구조
주요 소스코드는 `src/person_reid` 내부에 기능별로 모듈화되어 있습니다.
```text
person_reid_project/
├── main.py                         # 전체 프로그램 진입점 및 영상 분석 메인 루프
├── pyproject.toml                  # 프로젝트 의존성(Packages) 정의 파일
├── data/                           # (옵션) 비디오 샘플을 넣기 위한 데이터 폴더
└── src/person_reid/
    ├── api/                        # 실시간 상태 정보를 JSON으로 내려주는 FastAPI 모듈
    ├── behavior_analysis/          # 추적 ID 이력 및 관절 이력을 바탕으로 행동(쓰러짐/배회) 감지 분석 모듈
    ├── config/                     # 임계값(Threshold), 모델 정보 등 전역 설정 파일 관리 모듈
    ├── detection/                  # YOLOv8 기반 객체 탐지 및 ByteTrack 알고리즘 관리
    ├── pose_estimation/            # YOLOv8-pose 모델을 통한 사람의 골격 점 17개 검출
    └── utils/                      # 영상 읽기기/쓰기, OpenCV 시각화, 프로젝트 로거
```

---

## 🛠️ 설치 방법

본 프로젝트는 최신 패키지 매니저인 `uv`를 사용하도록 설정되어 있습니다.

```bash
# 1. uv를 통한 환경 구축 및 패키지 설치
uv sync

# 만약 모르는 패키지 경고가 뜬다면, 개별로 갱신 명령이 가능합니다.
uv pip install -e .
```

---

## 💻 사용 방법

`main.py` 스크립트를 통해 영상을 실시간으로 분석합니다. 터미널(또는 명령 프롬프트)에서 아래의 명령어를 입력하세요.

### 기본 실행 (웹캠 0번 카메라 연결 시)
```bash
uv run python main.py
```

### 테스트용 영상 파일 실행
CCTV 영상 파일이 있다면 파일 경로를 인자로 넣어 작동시킬 수 있습니다.
```bash
uv run python main.py --source ./data/cctv_sample.mp4
```

### 분석된 결과를 비디오 파일로 저장
시각화된 추적 결과 및 경고 표출 화면을 저장하고 싶다면 `--output` 옵션을 사용하세요.
```bash
uv run python main.py --source ./data/cctv_sample.mp4 --output ./data/result.mp4
```

### 기타 실행 옵션
- `--headless`: 화면(팝업 창)을 띄우지 않고 백그라운드에서 분석만 돌립니다. 서버에 배포할 때 유용합니다.
- `--no-api`: `http://localhost:8000/api/status` 로 열리는 상태 알람 API 서버를 끕니다. 간단한 화면 확인만 할 때 유용합니다.
- 종료 시: 실행된 팝업 영상창을 클릭하고 영어 키보드로 `q` 를 누르거나 터미널에서 `Ctrl+c`를 누르면 종료됩니다.

---

## ⚙️ 설정값 조정
`src/person_reid/config/settings.py`에서 민감도를 수정하여 시스템을 통제할 수 있습니다.
* `self.detection.confidence_threshold` : (기본 0.5) 사람이 인식되는 최소 확률 점수입니다. 오탐지가 발생하면 올리세요.
* `self.behavior.inactivity_threshold` : (기본 30초) 몇 초 이상 멈춰 있어야 '배회'로 판단할지 설정합니다.
