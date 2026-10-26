# 잔상 기반 비디오 해석 재현 패키지

[English / 전체 명령 안내](README.md)

![선택 프레임과 실제 잔상 비교](assets/afterimage_demo.gif)

고정 CCTV에서 주요 사건과 시간적 전개를 해석하면서 시각적 입력량을 줄이는 연구입니다. GIF는 Weizmann run의 선택 프레임 6회와 실제 잔상 2장을 비교합니다. 공통 시작·끝 2장을 더하면 입력은 각각 8장과 4장입니다. 같은 사람이 시간에 따라 여러 위치에 나타난 것이며, GIF 속도는 설명용입니다. [출처와 해시](assets/PROVENANCE.json).

## 제공하는 내용

- 비디오 폴더 → 잔상 및 비교 조건 생성 코드
- 독립 GT 자료: GT1 303건, GT2 90건
- 입력 순서와 중복 등장, 파일 해시를 보존하는 manifest
- GPT/Claude API 해석 및 비디오별 응답 저장 코드
- 저장된 응답을 별도 GT로 채점하는 코드
- 개별 점수 및 평균 내보내기

## 1. 내려받기와 환경 설치

```powershell
git clone https://github.com/Jeon-minseong/Afterimage.git
cd Afterimage
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-generation.txt
python -m unittest discover -s tests -v
```

잔상 생성은 Windows/Python 3.10/NVIDIA CUDA가 현재 검증 기준입니다. 다른 PC 신규 설치까지 검증한 것은 아닙니다. 해석·채점 단계는 Python 3.10 이상 표준 라이브러리만 사용하며 GPU가 필요 없습니다.

## 2. 원본 영상과 모델 가중치 준비

원본 303편과 가중치는 저장소에 포함하지 않습니다. 적법하게 확보한 영상을 `C:\datasets\afterimage`에 두고, `configs/dataset_catalog.json`의 해시와 대응시킵니다. 가중치 경로는 `C:\models\afterimage`이며 `yolo11n.pt`와 `rmbg2`의 네 파일이 필요합니다. [정확한 구조와 출처](README.md#2-obtain-videos-and-model-assets-separately)를 따릅니다.

## 3. 잔상과 네 조건 생성

```powershell
python -m afterimage generate --videos C:\datasets\afterimage --weights C:\models\afterimage --out runs/inputs
python -m afterimage validate --inputs runs/inputs
```

| 조건 | 입력 수 |
|---|---:|
| 균일 선택 원본 프레임 | M+2 |
| 잔상 합성에 사용한 선택 프레임 | M+2 |
| 잔상 시퀀스 | J+2 |
| 잔상과 동일 이미지 수의 균일 프레임 | J+2 |

M은 고유 프레임 수가 아니라 **중복을 포함한 등장 횟수**, J는 잔상 수입니다. 모든 조건은 동일한 시작·끝 원본을 포함합니다.

## 4. 요청 계획 및 무료 사전 점검

```powershell
python -m afterimage prepare-inference --inputs runs/inputs --out runs/inference_plan --repeats 1
python -m afterimage run --plan runs/inference_plan --inputs runs/inputs --out runs/answers
```

기본 실행은 API를 호출하지 않습니다. 두 모델 1회는 3,144요청, 3회는 9,432요청입니다. 모델·프롬프트·GT 검토·채점 기준·예산은 본실험 전에 확정해야 합니다. `configs/experiment.json`의 모델과 요금은 후보 설정이므로 실행 시점에 확인합니다.

## 5. API 해석 실행

키는 별도로 `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` 환경 변수에 입력합니다. 저장소나 보고서에 기록하지 않습니다. 검토한 계획 해시, `--execute`, 제공업체별 예산 상한을 명시해야 유료 실행됩니다. [실행 명령](README.md#5-generate-vlm-answers-paid-explicit-opt-in)을 따릅니다.

응답은 `runs/answers/videos/비디오ID/` 아래에 모델·조건·반복 번호별 JSON으로 저장됩니다. 원본 응답, 입력·프롬프트 해시와 실제 API 사용량을 함께 남깁니다. 해석 요청에는 GT가 들어가지 않습니다.

## 6. 별도 GT로 채점

```powershell
python -m afterimage prepare-scoring --answers runs/answers --gt gt/references.json --out runs/scoring_plan
python -m afterimage run --plan runs/scoring_plan --out runs/scores
```

이 명령도 사전 점검입니다. 유료 채점에는 별도 계획 해시 승인과 예산이 필요합니다. 해석과 채점에 같은 비용 장부를 사용합니다. GT는 모델 보조 작성이며 사람의 전수 검증 완료 자료가 아닙니다. 현재 0–5점 채점 기준은 확정 전 후보입니다.

## 7. 점수표 생성

```powershell
python -m afterimage summarize --scores runs/scores --out runs/tables
```

개별 점수 CSV와 요약 JSON을 생성합니다. 반복을 영상별로 먼저 평균내고 영상을 평균내며, 데이터셋과 GT1/GT2는 구분합니다.

## 현재 검증 범위

새 생성 진입점으로 Weizmann run과 UT push를 캐시 없이 생성하여 네 조건의 48회 입력 등장, 27개 PNG가 기존 입력과 일치했습니다. 전체 303편의 캐시 없는 재생성, 새 PC 설치 및 이 패키지를 이용한 실제 API 해석·채점은 아직 완료하지 않았습니다. 기존 CLI·구독앱·API 시험 결과는 본실험 결과와 혼합하지 않습니다.

[검증 기록](validation/STATUS.json) · [배포 상태](docs/RELEASE_STATUS.md) · [권리·출처](THIRD_PARTY_NOTICES.md)
