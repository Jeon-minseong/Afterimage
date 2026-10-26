# 잔상 기반 비디오 해석 재현 패키지

[English / 전체 명령 안내](README.md)

https://github.com/user-attachments/assets/0b2a896b-c771-4e6a-b6e3-baf3ce08f668

**UT-interaction Push: 원본 동작과 실제 잔상 두 장 비교.** 재생 버튼을 누르면 3.7초 원본을 원래 속도로 두 번 재생하고, 마지막 화면을 2초 유지합니다. 선택 프레임 6회가 잔상 2장으로 합성됩니다. 공통 시작·끝 원본을 포함하면 VLM 입력은 **8장 → 4장(50% 감소)**입니다.

[MP4 열기·다운로드](assets/ut_push_demo_v1.mp4) · [정지 화면](assets/ut_push_poster_v1.jpg) · [입력 해시와 표시 변환 기록](assets/UT_PUSH_PROVENANCE_v1.json)

오른쪽 잔상 두 장을 함께 배치한 것은 설명용입니다. 실제 API에는 순서가 정해진 개별 이미지를 전달하며, 시작·끝 원본은 입력 수에 포함하지만 이 화면에 따로 표시하지 않습니다. 이 영상은 모델의 해석 결과나 정확도 검증 결과가 아닙니다.

출처: M. S. Ryoo, J. K. Aggarwal, [UT-Interaction Dataset, ICPR SDHA contest (2010)](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html). 기존 Weizmann GIF와 출처 기록도 `assets/`에 보존했습니다.

고정 CCTV에서 주요 사건과 시간적 전개를 해석하면서 시각적 입력량을 줄이는 연구입니다.

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

이미지 수·실제 토큰·비용·대응 점수 비교표는 다음 명령으로 만듭니다. 원본 N은 영상 파일 해시로 대응한 메타데이터 프레임 수이며, 전수 디코딩 검증은 아닙니다.

```powershell
python tools/collect_frame_metadata.py --videos C:\datasets\afterimage --inputs runs/inputs --out runs/frame_counts.csv
python -m afterimage metrics --inputs runs/inputs --answers runs/answers --scores runs/scores --frame-counts runs/frame_counts.csv --out runs/paper_tables
```

원본 대비 선택은 `1-M/N`, 선택 대비 합성은 `1-J/M`, 시작·끝 포함 VLM 입력은 `1-(J+2)/(M+2)`로 계산합니다. 전체 합계 비율과 영상별 비율 평균을 구분합니다. 토큰 수는 이미지뿐 아니라 프롬프트와 JSON 구조를 포함한 실측 입력 토큰입니다. 이미지 수 감소율을 토큰 감소율로 대신 표기하지 않습니다. 비교표는 같은 반복의 선택 프레임 및 동일 이미지 수의 균일 프레임과 잔상을 각각 비교합니다.

## 현재 검증 범위

배치 실행은 [전체 명령 안내](README.md#batch-execution)를 따릅니다. `prepare-batch`는 유료 생성 없이 토큰 계산 API를 호출하고, `submit-batch`는 명시적인 실행 옵션과 배치 해시가 있어야 유료 접수합니다. 각 영상·조건·반복은 별도 요청이며, 같은 영상의 이미지를 여러 요청으로 쪼개지 않습니다. 접수한 모든 요청의 최대 출력 비용을 미리 예약하고, 비동기 결과는 요청 ID로 연결합니다. GT1은 세 데이터셋 사건 요약이고 GT2는 KISA 시간적 전개이므로, Weizmann 계획에는 GT1만 포함합니다.

새 생성 진입점으로 Weizmann run과 UT push를 캐시 없이 생성하여 네 조건의 48회 입력 등장, 27개 PNG가 기존 입력과 일치했습니다. 이후 Weizmann run·UT push·KISA 주정차 각 1편으로 두 모델·네 조건의 API 해석 32건과 별도 채점 32건을 완료했습니다. 정상 점검 비용은 $0.500936이며 해석 오류도 보존했습니다. 구조화 출력 설정은 JSON 형식을 보장하기 위한 것이며 해석의 정확성을 보장하지 않습니다. 배치 수집과 예산 예약을 포함한 오프라인 테스트 23개와 303편 입력/32개 응답·점수의 표 내보내기를 검증했습니다.

전체 303편의 캐시 없는 재생성, 새 PC 설치 및 전체 3회 실험은 아직 완료하지 않았습니다. Claude는 API 배치, GPT는 Codex 구독 실행으로 진행합니다. Claude Weizmann 1,116건은 수집을 마쳤으며, 누락됐던 UT 1,440건과 KISA GT1·GT2 각 1,080건의 후속 실행을 시작했습니다. 큰 KISA 입력의 배치 처리는 별도 확인 후 확대합니다. [큰 입력과 단계별 제출](docs/BATCH_LARGE_INPUTS.md)에 전송 한도와 예산 관리 방법을 기록했습니다. 이미지 순서·중복·원본 PNG 바이트를 유지하며, 과거 CLI·구독앱·진단 결과를 본실험 결과와 혼합하지 않습니다.

[검증 기록](validation/STATUS.json) · [배포 상태](docs/RELEASE_STATUS.md) · [권리·출처](THIRD_PARTY_NOTICES.md)

## GPT 구독 실행과 Claude API 분리

GPT는 현재 Codex의 ChatGPT 구독 인증으로 매 요청마다 새 독립 세션을 실행하고, Claude는 API 배치를 사용합니다. [실행 명령과 차이점](docs/SUBSCRIPTION_EXECUTION.md)을 확인하세요. GPT의 CLI 토큰에는 기본 지시문 등이 포함되므로 API 토큰과 동일한 측정값으로 합치지 않습니다. 구독 이용액을 API 단가로 계산하거나 무료라고 표기하지 않습니다.
