# iREAD 참고 재구현 — MIMIC-IV에서 ICU 퇴실 후 48시간 재입실·사망 예측

ICU에서 일반병동으로 옮긴 뒤 다시 중환자 치료가 필요해질 위험을 퇴실 시점에 예측할 수 있는지 탐색하는 코드입니다.

- 공개된 iREAD 방법(Lim et al., *eClinicalMedicine* 2025;81:103112, doi:10.1016/j.eclinm.2025.103112)을 참고했습니다.
- MIMIC-IV v3.1에서 새로 학습했습니다. 원 논문의 학습된 모델을 쓴 외부검증이 아닙니다.

> **이 저장소에는 데이터가 없습니다.**
> - MIMIC-IV는 PhysioNet 자격 심사(CITI 교육 이수, 데이터 사용 계약 서명)를 거쳐 직접 내려받아야 합니다: https://physionet.org/content/mimiciv/3.1/
> - 원자료, 환자 단위 파생 자료, 학습된 모델 가중치는 포함하지 않습니다.
> - 결과는 집계값만 있으며, 1–9건인 칸은 `<10`으로 가렸습니다.

## 결과 요약
- 대표 분석: 성인 ICU 퇴실 사례 83,792건(환자 59,261명). 실제 ICU 퇴실 시각까지의 정보로, 같은 입원 중 48시간 안의 ICU 재입실 또는 재입실 전 사망을 예측했습니다.
- 평가 자료(23,542건)의 결과
  - 복합결과 AUROC 0.728
  - 같은 경고 기준에서 재입실 포착 29.8%, 재입실 전 사망 포착 94.9%
  - 복합결과 성능을 재입실 예측 성능으로 해석할 수 없습니다.
- 평가 결과를 본 뒤 정의를 수정한 **탐색적 분석**입니다. 대상에 일반병동 전실 여부가 확인되지 않은 사례가 포함됩니다.

> 최종 보고서(`results/`)는 저장소에 포함되지 않습니다.

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `config.py`, `labels.py` | 경로·변수·결과 정의 설정 |
| `01_extract.py` – `03_features.py` | 추출 → 코호트·라벨 → 입력 변수 |
| `04_train.py` | 학습(설정 선택 자료로 튜닝·보정) → 평가 |
| `05_` – `13_*.py` | 진단, SHAP, 보정, 사건별·경로별 분석, 재입실 직접 학습, 퇴실 직후 위치, 민감도, 기술통계 |
| `safe/` | 안전 실행 도구: 출력 스키마를 검사해 집계값만 통과시키고, 1–9건을 가리고, 원문 로그를 분리 |
| `run_safe.sh` | 단계별 실행 진입점 |
| `internal/` | 소유자 전용 내부 열람 보고서 생성기. 결과물은 저장소 밖에 생성되며 공개하지 않음 |
| `results/` *(제외)* | 최종 보고서, 기술통계, 그림 — 저장소에 포함하지 않음 |
| `evidence/` *(제외)* | 합성 경계 사례 생성·검사 코드 및 시험 로그 — 저장소에 포함하지 않음 |

## 실행 방법
1. 환경을 준비합니다. Python 3.12 이상과 `duckdb`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `torch`, `shap`, `matplotlib`이 필요합니다.
2. MIMIC-IV v3.1을 내려받고 경로를 지정합니다.
   ```bash
   export MIMIC_DIR=/path/to/physionet.org/files/mimiciv/3.1
   export IREAD_WORK=/path/to/work        # 파생 자료 저장 위치 (공개 금지)
   ```
3. 단계별로 실행합니다.
   - 화면에는 단계별 요약 JSON만 나옵니다.
   - 원문 출력은 `SAFE_LOG_DIR`(기본 `~/KCD2026/private_logs`)에 저장되며, 이 로그는 공개하지 않습니다.
   ```bash
   bash run_safe.sh            # 01–03 데이터 준비
   bash run_safe.sh 04t B d1   # 설정 선택 자료로 튜닝
   bash run_safe.sh 04 B d1    # 평가 (정의별 1회)
   bash run_safe.sh 09 B d1    # 사건별·경로별 지표와 신뢰구간
   bash run_safe.sh 13         # 기술통계
   ```

## 원자료 없이 코드 확인하기 (합성 데이터)
`internal/synth_browse.py`는 손으로 만든 가상 사례를 MIMIC 형식으로 생성합니다. 실제 환자 기록에서 만든 것이 아닙니다.
```bash
python internal/synth_browse.py /tmp/synth/data /tmp/synth/work
MIMIC_DIR=/tmp/synth/data IREAD_WORK=/tmp/synth/work python 02_cohort.py
```
합성 데이터 시험은 코드가 규칙대로 동작하는지 확인하는 것이며, 실제 임상 라벨이 타당한지 검증하는 것이 아닙니다.

## 데이터 사용 계약 관련
- MIMIC-IV 데이터 사용 계약 9항에 따라 결과를 만든 코드를 공개합니다.
- 이 저장소를 쓰는 분도 각자 PhysioNet 자격과 계약을 갖춰야 합니다.
- 데이터나 환자 단위 산출물을 공유하지 마세요.

## 인용
- Lim L, et al. Multicenter validation of a machine learning model to predict intensive care unit readmission within 48 hours after discharge. eClinicalMedicine. 2025;81:103112.
- Johnson A, et al. MIMIC-IV (version 3.1). PhysioNet. 2024. doi:10.13026/kpb9-mt58

## 라이선스
MIT License (코드에 한함) — 데이터·결과물에는 MIMIC-IV DUA가 별도로 적용됩니다.
