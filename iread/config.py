"""iREAD 재현 (MIMIC-IV 3.1) 공통 설정.

DUA 주의: 이 프로젝트의 스크립트는 집계값(N, 비율, AUROC 등)만 출력한다.
환자 단위 행(row)을 화면에 출력하지 말 것.
"""
import os
from pathlib import Path

ROOT = Path.home() / "KCD2026"
DATA = Path(os.environ.get("MIMIC_DIR", ROOT / "datasets/mimic-iv/physionet.org/files/mimiciv/3.1"))
WORK = Path(os.environ.get("IREAD_WORK", ROOT / "work"))   # 파생 데이터 (DUA 적용 대상, 외부 공유 금지)
WORK.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("IREAD_QUICK") == "1"   # 합성 데이터 시험용: 학습 횟수 축소

# ---------------------------------------------------------------- 코호트/라벨
MERGE_GAP_HOURS = {"A": 24, "B": 0}   # A: MIMIC-III 방식(24h 내 재입실 병합), B: SNUH 방식(직접 이동만)
ICU_DEATH_TOL_HOURS = 1               # 퇴실 후 이 시간 내 사망은 ICU 내 사망으로 간주 (제외)
EARLY_HOURS = 48
MIN_AGE = 18
DEV_YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]   # 개발(학습+튜닝)
TEST_YEAR_GROUPS = ["2017 - 2019", "2020 - 2022"]                 # 시간적 내부검증

# Code status 값 분류 (대소문자 무시 정규식, 이슈 f047c0f5). CMO 를 먼저 검사한다
#   cmo: 임종 돌봄(comfort measures only) / dnr: DNR·DNI 등 일부 처치 거부 / 그 외 값: full
CMO_REGEX = r"comfort|cmo"
DNR_REGEX = r"dnr|dni|dnar|do not"
LST_REFUSAL_REGEX = f"{CMO_REGEX}|{DNR_REGEX}"   # (05_diag 등 기존 집계 호환용)

# ---------------------------------------------------------------- 입력 변수 30개
# 논문 Fig.4a / Table S1. 범위는 Table S2 (None = 무한대)
# chartevents (MetaVision itemid)
CHART_ITEMS = {
    "hr":        ([220045], (10, 190)),
    "rr":        ([220210, 224690], (5, 45)),
    "sbp":       ([220050, 220179, 225309], (40, 230)),
    "dbp":       ([220051, 220180, 225310], (20, 130)),
    "temp_c":    ([223762], (32, 41)),
    "temp_f":    ([223761], (32, 41)),          # 화씨 -> 섭씨 변환 후 범위 적용, temp 로 합침
    "spo2":      ([220277], (68, 100)),
    "gcs_eye":   ([220739], (1, 4)),
    "gcs_verbal":([223900], (1, 5)),
    "gcs_motor": ([223901], (1, 6)),
}
CODE_STATUS_ITEMS = [223758]

# labevents (hosp d_labitems itemid)
LAB_ITEMS = {
    "albumin":    ([50862], (0, 70)),
    "bun":        ([51006], (0, 300)),
    "bilirubin":  ([50885], (0, 1000)),
    "calcium":    ([50893], (4, 18)),
    "ion_calcium":([50808], (0, 1.5)),
    "creatinine": ([50912], (0, 1500)),
    "glucose":    ([50931, 50809], (0, 2000)),     # 논문은 상한 없음. 극단값(SD 1662) 때문에 2000 mg/dL 상한 추가 (이슈 05531bf3)
    "hco3":       ([50882], (0, 50)),
    "lactate":    ([50813], (0, 30)),
    "lymphocyte": ([51244], (0, 100)),
    "inr":        ([51237], (0, 8)),
    "platelet":   ([51265], (0, 1500)),
    "potassium":  ([50971, 50822], (0.5, 12)),
    "sodium":     ([50983, 50824], (105, 170)),
    "wbc":        ([51301, 51300], (0, None)),
    "ph":         ([50820], (6.1, 8.5)),
    "po2":        ([50821], (20, 500)),
}

# SWIFT 베이스라인 전용 보조 변수 (모델 입력 30개에는 포함하지 않음. 이슈 b4bd041e)
# FiO2: MetaVision "Inspired O2 Fraction" (%, 일부는 0~1 비율로 기록 -> 03 에서 % 로 통일)
# PaCO2: 혈액가스 pCO2
AUX_CHART_ITEMS = {"fio2": ([223835], (21, 100))}
AUX_LAB_ITEMS = {"paco2": ([50818], (5, 200))}

# 정적 변수: age, gender, icu_los(일), adm_to_icu(일)
STATIC_FEATURES = ["age", "gender", "icu_los", "adm_to_icu"]
DYNAMIC_FEATURES = ["hr", "rr", "sbp", "dbp", "temp", "spo2",
                    "gcs_eye", "gcs_verbal", "gcs_motor"] + list(LAB_ITEMS)
assert len(STATIC_FEATURES) + len(DYNAMIC_FEATURES) == 30


def csv(name: str) -> str:
    """'icu/icustays' -> csv.gz 경로 문자열 (DuckDB read_csv 용)."""
    return str(DATA / f"{name}.csv.gz")
