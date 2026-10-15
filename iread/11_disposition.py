"""11단계 (이슈 f8831309-2, 작업 19dbeb0d): ICU 퇴실 직후 어디로 갔는가 — 직접 퇴원 집계. 재학습·라벨 변경 없음.

입력: 03 산출물 cohort_{A,B}_final.parquet(분석 대상) + 원자료 transfers·admissions (집계만 출력)
퇴실 직후 위치 = 같은 입원의 transfers 중 입실 시각이 ICU 퇴실 시각(-5분 허용) 이후인 첫 행
  discharge   : 그 행이 eventtype='discharge' (병원 퇴원 행)   → "ICU에서 바로 퇴원"
  ward        : ICU·시술 이름이 아닌 유닛 (02 의 '나머지=병동' 규칙과 같음, 실제 병동인지는 미확인)
  procedure   : 수술실·회복실·시술 유닛
  icu_name    : ICU 이름 유닛 (병합되지 않은 ICU 기록 등)
  blank       : careunit 이 빈 비퇴원 행
  no_record   : 퇴실 이후 transfers 행이 없음
보조 지표: 병원 퇴원 시각이 ICU 퇴실 후 1시간 이내인 표본 수 (transfers 없이 코호트만으로 본 근사)
직접 퇴원의 퇴원 장소(discharge_location)는 고정 범주로 묶어 건수만.
실행: bash iread/run_safe.sh 11      (safe_run 스키마: disposition)
"""
import re

import duckdb
import numpy as np
import pandas as pd

from config import WORK, csv
from labels import apply_def
from safe.agg import emit

PROC = re.compile(r"(?i)pacu|operating|procedure")      # 02_cohort.py 와 같은 규칙
ICU = re.compile(r"(?i)icu|intensive|ccu|critical")
CATS = ("discharge", "ward", "procedure", "icu_name", "blank", "no_record")
LOC_RULES = [("died", r"(?i)died|expired|death"), ("hospice", r"(?i)hospice"),
             ("acute_hospital", r"(?i)acute hospital|other hospital"),
             ("facility", r"(?i)skilled|snf|rehab|chronic|long term|nursing|assisted|psych|facility"),
             ("home_health", r"(?i)home health"), ("home", r"(?i)^home$|^home\b"), ("against_advice", r"(?i)against")]
LOCS = tuple(k for k, _ in LOC_RULES) + ("other", "missing")
TOL = pd.Timedelta(minutes=5)

con = duckdb.connect()
xfer = con.sql(f"SELECT hadm_id, eventtype, careunit, intime FROM read_csv('{csv('hosp/transfers')}', "
               f"types={{'careunit': 'VARCHAR'}}) WHERE hadm_id IS NOT NULL AND intime IS NOT NULL").df()
adm = con.sql(f"SELECT hadm_id, discharge_location FROM read_csv('{csv('hosp/admissions')}')").df().set_index("hadm_id")
xb = {k: g.sort_values("intime") for k, g in xfer.groupby("hadm_id")}


def next_place(hadm, out):
    x = xb.get(hadm)
    if x is None:
        return "no_record"
    x = x[x.intime >= out - TOL]
    if len(x) == 0:
        return "no_record"
    r = x.iloc[0]
    if str(r.eventtype).lower() == "discharge":
        return "discharge"
    cu = "" if pd.isna(r.careunit) else str(r.careunit).strip()
    if cu == "":
        return "blank"
    if ICU.search(cu):
        return "icu_name"
    if PROC.search(cu):
        return "procedure"
    return "ward"


def loc_cat(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return "missing"
    for k, rx in LOC_RULES:
        if re.search(rx, str(v)):
            return k
    return "other"


def summarize(c):
    cnt = lambda s: {k: int((s == k).sum()) for k in CATS}
    test = c.split == "test"
    d = c[c.place == "discharge"]
    gap = (d.dischtime - d.outtime).dt.total_seconds() / 3600
    within1 = (c.dischtime - c.outtime) <= pd.Timedelta(hours=1)
    return {"n": len(c), "test_n": int(test.sum()),
            "next": cnt(c.place), "test_next": cnt(c.place[test]),
            "early_events": cnt(c.place[c.early == 1]), "test_early_events": cnt(c.place[test & (c.early == 1)]),
            "direct_discharge_location": {k: int((d.loc_cat == k).sum()) for k in LOCS},
            "direct_discharge_gap_hours": {"p25": float(gap.quantile(.25)) if len(d) else 0.0,
                                           "p50": float(gap.quantile(.5)) if len(d) else 0.0,
                                           "p75": float(gap.quantile(.75)) if len(d) else 0.0},
            "proxy_discharged_within_1h": int(within1.sum()),
            "proxy_and_direct": int((within1 & (c.place == "discharge")).sum())}


out = {}
for v in ("A", "B"):
    full = pd.read_parquet(WORK / f"cohort_{v}_final.parquet")
    full["place"] = [next_place(h, o) for h, o in zip(full.hadm_id, full.outtime)]
    full["loc_cat"] = [loc_cat(adm.discharge_location.get(h)) for h in full.hadm_id]
    out[v] = {"all": summarize(full.assign(early=full.early_d1)), "d1": summarize(apply_def(full, "d1"))}
    print(v, "all", out[v]["all"]["next"], "| d1 test", out[v]["d1"]["test_next"])
emit(cohorts=out)
