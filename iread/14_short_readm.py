"""14단계 (이슈 dc79440a, 작업 519dea22): 퇴실 후 짧은 간격의 재입실 성격 집계 — 재학습·라벨 변경 없음, 집계만.

대상: 코호트 B · 정의 ① 최종 대상 중 48시간 이내 첫 사건이 ICU 재입실인 표본, 퇴실→재입실 간격으로 세 군
  lt1h(<1시간) / h1_6(1–6시간) / h6_48(6–48시간)
집계
  - 간격 길이 분포
  - 간격 동안 겹치는 transfers 행의 유닛 범주(02 와 같은 이름 규칙 + 응급실·중간치료 키워드 구분)와 행 수
  - 퇴실한 ICU 유닛(직전 ICU 체류의 last_careunit)과 재입실 ICU 유닛(다음 체류의 first_careunit)이 같은지
  - 재입실한 ICU 체류의 길이 분포
실행: bash iread/run_safe.sh 14      (safe_run 스키마: shortreadm)
"""
import re

import duckdb
import numpy as np
import pandas as pd

from config import WORK, csv
from labels import apply_def, check_feature_snapshot
from safe.agg import emit

PROC = re.compile(r"(?i)pacu|operating|procedure")       # 02_cohort.py 와 같은 규칙
ICU = re.compile(r"(?i)icu|intensive|ccu|critical")
ED = re.compile(r"(?i)emergency")
STEP = re.compile(r"(?i)intermediate|step ?down")
UNIT_CATS = ("icu_name", "procedure", "emergency", "intermediate_stepdown", "ward_other", "blank")


def unit_cat(cu):
    cu = "" if cu is None or (isinstance(cu, float) and np.isnan(cu)) else str(cu).strip()
    if cu == "":
        return "blank"
    if ICU.search(cu):
        return "icu_name"
    if PROC.search(cu):
        return "procedure"
    if ED.search(cu):
        return "emergency"
    if STEP.search(cu):
        return "intermediate_stepdown"
    return "ward_other"


full = pd.read_parquet(WORK / "cohort_B_final.parquet").set_index("sample_id")
check_feature_snapshot(full, "B")
c = apply_def(full, "d1")
r = c[(c.early == 1) & (c.event_type == "readmission")].copy()

con = duckdb.connect()
icu = con.sql(f"SELECT hadm_id, stay_id, first_careunit, last_careunit, intime, outtime "
              f"FROM read_csv('{csv('icu/icustays')}')").df()
xfer = con.sql(f"SELECT hadm_id, careunit, intime, outtime FROM read_csv('{csv('hosp/transfers')}', "
               f"types={{'careunit': 'VARCHAR'}}) WHERE hadm_id IS NOT NULL").df()
xb = {k: g for k, g in xfer.groupby("hadm_id")}
stay = icu.set_index("stay_id")
icu_by = {k: g.sort_values("intime") for k, g in icu.groupby("hadm_id")}

rows = []
for sid, x in r.iterrows():
    t0 = x.outtime
    t1 = t0 + pd.Timedelta(hours=float(x.event_hours))
    last_unit = stay.loc[x.stay_ids[-1], "last_careunit"] if x.stay_ids[-1] in stay.index else None
    nxt = icu_by[x.hadm_id]
    nxt = nxt[nxt.intime >= t1 - pd.Timedelta(minutes=1)].head(1)
    next_unit = nxt.first_careunit.iloc[0] if len(nxt) else None
    next_los_h = (nxt.outtime.iloc[0] - nxt.intime.iloc[0]).total_seconds() / 3600 if len(nxt) else np.nan
    g = xb.get(x.hadm_id)
    g = g[(g.intime < t1) & (g.outtime > t0)] if g is not None else pd.DataFrame(columns=["careunit"])
    cats = [unit_cat(cu) for cu in g.careunit]
    rows.append(dict(hours=float(x.event_hours), n_rows=len(cats), cats=set(cats),
                     same=(last_unit is not None and next_unit is not None and str(last_unit) == str(next_unit)),
                     known=(last_unit is not None and next_unit is not None), next_los_h=next_los_h))
d = pd.DataFrame(rows)
d["grp"] = np.where(d.hours < 1, "lt1h", np.where(d.hours < 6, "h1_6", "h6_48"))


def cut_count(s, edges, labels):
    b = pd.cut(s, edges, labels=labels, right=False)
    return {k: int((b == k).sum()) for k in labels}


out = {}
for gname, g in d.groupby("grp"):
    out[gname] = {
        "n": len(g),
        "gap_minutes": cut_count(g.hours * 60, [0, 5, 15, 30, 60, 360, 2881],
                                 ["0-5", "5-15", "15-30", "30-60", "60-360", "360-2880"]),
        "gap_unit_any": {k: int(g.cats.apply(lambda s: k in s).sum()) for k in UNIT_CATS},
        "gap_only_icu_or_blank": int(g.cats.apply(lambda s: len(s) > 0 and s <= {"icu_name", "blank"}).sum()),
        "gap_no_rows": int((g.n_rows == 0).sum()),
        "gap_rows": cut_count(g.n_rows, [0, 1, 2, 3, 1000], ["0", "1", "2", "3+"]),
        "same_unit": int(g.same.sum()), "unit_known": int(g.known.sum()),
        "next_icu_los_hours": cut_count(g.next_los_h, [0, 4, 12, 24, 72, 1e6], ["<4", "4-12", "12-24", "24-72", "72+"]),
    }
    print(gname, out[gname]["n"], out[gname]["same_unit"])
emit(groups=out)
