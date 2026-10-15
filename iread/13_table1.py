"""13단계: 조원 공유용 기술통계(표 1) — 대표 분석 대상(코호트 B · 정의 ①) 집계만. 재학습·라벨 변경 없음.

열(집단): all / train / holdout / test (04 와 같은 분할 재현) / 48시간 결과별 early_readm, early_death, no_early_event
행: 표본·환자·입원 수, 나이, 성별, 첫 ICU 유형, ICU 재실 기간, 입원→ICU 기간, 응급실 경유, 퇴실 시점 치료목표,
    퇴실 직후 위치(11_disposition 과 같은 규칙), 병원 퇴원 장소, 원내 사망, 48h·후기 사건, 퇴실 전 마지막 측정값(중앙값·IQR)과 결측률
실행: bash iread/run_safe.sh 13      (safe_run 스키마: table1)
"""
import re

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import DYNAMIC_FEATURES, WORK, csv
from labels import apply_def, check_feature_snapshot
from safe.agg import emit

SEED = 42
VERSION, OUTCOME = "B", "d1"
TOL = pd.Timedelta(minutes=5)
UNIT_RULES = [("micu_sicu", r"(?i)micu/sicu"), ("cvicu", r"(?i)cvicu|cardiac vascular"), ("tsicu", r"(?i)tsicu|trauma"),
              ("neuro", r"(?i)neuro"), ("ccu", r"(?i)\bccu\b|coronary"), ("sicu", r"(?i)sicu|surgical"), ("micu", r"(?i)micu|medical")]
LOC_RULES = [("died", r"(?i)died|expired|death"), ("hospice", r"(?i)hospice"),
             ("acute_hospital", r"(?i)acute hospital|other hospital"),
             ("facility", r"(?i)skilled|snf|rehab|chronic|long term|nursing|assisted|psych|facility"),
             ("home_health", r"(?i)home health"), ("home", r"(?i)^home$|^home\b"), ("against_advice", r"(?i)against")]
PROC = re.compile(r"(?i)pacu|operating|procedure")            # 02/11 과 같은 규칙
ICU = re.compile(r"(?i)icu|intensive|ccu|critical")


def first_match(v, rules, default):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return "missing"
    for k, rx in rules:
        if re.search(rx, str(v)):
            return k
    return default


full = pd.read_parquet(WORK / f"cohort_{VERSION}_final.parquet").set_index("sample_id")
check_feature_snapshot(full, VERSION)
stats = pd.read_parquet(WORK / f"stats_{VERSION}.parquet").set_index("sample_id")

# 04 와 같은 분할 재현 (dev 환자 80/20 → train/holdout)
dev_all = full.index[full.split == "dev"]
_, ho = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
             .split(dev_all, groups=full.loc[dev_all, "subject_id"]))
ho_subj = set(full.loc[dev_all[ho], "subject_id"])
c = apply_def(full, OUTCOME)
c["part"] = np.where(c.split == "test", "test", np.where(c.subject_id.isin(ho_subj), "holdout", "train"))

con = duckdb.connect()
adm = con.sql(f"SELECT hadm_id, deathtime, discharge_location FROM read_csv('{csv('hosp/admissions')}')").df().set_index("hadm_id")
xfer = con.sql(f"SELECT hadm_id, eventtype, careunit, intime FROM read_csv('{csv('hosp/transfers')}', "
               f"types={{'careunit': 'VARCHAR'}}) WHERE hadm_id IS NOT NULL AND intime IS NOT NULL").df()
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
    return "blank" if cu == "" else "icu_name" if ICU.search(cu) else "procedure" if PROC.search(cu) else "ward"


c["place"] = [next_place(h, o) for h, o in zip(c.hadm_id, c.outtime)]
c["unit"] = [first_match(u, UNIT_RULES, "other") for u in c.first_careunit]
c["loc"] = [first_match(adm.discharge_location.get(h), LOC_RULES, "other") for h in c.hadm_id]
c["died_in_hosp"] = [int(pd.notna(adm.deathtime.get(h))) for h in c.hadm_id]
c["age_grp"] = pd.cut(c.age, [-1, 44, 64, 79, 200], labels=["18-44", "45-64", "65-79", "80+"]).astype(str)
last = stats[[f"{f}_last" for f in DYNAMIC_FEATURES if f"{f}_last" in stats.columns]].reindex(c.index)

groups = {"all": c, "train": c[c.part == "train"], "holdout": c[c.part == "holdout"], "test": c[c.part == "test"],
          "early_readm": c[(c.early == 1) & (c.event_type == "readmission")],
          "early_death": c[(c.early == 1) & (c.event_type == "ward_death")],
          "no_early_event": c[c.early == 0]}


def q(s):
    s = s.dropna()
    if len(s) == 0:
        return {"mean": 0.0, "sd": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {"mean": float(s.mean()), "sd": float(s.std()) if len(s) > 1 else 0.0,
            "p25": float(s.quantile(.25)), "p50": float(s.median()), "p75": float(s.quantile(.75))}


def cnt(s, keys):
    return {k: int((s == k).sum()) for k in keys}


out = {}
for g, d in groups.items():
    lv = last.loc[d.index]
    out[g] = {
        "n_samples": len(d), "n_patients": int(d.subject_id.nunique()), "n_admissions": int(d.hadm_id.nunique()),
        "age": q(d.age), "age_group": cnt(d.age_grp, ("18-44", "45-64", "65-79", "80+")),
        "male": int((d.gender == 1).sum()),
        "first_unit": cnt(d.unit, tuple(k for k, _ in UNIT_RULES) + ("other", "missing")),
        "icu_los_days": q(d.icu_los), "adm_to_icu_days": q(d.adm_to_icu), "from_ed": int(d.from_ed.sum()),
        "code_status": cnt(d.cs_detail, ("full", "none", "dnr", "cmo")),
        "next_place": cnt(d.place, ("discharge", "ward", "procedure", "icu_name", "blank", "no_record")),
        "discharge_location": cnt(d["loc"], tuple(k for k, _ in LOC_RULES) + ("other", "missing")),
        "died_in_hospital": int(d.died_in_hosp.sum()),
        "events": {"early_readm": int(((d.early == 1) & (d.event_type == "readmission")).sum()),
                   "early_death": int(((d.early == 1) & (d.event_type == "ward_death")).sum()),
                   "late_readm": int(((d.late == 1) & (d.event_type == "readmission")).sum()),
                   "late_death": int(((d.late == 1) & (d.event_type == "ward_death")).sum())},
        "last_value": {f: q(lv[f"{f}_last"]) for f in DYNAMIC_FEATURES if f"{f}_last" in lv.columns},
        "missing_rate": {f: float(lv[f"{f}_last"].isna().mean()) if len(d) else 0.0
                         for f in DYNAMIC_FEATURES if f"{f}_last" in lv.columns},
    }
    print(g, out[g]["n_samples"], out[g]["n_patients"])
emit(version=VERSION, outcome=OUTCOME, groups=out)
