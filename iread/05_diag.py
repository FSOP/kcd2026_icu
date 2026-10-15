"""진단: (1) 병동 사망 라벨 오염 가능성, (2) 혈당 이상치. 집계값만 출력.

(1) 병동 사망 이벤트의 퇴실 후 경과 시간 분포, ICU 체류 중 code status 기록 유무, 퇴원 장소 범주
    (비교용으로 재입실 이벤트도 같은 방식으로 집계)
(2) 혈당 값 분포와 상한(GLUCOSE_CAP) 적용 전후 평균·SD
실행: .venv/bin/python iread/05_diag.py   (safe_run 스키마: diag)
"""
import duckdb
import numpy as np
import pandas as pd

from config import CODE_STATUS_ITEMS, LAB_ITEMS, WORK, csv
from labels import apply_def
from safe.agg import emit

GLUCOSE_CAP = 2000          # mg/dL. 제안값 (논문 Table S2 는 상한 없음)
HBINS = [(1, 6, "1-6h"), (6, 24, "6-24h"), (24, 48, "24-48h"), (48, np.inf, ">48h")]


def hbin(h):
    for lo, hi, name in HBINS:
        if lo <= h < hi:
            return name
    return "<1h"


def dloc(x):
    """퇴원 장소 원문을 고정 범주로 (범주 이름은 코드에서 정함)"""
    x = (x or "").upper()
    if not x:
        return "missing"
    if "DIED" in x:
        return "died"
    if "HOSPICE" in x:
        return "hospice"
    if x.startswith("HOME"):
        return "home"
    if any(k in x for k in ("NURSING", "REHAB", "CHRONIC", "FACILITY", "ASSISTED", "ACUTE HOSPITAL", "PSYCH")):
        return "facility"
    return "other"


con = duckdb.connect()
con.execute("PRAGMA threads=16")
adm = con.sql(f"SELECT hadm_id, discharge_location FROM read_csv('{csv('hosp/admissions')}')").df()
cs_stays = set(con.sql(f"SELECT DISTINCT stay_id FROM '{WORK}/chart_raw.parquet' "
                       f"WHERE itemid IN ({','.join(map(str, CODE_STATUS_ITEMS))})").df().stay_id)

out = {}
for v in ("A", "B"):
    # 주분석 정의(d1: 퇴실 시점 CMO 제외) 기준으로 진단
    c = apply_def(pd.read_parquet(WORK / f"cohort_{v}_final.parquet"), "d1").merge(adm, on="hadm_id", how="left")
    c["has_cs"] = c.stay_ids.apply(lambda ids: any(s in cs_stays for s in ids))
    rec = {}
    for etype, key in (("ward_death", "ward_death"), ("readmission", "readmission")):
        for win in ("early", "late"):
            sub = c[(c[win] == 1) & (c.event_type == etype)]
            rec.setdefault(f"{key}_hours", {})[win] = sub.event_hours.map(hbin).value_counts().to_dict()
            rec.setdefault(f"{key}_code_status", {})[win] = {
                "any_record": int(sub.has_cs.sum()), "no_record": int((~sub.has_cs).sum())}
            rec.setdefault(f"{key}_discharge", {})[win] = sub.discharge_location.map(dloc).value_counts().to_dict()
    rec["all_code_status"] = {"any_record": int(c.has_cs.sum()), "no_record": int((~c.has_cs).sum())}
    out[v] = {k: ({w: {kk: int(vv) for kk, vv in d.items()} for w, d in val.items()}
                  if k != "all_code_status" else val) for k, val in rec.items()}

# ---- 혈당: 03 과 같은 창(체류 안 + storetime 이 퇴실 이전), 코호트 A 최종 표본
c = pd.read_parquet(WORK / "cohort_A_final.parquet")[["sample_id", "subject_id", "intime", "outtime"]]
con.register("samples", c)
ids = ",".join(map(str, LAB_ITEMS["glucose"][0]))
g = con.sql(f"""
    SELECT s.sample_id, r.valuenum AS v FROM '{WORK}/lab_raw.parquet' r
    JOIN samples s ON s.subject_id = r.subject_id AND r.charttime BETWEEN s.intime AND s.outtime
    WHERE r.itemid IN ({ids}) AND coalesce(r.storetime, r.charttime) <= s.outtime AND r.valuenum >= 0""").df()
capped = g[g.v <= GLUCOSE_CAP]
glu = {"n_values": len(g), "over_1000": int((g.v > 1000).sum()), "over_2000": int((g.v > 2000).sum()),
       "over_5000": int((g.v > 5000).sum()),
       "samples_over_cap": int(g.loc[g.v > GLUCOSE_CAP, "sample_id"].nunique()),
       "mean_raw": float(g.v.mean()), "sd_raw": float(g.v.std()),
       "mean_capped": float(capped.v.mean()), "sd_capped": float(capped.v.std()),
       "p50": float(g.v.quantile(.5)), "p99": float(g.v.quantile(.99)), "cap": GLUCOSE_CAP}

print("== 진단 (집계)")
for v, rec in out.items():
    print(v, rec)
print("glucose", glu)
emit(cohort=out, glucose=glu)
