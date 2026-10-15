"""8단계: iREAD(앙상블) 보정 곡선(10분위)과 하위집단 성능. 집계값만 출력.

04 final 이 저장한 preds_{TAG}.parquet 의 holdout 재보정 확률(iREAD_cal)을 사용한다.
- 보정 곡선: 예측 확률 10분위마다 평균 예측값, 실제 이벤트율, 표본 수
- 하위집단: 연령(<65 / 65-79 / ≥80), ICU 재실 기간(<2일 / 2-7일 / >7일), 퇴실 시점 code status(full / dnr)별 표본·이벤트 수와 AUROC
  (이벤트나 비이벤트가 MIN_CELL 미만이면 AUROC 를 내지 않음)
실행: .venv/bin/python iread/08_calib.py A [d1|d2|d3]   (safe_run 스키마: calib)
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import WORK
from labels import DEFS, apply_def
from safe.agg import emit
from safe.schemas import MIN_CELL

TASKS = ["early", "late", "overall"]
version = sys.argv[1] if len(sys.argv) > 1 else "A"
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "d1"
assert OUTCOME in DEFS, OUTCOME
TAG = f"{version}_{OUTCOME}"

preds = pd.read_parquet(WORK / f"preds_{TAG}.parquet")
cohort = pd.read_parquet(WORK / f"cohort_{version}_final.parquet").set_index("sample_id")
# 정렬 검증 (F11): preds 의 각 행이 같은 (subject_id, hadm_id, outtime) 의 표본인지 확인
key = cohort[["subject_id", "hadm_id", "outtime"]]
assert {"subject_id", "hadm_id", "outtime"} <= set(preds.columns), "preds 에 정렬 키 없음"
chk = preds[["sample_id", "subject_id", "hadm_id", "outtime"]].join(key, on="sample_id", rsuffix="_c")
assert len(chk) and (chk.subject_id == chk.subject_id_c).all() and (chk.hadm_id == chk.hadm_id_c).all() \
    and (chk.outtime == chk.outtime_c).all(), "preds 와 cohort 의 표본 정렬 불일치"
preds = preds.join(cohort[["age", "icu_los", "cs_detail", f"event_type_{OUTCOME}"]], on="sample_id")
preds["event_type"] = preds.pop(f"event_type_{OUTCOME}")
preds["age_grp"] = pd.cut(preds.age, [-np.inf, 65, 80, np.inf], right=False, labels=["<65", "65-79", ">=80"])
preds["los_grp"] = pd.cut(preds.icu_los, [-np.inf, 2, 7, np.inf], right=False, labels=["<2d", "2-7d", ">7d"])

curve, sub, by_type, alert = {}, {}, {}, {}
THR = json.loads((WORK / f"thresholds_{TAG}.json").read_text())
for task in TASKS:
    d = preds[preds.task == task]
    dec = pd.qcut(d.iREAD_cal.rank(method="first"), 10, labels=[f"d{i}" for i in range(1, 11)])
    curve[task] = {k: {"mean_pred": float(g.iREAD_cal.mean()), "obs": float(g.y.mean()), "n": len(g)}
                   for k, g in d.groupby(dec, observed=True)}
    sub[task] = {}
    for col in ("age_grp", "los_grp", "cs_detail"):
        for k, g in d.groupby(col, observed=True):
            ev = int(g.y.sum())
            rec = {"n": len(g), "events": ev}
            if ev >= MIN_CELL and len(g) - ev >= MIN_CELL:
                rec["auroc"] = float(roc_auc_score(g.y, g.iREAD_cal))
            sub[task][str(k)] = rec
    # 이벤트 종류별 판별력: 해당 종류 이벤트 vs 비이벤트 (다른 종류 이벤트는 제외)
    by_type[task] = {}
    for et in ("readmission", "ward_death", "ward_dnr"):
        # N2: 이 과제의 이벤트(y==1) 중 해당 종류 vs 비이벤트(y==0). early 과제의 late 이벤트는 y==0 이라 비이벤트
        pos = (d.event_type == et) & (d.y == 1)
        g = d[pos | (d.y == 0)]
        ev = int(pos.sum())
        rec = {"events": ev}
        if ev >= MIN_CELL:
            rec["auroc"] = float(roc_auc_score(pos.loc[g.index], g.iREAD_cal))
        by_type[task][et] = rec
    # 같은 점수로 경고군의 실제 결과 구성: holdout 고정 임계값(특이도 0.9) / 위험 상위 10%(순위 기반 기술 분석)
    outcome = np.where(d.y == 1, d.event_type, np.where(d.event_type != "none", "later_event", "none"))
    for key, flag in (("holdout_thr", d.iREAD_cal.to_numpy() > THR[task]),
                      ("top10", d.iREAD_cal.rank(method="first", ascending=False).to_numpy() <= np.ceil(0.1 * len(d)))):
        comp = {k: int(((outcome == k) & flag).sum()) for k in ("readmission", "ward_death", "ward_dnr", "later_event", "none")}
        cap = {f"capture_{k}": float(((outcome == k) & flag).sum() / max((outcome == k).sum(), 1))
               for k in ("readmission", "ward_death", "ward_dnr")}
        alert.setdefault(task, {})[key] = {"flagged": int(flag.sum()), **comp, **cap}
    alert[task]["totals"] = {k: int((outcome == k).sum()) for k in ("readmission", "ward_death", "ward_dnr", "later_event", "none")}
    print(f"== {TAG} {task}")
    print(pd.DataFrame(curve[task]).T.round(4).to_string())
    print(pd.DataFrame(sub[task]).T.to_string())

emit(version=version, outcome=OUTCOME, curve=curve, subgroup=sub, by_event_type=by_type, alert=alert)
