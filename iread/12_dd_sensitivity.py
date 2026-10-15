"""12단계 (이슈 85db75ca, 작업 8e9935fe): '바로 퇴원' 제외 민감도 분석 — 재학습 없음, 저장된 test 예측 재집계만.

대상 범위 민감도: 04 가 저장한 test 예측(preds_{v}_{d}) 과 holdout 고정 임계값(thresholds_{v}_{d}) 을 그대로 쓰고,
ICU 퇴실 직후 첫 이동 기록이 병원 퇴원인 표본(= 11_disposition.py 의 'discharge')만 빼서 09_w6 와 같은 지표를 다시 계산.
  - 복합결과 AUROC, 경고군 양성예측도, 사건별 포착률, 사건별 AUROC(해당 사건 vs 공통 비이벤트)
  - 환자 단위 군집 부트스트랩 95% CI (전체 / 바로 퇴원 제외 각각)
모델·경고 기준·라벨은 바꾸지 않는다. 바로 퇴원 판정 규칙은 11_disposition.py 와 동일(코드 복제, 결과 건수로 대조).
실행: bash iread/run_safe.sh 12 B d1   (safe_run 스키마: ddsens)
"""
import json
import re
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import WORK, csv
from labels import apply_def, check_feature_snapshot
from safe.agg import emit
from safe.schemas import MIN_CELL

SEED, N_BOOT = 42, 1000
TASKS = ["early", "late", "overall"]
TOL = pd.Timedelta(minutes=5)
version = sys.argv[1] if len(sys.argv) > 1 else "B"
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "d1"
TAG = f"{version}_{OUTCOME}"

full = pd.read_parquet(WORK / f"cohort_{version}_final.parquet").set_index("sample_id")
check_feature_snapshot(full, version)
c = apply_def(full, OUTCOME)

# ---- 바로 퇴원 판정 (11_disposition.py 와 같은 규칙: 퇴실 시각 -5분 이후 첫 transfers 행이 discharge)
con = duckdb.connect()
xfer = con.sql(f"SELECT hadm_id, eventtype, intime FROM read_csv('{csv('hosp/transfers')}') "
               f"WHERE hadm_id IS NOT NULL AND intime IS NOT NULL").df()
xb = {k: g.sort_values("intime") for k, g in xfer.groupby("hadm_id")}


def direct_discharge(hadm, out):
    x = xb.get(hadm)
    if x is None:
        return False
    x = x[x.intime >= out - TOL]
    return len(x) > 0 and str(x.iloc[0].eventtype).lower() == "discharge"


c["dd"] = [direct_discharge(h, o) for h, o in zip(c.hadm_id, c.outtime)]

preds = pd.read_parquet(WORK / f"preds_{TAG}.parquet")
chk = preds[["sample_id", "subject_id", "outtime"]].join(c[["subject_id", "outtime"]], on="sample_id", rsuffix="_c")
assert (chk.subject_id == chk.subject_id_c).all() and (chk.outtime == chk.outtime_c).all(), "정렬 불일치"
preds = preds.join(c[["event_type", "dd"]], on="sample_id")
THR = json.loads((WORK / f"thresholds_{TAG}.json").read_text())
rng = np.random.default_rng(SEED + 12)


def stats_of(d, thr):
    s = d.iREAD_cal.to_numpy(); y = d.y.to_numpy(); et = d.event_type.to_numpy()
    flag = s > thr
    out = {"auroc_composite": roc_auc_score(y, s), "ppv": flag[y == 1].sum() / max(flag.sum(), 1)}
    for k in ("readmission", "ward_death"):
        pos = (y == 1) & (et == k)
        out[f"capture_{k}"] = flag[pos].sum() / max(pos.sum(), 1)
        sel = pos | (y == 0)
        out[f"auroc_{k}"] = roc_auc_score(pos[sel], s[sel]) if pos.sum() >= MIN_CELL else np.nan
    return out


def with_ci(d, thr):
    est = stats_of(d, thr)
    codes, _ = pd.factorize(d.subject_id)
    members = pd.Series(np.arange(len(d))).groupby(codes).apply(np.asarray).to_list()
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        try:
            boots.append(stats_of(d.iloc[idx], thr))
        except ValueError:
            continue
    bt = pd.DataFrame(boots)
    return {k: {"est": float(v), "lo": float(np.nanpercentile(bt[k], 2.5)), "hi": float(np.nanpercentile(bt[k], 97.5))}
            for k, v in est.items() if pd.notna(v)}


counts, metrics = {}, {}
for task in TASKS:
    d = preds[preds.task == task].reset_index(drop=True)
    pops = {"all": d, "excl_dd": d[~d.dd].reset_index(drop=True)}
    counts[task], metrics[task] = {}, {}
    for name, p in pops.items():
        flag = p.iREAD_cal.to_numpy() > THR[task]
        counts[task][name] = {"n": len(p), "events": int(p.y.sum()),
                              "readmission": int(((p.y == 1) & (p.event_type == "readmission")).sum()),
                              "ward_death": int(((p.y == 1) & (p.event_type == "ward_death")).sum()),
                              "alerts": int(flag.sum()), "direct_discharge": int(p.dd.sum()),
                              "direct_discharge_alerted": int((flag & p.dd.to_numpy()).sum())}
        metrics[task][name] = with_ci(p, THR[task])
    print(task, counts[task], {n: round(m["auroc_composite"]["est"], 3) if "auroc_composite" in m else None for n, m in metrics[task].items()})

emit(version=version, outcome=OUTCOME, counts=counts, metrics=metrics)
