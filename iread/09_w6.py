"""9단계 (W6, 이슈 25e67e4a 댓글): 재학습·test 재평가 없이 기존 코호트·저장된 예측만 다시 집계. 집계값만 출력.

W6-a 재입실 라벨 확인: 정의 d1 의 재입실 표본을 퇴실~다음 ICU 입실 사이 이동 경로로 분류
     - via_procedure: 간격 중 수술실·회복실·시술 유닛을 거침 (계획 수술 후 ICU 입실 가능성 → 비계획 악화와 구분 불가)
     - ward_only: 비ICU·비시술 유닛만 (이동 기록상 병동 경유 재입실)
     - unclear: 간격에 transfers 기록이 없거나 위 둘로 분류 불가
     test 의 재입실 경로별로 같은 iREAD 점수의 AUROC (해당 경로 재입실 vs 공통 비이벤트)
W6-b 분할 기술 통계: train/holdout/test 별 고유 환자·전실·조기 재입실·재입실 전 사망 (04 와 같은 분할 재현)
     + W4: 시간 분할 holdout 환자 중 무작위 분할 test 에 들어간 환자·표본 수
W6-c 불확실성: 사건별 포착률·양성예측도·사건별 AUROC 의 환자 단위 군집 부트스트랩 95% CI
실행: .venv/bin/python iread/09_w6.py B d1   (safe_run 스키마: w6)
"""
import json
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from config import WORK, csv
from labels import apply_def, check_feature_snapshot
from safe.agg import emit
from safe.schemas import MIN_CELL

SEED, N_BOOT = 42, 1000
TASKS = ["early", "late", "overall"]
PROC = r"(?i)pacu|operating|procedure"
ICU = r"(?i)icu|intensive|ccu|critical"
version = sys.argv[1] if len(sys.argv) > 1 else "B"
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "d1"
TAG = f"{version}_{OUTCOME}"

full = pd.read_parquet(WORK / f"cohort_{version}_final.parquet").set_index("sample_id")
check_feature_snapshot(full, version)


# ---------------------------------------------------------------- W6-b 분할 (04 와 동일 절차 재현)
def holdout_subjects(c):
    dev_all = c.index[c.split == "dev"]
    _, ho = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                 .split(dev_all, groups=c.loc[dev_all, "subject_id"]))
    return set(c.loc[dev_all[ho], "subject_id"])


ho_subj = holdout_subjects(full)
c = apply_def(full, OUTCOME)
part = np.where(c.split == "test", "test",
                np.where(c.split == "dev", np.where(c.subject_id.isin(ho_subj), "holdout", "train"), "other"))
c["part"] = part
split_stats = {}
for p, g in c.groupby("part"):
    split_stats[p] = {"subjects": int(g.subject_id.nunique()), "samples": len(g),
                      "early_readmission": int(((g.early == 1) & (g.event_type == "readmission")).sum()),
                      "early_ward_death": int(((g.early == 1) & (g.event_type == "ward_death")).sum()),
                      "late_readmission": int(((g.late == 1) & (g.event_type == "readmission")).sum()),
                      "late_ward_death": int(((g.late == 1) & (g.event_type == "ward_death")).sum())}

# W4: 무작위 분할 test (04 의 IREAD_SPLIT=random 과 동일 절차) 에 들어간 시간 분할 holdout 환자
elig = full.index[full.split.isin(["dev", "test"])]
_, te_pos = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 7)
                 .split(elig, groups=full.loc[elig, "subject_id"]))
rand_test = full.loc[elig[te_pos]]
w4_overlap = {"rand_test_subjects": int(rand_test.subject_id.nunique()),
              "overlap_subjects": int(rand_test.subject_id.isin(ho_subj).groupby(rand_test.subject_id).first().sum()),
              "overlap_samples": int(rand_test.subject_id.isin(ho_subj).sum())}

# ---------------------------------------------------------------- W6-a 재입실 경로
con = duckdb.connect()
xfer = con.sql(f"SELECT hadm_id, careunit, intime, outtime FROM read_csv('{csv('hosp/transfers')}') "
               f"WHERE hadm_id IS NOT NULL").df()
xb = {k: g for k, g in xfer.groupby("hadm_id")}


def path_of(r):
    """퇴실 ~ (다음 ICU 입실 = 퇴실 + event_hours) 사이 이동 경로 범주"""
    t0 = r.outtime; t1 = t0 + pd.Timedelta(hours=float(r.event_hours))
    x = xb.get(r.hadm_id)
    if x is None:
        return "unclear"
    x = x[(x.intime < t1) & (x.outtime > t0)]
    cu = x.careunit.fillna("")
    cu = cu[~cu.str.contains(ICU)]
    if len(cu) == 0:
        return "unclear"
    if cu.str.contains(PROC).any():
        return "via_procedure"
    return "ward_only"


ra = c[c.event_type == "readmission"].copy()
ra["path"] = [path_of(r) for r in ra.itertuples()]
c["path"] = ra.path.reindex(c.index)
paths = ("ward_only", "via_procedure", "unclear")
path_stats = {win: {p: int(((ra[win] == 1) & (ra.path == p)).sum()) for p in paths} for win in ("early", "late")}

# ---------------------------------------------------------------- W6-c / W6-a: test 예측 재집계 (저장된 preds 만 사용)
preds = pd.read_parquet(WORK / f"preds_{TAG}.parquet")
key = c[["subject_id", "hadm_id", "outtime"]]
chk = preds[["sample_id", "subject_id", "hadm_id", "outtime"]].join(key, on="sample_id", rsuffix="_c")
assert (chk.subject_id == chk.subject_id_c).all() and (chk.outtime == chk.outtime_c).all(), "정렬 불일치"
preds = preds.join(c[["event_type", "path"]], on="sample_id")
THR = json.loads((WORK / f"thresholds_{TAG}.json").read_text())
rng = np.random.default_rng(SEED + 11)


def stats_of(d, thr):
    s = d.iREAD_cal.to_numpy(); y = d.y.to_numpy(); et = d.event_type.to_numpy()
    flag = s > thr
    out = {"ppv": flag[y == 1].sum() / max(flag.sum(), 1)}
    for k in ("readmission", "ward_death"):
        pos = (y == 1) & (et == k)
        out[f"capture_{k}"] = flag[pos].sum() / max(pos.sum(), 1)
        sel = pos | (y == 0)
        out[f"auroc_{k}"] = roc_auc_score(pos[sel], s[sel]) if pos.sum() >= MIN_CELL else np.nan
    for pth in paths:      # 경로별 재입실 AUROC (해당 경로 재입실 vs 공통 비이벤트)
        pos = (y == 1) & (et == "readmission") & (d.path.to_numpy() == pth)
        sel = pos | (y == 0)
        out[f"auroc_readm_{pth}"] = roc_auc_score(pos[sel], s[sel]) if pos.sum() >= MIN_CELL else np.nan
    return out


ci = {}
for task in TASKS:
    d = preds[preds.task == task].reset_index(drop=True)
    est = stats_of(d, THR[task])
    codes, _ = pd.factorize(d.subject_id)
    members = pd.Series(np.arange(len(d))).groupby(codes).apply(np.asarray).to_list()
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        try:
            boots.append(stats_of(d.iloc[idx], THR[task]))
        except ValueError:
            continue
    bt = pd.DataFrame(boots)
    ci[task] = {}
    for k, v in est.items():
        if pd.notna(v):
            ci[task][k] = {"est": float(v), "lo": float(np.nanpercentile(bt[k], 2.5)), "hi": float(np.nanpercentile(bt[k], 97.5))}
    ci[task]["n_readm_by_path"] = {p: int(((d.y == 1) & (d.event_type == "readmission") & (d.path == p)).sum()) for p in paths}

print(json.dumps({"split": split_stats, "w4": w4_overlap, "paths": path_stats}, ensure_ascii=False))
emit(version=version, outcome=OUTCOME, split_stats=split_stats, w4_overlap=w4_overlap,
     readmission_paths=path_stats,
     test_ci={t: {k: v for k, v in x.items() if k != "n_readm_by_path"} for t, x in ci.items()},
     test_readm_by_path={t: x["n_readm_by_path"] for t, x in ci.items()})
