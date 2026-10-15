"""10단계 (W6-d): ICU 재입실을 직접 목표로 한 단순 모델(LightGBM, LR)과 복합결과 점수(iREAD, d1)를 같은 평가 집단·같은 경고 조건에서 비교.

평가 집단: 결과 정의 d1 의 test 표본 그대로 (병동 사망 표본도 삭제하지 않음). 과제별 표본 집합도 04 와 같다
           (late 과제는 d1 의 early 이벤트 표본 제외).
직접 모델 목표: 해당 시간창에서 '가장 먼저 일어난 사건'이 ICU 재입실이면 1, 그 외(재입실 전 병동 사망·퇴원·무사건) 0.
           ※ 사망을 '재입실 아님'으로 두는 분류 목표이며 경쟁위험 모형이 아니다.
학습: 04 와 같은 분할 재현 (dev 환자 80/20 train/holdout, test = d1 test). LightGBM 은 B_d1 의 과제별 튜닝값 재사용,
      LR 은 04 와 같은 설정. holdout 으로 조기 종료·Platt 재보정. test 평가는 TAG 당 1회 (final_done 표시).
비교 지표 (test, 환자 단위 군집 부트스트랩 CI):
  - 재입실 AUROC: (a) 재입실 vs 나머지 전체, (b) 재입실 vs 공통 비이벤트
  - 재입실 전 병동 사망 AUROC (vs 공통 비이벤트)
  - 같은 경고 수(복합 점수가 holdout 고정 임계값으로 낸 test 경고 수 = k) 상위 k 명 경고 시 재입실·사망 포착률, 경고군 재입실 비율
실행: .venv/bin/python iread/10_direct_readm.py B d1   (safe_run 스키마: direct)
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from config import STATIC_FEATURES, WORK
from labels import apply_def, check_feature_snapshot
from safe.agg import emit
from safe.schemas import MIN_CELL

SEED, N_BOOT = 42, 500
TASKS = ["early", "late", "overall"]
version = sys.argv[1] if len(sys.argv) > 1 else "B"
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "d1"
BASE = f"{version}_{OUTCOME}"
TAG = f"{BASE}_direct_readm"
MARK = WORK / f"final_done_{TAG}.json"
if MARK.exists() and os.environ.get("IREAD_ALLOW_FINAL_RERUN") != "1":
    raise SystemExit("already evaluated")

full = pd.read_parquet(WORK / f"cohort_{version}_final.parquet").set_index("sample_id")
check_feature_snapshot(full, version)
stats = pd.read_parquet(WORK / f"stats_{version}.parquet").set_index("sample_id")

dev_all = full.index[full.split == "dev"]
_, ho = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
             .split(dev_all, groups=full.loc[dev_all, "subject_id"]))
ho_subj = set(full.loc[dev_all[ho], "subject_id"])
c = apply_def(full, OUTCOME)
stats = stats.reindex(c.index)
tab = pd.concat([c[STATIC_FEATURES], stats], axis=1)
dev = c.index[c.split == "dev"]
is_ho = c.loc[dev, "subject_id"].isin(ho_subj).to_numpy()
train_idx, hold_idx, test_idx = dev[~is_ho], dev[is_ho], c.index[c.split == "test"]
tab_mean, tab_std = tab.loc[train_idx].mean(), tab.loc[train_idx].std().replace(0, 1)
tab_z = ((tab - tab_mean) / tab_std).fillna(0)
PARAMS = json.loads((WORK / f"params_{BASE}.json").read_text())["chosen"]
preds = pd.read_parquet(WORK / f"preds_{BASE}.parquet")
THR = json.loads((WORK / f"thresholds_{BASE}.json").read_text())


def task_rows(task):
    """04 와 같은 과제별 표본 집합과 복합 라벨"""
    if task == "late":
        keep = c.index[c.early == 0]
        y = c.late
    else:
        keep = c.index
        y = c.early if task == "early" else c.event
    return keep, y


def platt(p_ho, y_ho):
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    m = LogisticRegression(C=1e6).fit(z(p_ho), y_ho)
    return lambda p: m.predict_proba(z(p))[:, 1]


def metrics(d, k):
    """d: test 표본 (score 열들, y_comp, etype). k: 경고 수"""
    out = {}
    readm = ((d.y_comp == 1) & (d.etype == "readmission")).to_numpy()
    death = ((d.y_comp == 1) & (d.etype == "ward_death")).to_numpy()
    nonev = (d.y_comp == 0).to_numpy()
    for name in ("composite", "direct_lgbm", "direct_lr"):
        s = d[name].to_numpy()
        r = {}
        r["auroc_readm_vs_all"] = roc_auc_score(readm, s)
        sel = readm | nonev
        r["auroc_readm_vs_nonevent"] = roc_auc_score(readm[sel], s[sel])
        sel = death | nonev
        r["auroc_death_vs_nonevent"] = roc_auc_score(death[sel], s[sel]) if death.sum() >= MIN_CELL else np.nan
        flag = np.zeros(len(s), bool); flag[np.argsort(-s, kind="stable")[:k]] = True
        r["capture_readm_topk"] = flag[readm].sum() / max(readm.sum(), 1)
        r["capture_death_topk"] = flag[death].sum() / max(death.sum(), 1)
        r["readm_share_topk"] = flag[readm].sum() / max(k, 1)
        out[name] = r
    return out


rng = np.random.default_rng(SEED + 23)
res, ns = {}, {}
for task in TASKS:
    keep, y_comp = task_rows(task)
    win = {"early": c.early, "late": c.late, "overall": c.event}[task]
    y_readm = ((win == 1) & (c.event_type == "readmission")).astype(int)
    tr, hoi, te = (idx[np.isin(idx, keep)] for idx in (train_idx, hold_idx, test_idx))
    ytr, yho = y_readm.loc[tr].to_numpy(), y_readm.loc[hoi].to_numpy()

    lg = PARAMS[task]["lgbm"]
    ratio = (ytr == 0).sum() / max(ytr.sum(), 1)
    gbm = lgb.LGBMClassifier(n_estimators=3000, learning_rate=lg["learning_rate"], num_leaves=lg["num_leaves"],
                             min_child_samples=lg["min_child_samples"], subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                             scale_pos_weight={"ratio": ratio, "sqrt": ratio ** 0.5, "none": 1}[lg["spw"]],
                             verbose=-1, random_state=SEED)
    gbm.fit(tab.loc[tr], ytr, eval_set=[(tab.loc[hoi], yho)], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(tab_z.loc[tr], ytr)
    f_g = platt(gbm.predict_proba(tab.loc[hoi])[:, 1], yho)
    f_l = platt(lr.predict_proba(tab_z.loc[hoi])[:, 1], yho)

    p = preds[preds.task == task].set_index("sample_id").reindex(te)
    assert p.iREAD_cal.notna().all() and (p.y.to_numpy() == y_comp.loc[te].to_numpy()).all(), "복합 점수 표본 불일치"
    d = pd.DataFrame({"composite": p.iREAD_cal.to_numpy(),
                      "direct_lgbm": f_g(gbm.predict_proba(tab.loc[te])[:, 1]),
                      "direct_lr": f_l(lr.predict_proba(tab_z.loc[te])[:, 1]),
                      "y_comp": y_comp.loc[te].to_numpy(), "etype": c.loc[te, "event_type"].to_numpy(),
                      "subject_id": c.loc[te, "subject_id"].to_numpy()})
    k = int((d.composite > THR[task]).sum())            # 복합 점수의 holdout 고정 임계값 경고 수
    est = metrics(d, k)
    codes, _ = pd.factorize(d.subject_id)
    members = pd.Series(np.arange(len(d))).groupby(codes).apply(np.asarray).to_list()
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        b = d.iloc[idx].reset_index(drop=True)
        try:
            boots.append(metrics(b, int(round(k * len(b) / len(d)))))
        except ValueError:
            continue
    res[task] = {}
    for m, r in est.items():
        res[task][m] = {}
        for met, v in r.items():
            if pd.notna(v):
                vals = [bb[m][met] for bb in boots if pd.notna(bb[m][met])]
                res[task][m][met] = {"est": float(v), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}
    ns[task] = {"test": len(d), "alert_k": k, "readmission": int(((d.y_comp == 1) & (d.etype == "readmission")).sum()),
                "ward_death": int(((d.y_comp == 1) & (d.etype == "ward_death")).sum()),
                "train_readm": int(ytr.sum()), "holdout_readm": int(yho.sum())}
    print(task, ns[task], {m: round(r["auroc_readm_vs_all"], 3) for m, r in est.items()})

MARK.write_text(json.dumps({"tag": TAG}))
emit(version=version, outcome=OUTCOME, counts=ns, results=res)
