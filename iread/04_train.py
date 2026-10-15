"""4단계: 모델 학습과 평가 (논문 Table 1 / S5 형식).

모델: iREAD(LSTM 0.5 + LightGBM 0.5 앙상블), LR, RF, Cox, NEWS, MEWS
과제: early(≤48h), late(>48h, early 이벤트는 제외), overall
분할: dev(2008-2016) -> train/holdout(80/20, 환자 단위), test(2017-2022) 로 평가
모든 모델은 holdout 으로 재보정(Platt). 출력: 집계 지표만.
실행: .venv/bin/python iread/04_train.py A   (또는 B)
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from config import DYNAMIC_FEATURES, EARLY_HOURS, QUICK, STATIC_FEATURES, WORK
from labels import DEFS, apply_def, check_feature_snapshot
from safe.agg import emit

MAX_HOURS = 168
SEED = 42
N_BOOT = 50 if QUICK else 1000
TASKS = ["early", "late", "overall"]
GCS = {"gcs_eye": 4, "gcs_verbal": 5, "gcs_motor": 6}
CONT = [f for f in DYNAMIC_FEATURES if f not in GCS]

version = sys.argv[1] if len(sys.argv) > 1 else "A"
# 결과 정의 (이슈 f047c0f5): d1 주분석(CMO 제외; 재입실·병동 사망) / d2 논문 정의 / d3 재입실만
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "d1"
assert OUTCOME in DEFS, OUTCOME
# 분할 방식 (W4): temporal = 2008–16 dev / 2017–22 test (기본), random = 전체 연도에서 환자 단위 무작위 25% test
SPLIT_MODE = os.environ.get("IREAD_SPLIT", "temporal")
assert SPLIT_MODE in ("temporal", "random"), SPLIT_MODE
TAG = f"{version}_{OUTCOME}" + ("_rand" if SPLIT_MODE == "random" else "")
# 실행 모드: tune = holdout 만으로 하이퍼파라미터 탐색 (test 라벨·예측은 만들지도 평가하지도 않음)
#            final = (튜닝 결과가 있으면 그 값으로) 학습 후 test 를 1회 평가
MODE = sys.argv[3] if len(sys.argv) > 3 else "final"
assert MODE in ("tune", "final"), MODE
DEFAULT = {"lgbm": {"num_leaves": 31, "min_child_samples": 50, "learning_rate": 0.02, "spw": "ratio"},
           "lstm": {"d": 64, "p": 0.2, "lr": 1e-3}}
GRID = {"lgbm": [{"num_leaves": nl, "min_child_samples": mc, "learning_rate": lr, "spw": w}
                 for nl in (15, 31, 63) for mc in (50, 200) for lr in (0.02, 0.05) for w in ("ratio", "sqrt", "none")],
        "lstm": [{"d": d, "p": p, "lr": lr} for d in (64, 128) for p in (0.2, 0.4) for lr in (1e-3, 3e-4)]}
PARAMS_FILE = WORK / f"params_{TAG}.json"
rng = np.random.default_rng(SEED)                 # LSTM 균형 배치용
boot_rng = np.random.default_rng(SEED + 1)        # 부트스트랩 전용 (F9: 학습과 분리)
torch.manual_seed(SEED)
torch.set_num_threads(16)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LSTM 학습 장치: {DEVICE}")

cohort = pd.read_parquet(WORK / f"cohort_{version}_final.parquet").set_index("sample_id")
stats = pd.read_parquet(WORK / f"stats_{version}.parquet").set_index("sample_id").reindex(cohort.index)
ts = pd.read_parquet(WORK / f"ts_{version}.parquet")
check_feature_snapshot(cohort, version)               # N10: 03 산출물과 같은 코호트인지 확인
if SPLIT_MODE == "random":                             # 논문 MIMIC-III 내부검증과 비슷한 조건 (연도 무관 환자 무작위)
    elig = cohort.index[cohort.split.isin(["dev", "test"])]
    _, te_pos = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 7)
                     .split(elig, groups=cohort.loc[elig, "subject_id"]))
    cohort["split"] = np.where(cohort.index.isin(elig[te_pos]), "test",
                               np.where(cohort.index.isin(elig), "dev", "other"))

# ---------------------------------------------------------------- 분할 (N8: 정의 적용 전 전체 dev 환자로 한 번만 → 정의 간 동일 holdout 환자)
dev_all = cohort.index[cohort.split == "dev"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_pos, ho_pos = next(gss.split(dev_all, groups=cohort.loc[dev_all, "subject_id"]))
hold_subjects = set(cohort.loc[dev_all[ho_pos], "subject_id"])

cohort = apply_def(cohort, OUTCOME)
stats = stats.reindex(cohort.index)
ts = ts[ts.sample_id.isin(cohort.index)]
print(f"결과 정의: {OUTCOME}  표본 {len(cohort):,}")
dev_idx = cohort.index[cohort.split == "dev"]
test_idx = cohort.index[cohort.split == "test"]
is_ho = cohort.loc[dev_idx, "subject_id"].isin(hold_subjects).to_numpy()
train_idx, hold_idx = dev_idx[~is_ho], dev_idx[is_ho]

# N6: test 평가는 결과 정의마다 1회만. 다시 평가하려면 IREAD_ALLOW_FINAL_RERUN=1 로 명시 (사유를 보고에 기록)
FINAL_MARK = WORK / f"final_done_{TAG}.json"
if FINAL_MARK.exists() and os.environ.get("IREAD_ALLOW_FINAL_RERUN") != "1":   # final 이후엔 tune 재실행도 금지
    raise SystemExit("final already evaluated for this TAG")


def labels(task: str) -> pd.Series:
    """과제별 라벨. late 과제에서는 early 이벤트를 NaN(제외)으로 둔다."""
    if task == "early":
        return cohort.early.astype(float)
    if task == "late":
        return cohort.late.astype(float).where(cohort.early == 0)
    return cohort.event.astype(float)


# ---------------------------------------------------------------- 표 형식 입력 (LightGBM / LR / RF / Cox)
tab = pd.concat([cohort[STATIC_FEATURES], stats], axis=1)
tab_mean = tab.loc[train_idx].mean()
tab_std = tab.loc[train_idx].std().replace(0, 1)
tab_z = ((tab - tab_mean) / tab_std).fillna(0)          # LR / Cox 용 (결측 = 평균)

# ---------------------------------------------------------------- 시계열 텐서 (LSTM)
los_h = (cohort.icu_los * 24).to_numpy()
length = np.clip(np.floor(los_h).astype(int) + 1, 1, MAX_HOURS)
pos_of = pd.Series(np.arange(len(cohort)), index=cohort.index)
F_cont = len(CONT)
F = F_cont + sum(GCS.values()) + len(STATIC_FEATURES)
X = np.full((len(cohort), MAX_HOURS, F_cont), np.nan, dtype=np.float32)
G = np.full((len(cohort), MAX_HOURS, len(GCS)), np.nan, dtype=np.float32)

ts = ts[ts.h_before < MAX_HOURS]
r = pos_of.loc[ts.sample_id].to_numpy()
t = length[r] - 1 - ts.h_before.to_numpy()
ok = t >= 0
feat_pos = {f: i for i, f in enumerate(CONT)}
gcs_pos = {f: i for i, f in enumerate(GCS)}
fc = ts.feature.to_numpy()
for f, i in feat_pos.items():
    m = ok & (fc == f)
    X[r[m], t[m], i] = ts.value.to_numpy()[m]
for f, i in gcs_pos.items():
    m = ok & (fc == f)
    G[r[m], t[m], i] = ts.value.to_numpy()[m]


def ffill_time(a: np.ndarray) -> np.ndarray:
    """시간축(axis=1) forward fill."""
    n, T, f = a.shape
    idx = np.where(~np.isnan(a), np.arange(T)[None, :, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return np.take_along_axis(a, idx, axis=1)


X, G = ffill_time(X), ffill_time(G)
tr_rows = pos_of.loc[train_idx].to_numpy()
mu = np.nanmean(X[tr_rows].reshape(-1, F_cont), axis=0)
sd = np.nanstd(X[tr_rows].reshape(-1, F_cont), axis=0) + 1e-6
X = np.nan_to_num((X - mu) / sd)                          # 남은 결측 = 평균(0)
onehots = []
for f, k in GCS.items():                                   # GCS 는 범주형 one-hot (결측 = 전부 0)
    g = np.rint(G[:, :, gcs_pos[f]])
    onehots.append(np.stack([(g == v) for v in range(1, k + 1)], axis=-1).astype(np.float32))
st = ((cohort[STATIC_FEATURES] - tab_mean[STATIC_FEATURES]) / tab_std[STATIC_FEATURES]).fillna(0).to_numpy(np.float32)
SEQ = np.concatenate([X] + onehots + [np.repeat(st[:, None, :], MAX_HOURS, axis=1)], axis=-1)
del X, G, onehots
LEN = torch.as_tensor(length)


class IREAD(nn.Module):
    """Fig.S2: Linear embedding -> (FCN+LayerNorm+Dropout) 잔차 2회 -> LSTM 3층 -> FCN+LayerNorm -> softmax."""

    def __init__(self, f_in: int, d: int = 64, p: float = 0.2):
        super().__init__()
        self.emb = nn.Linear(f_in, d)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(p))
                                    for _ in range(2))
        self.lstm = nn.LSTM(d, d, num_layers=3, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, x, lengths):
        h = self.emb(x)
        for b in self.blocks:
            h = h + b(h)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.head(hn[-1])


def predict_lstm(model, rows, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            b = rows[i:i + bs]
            out.append(torch.softmax(model(torch.as_tensor(SEQ[b]).to(DEVICE), LEN[b]), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_lstm(tr_rows, y_tr, ho_rows, y_ho, d=64, p=0.2, lr=1e-3, epochs=2 if QUICK else 40,
               steps=5 if QUICK else 200, bs=256, patience=6):
    """이벤트/비이벤트를 반반 뽑는 균형 미니배치 + 학습률 스케줄 + early stopping. (모델, 최고 holdout AUROC) 반환"""
    model = IREAD(SEQ.shape[-1], d=d, p=p).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    pos, neg = tr_rows[y_tr == 1], tr_rows[y_tr == 0]
    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train()
        for _ in range(steps):
            b = np.concatenate([rng.choice(pos, bs // 2), rng.choice(neg, bs // 2)])
            yb = torch.as_tensor(np.r_[np.ones(bs // 2), np.zeros(bs // 2)], dtype=torch.long, device=DEVICE)
            loss = nn.functional.cross_entropy(model(torch.as_tensor(SEQ[b]).to(DEVICE), LEN[b]), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        auc = roc_auc_score(y_ho, predict_lstm(model, ho_rows))
        sched.step(auc)
        print(f"      epoch {ep + 1:>2}  holdout AUROC {auc:.4f}", flush=True)
        if auc > best:
            best, best_state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def fit_lgbm(tr, y_tr, ho, y_ho, num_leaves, min_child_samples, learning_rate, spw):
    ratio = (y_tr == 0).sum() / max(y_tr.sum(), 1)
    gbm = lgb.LGBMClassifier(n_estimators=3000, learning_rate=learning_rate, num_leaves=num_leaves,
                             min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, scale_pos_weight={"ratio": ratio, "sqrt": ratio ** 0.5, "none": 1}[spw],
                             verbose=-1, random_state=SEED)
    gbm.fit(tab.loc[tr], y_tr, eval_set=[(tab.loc[ho], y_ho)], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    return gbm


def platt(p_ho, y_ho, logit=True):
    """holdout 으로 재보정. 확률은 logit 에, 임상 점수는 점수 자체에 로지스틱 회귀."""
    if logit:
        z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    else:
        z = lambda p: np.asarray(p, dtype=float)[:, None]
    lr = LogisticRegression(C=1e6).fit(z(p_ho), y_ho)
    return lambda p: lr.predict_proba(z(p))[:, 1]


# ---------------------------------------------------------------- 임상 점수 (퇴실 시점 마지막 값)
def band(x, edges, scores):
    """x 가 edges 로 나눈 구간 중 어디에 속하는지에 따라 점수. 결측은 0점."""
    s = np.select([x <= e for e in edges] + [x > edges[-1]], scores, 0)
    return np.where(np.isnan(x), 0, s)


def last(f):
    return stats[f"{f}_last"].to_numpy(dtype=float) if f"{f}_last" in stats else np.full(len(stats), np.nan)


gcs_total = last("gcs_eye") + last("gcs_verbal") + last("gcs_motor")
NEWS = (band(last("rr"), [8, 11, 20, 24], [3, 1, 0, 2, 3])
        + band(last("spo2"), [91, 93, 95], [3, 2, 1, 0])
        + band(last("temp"), [35, 36, 38, 39], [3, 1, 0, 1, 2])
        + band(last("sbp"), [90, 100, 110, 219], [3, 2, 1, 0, 3])
        + band(last("hr"), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
        + np.where(gcs_total < 15, 3, 0))
MEWS = (band(last("sbp"), [70, 80, 100, 199], [3, 2, 1, 0, 2])
        + band(last("hr"), [40, 50, 100, 110, 129], [2, 1, 0, 1, 2, 3])
        + band(last("rr"), [8, 14, 20, 29], [2, 0, 1, 2, 3])
        + band(last("temp"), [34.9, 38.4], [2, 0, 2])
        + band(gcs_total, [8, 12, 14], [3, 2, 1, 0]))
# SWIFT (Gajic 2008): 입실 경로, 총 ICU 재실 기간, 마지막 PaO2/FiO2, 마지막 GCS, 마지막 PaCO2. 결측은 0점
aux = pd.read_parquet(WORK / f"aux_{version}.parquet").set_index("sample_id").reindex(cohort.index)
pf = last("po2") / (aux.fio2_last.to_numpy(dtype=float) / 100)
SWIFT = (np.where(cohort.from_ed.to_numpy() == 1, 0, 8)
         + band(cohort.icu_los.to_numpy(dtype=float), [2 - 1e-9, 10], [0, 1, 14])
         + np.where(np.isnan(pf), 0, np.select([pf >= 400, pf >= 150, pf >= 100], [0, 5, 10], 13))
         + np.where(np.isnan(gcs_total), 0, np.select([gcs_total >= 15, gcs_total >= 11, gcs_total >= 8], [0, 6, 14], 24))
         + np.where(aux.paco2_last.to_numpy(dtype=float) > 45, 5, 0))
SCORES = {"NEWS": pd.Series(NEWS, index=cohort.index), "MEWS": pd.Series(MEWS, index=cohort.index),
          "SWIFT": pd.Series(SWIFT, index=cohort.index)}


# ---------------------------------------------------------------- 평가 지표
def ece(p, y, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def metrics(p, y, groups, spec_target=0.9):
    thr = np.quantile(p[y == 0], spec_target)
    pred = p > thr
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, tn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else np.nan
    # 환자(subject_id) 단위 군집 부트스트랩 (F9): 같은 환자의 여러 퇴실을 함께 재추출
    codes, uniq = pd.factorize(groups)
    members = pd.Series(np.arange(len(y))).groupby(codes).apply(np.asarray).to_list()
    boot = []
    for _ in range(N_BOOT):
        pick = boot_rng.integers(0, len(members), len(members))
        i = np.concatenate([members[k] for k in pick])
        if 0 < y[i].sum() < len(i):
            boot.append(roc_auc_score(y[i], p[i]))
    return {"AUROC": roc_auc_score(y, p), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
            "Acc": (tp + tn) / len(y), "Sens": sens, "Spec": spec, "PPV": ppv,
            "NPV": tn / (tn + fn), "F1": 2 * tp / (2 * tp + fp + fn), "Youden": sens + spec - 1,
            "ECE": ece(p, y)}


# ---------------------------------------------------------------- 튜닝 (holdout 만 사용, test 미사용)
def task_sets(task, use_test):
    y = labels(task)
    tr = train_idx[y.loc[train_idx].notna()]
    ho = hold_idx[y.loc[hold_idx].notna()]
    te = test_idx[y.loc[test_idx].notna()] if use_test else None
    return y, tr, ho, te


if MODE == "tune":
    chosen, log = {}, {}
    for task in TASKS:
        y, tr, ho, _ = task_sets(task, use_test=False)          # test 는 만들지도 않음
        y_tr, y_ho = (y.loc[i].to_numpy().astype(int) for i in (tr, ho))
        res_l = []
        for cfg in GRID["lgbm"]:
            gbm = fit_lgbm(tr, y_tr, ho, y_ho, **cfg)
            res_l.append(roc_auc_score(y_ho, gbm.predict_proba(tab.loc[ho])[:, 1]))
            print(f"   [{task}] lgbm {cfg} holdout AUROC {res_l[-1]:.4f}", flush=True)
        res_s = []
        for cfg in GRID["lstm"]:
            _, best = train_lstm(pos_of.loc[tr].to_numpy(), y_tr, pos_of.loc[ho].to_numpy(), y_ho, **cfg)
            res_s.append(best)
            print(f"   [{task}] lstm {cfg} holdout AUROC {best:.4f}", flush=True)
        chosen[task] = {"lgbm": GRID["lgbm"][int(np.argmax(res_l))], "lstm": GRID["lstm"][int(np.argmax(res_s))]}
        log[task] = {"lgbm": res_l, "lstm": res_s,
                     "lgbm_default": res_l[GRID["lgbm"].index(DEFAULT["lgbm"])],
                     "lstm_default": res_s[GRID["lstm"].index(DEFAULT["lstm"])]}
    PARAMS_FILE.write_text(json.dumps({"chosen": chosen, "grid": GRID, "holdout_auroc": log}, indent=1))
    emit(version=version, outcome=OUTCOME, n_configs={"lgbm": len(GRID["lgbm"]), "lstm": len(GRID["lstm"])},
         holdout={t: {"lgbm_best": max(log[t]["lgbm"]), "lgbm_default": log[t]["lgbm_default"],
                      "lstm_best": max(log[t]["lstm"]), "lstm_default": log[t]["lstm_default"]} for t in TASKS},
         chosen={t: {"num_leaves": c["lgbm"]["num_leaves"], "min_child_samples": c["lgbm"]["min_child_samples"],
                     "lgbm_lr": c["lgbm"]["learning_rate"], "spw": c["lgbm"]["spw"], "d": c["lstm"]["d"], "p": c["lstm"]["p"], "lstm_lr": c["lstm"]["lr"]}
                 for t, c in chosen.items()})
    raise SystemExit(0)

if not PARAMS_FILE.exists() and SPLIT_MODE == "random":    # W4: 시간 분할에서 고른 튜닝값 재사용
    PARAMS_FILE = WORK / f"params_{version}_{OUTCOME}.json"
PARAMS = json.loads(PARAMS_FILE.read_text())["chosen"] if PARAMS_FILE.exists() else {t: DEFAULT for t in TASKS}
PARAMS_SOURCE = "tuned" if PARAMS_FILE.exists() else "default"
print(f"하이퍼파라미터: {PARAMS_SOURCE} {PARAMS}")

# ---------------------------------------------------------------- 학습 루프 (final: test 1회 평가)
results, preds, THR = [], [], {}
for task in TASKS:
    y, tr, ho, te = task_sets(task, use_test=True)
    y_tr, y_ho, y_te = (y.loc[i].to_numpy().astype(int) for i in (tr, ho, te))
    print(f"\n==== [{version}] {task}: train {len(tr):,} ({y_tr.sum():,} 이벤트) / holdout {len(ho):,} "
          f"({y_ho.sum():,}) / test {len(te):,} ({y_te.sum():,})", flush=True)
    p = {}

    print("   LightGBM", flush=True)
    gbm = fit_lgbm(tr, y_tr, ho, y_ho, **PARAMS[task]["lgbm"])
    p_gbm = {k: gbm.predict_proba(tab.loc[i])[:, 1] for k, i in (("ho", ho), ("te", te))}

    print("   LSTM", flush=True)
    lstm, _ = train_lstm(pos_of.loc[tr].to_numpy(), y_tr, pos_of.loc[ho].to_numpy(), y_ho, **PARAMS[task]["lstm"])
    p_lstm = {k: predict_lstm(lstm, pos_of.loc[i].to_numpy()) for k, i in (("ho", ho), ("te", te))}
    torch.save(lstm.state_dict(), WORK / f"lstm_{TAG}_{task}.pt")
    gbm.booster_.save_model(str(WORK / f"lgbm_{TAG}_{task}.txt"))

    # 논문: 두 모델 확률을 0.5 씩 앙상블
    p["iREAD"] = {k: 0.5 * p_lstm[k] + 0.5 * p_gbm[k] for k in ("ho", "te")}
    p["LSTM only"] = p_lstm
    p["LightGBM only"] = p_gbm

    print("   LR / RF / Cox", flush=True)
    lr_m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(tab_z.loc[tr], y_tr)
    p["LR"] = {k: lr_m.predict_proba(tab_z.loc[i])[:, 1] for k, i in (("ho", ho), ("te", te))}
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED).fit(tab.loc[tr].fillna(tab_mean), y_tr)
    p["RF"] = {k: rf.predict_proba(tab.loc[i].fillna(tab_mean))[:, 1] for k, i in (("ho", ho), ("te", te))}

    # Cox: 시간 = 이벤트까지 / 비이벤트는 병원 퇴원까지 (early 과제는 48h 에서 중도절단)
    surv_h = ((cohort.dischtime - cohort.outtime).dt.total_seconds() / 3600).clip(lower=0.1)
    dur = cohort.event_hours.where(cohort.event == 1, surv_h).clip(lower=0.1)
    if task == "early":
        dur = dur.clip(upper=EARLY_HOURS)
    cox_df = tab_z.loc[tr].assign(T=dur.loc[tr].to_numpy(), E=y_tr)
    cph = CoxPHFitter(penalizer=0.1).fit(cox_df, "T", "E")
    p["Cox"] = {k: cph.predict_partial_hazard(tab_z.loc[i]).to_numpy() for k, i in (("ho", ho), ("te", te))}

    for name, s in SCORES.items():
        p[name] = {"ho": s.loc[ho].to_numpy(dtype=float), "te": s.loc[te].to_numpy(dtype=float)}

    cal = {}
    for name, pp in p.items():
        if name == "Cox":                       # partial hazard -> log hazard 를 점수로 사용
            pp = {k: np.log(v) for k, v in pp.items()}
        f_cal = platt(pp["ho"], y_ho, logit=name not in SCORES and name != "Cox")
        cal[name] = f_cal(pp["te"])
        if name == "iREAD":                     # 경고 기준: holdout 비이벤트의 90 백분위(특이도 0.9)를 test 에 고정 적용
            THR[task] = float(np.quantile(f_cal(pp["ho"])[y_ho == 0], 0.9))
        m = metrics(cal[name], y_te, cohort.loc[te, "subject_id"].to_numpy())
        results.append({"cohort": version, "task": task, "model": name, **m})
        print(f"   {name:<14} AUROC {m['AUROC']:.3f} ({m['lo']:.3f}-{m['hi']:.3f})  ECE {m['ECE']:.3f}", flush=True)
    preds.append(pd.DataFrame({"sample_id": te, "task": task, "y": y_te,
                               "subject_id": cohort.loc[te, "subject_id"].to_numpy(),
                               "hadm_id": cohort.loc[te, "hadm_id"].to_numpy(),
                               "outtime": cohort.loc[te, "outtime"].to_numpy(),
                               **{k: v["te"] for k, v in p.items()}, **{f"{k}_cal": v for k, v in cal.items()}}))

res = pd.DataFrame(results)
FINAL_MARK.write_text(json.dumps({"tag": TAG, "params": PARAMS_SOURCE}))
(WORK / f"thresholds_{TAG}.json").write_text(json.dumps(THR))
res.to_csv(WORK / f"results_{TAG}.csv", index=False)
pd.concat(preds).to_parquet(WORK / f"preds_{TAG}.parquet", index=False)

print(f"\n================ 결과 요약 (코호트 {version}, test 2017-2022) ================")
print(res.pivot(index="model", columns="task", values="AUROC")[TASKS].round(3).to_string())
print("\n논문 MIMIC-III 내부학습 iREAD (Table S6): early 0.799 / late 0.820 / overall 0.815")

# ---- 안전 래퍼용 집계
piv = lambda col: {mdl: {t: float(v) for t, v in g.set_index("task")[col].items()} for mdl, g in res.groupby("model")}
te_all = pd.concat(preds)
emit(version=version, outcome=OUTCOME, params=PARAMS_SOURCE, split=SPLIT_MODE,
     n_test={t: int((te_all.task == t).sum()) for t in TASKS},
     n_test_events={t: int(te_all.loc[te_all.task == t, "y"].sum()) for t in TASKS},
     auroc=piv("AUROC"), auroc_lo=piv("lo"), auroc_hi=piv("hi"), ece=piv("ECE"))
