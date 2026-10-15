"""6단계: LightGBM TreeSHAP 변수 중요도 (30개 변수 단위로 합산한 비율). 집계값만 출력.

04 에서 저장한 lgbm_{TAG}_{task}.txt 를 불러와 test 표본(최대 N_EXPLAIN 명)의 SHAP 값을 계산하고,
변수별 통계 요약 열(first/last/min/...)의 |SHAP| 평균을 원래 변수 30개로 합쳐 전체 대비 비율로 낸다.
LSTM 부분의 SHAP 은 계산 비용 때문에 제외 (iREAD 앙상블의 LightGBM 쪽만 설명).
실행: .venv/bin/python iread/06_shap.py A [d1|d2|d3]   (safe_run 스키마: shap)
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import DYNAMIC_FEATURES, STATIC_FEATURES, WORK
from labels import DEFS, apply_def, check_feature_snapshot
from safe.agg import emit

N_EXPLAIN = 5000
TASKS = ["early", "late", "overall"]
version = sys.argv[1] if len(sys.argv) > 1 else "A"
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "d1"
assert OUTCOME in DEFS, OUTCOME
TAG = f"{version}_{OUTCOME}"

_full = pd.read_parquet(WORK / f"cohort_{version}_final.parquet").set_index("sample_id")
check_feature_snapshot(_full, version)                     # N10
cohort = apply_def(_full, OUTCOME)
stats = pd.read_parquet(WORK / f"stats_{version}.parquet").set_index("sample_id").reindex(cohort.index)
tab = pd.concat([cohort[STATIC_FEATURES], stats], axis=1)
test_all = tab.loc[cohort.split == "test"]

base_of = {c: c for c in STATIC_FEATURES}
for f in DYNAMIC_FEATURES:
    for c in tab.columns:
        if c.startswith(f + "_") and c[len(f) + 1:] in ("first", "last", "min", "max", "mean", "std", "cnt"):
            base_of[c] = f

share = {}
for task in TASKS:
    # N7: 과제에 라벨이 있는 test 표본만 (late 과제는 early 이벤트 제외)
    test = test_all.loc[~cohort.loc[test_all.index, "early"].astype(bool)] if task == "late" else test_all
    rng = np.random.default_rng(42)
    test = test.iloc[np.sort(rng.choice(len(test), min(N_EXPLAIN, len(test)), replace=False))]
    booster = lgb.Booster(model_file=str(WORK / f"lgbm_{TAG}_{task}.txt"))
    cols = booster.feature_name()
    contrib = booster.predict(test[cols], pred_contrib=True)[:, :-1]      # 마지막 열 = 기댓값(bias)
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=cols).groupby(lambda c: base_of.get(c, "other")).sum()
    imp = imp.drop("other", errors="ignore")
    share[task] = (imp / imp.sum()).to_dict()
    print(f"== {TAG} {task} 상위 10")
    print((imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_string())

emit(version=version, outcome=OUTCOME, n_explained=min(N_EXPLAIN, len(test_all)),
     share={t: {k: float(v) for k, v in s.items()} for t, s in share.items()})
