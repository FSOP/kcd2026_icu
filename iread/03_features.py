"""3단계: 입력 변수 30개 생성.

- 이상치 제거: Table S2 범위 밖 값 삭제
- 시계열(LSTM 용): ICU 퇴실 직전 MAX_HOURS 시간을 1시간 단위 평균으로 정리 (long 형식)
- 통계 요약(LightGBM/LR/RF 용): 체류 전체의 first/last/min/max/mean/std/count
- 활력징후 또는 검사값이 전혀 없는 체류 제외 (Fig.S1)
출력: 집계값만 (변수별 결측률, 평균±SD).
실행: .venv/bin/python iread/03_features.py A   (또는 B)
"""
import json
import sys

import duckdb
import numpy as np
import pandas as pd

from config import AUX_CHART_ITEMS, AUX_LAB_ITEMS, CHART_ITEMS, DYNAMIC_FEATURES, LAB_ITEMS, WORK
from labels import DEFS, cohort_key_hash, def_summary, file_hash
from safe.agg import emit

MAX_HOURS = 168          # 퇴실 전 7일까지 시계열로 사용
VITALS = ["hr", "rr", "sbp", "dbp", "temp", "spo2"]

version = sys.argv[1] if len(sys.argv) > 1 else "A"
con = duckdb.connect()
con.execute("PRAGMA threads=16")

cohort = pd.read_parquet(WORK / f"cohort_{version}.parquet")
stays = cohort[["sample_id", "subject_id", "stay_ids", "intime", "outtime"]].explode("stay_ids") \
    .rename(columns={"stay_ids": "stay_id"}).astype({"stay_id": "int64"})
con.register("stays", stays)
con.register("samples", cohort[["sample_id", "subject_id", "intime", "outtime"]])

rows = []
for src, items in [("chart", CHART_ITEMS), ("lab", LAB_ITEMS)]:
    for feat, (ids, (lo, hi)) in items.items():
        for i in ids:
            rows.append((src, i, "temp" if feat.startswith("temp_") else feat,
                         feat == "temp_f", lo, hi))
con.register("itemmap", pd.DataFrame(rows, columns=["src", "itemid", "feature", "is_f", "lo", "hi"]))

# 원시 측정값 (체류 기간 내, 범위 필터 적용)
con.execute(f"""
CREATE TABLE ev AS
WITH raw AS (
    -- 누수 방지: 측정 시각(charttime)이 체류 기간 안이고, 결과가 기록된 시각(storetime)도 퇴실 시점 이전인 값만
    -- (퇴실 시점에 의료진이 알 수 없었던 값 제외. storetime 이 없으면 charttime 으로 대체)
    SELECT s.sample_id, m.feature, r.charttime,
           CASE WHEN m.is_f THEN (r.valuenum - 32) * 5 / 9 ELSE r.valuenum END AS value, m.lo, m.hi
    FROM '{WORK}/chart_raw.parquet' r
    JOIN itemmap m ON m.src = 'chart' AND m.itemid = r.itemid
    JOIN stays s ON s.stay_id = r.stay_id
    WHERE r.charttime BETWEEN s.intime AND s.outtime AND r.valuenum IS NOT NULL
      AND coalesce(r.storetime, r.charttime) <= s.outtime
    UNION ALL
    SELECT s.sample_id, m.feature, r.charttime, r.valuenum, m.lo, m.hi
    FROM '{WORK}/lab_raw.parquet' r
    JOIN itemmap m ON m.src = 'lab' AND m.itemid = r.itemid
    JOIN samples s ON s.subject_id = r.subject_id AND r.charttime BETWEEN s.intime AND s.outtime
    WHERE coalesce(r.storetime, r.charttime) <= s.outtime
)
SELECT sample_id, feature, charttime, value FROM raw
WHERE value >= lo AND (hi IS NULL OR value <= hi)
""")

# 누수 점검: charttime 은 체류 안이지만 storetime 이 퇴실 이후라 제외된 행 수
leak = con.sql(f"""
    SELECT 'chart' src, count(*) n_in, count(*) FILTER (WHERE coalesce(r.storetime, r.charttime) > s.outtime) n_late
    FROM '{WORK}/chart_raw.parquet' r JOIN stays s ON s.stay_id = r.stay_id
    WHERE r.charttime BETWEEN s.intime AND s.outtime AND r.valuenum IS NOT NULL
    UNION ALL
    SELECT 'lab', count(*), count(*) FILTER (WHERE coalesce(r.storetime, r.charttime) > s.outtime)
    FROM '{WORK}/lab_raw.parquet' r JOIN samples s ON s.subject_id = r.subject_id
    WHERE r.charttime BETWEEN s.intime AND s.outtime""").df().set_index("src")
print("== 누수 점검 (storetime > 퇴실 시점이라 제외된 행)")
print(leak.to_string())

# 활력징후/검사값 없는 체류 제외
has = con.sql(f"""
    SELECT sample_id,
           bool_or(feature IN ({",".join(f"'{v}'" for v in VITALS)})) AS has_vital,
           bool_or(feature IN ({",".join(f"'{v}'" for v in LAB_ITEMS)})) AS has_lab
    FROM ev GROUP BY sample_id""").df()
keep = set(has[has.has_vital & has.has_lab].sample_id)
n0 = len(cohort)
cohort = cohort[cohort.sample_id.isin(keep)]
con.register("kept", cohort[["sample_id", "outtime"]])

# 시계열: 퇴실 전 시간(h_before, 0 = 마지막 1시간) 단위 평균
con.execute(f"""
COPY (
    SELECT e.sample_id, CAST(floor(epoch(k.outtime - e.charttime) / 3600) AS INTEGER) AS h_before,
           e.feature, avg(e.value) AS value
    FROM ev e JOIN kept k USING (sample_id)
    WHERE epoch(k.outtime - e.charttime) < {MAX_HOURS} * 3600
    GROUP BY ALL
) TO '{WORK}/ts_{version}.parquet' (FORMAT parquet)""")

# 통계 요약 (wide)
stats = con.sql("""
    SELECT e.sample_id, e.feature,
           arg_min(value, charttime) AS first, arg_max(value, charttime) AS last,
           min(value) AS min, max(value) AS max, avg(value) AS mean,
           stddev_samp(value) AS std, count(*) AS cnt
    FROM ev e JOIN kept USING (sample_id) GROUP BY ALL""").df()
wide = stats.pivot(index="sample_id", columns="feature")
wide.columns = [f"{f}_{s}" for s, f in wide.columns]
wide = wide.reset_index()
wide.to_parquet(WORK / f"stats_{version}.parquet", index=False)
cohort.to_parquet(WORK / f"cohort_{version}_final.parquet", index=False)


# SWIFT 보조 변수: 퇴실 시점 기준 마지막 FiO2(%), PaCO2 (같은 시간 창·storetime 규칙)
(fio2_ids, (f_lo, f_hi)), = AUX_CHART_ITEMS.values()
(pco2_ids, (p_lo, p_hi)), = AUX_LAB_ITEMS.values()
aux = con.sql(f"""
    WITH f AS (
        SELECT s.sample_id, r.charttime,
               CASE WHEN r.valuenum <= 1 THEN r.valuenum * 100 ELSE r.valuenum END AS v
        FROM '{WORK}/chart_raw.parquet' r JOIN stays s ON s.stay_id = r.stay_id
        WHERE r.itemid IN ({",".join(map(str, fio2_ids))}) AND r.charttime BETWEEN s.intime AND s.outtime
          AND coalesce(r.storetime, r.charttime) <= s.outtime),
    p AS (
        SELECT s.sample_id, r.charttime, r.valuenum AS v
        FROM '{WORK}/lab_raw.parquet' r JOIN samples s ON s.subject_id = r.subject_id
        WHERE r.itemid IN ({",".join(map(str, pco2_ids))}) AND r.charttime BETWEEN s.intime AND s.outtime
          AND coalesce(r.storetime, r.charttime) <= s.outtime)
    SELECT k.sample_id, fa.fio2_last, pa.paco2_last
    FROM kept k
    LEFT JOIN (SELECT sample_id, arg_max(v, charttime) AS fio2_last FROM f
               WHERE v BETWEEN {f_lo} AND {f_hi} GROUP BY sample_id) fa USING (sample_id)
    LEFT JOIN (SELECT sample_id, arg_max(v, charttime) AS paco2_last FROM p
               WHERE v BETWEEN {p_lo} AND {p_hi} GROUP BY sample_id) pa USING (sample_id)""").df()
aux.to_parquet(WORK / f"aux_{version}.parquet", index=False)
(WORK / f"features_{version}.meta.json").write_text(json.dumps({                       # N10: 코호트·라벨·산출물 해시
    "cohort_hash": cohort_key_hash(cohort),
    "files": {n: file_hash(WORK / n) for n in (f"stats_{version}.parquet", f"ts_{version}.parquet", f"aux_{version}.parquet")}}))

# ---- 집계 출력
n = len(cohort)
print(f"== 코호트 {version}: 활력징후/검사값 없음 제외 {n0 - n:,} -> 최종 {n:,}")
for d in DEFS:
    print(f"   [{d}] {def_summary(cohort, d)}")
print(f"   퇴실 시점 code status: {cohort.cs_at_out.value_counts().to_dict()}")

print("\n== 변수별 요약 (Table S1/S7 비교용)  환자기준 결측률% / 측정값 평균±SD")
summary = con.sql("""
    SELECT feature, count(DISTINCT e.sample_id) AS n_pat, avg(value) AS mean, stddev_samp(value) AS sd
    FROM ev e JOIN kept USING (sample_id) GROUP BY feature""").df().set_index("feature")
for f in DYNAMIC_FEATURES:
    r = summary.loc[f] if f in summary.index else None
    miss = 100 * (1 - (r.n_pat if r is not None else 0) / n)
    ms = f"{r['mean']:.2f} ± {r.sd:.2f}" if r is not None else "-"
    print(f"   {f:<12} 결측 {miss:6.2f}%   {ms}")
for f in ["age", "icu_los", "adm_to_icu"]:
    print(f"   {f:<12} {cohort[f].mean():.2f} ± {cohort[f].std():.2f}")
print(f"   {'male':<12} {cohort.gender.mean():.2%}")

# ---- 안전 래퍼용 집계
stat_mean = {f: float(summary.loc[f, "mean"]) for f in DYNAMIC_FEATURES if f in summary.index}
stat_sd = {f: float(summary.loc[f, "sd"]) for f in DYNAMIC_FEATURES if f in summary.index and pd.notna(summary.loc[f, "sd"])}
for f in ["age", "icu_los", "adm_to_icu"]:
    stat_mean[f], stat_sd[f] = float(cohort[f].mean()), float(cohort[f].std())
emit(version=version, n_before=n0, excluded_no_data=n0 - n, n_final=n,
     split_n={k: int(v) for k, v in cohort.split.value_counts().items()},
     cs_at_out={c: int((cohort.cs_at_out == c).sum()) for c in ("full", "dnr", "cmo")},
     defs={d: def_summary(cohort, d) for d in DEFS},
     missing={f: 1 - (float(summary.loc[f, "n_pat"]) if f in summary.index else 0) / n for f in DYNAMIC_FEATURES},
     mean=stat_mean, sd=stat_sd, male=float(cohort.gender.mean()),
     swift_inputs={"fio2_missing": float(aux.fio2_last.isna().mean()), "paco2_missing": float(aux.paco2_last.isna().mean()),
                   "from_ed": float(cohort.from_ed.mean())},
     leak_check={"chart_in_window": int(leak.loc["chart", "n_in"]), "chart_store_after_out": int(leak.loc["chart", "n_late"]),
                 "lab_in_window": int(leak.loc["lab", "n_in"]), "lab_store_after_out": int(leak.loc["lab", "n_late"])})
