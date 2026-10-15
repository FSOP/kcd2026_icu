"""1단계: chartevents / labevents 에서 필요한 itemid 만 뽑아 parquet 으로 저장.

큰 csv.gz 를 한 번만 읽도록 분리한 단계. 출력은 itemid 별 행 수 등 집계값뿐.
실행: .venv/bin/python iread/01_extract.py
"""
import duckdb

from config import (AUX_CHART_ITEMS, AUX_LAB_ITEMS, CHART_ITEMS, CODE_STATUS_ITEMS, DATA, LAB_ITEMS,
                    CMO_REGEX, DNR_REGEX, LST_REFUSAL_REGEX, WORK, csv)
from safe.agg import emit

NEEDED = ["icu/chartevents", "icu/d_items", "hosp/labevents", "hosp/d_labitems"]
for name in NEEDED:
    p = DATA / f"{name}.csv.gz"
    if not p.exists() or p.with_name(p.name + ".aria2").exists():
        raise SystemExit(f"아직 다운로드가 끝나지 않았습니다: {name}")

con = duckdb.connect()
con.execute("PRAGMA threads=16")

CHART_ALL = {**CHART_ITEMS, **AUX_CHART_ITEMS}      # 모델 입력 + SWIFT 보조 변수
LAB_ALL = {**LAB_ITEMS, **AUX_LAB_ITEMS}
chart_ids = sorted({i for ids, _ in CHART_ALL.values() for i in ids} | set(CODE_STATUS_ITEMS))
lab_ids = sorted({i for ids, _ in LAB_ALL.values() for i in ids})

# itemid 가 의도한 항목인지 라벨로 확인 (사전 테이블 = 메타데이터)
print("== chartevents itemid 확인")
print(con.sql(f"""
    SELECT itemid, label, unitname FROM read_csv('{csv("icu/d_items")}')
    WHERE itemid IN ({",".join(map(str, chart_ids))}) ORDER BY itemid""").df().to_string(index=False))
print("\n== labevents itemid 확인")
print(con.sql(f"""
    SELECT itemid, label, fluid, category FROM read_csv('{csv("hosp/d_labitems")}')
    WHERE itemid IN ({",".join(map(str, lab_ids))}) ORDER BY itemid""").df().to_string(index=False))

print("\n== chartevents 추출 중 (수십 분 걸릴 수 있음)")
con.execute(f"""
    COPY (
        SELECT stay_id, itemid, charttime, storetime, value, valuenum
        FROM read_csv('{csv("icu/chartevents")}', types={{'value': 'VARCHAR', 'valueuom': 'VARCHAR'}})
        WHERE itemid IN ({",".join(map(str, chart_ids))}) AND stay_id IS NOT NULL
    ) TO '{WORK}/chart_raw.parquet' (FORMAT parquet)""")

print("== labevents 추출 중")
con.execute(f"""
    COPY (
        SELECT subject_id, hadm_id, itemid, charttime, storetime, valuenum
        FROM read_csv('{csv("hosp/labevents")}',
                      types={{'value': 'VARCHAR', 'valueuom': 'VARCHAR', 'comments': 'VARCHAR',
                              'flag': 'VARCHAR', 'priority': 'VARCHAR', 'order_provider_id': 'VARCHAR'}})
        WHERE itemid IN ({",".join(map(str, lab_ids))}) AND valuenum IS NOT NULL
    ) TO '{WORK}/lab_raw.parquet' (FORMAT parquet)""")

print("\n== itemid 별 추출 행 수")
print(con.sql(f"""
    SELECT 'chart' src, itemid, count(*) n FROM '{WORK}/chart_raw.parquet' GROUP BY ALL
    UNION ALL
    SELECT 'lab', itemid, count(*) FROM '{WORK}/lab_raw.parquet' GROUP BY ALL
    ORDER BY src, itemid""").df().to_string(index=False))

# Code status 값 종류 (범주형 문자열과 건수만)
print("\n== Code status 값 분포")
print(con.sql(f"""
    SELECT value, count(*) n FROM '{WORK}/chart_raw.parquet'
    WHERE itemid IN ({",".join(map(str, CODE_STATUS_ITEMS))})
    GROUP BY value ORDER BY n DESC""").df().to_string(index=False))

# ---- 안전 래퍼용 집계 (변수 이름 기준 건수만)
feat_of = {i: f for f, (ids, _) in CHART_ALL.items() for i in ids}
feat_of.update({i: f for f, (ids, _) in LAB_ALL.items() for i in ids})
feat_of.update({i: "code_status" for i in CODE_STATUS_ITEMS})
in_dict = set(con.sql(f"SELECT itemid FROM read_csv('{csv("icu/d_items")}') WHERE itemid IN ({",".join(map(str, chart_ids))})"
                      ).df().itemid) | set(con.sql(f"SELECT itemid FROM read_csv('{csv("hosp/d_labitems")}') "
                                                   f"WHERE itemid IN ({",".join(map(str, lab_ids))})").df().itemid)
rows = con.sql(f"""
    SELECT itemid, count(*) n FROM '{WORK}/chart_raw.parquet' GROUP BY ALL
    UNION ALL SELECT itemid, count(*) FROM '{WORK}/lab_raw.parquet' GROUP BY ALL""").df()
agg = {k: {} for k in ("itemids_expected", "itemids_in_dict", "rows")}
for i, f in feat_of.items():
    agg["itemids_expected"][f] = agg["itemids_expected"].get(f, 0) + 1
    agg["itemids_in_dict"][f] = agg["itemids_in_dict"].get(f, 0) + (i in in_dict)
    agg["rows"][f] = agg["rows"].get(f, 0) + int(rows.loc[rows.itemid == i, "n"].sum())
cs = con.sql(f"""
    SELECT count(*) FILTER (WHERE regexp_matches(lower(value), '{CMO_REGEX}')) cmo,
           count(*) FILTER (WHERE NOT regexp_matches(lower(value), '{CMO_REGEX}') AND regexp_matches(lower(value), '{DNR_REGEX}')) dnr,
           count(*) FILTER (WHERE NOT regexp_matches(lower(value), '{CMO_REGEX}|{DNR_REGEX}')) other,
           count(DISTINCT value) nv
    FROM '{WORK}/chart_raw.parquet' WHERE itemid IN ({",".join(map(str, CODE_STATUS_ITEMS))})""").fetchone()
emit(**agg, code_status={"cmo": cs[0], "dnr": cs[1], "other": cs[2]}, code_status_n_values=cs[3])
