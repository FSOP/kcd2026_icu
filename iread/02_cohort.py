"""2단계: 코호트와 라벨 생성 (A: 24h 병합 / B: 직접 이동·수술실 경유·1h 이내 빈 간격만 병합).

공통 제외 (논문 Fig.S1 순서): ICU 체류 병합 -> 18세 미만 -> ICU 내 사망(퇴실 후 1h 이내 포함)
  -> 시간 기록 오류(퇴실·입실 시각 결측 포함). (활력징후·검사값 없음은 03 에서 제외)

결과 정의 3종 (이슈 f047c0f5, 소유자 확인). 제외는 "퇴실 시점에 알 수 있는 정보"만 쓴다.
  퇴실 시점 code status = chartevents 223758 + POE 'Code status' 오더를 합친 시간순 "지정" 기록 중
  퇴실 시각 이전의 마지막 값. code status 는 새로 지정되기 전까지 유지되는 것으로 본다:
  POE 오더의 D/C 만으로는 상태를 바꾸지 않는다(전동 시 오더 정리·재입력이 흔함. W2 검토 N1).
  D/C 를 '지정 없음'으로 보는 규칙과 판정이 갈리는 표본 수는 진단 집계로 함께 낸다.
  chart 기록 시각은 max(charttime, storetime) (퇴실 후 입력된 기록은 퇴실 시점 판정에 쓰지 않음. N4).
  값이 비어 있는 기록은 상태를 바꾸지 않는다. 기록이 없으면 full. 같은 시각 기록이 여럿이면 가장 제한적인 값(N12).
  ① d1 (주분석): 퇴실 시점 CMO 만 제외 / 이벤트 = ICU 재입실 또는 병동 사망
  ② d2 (논문 정의): 퇴실 시점 DNR·DNI·CMO 제외 / 이벤트 = 재입실, 병동 사망, 병동 DNR(퇴실 후 첫 거부 기록)
  ③ d3 (재입실만): ① 에서 병동 사망 표본을 추가 제외 / 이벤트 = ICU 재입실
  이벤트 시각은 후보 중 가장 이른 것 (F8). early ≤ 48h, late > 48h.
출력: 집계값만.
실행: .venv/bin/python iread/02_cohort.py
"""
import duckdb
import numpy as np
import pandas as pd

from config import (CMO_REGEX, CODE_STATUS_ITEMS, DEV_YEAR_GROUPS, DNR_REGEX, EARLY_HOURS, ICU_DEATH_TOL_HOURS,
                    MERGE_GAP_HOURS, MIN_AGE, TEST_YEAR_GROUPS, WORK, csv)
from labels import DEFS, def_summary
from safe.agg import emit

PROCEDURE_UNIT_REGEX = r"(?i)pacu|operating|procedure"   # 수술실/회복실 경유는 B 에서도 병합
ICU_UNIT_REGEX = r"(?i)icu|intensive|ccu|critical"        # 병합 간격 분류용 (진단)
B_EMPTY_GAP_HOURS = 1        # B: 간격 안에 transfers 기록이 없고 이 시간 이하면 직접 이동으로 보고 병합 (F6)
CAT_RANK = {"none": 0, "full": 1, "dnr": 2, "cmo": 3}   # 같은 시각 기록이 여럿이면 가장 제한적인 값이 마지막

con = duckdb.connect()
icu = con.sql(f"SELECT subject_id, hadm_id, stay_id, first_careunit, intime, outtime "
              f"FROM read_csv('{csv('icu/icustays')}')").df()
adm = con.sql(f"SELECT hadm_id, admittime, dischtime, deathtime "
              f"FROM read_csv('{csv('hosp/admissions')}')").df()
pat = con.sql(f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group "
              f"FROM read_csv('{csv('hosp/patients')}')").df()
xfer = con.sql(f"SELECT hadm_id, careunit, intime, outtime "
               f"FROM read_csv('{csv('hosp/transfers')}') WHERE hadm_id IS NOT NULL").df()


def cat_sql(col):
    """code status 값 -> 범주 (값 원문은 SQL 밖으로 나가지 않음)"""
    return (f"CASE WHEN {col} IS NULL OR trim({col}) = '' THEN NULL "
            f"WHEN regexp_matches(lower({col}), '{CMO_REGEX}') THEN 'cmo' "
            f"WHEN regexp_matches(lower({col}), '{DNR_REGEX}') THEN 'dnr' ELSE 'full' END")


# ---- code status 시간순 기록 (hadm 단위). 값이 빈 기록은 제외(상태 변화 없음)
chart_cs = con.sql(f"""
    SELECT stay_id, greatest(charttime, coalesce(storetime, charttime)) AS t, charttime AS t_chart,
           {cat_sql('value')} AS cat
    FROM '{WORK}/chart_raw.parquet'
    WHERE itemid IN ({",".join(map(str, CODE_STATUS_ITEMS))}) AND charttime IS NOT NULL""").df()
chart_cs = chart_cs.merge(icu[["stay_id", "hadm_id"]], on="stay_id").dropna(subset=["cat"])
con.execute(f"""
    CREATE TABLE cs_detail AS SELECT poe_id, max(field_value) AS v
    FROM read_csv('{csv('hosp/poe_detail')}') WHERE field_name = 'Code status' GROUP BY poe_id""")
con.execute(f"""
    CREATE TABLE poe_cs AS SELECT p.poe_id, p.hadm_id, p.ordertime, p.transaction_type, p.discontinue_of_poe_id
    FROM read_csv('{csv('hosp/poe')}', types={{'order_provider_id': 'VARCHAR', 'discontinue_of_poe_id': 'VARCHAR',
                                              'discontinued_by_poe_id': 'VARCHAR', 'poe_id': 'VARCHAR'}}) p
    WHERE p.poe_id IN (SELECT poe_id FROM cs_detail) OR p.discontinue_of_poe_id IN (SELECT poe_id FROM cs_detail)""")
poe_orders = con.sql(f"""
    SELECT p.hadm_id, p.ordertime AS t, {cat_sql('d.v')} AS cat FROM poe_cs p JOIN cs_detail d USING (poe_id)
    WHERE p.hadm_id IS NOT NULL AND p.ordertime IS NOT NULL""").df().dropna(subset=["cat"])
# D/C: code status 오더를 끊는 행(자기 자신은 code status 값이 없음) -> 그 시점부터 '지정 없음'
poe_dc = con.sql("""
    SELECT hadm_id, ordertime AS t, 'none' AS cat FROM poe_cs
    WHERE discontinue_of_poe_id IN (SELECT poe_id FROM cs_detail) AND poe_id NOT IN (SELECT poe_id FROM cs_detail)
      AND hadm_id IS NOT NULL AND ordertime IS NOT NULL""").df()
def prep(frames):
    e = pd.concat(frames, ignore_index=True).astype({"hadm_id": "int64", "cat": "str"})   # 0건이어도 dtype 고정 (F15)
    e["t"] = pd.to_datetime(e.t).astype("datetime64[us]")
    e["rank"] = e.cat.map(CAT_RANK)
    return e.sort_values(["t", "rank"]).reset_index(drop=True)


cs_events = prep([chart_cs[["hadm_id", "t", "cat"]], poe_orders])            # 주 규칙: 지정 기록만
cs_events_dc = prep([chart_cs[["hadm_id", "t", "cat"]], poe_orders, poe_dc])  # 진단용: D/C = 지정 없음
cs_events_ct = prep([chart_cs[["hadm_id", "t_chart", "cat"]].rename(columns={"t_chart": "t"}), poe_orders])  # 진단용: charttime 기준
CS_STATS = {"chart_records": len(chart_cs), "poe_orders": len(poe_orders), "poe_dc": len(poe_dc),
            **{f"chart_{k}": int((chart_cs.cat == k).sum()) for k in ("full", "dnr", "cmo")},
            **{f"poe_{k}": int((poe_orders.cat == k).sum()) for k in ("full", "dnr", "cmo")}}

icu = icu.sort_values(["hadm_id", "intime"]).reset_index(drop=True)
MERGE_DIAG = {}
xfer_by_hadm = {k: g for k, g in xfer.groupby("hadm_id")}


def units_between(hadm_id, t0, t1):
    x = xfer_by_hadm.get(hadm_id)
    if x is None:
        return x
    return x[(x.intime < t1) & (x.outtime > t0)]


def covers(x, t0, t1, tol=pd.Timedelta(minutes=5)):
    """x 의 구간 합집합이 [t0, t1] 을 (허용 오차 안에서) 빈틈없이 덮는지 (N3)."""
    cur = t0
    for a, b in x.sort_values("intime")[["intime", "outtime"]].itertuples(index=False):
        if pd.isna(a) or pd.isna(b) or a > cur + tol:
            return False
        cur = max(cur, b)
    return cur >= t1 - tol


def merge_stays(gap_hours: float, version_b: bool) -> pd.DataFrame:
    """같은 입원 안에서 앞 체류들 중 가장 늦은 퇴실 이후 gap_hours 이내 재입실을 병합 (F5: 누적 최대).
    B 는 추가로 (a) 간격 동안 머문 곳이 모두 수술실/회복실 (b) 간격 1h 이하이고 transfers 기록이 없음 이면 병합 (F6)."""
    prev_out = icu.groupby("hadm_id").outtime.transform(lambda s: s.cummax().shift())
    gap_h = (icu.intime - prev_out).dt.total_seconds() / 3600
    merge = gap_h.notna() & (gap_h <= gap_hours)
    if version_b:
        extra = []
        for i in range(len(icu)):
            if not gap_h.notna()[i] or merge[i] or gap_h[i] >= 24:
                extra.append(False); continue
            x = units_between(icu.hadm_id[i], prev_out[i], icu.intime[i])
            empty = x is None or len(x) == 0
            extra.append((empty and gap_h[i] <= B_EMPTY_GAP_HOURS)
                         or (not empty and x.careunit.fillna("").str.contains(PROCEDURE_UNIT_REGEX).all()
                             and covers(x, prev_out[i], icu.intime[i])))
        merge |= pd.Series(extra)
    # 진단: 병합된 간격(>0h) 동안 어디에 있었는지 — 확인된 병동 체류가 있으면 실제 재입실이 병합으로 사라진 것
    gap_kind = {"ward": 0, "procedure": 0, "icu_or_direct": 0, "unknown": 0}
    for i in np.where(merge & (gap_h > 0))[0]:
        x = units_between(icu.hadm_id[i], prev_out[i], icu.intime[i])
        if x is None or len(x) == 0:
            gap_kind["unknown"] += 1; continue
        cu = x.careunit.fillna("")
        if (~cu.str.contains(PROCEDURE_UNIT_REGEX) & ~cu.str.contains(ICU_UNIT_REGEX) & (cu != "")).any():
            gap_kind["ward"] += 1
        elif cu.str.contains(PROCEDURE_UNIT_REGEX).any():
            gap_kind["procedure"] += 1
        else:
            gap_kind["icu_or_direct"] += 1
    gap_kind["direct_0h"] = int((merge & (gap_h <= 0)).sum())
    MERGE_DIAG[gap_hours, version_b] = gap_kind
    grp = (~merge).cumsum()
    return (icu.assign(grp=grp).groupby("grp")
            .agg(subject_id=("subject_id", "first"), hadm_id=("hadm_id", "first"),
                 stay_ids=("stay_id", list), first_careunit=("first_careunit", "first"),
                 intime=("intime", "min"),
                 outtime=("outtime", lambda s: s.max() if s.notna().all() else pd.NaT))   # N5
            .reset_index(drop=True))


def status_at(df, events=None):
    """퇴실 시점 마지막 code status (none/기록 없음 -> full) 와 퇴실 후 첫 거부(dnr/cmo) 기록 시각."""
    events = cs_events if events is None else events
    base = df[["hadm_id", "outtime"]].reset_index().dropna(subset=["outtime"])
    base["outtime"] = base.outtime.astype("datetime64[us]")
    base = base.sort_values("outtime")
    last = pd.merge_asof(base, events[["hadm_id", "t", "cat"]], left_on="outtime", right_on="t",
                         by="hadm_id", direction="backward").set_index("index").reindex(df.index)
    lst_ev = events[events.cat.isin(["dnr", "cmo"])][["hadm_id", "t"]]
    nxt = pd.merge_asof(base, lst_ev, left_on="outtime", right_on="t", by="hadm_id",
                        direction="forward", allow_exact_matches=False).set_index("index").reindex(df.index)
    return last.cat.replace({"none": "full"}).fillna("full"), nxt.t, last.cat.fillna("none")


def labels(df, next_in, t_death, t_dnr, k):
    """후보 사건 중 가장 이른 것을 이벤트로 (F8)."""
    cand = {"readmission": next_in, "ward_death": t_death}
    if k == "d2":
        cand["ward_dnr"] = t_dnr
    cand = pd.DataFrame(cand)
    has = cand.notna().any(axis=1)
    etype = pd.Series("none", index=df.index)
    etype[has] = cand[has].apply(pd.to_datetime).idxmin(axis=1)
    hours = (cand.min(axis=1) - df.outtime).dt.total_seconds() / 3600
    event = (etype != "none").astype(int)
    return etype, hours, event, ((event == 1) & (hours <= EARLY_HOURS)).astype(int), \
        ((event == 1) & (hours > EARLY_HOURS)).astype(int)


def build(version: str) -> pd.DataFrame:
    flow = [("ICU 체류 (원자료)", len(icu))]
    df = merge_stays(MERGE_GAP_HOURS[version], version_b=(version == "B"))
    merged_gaps = MERGE_DIAG[MERGE_GAP_HOURS[version], version == "B"]
    flow.append(("병합 후", len(df)))

    df = df.merge(adm, on="hadm_id", how="left").merge(pat, on="subject_id", how="left")
    df = df.sort_values(["hadm_id", "intime"]).reset_index(drop=True)

    # ---- 사건 후보 (제외 전 전체 체류 기준: 재입실 체류가 나중에 제외돼도 앞 체류의 이벤트는 유지)
    next_in = df.groupby("hadm_id").intime.shift(-1)
    tol = pd.Timedelta(hours=ICU_DEATH_TOL_HOURS)
    icu_death = df.deathtime.notna() & (df.deathtime <= df.outtime + tol)
    t_death = df.deathtime.where(df.deathtime.notna() & ~icu_death & df.outtime.notna())
    df["cs_at_out"], t_lst_after, df["cs_detail"] = status_at(df)   # cs_detail: full=명시적 full, none=기록 없음
    cs_dc, _, _ = status_at(df, cs_events_dc)              # 진단: D/C 를 '지정 없음'으로 볼 때
    diverge = df.cs_at_out.isin(["dnr", "cmo"]) & (cs_dc == "full")
    reenter = diverge & ((t_lst_after - df.outtime) <= pd.Timedelta(hours=24))
    cs_ct, _, _ = status_at(df, cs_events_ct)              # 진단: ICU 에서 차팅했지만 퇴실 후 저장된 거부 기록 (W2 확인검토 §3-1)
    stored_late = (df.cs_at_out == "full") & cs_ct.isin(["dnr", "cmo"])
    end = next_in.fillna(df.dischtime)                       # 병동 DNR 은 다음 ICU 입실 전 / 퇴원 전까지만
    t_dnr = t_lst_after.where(t_lst_after <= end)

    for k in DEFS:
        src = "d1" if k == "d3" else k
        (df[f"event_type_{k}"], df[f"event_hours_{k}"], df[f"event_{k}"],
         df[f"early_{k}"], df[f"late_{k}"]) = labels(df, next_in, t_death, t_dnr, src)
        if k == "d3":                                        # 재입실만: 병동 사망은 이벤트 아님(표본 제외)
            for c in ("event", "early", "late"):
                df.loc[df.event_type_d3 == "ward_death", f"{c}_d3"] = 0
    df["excl_d1"] = (df.cs_at_out == "cmo").astype(int)
    df["excl_d2"] = df.cs_at_out.isin(["dnr", "cmo"]).astype(int)
    df["excl_d3"] = (df.excl_d1.astype(bool) | (df.event_type_d1 == "ward_death")).astype(int)

    # ---- SWIFT 용: ICU 입실 직전 위치가 응급실인지 (transfers 에서 입실 시각 직전 1h 안에 끝난 이동)
    prev = pd.merge_asof(df[["intime", "hadm_id"]].reset_index().dropna(subset=["intime"]).sort_values("intime"),
                         xfer.dropna(subset=["outtime"]).sort_values("outtime")[["hadm_id", "outtime", "careunit"]],
                         left_on="intime", right_on="outtime", by="hadm_id", direction="backward",
                         tolerance=pd.Timedelta(hours=1)).set_index("index").reindex(df.index)
    df["from_ed"] = prev.careunit.fillna("").str.contains("(?i)emergency").astype(int)

    # ---- 정적 변수
    df["age"] = df.anchor_age + (df.intime.dt.year - df.anchor_year)
    df["gender"] = (df.gender == "M").astype(int)
    df["icu_los"] = (df.outtime - df.intime).dt.total_seconds() / 86400
    df["adm_to_icu"] = (df.intime - df.admittime).dt.total_seconds() / 86400

    # ---- 공통 제외 (논문 순서). 연명치료 관련 제외는 정의별 excl_* 열로 (04 에서 적용)
    df = df[df.age >= MIN_AGE]; flow.append((f"{MIN_AGE}세 미만 제외", len(df)))
    df = df[~icu_death.loc[df.index]]; flow.append(("ICU 내 사망 제외", len(df)))
    bad_time = (df.outtime.isna() | df.intime.isna() | df.admittime.isna()                  # F7
                | (df.outtime <= df.intime) | (df.outtime > df.dischtime + pd.Timedelta(hours=24))
                | (df.intime < df.admittime - pd.Timedelta(hours=24)))
    df = df[~bad_time]; flow.append(("시간 기록 오류 제외", len(df)))

    df["split"] = np.where(df.anchor_year_group.isin(DEV_YEAR_GROUPS), "dev",
                           np.where(df.anchor_year_group.isin(TEST_YEAR_GROUPS), "test", "other"))

    # ---- 집계 출력
    print(f"\n================ 코호트 {version} (병합 기준 {MERGE_GAP_HOURS[version]}h"
          f"{' + 수술실 경유·1h 빈 간격' if version == 'B' else ''}) ================")
    for name, n in flow:
        print(f"  {name:<28}{n:>8,}")
    print(f"  퇴실 시점 code status: {df.cs_at_out.value_counts().to_dict()}")
    for k in DEFS:
        print(f"  [{k}] {def_summary(df, k)}")

    keys = ["raw", "merged", "age", "icu_death", "bad_time"]
    AGG[version] = {
        "flow": {k: n for k, (_, n) in zip(keys, flow)},
        "n_patients": df.subject_id.nunique(), "n_hadm": df.hadm_id.nunique(), "n_samples": len(df),
        "cs_at_out": {c: int((df.cs_at_out == c).sum()) for c in ("full", "dnr", "cmo")},
        "cs_detail": {c: int((df.cs_detail == c).sum()) for c in ("full", "none", "dnr", "cmo")},
        "merged_gaps": merged_gaps,
        "dc_rule_diverge": {"diverge": int(diverge.loc[df.index].sum()),
                            "diverge_cmo": int((diverge & (df.cs_at_out == "cmo")).loc[df.index].sum()),
                            "reentered_24h": int(reenter.loc[df.index].sum()),
                            "stored_after_out_lst": int(stored_late.loc[df.index].sum()),
                            "stored_after_out_d2_ward_dnr": int((stored_late & (df.event_type_d2 == "ward_dnr")
                                                                 & (df.excl_d2 == 0)).loc[df.index].sum())},
        "defs": {k: def_summary(df, k) for k in DEFS},
    }

    df = df.reset_index(drop=True)
    df["sample_id"] = np.arange(len(df))
    cols = (["sample_id", "subject_id", "hadm_id", "stay_ids", "first_careunit", "intime", "outtime", "dischtime",
             "age", "gender", "icu_los", "adm_to_icu", "from_ed", "cs_at_out", "cs_detail", "split", "anchor_year_group"]
            + [f"{c}_{k}" for k in DEFS for c in ("excl", "event_type", "event_hours", "event", "early", "late")])
    df[cols].to_parquet(WORK / f"cohort_{version}.parquet", index=False)
    return df


if __name__ == "__main__":
    units = xfer.careunit.fillna("(없음)")
    print("== transfers 의 수술실/회복실 계열 careunit")
    print(units[units.str.contains(PROCEDURE_UNIT_REGEX)].value_counts().to_string())
    print(f"== code status 기록: {CS_STATS}")
    AGG = {}
    for v in ["A", "B"]:
        build(v)
    emit(procedure_unit_transfers=int(units.str.contains(PROCEDURE_UNIT_REGEX).sum()), cohort=AGG,
         code_status_sources=CS_STATS)
