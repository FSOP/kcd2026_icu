"""내부 열람용 데이터 둘러보기 생성기 (이슈 ad40b868). 읽기 전용: 코호트·모델·라벨을 바꾸지 않는다.

입력: 원자료(MIMIC_DIR) + 02/03 산출물(IREAD_WORK: cohort_{A,B}.parquet, cohort_{A,B}_final.parquet, chart_raw.parquet)
출력: --out 폴더에 정적 HTML 하나(index.html, 외부 요청·스크립트 없음) + 사례 번호↔원기록 키 대응표(keymap.csv, 별도 파일)
      + meta.json(코드 해시·표본추출 설정). 화면(stdout)에는 성공/실패 한 줄만. 상세 오류는 --log 파일에만.
  1. 유닛 분류 확인표: transfers careunit 명칭·빈도·현재 규칙의 분류(ICU / 시술 / 병동=나머지 / 빈 값)
  2. 유형별 이동 타임라인: 유형마다 고정 seed 로 n 사례 추출. 원기록 → 규칙별 판정 → A/B 병합 → 정의별 라벨
  3. 같은 사례의 퇴실 전 대표 입력(hr, sbp, rr)과 code status 기록 (포함/제외 이유)
규칙 상수는 02_cohort.py 원문에서 읽어 와 재계산하고, 02 실제 결과와 다르면 ⚠ 로 표시한다.
시술 경유를 계획 수술로, 병동 경유를 악화로 확정하지 않는다. 소수 사례로 라벨 정확도를 추정하지 않는다.

실행 (합성): python browse_gen.py --data <합성 원자료> --work <합성 WORK> --out <폴더> --log <파일>
실행 (실제): bash iread/internal/run_browse.sh   ← 소유자가 직접. 출력은 ~/KCD2026/private_reports/ (LLM 비열람)
"""
import argparse
import ast
import hashlib
import html
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--log", required=True)
ap.add_argument("--seed", type=int, default=20261011); ap.add_argument("--n", type=int, default=3)
ap.add_argument("--label", default="")
args = ap.parse_args()
os.umask(0o077)
OUT, LOG = Path(args.out), Path(args.log)
OUT.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
log = open(LOG, "a", buffering=1)
sys.stdout = sys.stderr = log          # 이후 모든 출력·경고·traceback 은 제한 로그로만
os.environ["MIMIC_DIR"], os.environ["IREAD_WORK"] = args.data, args.work
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
STATUS = sys.__stdout__


def done(ok, msg):
    print(("OK " if ok else "FAIL ") + msg, file=STATUS, flush=True)
    sys.exit(0 if ok else 1)


try:
    import duckdb
    import numpy as np
    import pandas as pd
    from config import CHART_ITEMS, CMO_REGEX, CODE_STATUS_ITEMS, DNR_REGEX, EARLY_HOURS, MERGE_GAP_HOURS, WORK, csv
    from labels import DEFS
except Exception:
    traceback.print_exc(); done(False, f"모듈 적재 실패 — 상세는 제한 로그({LOG})")


def consts_from(path, names):
    """02_cohort.py 의 모듈 상수를 실행 없이 원문에서 읽는다 (규칙 표류 방지)."""
    tree = ast.parse(Path(path).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(out) == set(names), f"02 상수 누락: {set(names) - set(out)}"
    return out


C02 = consts_from(HERE.parent / "02_cohort.py", ["PROCEDURE_UNIT_REGEX", "ICU_UNIT_REGEX", "B_EMPTY_GAP_HOURS"])
PROC, ICU_RE, B_EMPTY = C02["PROCEDURE_UNIT_REGEX"], C02["ICU_UNIT_REGEX"], C02["B_EMPTY_GAP_HOURS"]
REVIEW_RE = r"(?i)step|intermediate|observation|emergency|unknown|labor|delivery|hospice|lounge|transplant|neuro|psych|cath|endoscopy|radiology|dialysis"
VITAL_ITEMS = {f: CHART_ITEMS[f] for f in ("hr", "sbp", "rr")}
TYPES = [
    ("T1", "병동 전실 후 관찰창 내 재입실·사망 없음", "B·d1 사건 없음, 병원 퇴원(또는 관찰 종료)까지 추적"),
    ("T2", "24시간 이내 재입실 후보", "B·d1 첫 사건 = 재입실, 0–24h"),
    ("T3", "24–48시간 재입실 후보", "B·d1 첫 사건 = 재입실, 24–48h"),
    ("T4a", "ICU → 수술실·회복실 → ICU (병합된 간격)", "B 표본 안에 시술 유닛만 거친 병합 간격이 있음"),
    ("T4b", "ICU → (병동 등) → 수술실·회복실 → ICU (재입실)", "B·d1 재입실이고 간격 경로에 시술 유닛 포함"),
    ("T5", "재입실 전 병동 사망", "B·d1 첫 사건 = 병동 사망"),
    ("T6a", "이동 경로 불명확", "B·d1 재입실인데 간격에 이동 기록이 없거나 ICU 명칭만 있음"),
    ("T6b", "code status 불명확·충돌", "퇴실 시점 기록 없음, 출처 간 마지막 값 불일치, 또는 퇴실 후 저장·D/C 기록 존재"),
]


def h(x):
    return html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))


def unit_cat(cu, eventtype=None):
    if cu is None or (isinstance(cu, float) and np.isnan(cu)) or str(cu).strip() == "":
        return "퇴원 행(careunit 없음)" if eventtype == "discharge" else "빈 값"
    s = str(cu)
    if re.search(ICU_RE, s):
        return "ICU"
    if re.search(PROC, s):
        return "시술"
    return "병동(나머지)"


def cs_cat(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    s = str(v).lower()
    return "cmo" if re.search(CMO_REGEX, s) else "dnr" if re.search(DNR_REGEX, s) else "full"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


try:
    t_start = time.time()
    con = duckdb.connect()
    icu = con.sql(f"SELECT subject_id, hadm_id, stay_id, first_careunit, intime, outtime FROM read_csv('{csv('icu/icustays')}')").df()
    adm = con.sql(f"SELECT hadm_id, admittime, dischtime, deathtime, discharge_location FROM read_csv('{csv('hosp/admissions')}')").df()
    xfer_all = con.sql(f"SELECT hadm_id, eventtype, careunit, intime, outtime FROM read_csv('{csv('hosp/transfers')}', "
                       f"types={{'careunit': 'VARCHAR'}})").df()
    coh = {v: pd.read_parquet(WORK / f"cohort_{v}.parquet") for v in ("A", "B")}
    final = {v: (set(pd.read_parquet(WORK / f"cohort_{v}_final.parquet", columns=["sample_id"]).sample_id)
                 if (WORK / f"cohort_{v}_final.parquet").exists() else None) for v in ("A", "B")}
    print("loaded", round(time.time() - t_start), "s")

    # ------------------------------------------------------------------ 1. 유닛 분류표
    xa = xfer_all.assign(cu=xfer_all.careunit.fillna("").str.strip())
    ut = (xa.groupby(["cu"], dropna=False)
          .agg(n=("hadm_id", "size"), n_hadm=("hadm_id", "nunique"),
               ev=("eventtype", lambda s: ", ".join(f"{k} {v:,}" for k, v in s.value_counts().items())))
          .reset_index().sort_values("n", ascending=False))
    ut["cat"] = [unit_cat(c if c else None) for c in ut.cu]
    ut["both"] = ut.cu.str.contains(ICU_RE) & ut.cu.str.contains(PROC)
    ut["review"] = ut.cu.str.contains(REVIEW_RE) | ut.both

    # ------------------------------------------------------------------ 2. 사례 후보
    xfer = xfer_all[xfer_all.hadm_id.notna()].copy()
    xb = {k: g.sort_values("intime") for k, g in xfer.groupby("hadm_id")}
    icu_b = {k: g.sort_values("intime") for k, g in icu.groupby("hadm_id")}

    def gap_units(hadm, t0, t1):
        x = xb.get(hadm)
        return x.iloc[0:0] if x is None else x[(x.intime < t1) & (x.outtime > t0)]

    def path_of(hadm, t0, t1):
        """09_w6.py 와 같은 규칙: ICU 명칭 제외 후 시술 → via_procedure, 나머지 → ward_only, 없음 → unclear"""
        cu = gap_units(hadm, t0, t1).careunit.fillna("")
        cu = cu[~cu.str.contains(ICU_RE)]
        return "unclear" if len(cu) == 0 else "via_procedure" if cu.str.contains(PROC).any() else "ward_only"

    B = coh["B"].copy()
    B["path"] = None
    r_mask = B.event_type_d1 == "readmission"
    B.loc[r_mask, "path"] = [path_of(r.hadm_id, r.outtime, r.outtime + pd.Timedelta(hours=float(r.event_hours_d1)))
                             for r in B[r_mask].itertuples()]

    def merged_proc(r):
        st = icu_b[r.hadm_id]; st = st[st.stay_id.isin(r.stay_ids)].sort_values("intime")
        for a, b in zip(st.outtime.iloc[:-1], st.intime.iloc[1:]):
            if b > a:
                cu = gap_units(r.hadm_id, a, b).careunit.fillna("")
                if len(cu) and cu.str.contains(PROC).all():
                    return True
        return False

    multi = B[B.stay_ids.map(len) > 1]
    t4a = set(multi[[merged_proc(r) for r in multi.itertuples()]].sample_id) if len(multi) else set()

    # code status 원기록 (실제 값 문자열 포함 — 내부 보고서 전용)
    hadms_B = set(B.hadm_id)
    chart_cs = con.sql(f"""SELECT stay_id, charttime, storetime, value FROM '{WORK}/chart_raw.parquet'
                           WHERE itemid IN ({",".join(map(str, CODE_STATUS_ITEMS))})""").df().merge(icu[["stay_id", "hadm_id"]], on="stay_id")
    con.execute(f"""CREATE TABLE csd AS SELECT poe_id, max(field_value) AS v FROM read_csv('{csv('hosp/poe_detail')}')
                    WHERE field_name = 'Code status' GROUP BY poe_id""")
    poe_cs = con.sql(f"""SELECT p.poe_id, p.hadm_id, p.ordertime, p.transaction_type, p.discontinue_of_poe_id, d.v
        FROM read_csv('{csv('hosp/poe')}', types={{'order_provider_id': 'VARCHAR', 'discontinue_of_poe_id': 'VARCHAR',
             'discontinued_by_poe_id': 'VARCHAR', 'poe_id': 'VARCHAR'}}) p LEFT JOIN csd d USING (poe_id)
        WHERE p.poe_id IN (SELECT poe_id FROM csd) OR p.discontinue_of_poe_id IN (SELECT poe_id FROM csd)""").df()

    chart_by = {k: g for k, g in chart_cs.groupby("hadm_id")}
    poe_by = {k: g for k, g in poe_cs.dropna(subset=["hadm_id"]).groupby("hadm_id")}
    EMPTY_C, EMPTY_P = chart_cs.iloc[0:0], poe_cs.iloc[0:0]

    def cs_records(hadm):
        rows = []
        for r in chart_by.get(hadm, EMPTY_C).itertuples():
            t_used = max(r.charttime, r.storetime) if pd.notna(r.storetime) else r.charttime
            rows.append(dict(src="chart(223758)", t=t_used, t_chart=r.charttime, t_store=r.storetime, kind="기록",
                             value=r.value, ccat=cs_cat(r.value)))
        for r in poe_by.get(hadm, EMPTY_P).itertuples():
            dc = pd.notna(r.discontinue_of_poe_id) and pd.isna(r.v)
            rows.append(dict(src="POE", t=r.ordertime, t_chart=r.ordertime, t_store=None,
                             kind="D/C(주 규칙: 상태 유지)" if dc else str(r.transaction_type), value=r.v,
                             ccat=None if dc else cs_cat(r.v)))
        return pd.DataFrame(rows, columns=["src", "t", "t_chart", "t_store", "kind", "value", "ccat"])

    def cs_flags(r):
        rec = cs_records(r.hadm_id)
        if len(rec) == 0:
            return ["기록 없음"]
        f = []
        before = rec[(rec.t <= r.outtime) & rec.ccat.notna()]
        if len(before) == 0:
            f.append("퇴실 전 지정 기록 없음")
        last = before.sort_values("t").groupby("src").ccat.last()
        if len(set(last)) > 1:
            f.append("출처별 마지막 값 불일치")
        if ((rec.t_chart <= r.outtime) & (rec.t > r.outtime) & rec.ccat.notna()).any():
            f.append("퇴실 전 기록·퇴실 후 저장")
        if rec.kind.str.startswith("D/C").any():
            f.append("D/C 기록")
        return f

    B_hadm_cs = {}
    pools = {t: [] for t, _, _ in TYPES}
    for r in B.itertuples():
        e, hrs = r.event_type_d1, r.event_hours_d1
        if r.excl_d1:                      # 정의 ① 대상만 (퇴실 시점 CMO 제외)
            continue
        if e == "none":
            pools["T1"].append(r.sample_id)
        if e == "readmission" and hrs <= 24:
            pools["T2"].append(r.sample_id)
        if e == "readmission" and 24 < hrs <= EARLY_HOURS:
            pools["T3"].append(r.sample_id)
        if r.sample_id in t4a:
            pools["T4a"].append(r.sample_id)
        if e == "readmission" and r.path == "via_procedure":
            pools["T4b"].append(r.sample_id)
        if e == "ward_death":
            pools["T5"].append(r.sample_id)
        if e == "readmission" and r.path == "unclear":
            pools["T6a"].append(r.sample_id)
    # T6b: code status 플래그는 비용이 커서 무작위 순서로 훑어 n 개만 찾는다 (탐색 순서도 seed 고정)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(B.sample_id.to_numpy())
    Bi = B.set_index("sample_id")
    scanned, conflict, nonrec = 0, [], []
    CONFLICT = {"출처별 마지막 값 불일치", "퇴실 전 기록·퇴실 후 저장", "D/C 기록"}
    for sid in order:                      # 충돌·D/C·퇴실 후 저장을 먼저 채우고, 모자라면 '퇴실 시점 기록 없음'으로 채움
        r = Bi.loc[sid].copy(); r["sample_id"] = sid
        fl = set(cs_flags(r)); scanned += 1
        if fl & CONFLICT:
            conflict.append(sid)
        elif r.cs_detail == "none" and len(nonrec) < args.n:
            nonrec.append(sid)
        if len(conflict) >= args.n or scanned >= 20000:
            break
    pools["T6b"] = (conflict + nonrec)[:args.n]
    chosen = {}
    for t, _, _ in TYPES:
        p = sorted(pools[t])
        chosen[t] = (sorted(rng.choice(p, size=min(args.n, len(p)), replace=False).tolist()) if t != "T6b"
                     else pools["T6b"][:args.n])
    pool_n = {t: (f"{len(pools[t]):,}" if t != "T6b" else f"seed 순서로 {scanned:,}건 훑음: 충돌·D/C·퇴실 후 저장 {len(conflict)}, 기록 없음 {len(nonrec)}")
              for t in pools}
    print("pools", {t: (len(v) if t != "T6b" else len(v)) for t, v in pools.items()})

    # 측정값 원기록
    def vitals(r, stays):
        ids = [i for ids_, _ in VITAL_ITEMS.values() for i in ids_]
        st = ",".join(map(str, stays))
        v = con.sql(f"""SELECT stay_id, itemid, charttime, storetime, value, valuenum FROM '{WORK}/chart_raw.parquet'
                        WHERE stay_id IN ({st}) AND itemid IN ({",".join(map(str, ids))})""").df()
        inv = {i: (f, rng_) for f, (ids_, rng_) in VITAL_ITEMS.items() for i in ids_}
        v["feature"] = v.itemid.map(lambda i: inv[i][0])
        rows = []
        for x in v.itertuples():
            lo, hi = inv[x.itemid][1]
            store = x.storetime if pd.notna(x.storetime) else x.charttime
            if pd.isna(x.valuenum):
                why = "제외: 숫자 값 없음"
            elif not (r.intime <= x.charttime <= r.outtime):
                why = "제외: 체류 기간 밖 측정"
            elif store > r.outtime:
                why = "제외: 퇴실 후 저장(퇴실 시점에 알 수 없음)"
            elif x.valuenum < lo or (hi is not None and x.valuenum > hi):
                why = f"제외: 허용 범위 밖({lo}–{hi})"
            else:
                why = "포함"
            rows.append((x.feature, x.itemid, x.charttime, x.storetime, x.value, x.valuenum, why))
        return pd.DataFrame(rows, columns=["feature", "itemid", "charttime", "storetime", "value", "valuenum", "why"])

    try:
        units = con.sql(f"SELECT itemid, unitname FROM read_csv('{csv('icu/d_items')}')").df().set_index("itemid").unitname.to_dict()
    except Exception:
        units = {}

    # ------------------------------------------------------------------ 렌더링
    def rel(t, t0):
        return None if pd.isna(t) else round((t - t0).total_seconds() / 3600, 1)

    CAT_COLOR = {"ICU": "#d9534f", "시술": "#f0ad4e", "병동(나머지)": "#5b9bd5", "빈 값": "#999", "퇴원 행(careunit 없음)": "#999"}

    def svg_timeline(r, t0, A_rows, B_rows, stays, x, cs, adm_r):
        lo = min(rel(stays.intime.min(), t0) or 0, -24); lo = max(lo, -240)
        ends = [rel(adm_r.dischtime, t0), rel(adm_r.deathtime, t0), 72]
        hi = min(max(e for e in ends if e is not None) + 6, 24 * 21)
        W_, L = 900, 150
        sx = lambda v: L + (min(max(v, lo), hi) - lo) / (hi - lo) * (W_ - L - 90)
        lanes = [("이동 기록(transfers)", x), ("ICU 체류(원자료)", stays), ("A 병합 표본", A_rows), ("B 병합 표본", B_rows)]
        y, out = 20, []
        for name, df in lanes:
            out.append(f'<text x="4" y="{y + 12}" class="lab">{h(name)}</text>')
            for k, row in enumerate(df.itertuples()):
                a, b = rel(row.intime, t0), rel(row.outtime, t0)
                if a is None:
                    continue
                b = hi if b is None else b
                if name.startswith("이동"):
                    col = CAT_COLOR.get(unit_cat(row.careunit, row.eventtype), "#999"); tip = f"{row.careunit} ({unit_cat(row.careunit, row.eventtype)})"
                elif name.startswith("ICU"):
                    col = "#b52b27"; tip = f"ICU 체류 {row.first_careunit}"
                else:
                    col = "#555" if not getattr(row, "is_index", False) else "#2a7"; tip = f"표본 {a}–{b}h"
                yy = y + (k % 2) * 7 if name.startswith("이동") else y
                out.append(f'<rect x="{sx(a):.1f}" y="{yy}" width="{max(sx(b) - sx(a), 1.5):.1f}" height="12" fill="{col}" opacity=".85">'
                           f'<title>{h(tip)}: {a}h → {b}h</title></rect>')
            y += 30
        marks = [("ICU 퇴실(기준 0h)", 0, "#000")]
        for lab_, t in (("사망", adm_r.deathtime), ("병원 퇴원", adm_r.dischtime)):
            if pd.notna(t):
                marks.append((lab_, rel(t, t0), "#000"))
        if r.event_type_d1 == "readmission":
            marks.append(("다음 ICU 입실", float(r.event_hours_d1), "#d9534f"))
        for _, c in cs.iterrows():
            if pd.notna(c.ccat) and lo <= rel(c.t, t0) <= hi:
                out.append(f'<circle cx="{sx(rel(c.t, t0)):.1f}" cy="{y + 6}" r="4" fill="{ {"full": "#5cb85c", "dnr": "#f0ad4e", "cmo": "#7d3c98"}[c.ccat]}">'
                           f'<title>code status {c.src} {h(c.ccat)} @ {rel(c.t, t0)}h</title></circle>')
        out.append(f'<text x="4" y="{y + 10}" class="lab">code status 기록</text>')
        y += 22
        for lab_, v, col in marks:
            if v is not None and lo <= v <= hi:
                out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="10" y2="{y}" stroke="{col}" stroke-dasharray="3,2"/>'
                           f'<text x="{sx(v) + 2:.1f}" y="{y + 12}" class="mk">{h(lab_)} {v}h</text>')
        for v in (0, 24, 48):
            if lo <= v <= hi:
                out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="10" y2="{y}" stroke="#ccc"/>')
        tick = 24 if hi - lo <= 24 * 8 else 72
        for v in range(int(np.ceil(lo / tick) * tick), int(hi) + 1, tick):
            out.append(f'<text x="{sx(v):.1f}" y="{y + 26}" class="tk">{v}h</text>')
        return f'<svg viewBox="0 0 {W_} {y + 32}" width="100%" role="img">{"".join(out)}</svg>'

    def gap_rules(hadm, t0):
        """원자료 ICU 체류 사이 간격마다 A/B 병합 규칙 재계산"""
        st = icu_b[hadm].reset_index(drop=True)
        rows, cm = [], st.outtime.iloc[0]
        for i in range(1, len(st)):
            a, b = cm, st.intime[i]
            gap = (b - a).total_seconds() / 3600
            x = gap_units(hadm, a, b); cu = x.careunit.fillna("")
            seq = " → ".join(f"{c}[{unit_cat(c)}]" for c in cu) or "(이동 기록 없음)"
            A_m = gap <= MERGE_GAP_HOURS["A"]
            if gap <= 0:
                B_m, why = True, "간격 0 이하(직접 이동)"
            elif gap >= 24:
                B_m, why = False, "간격 ≥24h"
            elif len(x) == 0:
                B_m, why = gap <= B_EMPTY, f"이동 기록 없음 + 간격 {'≤' if gap <= B_EMPTY else '>'}{B_EMPTY}h"
            else:
                allp = cu.str.contains(PROC).all()
                cur, ok = a, True
                for s_, e_ in x.sort_values("intime")[["intime", "outtime"]].itertuples(index=False):
                    if pd.isna(s_) or pd.isna(e_) or s_ > cur + pd.Timedelta(minutes=5):
                        ok = False; break
                    cur = max(cur, e_)
                ok = ok and cur >= b - pd.Timedelta(minutes=5)
                B_m = allp and ok
                why = ("간격 전체가 시술 유닛이고 빈틈없음" if B_m else
                       "시술 외 유닛 경유" if not allp else "시술 유닛이지만 간격을 다 덮지 못함")
            rows.append(dict(gap=f"{rel(a, t0)}h → {rel(b, t0)}h", hours=round(gap, 2), seq=seq,
                             A=f"{'병합' if A_m else '분리'} (간격 {'≤' if A_m else '>'}{MERGE_GAP_HOURS['A']}h)",
                             B=f"{'병합' if B_m else '분리'} ({why})", A_m=A_m, B_m=B_m, i=i))
            cm = max(cm, st.outtime[i])
        return st, rows

    def label_rows(v, hadm, t0, index_out):
        c = coh[v][coh[v].hadm_id == hadm].sort_values("intime")
        out = []
        for r in c.itertuples():
            cells = []
            for d in DEFS:
                if getattr(r, f"excl_{d}"):
                    cells.append("대상 제외")
                else:
                    e = getattr(r, f"event_type_{d}")
                    cells.append("사건 없음" if e == "none" else
                                 f"{e} {round(getattr(r, f'event_hours_{d}'), 1)}h ({'early' if getattr(r, f'early_{d}') else 'late'})")
            fin = final[v]
            out.append(dict(idx=r.outtime == index_out, span=f"{rel(r.intime, t0)}h → {rel(r.outtime, t0)}h", n=len(r.stay_ids),
                            cs=f"{r.cs_at_out} / {r.cs_detail}", split=r.split,
                            fin="—" if fin is None else ("포함" if r.sample_id in fin else "제외(퇴실 전 측정값 없음)"),
                            cells=cells))
        return out

    case_html, keymap, n_case = [], [], 0
    for t, title, rule in TYPES:
        sec = [f'<h3 id="{t}">{t}. {h(title)}</h3><p class="rule">선정 규칙: {h(rule)} · 후보 {h(pool_n[t])}건 · 추출 {len(chosen[t])}건</p>']
        if not chosen[t]:
            sec.append('<p class="none">해당 사례 없음</p>')
        for sid in chosen[t]:
            n_case += 1; cno = f"C{n_case:02d}"
            r = Bi.loc[sid].copy(); r["sample_id"] = sid
            t0 = r.outtime; hadm = r.hadm_id
            keymap.append(dict(case=cno, type=t, subject_id=r.subject_id, hadm_id=hadm, stay_ids=" ".join(map(str, r.stay_ids)), sample_id_B=sid))
            adm_r = adm[adm.hadm_id == hadm].iloc[0]
            stays, gaps = gap_rules(hadm, t0)
            x = xb.get(hadm, xfer.iloc[0:0])
            A_rows = coh["A"][coh["A"].hadm_id == hadm].assign(is_index=lambda d: (d.intime <= t0) & (d.outtime >= t0))
            B_rows = coh["B"][coh["B"].hadm_id == hadm].assign(is_index=lambda d: d.outtime == t0)
            cs = cs_records(hadm)
            # 02 실제 병합과 재계산 비교
            warn = []
            for v, rows_ in (("A", A_rows), ("B", B_rows)):
                groups = [set(s) for s in rows_.stay_ids]
                for g in gaps:
                    a_id, b_id = stays.stay_id[g["i"] - 1], stays.stay_id[g["i"]]
                    same = any(a_id in s and b_id in s for s in groups)
                    both_in = any(a_id in s for s in groups) and any(b_id in s for s in groups)
                    if both_in and same != g[f"{v}_m"]:
                        warn.append(f"{v}: 간격 {g['gap']} 재계산={'병합' if g[f'{v}_m'] else '분리'}, 02 결과={'병합' if same else '분리'}")
            body = [f'<div class="case"><h4>{cno} <span class="tag">{t}</span></h4>']
            if warn:
                body.append('<p class="warn">⚠ 규칙 재계산과 02 결과 불일치: ' + h("; ".join(warn)) + "</p>")
            body.append('<p class="note">경로 해석 주의: 시술 경유 ≠ 계획 수술 확정, 병동 경유 ≠ 악화 확정. 아래 시각은 모두 ICU 퇴실(B 표본) 기준 상대시간입니다.</p>')
            body.append(svg_timeline(r, t0, A_rows, B_rows, stays, x, cs, adm_r))
            body.append('<p class="legend"><span style="background:#d9534f"></span>ICU <span style="background:#f0ad4e"></span>시술 '
                        '<span style="background:#5b9bd5"></span>병동(나머지) <span style="background:#999"></span>빈 값 '
                        '<span style="background:#2a7"></span>기준 B 표본 ● code status(초록 full, 주황 DNR, 보라 CMO)</p>')
            # (1) 원기록
            tr = "".join(f"<tr><td>{h(row.eventtype)}</td><td>{h(row.careunit)}</td><td>{h(unit_cat(row.careunit, row.eventtype))}</td>"
                         f"<td>{h(rel(row.intime, t0))}</td><td>{h(rel(row.outtime, t0))}</td></tr>" for row in x.itertuples())
            body.append(f'<details open><summary>① 원래 기록: 이동(transfers) {len(x)}행, ICU 체류 {len(stays)}건</summary>'
                        f'<table><tr><th>eventtype</th><th>careunit</th><th>현재 분류</th><th>시작 h</th><th>종료 h</th></tr>{tr}</table>')
            ov = []
            xs = x.dropna(subset=["intime"]).sort_values("intime").reset_index(drop=True)
            for i in range(1, len(xs)):
                p_out = xs.outtime[i - 1]
                if pd.notna(p_out):
                    d_ = (xs.intime[i] - p_out).total_seconds() / 3600
                    if abs(d_) > 1 / 60:
                        ov.append(f"{'공백' if d_ > 0 else '중첩'} {abs(round(d_, 2))}h @ {rel(p_out, t0)}h")
            body.append(f'<p>이동 기록 공백·중첩: {h("; ".join(ov) or "없음(연속)")}</p>')
            st_tr = "".join(f"<tr><td>ICU {k + 1}</td><td>{h(s.first_careunit)}</td><td>{h(rel(s.intime, t0))}</td><td>{h(rel(s.outtime, t0))}</td></tr>"
                            for k, s in enumerate(stays.itertuples()))
            body.append(f'<table><tr><th>ICU 체류</th><th>유닛</th><th>입실 h</th><th>퇴실 h</th></tr>{st_tr}</table>'
                        f'<p>입원 퇴원: {h(rel(adm_r.dischtime, t0))}h ({h(adm_r.discharge_location)}) · 사망: {h(rel(adm_r.deathtime, t0)) or "기록 없음"}</p></details>')
            # (2) 규칙별 판정
            gr = "".join(f"<tr><td>{h(g['gap'])}</td><td>{g['hours']}</td><td>{h(g['seq'])}</td><td>{h(g['A'])}</td><td>{h(g['B'])}</td></tr>" for g in gaps)
            body.append('<details open><summary>② 규칙별 판정 (ICU 체류 사이 간격마다)</summary>' +
                        (f'<table><tr><th>간격</th><th>시간</th><th>간격 중 경로[분류]</th><th>A 규칙</th><th>B 규칙</th></tr>{gr}</table>'
                         if gaps else "<p>ICU 체류가 1건이라 병합 판단 없음</p>") +
                        (f'<p>재입실 경로 분류(09 규칙): <b>{h(r.path)}</b></p>' if r.path else "") + "</details>")
            # (3) A/B 표본과 정의별 라벨
            lt = ""
            for v in ("A", "B"):
                for L_ in label_rows(v, hadm, t0, t0):
                    lt += (f"<tr class={'idx' if L_['idx'] else ''}><td>{v}</td><td>{h(L_['span'])}</td><td>{L_['n']}</td><td>{h(L_['cs'])}</td>"
                           f"<td>{h(L_['split'])}</td><td>{h(L_['fin'])}</td>" + "".join(f"<td>{h(c)}</td>" for c in L_["cells"]) + "</tr>")
            body.append('<details open><summary>③ A/B 병합 결과 → ④ 정의별 최종 라벨</summary>'
                        '<table><tr><th>코호트</th><th>표본 구간</th><th>ICU 체류 수</th><th>퇴실 시점 cs (판정/세부)</th><th>분할</th>'
                        '<th>03 포함</th><th>① d1</th><th>② d2</th><th>③ d3</th></tr>' + lt +
                        '</table><p class="note">초록 행 = 이 사례의 기준 표본(A는 기준 퇴실을 포함하는 표본). '
                        '공통 제외(18세 미만·ICU 내 사망·시간 오류)된 표본은 표에 나오지 않습니다.</p></details>')
            # (5) code status
            cs_s = cs.sort_values("t")
            before = cs_s[(cs_s.t <= t0) & cs_s.ccat.notna()]
            decide = before.index[-1] if len(before) else None
            ct = "".join(f"<tr class={'idx' if i == decide else ''}><td>{h(c.src)}</td><td>{h(c.kind)}</td><td>{h(rel(c.t_chart, t0))}</td>"
                         f"<td>{h(rel(c.t_store, t0))}</td><td>{h(rel(c.t, t0))}</td><td>{h(c.value)}</td><td>{h(c.ccat if pd.notna(c.ccat) else '—')}</td>"
                         f"<td>{'퇴실 후 → 퇴실 판정 미사용' if c.t > t0 else ('판정 근거' if i == decide else '')}</td></tr>"
                         for i, c in cs_s.iterrows())
            fl = cs_flags(r)
            body.append(f'<details><summary>⑤ code status 기록 {len(cs)}건 · 퇴실 시점 판정 {h(r.cs_at_out)} '
                        f'(세부 {h(r.cs_detail)}) · 표시: {h(", ".join(fl) or "없음")}</summary>'
                        '<table><tr><th>출처</th><th>종류</th><th>기록 h</th><th>저장 h</th><th>판정에 쓰는 시각 h</th><th>값(원문)</th><th>범주</th><th>비고</th></tr>'
                        + ct + '</table><p class="note">규칙: 퇴실 시각까지의 마지막 지정 기록. chart 는 max(측정, 저장) 시각 사용. '
                        'POE D/C 는 상태를 바꾸지 않음. 기록 없음은 full 로 확정하지 않고 세부 "none" 으로 따로 표시.</p></details>')
            # (6) 측정값
            vt = vitals(r, list(r.stay_ids))
            vt["rel"] = [rel(c, t0) for c in vt.charttime]
            near = vt[(vt.rel >= -24) & (vt.rel <= 6)].sort_values(["feature", "charttime"])
            summ = vt.groupby(["feature", "why"]).size().unstack(fill_value=0)
            vtr = "".join(f"<tr class={'' if w.why == '포함' else 'ex'}><td>{h(w.feature)}</td><td>{h(units.get(w.itemid, '단위 미상'))}</td>"
                          f"<td>{h(w.rel)}</td><td>{h(rel(w.storetime, t0))}</td><td>{h(w.value)}</td><td>{h(w.why)}</td></tr>"
                          for w in near.head(60).itertuples())
            body.append(f'<details><summary>⑥ 퇴실 전 대표 입력 (hr·sbp·rr): 체류 중 전체 {len(vt)}건 — 아래는 퇴실 전 24h ~ 후 6h</summary>'
                        f'{summ.to_html(classes="mini")}<table><tr><th>변수</th><th>단위</th><th>측정 h</th><th>저장 h</th><th>값</th><th>모델 사용</th></tr>{vtr}</table>'
                        f'<p class="note">{"60행까지만 표시. " if len(near) > 60 else ""}모델 입력은 체류 전체의 요약(처음·마지막·최소·최대·평균·SD·개수)과 '
                        '퇴실 전 168h 1시간 평균입니다. 같은 시간대 반복 측정은 평균됩니다.</p></details></div>')
            sec.append("".join(body))
        case_html.append("".join(sec))

    # ------------------------------------------------------------------ 문서 조립
    files = {p: sha(HERE.parent / p) for p in ("02_cohort.py", "03_features.py", "config.py", "labels.py", "09_w6.py")}
    files["internal/browse_gen.py"] = sha(__file__)
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), label=args.label, seed=args.seed, n_per_type=args.n,
                data_dir_name=Path(args.data).name, code_sha256_16=files,
                cohort_sha256_16={f"cohort_{v}.parquet": sha(WORK / f"cohort_{v}.parquet") for v in ("A", "B")},
                pool={t: pool_n[t] for t in pool_n}, chosen_count={t: len(v) for t, v in chosen.items()})
    ut_rows = "".join(f"<tr class={'rev' if rv else ''}><td>{h(cu) if cu else '<i>(NULL·빈 값)</i>'}</td><td>{n:,}</td><td>{nh:,}</td><td>{h(ev)}</td>"
                      f"<td>{h(cat)}</td><td>{'⚠ ICU·시술 둘 다 일치' if b else ('확인 권장(명칭 키워드)' if rv else '')}</td></tr>"
                      for cu, n, nh, ev, cat, b, rv in ut[["cu", "n", "n_hadm", "ev", "cat", "both", "review"]].itertuples(index=False))
    cat_sum = ut.groupby("cat").n.sum().to_dict()
    toc = "".join(f'<li><a href="#{t}">{t}. {h(ti)}</a> ({len(chosen[t])})</li>' for t, ti, _ in TYPES)
    css = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--head:#f3f4f6;--idx:#e5f6ea;--ex:#fbeaea;--rev:#fff4d6}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#aaa;--line:#3a3d44;--head:#22252b;--idx:#1f3a29;--ex:#3a2222;--rev:#3a3320}}
body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,'Noto Sans KR',sans-serif;max-width:1100px;margin:0 auto;padding:20px 16px}
table{border-collapse:collapse;margin:8px 0;font-size:.9em;display:block;overflow-x:auto}th,td{border:1px solid var(--line);padding:3px 7px;vertical-align:top}
th{background:var(--head)}tr.idx td{background:var(--idx)}tr.ex td{background:var(--ex)}tr.rev td{background:var(--rev)}
.case{border:1px solid var(--line);border-radius:8px;padding:8px 14px;margin:14px 0}.tag{font-size:.75em;background:var(--head);padding:1px 6px;border-radius:4px}
.note,.rule{color:var(--mut);font-size:.9em}.warn{color:#c0392b;font-weight:600}.none{color:var(--mut);font-style:italic}
svg{background:#fff;border:1px solid var(--line);border-radius:6px}svg .lab{font-size:11px}svg .mk{font-size:10px}svg .tk{font-size:10px;fill:#666}
.legend span{display:inline-block;width:12px;height:10px;margin:0 3px 0 10px}summary{cursor:pointer;font-weight:600;margin:6px 0}
.banner{border:2px solid #c0392b;padding:8px 12px;border-radius:6px}"""
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>MIMIC-IV 내부 둘러보기</title><style>{css}</style></head><body>
<p class="banner"><b>내부 전용 · 반출 금지{h(' · ' + args.label) if args.label else ''}</b> — 환자별 기록이 있을 수 있습니다. 보드·GitHub·외부 LLM·메신저에 올리거나 화면을 캡처해 공유하지 마세요.
의견은 아래 "의견 기록 방법"대로 환자 원문 없이 적어 주세요.</p>
<h1>MIMIC-IV 데이터 둘러보기 — 원기록 → 규칙 → 라벨</h1>
<p>생성 {h(meta['generated'])} · seed {args.seed} · 유형별 최대 {args.n}건 · 데이터 폴더 <code>{h(meta['data_dir_name'])}</code></p>
<p class="note">소규모 둘러보기입니다. 모집단 오류율을 추정하는 표본이 아니며, 몇 사례로 라벨 전체의 정확성이나 전실의 임상적 적절성을 입증하지 않습니다.</p>
<h2>목차</h2><ol><li><a href="#units">유닛 분류 확인표</a></li><li>유형별 사례<ul>{toc}</ul></li><li><a href="#howto">의견 기록 방법</a></li><li><a href="#meta">실행 버전</a></li></ol>
<h2 id="units">1. 유닛 분류 확인표 (transfers.careunit 전체)</h2>
<p>현재 코드 규칙(02_cohort.py에서 읽음): ICU = <code>{h(ICU_RE)}</code> · 시술 = <code>{h(PROC)}</code> · <b>그 외 이름 = 병동(나머지)</b> · NULL/빈 값 별도.
B 병합: 간격 0 이하, 또는 이동 기록 없고 ≤{B_EMPTY}h, 또는 간격 전체가 시술 유닛으로 빈틈없이 덮임. A 병합: 간격 ≤{MERGE_GAP_HOURS['A']}h.
노란 행 = 명칭상 다시 볼 만한 것(키워드 <code>{h(REVIEW_RE)}</code>) — 판정이 틀렸다는 뜻이 아닙니다.</p>
<p>분류별 행 수: {h(", ".join(f"{k} {v:,}" for k, v in cat_sum.items()))}</p>
<table><tr><th>careunit</th><th>행 수</th><th>입원 수</th><th>eventtype</th><th>현재 분류</th><th>표시</th></tr>{ut_rows}</table>
<h2>2. 유형별 사례 (기준: 코호트 B · 정의 ①)</h2>
<p class="note">추출: 유형별 후보를 sample_id 순으로 정렬한 뒤 seed 고정 무작위 비복원 추출(T6b는 seed 고정 순서로 훑어 처음 찾은 n건). 사례 번호 C01…는 이 보고서 안에서만 쓰는 번호이고,
원기록 대조용 키는 같은 폴더의 별도 제한 파일 <code>keymap.csv</code>에만 있습니다.</p>
{"".join(case_html)}
<h2 id="howto">3. 의견 기록 방법 (환자 원문 없이)</h2>
<p>보드에는 아래 형식으로만 적어 주세요. 사례 번호·시각·값·유닛 실명·화면 캡처는 쓰지 않습니다.</p>
<pre>규칙: (예) "그 외 이름 = 병동" / B 1시간 빈 간격 병합 / code status D/C 유지 / 퇴실 후 저장 기록 무시
의심 이유: (예) 명칭상 ICU와 비슷한 단계적 감시 유닛이 병동으로 분류되는 것으로 보임
해당 유형: T1–T6b 중
예상 영향: 재입실 과대/과소, 제외 과다 등 방향만
제안: 규칙 수정안 또는 추가 확인 방법</pre>
<h2 id="meta">4. 실행 버전</h2><pre>{h(json.dumps(meta, ensure_ascii=False, indent=1))}</pre>
</body></html>"""
    (OUT / "index.html").write_text(page)
    pd.DataFrame(keymap).to_csv(OUT / "keymap.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    for f in ("index.html", "keymap.csv", "meta.json"):
        os.chmod(OUT / f, 0o600)
    print("done", round(time.time() - t_start), "s")
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    done(False, f"생성 실패 — 상세는 제한 로그({LOG})")
done(True, f"생성 완료 — {OUT / 'index.html'} (사례 {n_case}건)")
