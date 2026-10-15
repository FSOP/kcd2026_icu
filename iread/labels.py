"""결과 정의(d1/d2/d3) 공통 도우미. 02 가 만든 정의별 열(excl_*, event_*, early_*, ...)을 사용한다 (이슈 f047c0f5).

d1 주분석: 퇴실 시점 CMO 제외, 이벤트 = 재입실·병동 사망
d2 논문 정의: 퇴실 시점 DNR·DNI·CMO 제외, 이벤트 = 재입실·병동 사망·병동 DNR
d3 재입실만: d1 + 병동 사망 표본 제외, 이벤트 = 재입실
"""
DEFS = ("d1", "d2", "d3")
EVENT_TYPES = ("readmission", "ward_death", "ward_dnr")


def apply_def(cohort, d):
    """정의 d 의 제외를 적용하고 early/late/event/event_type/event_hours 를 그 정의의 값으로 맞춘 사본."""
    assert d in DEFS, d
    c = cohort[cohort[f"excl_{d}"] == 0].copy()
    for col in ("early", "late", "event", "event_type", "event_hours"):
        c[col] = c[f"{col}_{d}"]
    return c


def def_summary(cohort, d):
    """정의 d 의 표본·이벤트 구성 (집계)."""
    k = cohort[cohort[f"excl_{d}"] == 0]
    typ = lambda win: {t: int(((k[f"{win}_{d}"] == 1) & (k[f"event_type_{d}"] == t)).sum()) for t in EVENT_TYPES}
    by = lambda col: {s: int(v) for s, v in k.groupby("split")[col].sum().items()}
    return {"n": len(k), "events": int(k[f"event_{d}"].sum()),
            "early": int(k[f"early_{d}"].sum()), "late": int(k[f"late_{d}"].sum()),
            "early_by_type": typ("early"), "late_by_type": typ("late"),
            "split_n": {s: int(v) for s, v in k.split.value_counts().items()},
            "split_early": by(f"early_{d}"), "split_late": by(f"late_{d}")}


def cohort_key_hash(cohort):
    """표본 정렬 키 (sample_id, subject_id, hadm_id, outtime) 의 해시. 03 산출물과 04/06 입력이 같은 코호트인지 확인용 (N10)."""
    import hashlib
    c = cohort.reset_index() if "sample_id" not in cohort.columns else cohort
    cols = ["sample_id", "subject_id", "hadm_id", "outtime", "cs_at_out"] + sorted(
        c_ for c_ in c.columns if c_.split("_")[-1] in DEFS and c_.rsplit("_", 1)[0] in
        ("excl", "event_type", "event_hours", "event", "early", "late"))
    key = c[[k for k in cols if k in c.columns]].sort_values("sample_id").astype(str).to_csv(index=False)
    return hashlib.sha256(key.encode()).hexdigest()


def check_feature_snapshot(cohort, version):
    """03 이 기록한 코호트 해시와 지금 읽은 cohort_{v}_final 이 같은지 확인. 다르면 즉시 실패."""
    import json
    from config import WORK
    meta = json.loads((WORK / f"features_{version}.meta.json").read_text())
    assert meta["cohort_hash"] == cohort_key_hash(cohort), "03 산출물(stats/ts)과 코호트가 다른 실행본입니다. 03 부터 다시 실행하세요."
    for name, h in meta.get("files", {}).items():
        assert file_hash(WORK / name) == h, f"{name} 이 03 실행 이후 바뀌었습니다. 03 부터 다시 실행하세요."


def file_hash(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
