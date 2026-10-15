"""단계별 집계 결과 허용 스키마 (allowlist).

규칙
- 허용된 필드만 통과. 모르는 필드가 하나라도 있으면 결과 전체를 보류.
- 문자열 값은 금지. 열거형(enum)으로 미리 정한 값만 허용.
- 숫자는 유한값만 허용(NaN, Infinity 금지). 범위를 벗어나면 보류.
- 인원수(count)가 1~MIN_CELL-1 이면 "<MIN_CELL" 로 가린다(소표본 억제).
"""
MIN_CELL = 10

# 값 종류
COUNT = ("count",)                       # 0 이상 정수, 소표본 억제 대상
META_INT = ("int",)
META_INT_BIG = ("float", 0, 1e6)          # 설정값(상한 등)                      # 환자 수가 아닌 메타데이터 개수(itemid 개수 등). 억제하지 않음
RATE = ("float", 0.0, 1.0)               # 비율, AUROC 등
FLOAT = ("float", -1e9, 1e9)             # 평균 등 일반 실수


def enum(*vals):
    return ("enum", frozenset(vals))


def table(keys, value):
    """키가 허용 목록에 있는 dict. 예: {변수명: 결측률}"""
    return ("table", frozenset(keys), value)


def record(**fields):
    """필드마다 형식이 다른 고정 구조 dict. 예: 코호트 A 의 {n_samples: COUNT, flow: {...}}"""
    return ("record", fields)


VERSIONS = ("A", "B")
TASKS = ("early", "late", "overall")
MODELS = ("iREAD", "LSTM only", "LightGBM only", "LR", "RF", "Cox", "MEWS", "NEWS", "SWIFT")
SPLITS = ("dev", "test", "other")
VITAL_KEYS = ("hr", "rr", "sbp", "dbp", "temp", "spo2", "gcs_eye", "gcs_verbal", "gcs_motor")
LAB_KEYS = ("albumin", "bun", "bilirubin", "calcium", "ion_calcium", "creatinine", "glucose", "hco3",
            "lactate", "lymphocyte", "inr", "platelet", "potassium", "sodium", "wbc", "ph", "po2")
FEATURES = VITAL_KEYS + LAB_KEYS
STATIC = ("age", "icu_los", "adm_to_icu")
DEFS = ("d1", "d2", "d3")
OUTCOMES = DEFS
EVENT_TYPES = ("readmission", "ward_death", "ward_dnr")
DEF_SUMMARY = record(n=COUNT, events=COUNT, early=COUNT, late=COUNT,
                     early_by_type=table(EVENT_TYPES, COUNT), late_by_type=table(EVENT_TYPES, COUNT),
                     split_n=table(SPLITS, COUNT), split_early=table(SPLITS, COUNT), split_late=table(SPLITS, COUNT))
DISPO_CATS = ("discharge", "ward", "procedure", "icu_name", "blank", "no_record")
DISCH_LOCS = ("died", "hospice", "acute_hospital", "facility", "home_health", "home", "against_advice", "other", "missing")
T1_Q = ("record", {"mean": FLOAT, "sd": FLOAT, "p25": FLOAT, "p50": FLOAT, "p75": FLOAT})
EXTRACT_KEYS = FEATURES + ("temp_c", "temp_f", "code_status", "fio2", "paco2")

SCHEMAS = {
    # 시험용
    "demo": {
        "n": COUNT,
        "event_rate": RATE,
        "auroc": RATE,
        "version": enum(*VERSIONS),
        "missing": table(("hr", "rr", "lactate"), RATE),
    },
    # 01_extract: 변수별 사전(d_items) 확인 개수와 추출 행 수, code status 분류
    "extract": {
        "itemids_expected": table(EXTRACT_KEYS, META_INT),
        "itemids_in_dict": table(EXTRACT_KEYS, META_INT),
        "rows": table(EXTRACT_KEYS, COUNT),
        "code_status": table(("cmo", "dnr", "other"), COUNT),
        "code_status_n_values": COUNT,
    },
    # 02_cohort: 코호트 크기와 결과 정의 3종 구성 (A, B 한 번에)
    "cohort": {
        "procedure_unit_transfers": COUNT,
        "code_status_sources": record(chart_records=COUNT, poe_orders=COUNT, poe_dc=COUNT,
                                      chart_full=COUNT, chart_dnr=COUNT, chart_cmo=COUNT,
                                      poe_full=COUNT, poe_dnr=COUNT, poe_cmo=COUNT),
        "cohort": table(VERSIONS, record(
            flow=table(("raw", "merged", "age", "icu_death", "bad_time"), COUNT),
            n_patients=COUNT, n_hadm=COUNT, n_samples=COUNT,
            cs_at_out=table(("full", "dnr", "cmo"), COUNT),
            cs_detail=table(("full", "none", "dnr", "cmo"), COUNT),
            merged_gaps=table(("ward", "procedure", "icu_or_direct", "unknown", "direct_0h"), COUNT),
            dc_rule_diverge=table(("diverge", "diverge_cmo", "reentered_24h", "stored_after_out_lst", "stored_after_out_d2_ward_dnr"), COUNT),
            defs=table(DEFS, DEF_SUMMARY),
        )),
    },
    # 03_features: 최종 코호트, 변수별 결측률·평균·SD
    "features": {
        "version": enum(*VERSIONS),
        "n_before": COUNT, "excluded_no_data": COUNT, "n_final": COUNT,
        "split_n": table(SPLITS, COUNT),
        "cs_at_out": table(("full", "dnr", "cmo"), COUNT),
        "defs": table(DEFS, DEF_SUMMARY),
        "missing": table(FEATURES, RATE),
        "mean": table(FEATURES + STATIC, FLOAT),
        "sd": table(FEATURES + STATIC, FLOAT),
        "male": RATE,
        "swift_inputs": table(("fio2_missing", "paco2_missing", "from_ed"), RATE),
        "leak_check": table(("chart_in_window", "chart_store_after_out", "lab_in_window", "lab_store_after_out"), COUNT),
    },
    # 04_train: 과제·모델별 test 성능
    "train": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "params": enum("default", "tuned"),
        "split": enum("temporal", "random"),
        "n_test": table(TASKS, COUNT),
        "n_test_events": table(TASKS, COUNT),
        "auroc": table(MODELS, table(TASKS, RATE)),
        "auroc_lo": table(MODELS, table(TASKS, RATE)),
        "auroc_hi": table(MODELS, table(TASKS, RATE)),
        "ece": table(MODELS, table(TASKS, RATE)),
    },
    # 05_diag: 병동 사망 라벨 진단 + 혈당 이상치
    "diag": {
        "cohort": table(VERSIONS, record(**{
            f"{e}_{k}": table(("early", "late"), table(keys, COUNT))
            for e in ("ward_death", "readmission")
            for k, keys in (("hours", ("<1h", "1-6h", "6-24h", "24-48h", ">48h")),
                            ("code_status", ("any_record", "no_record")),
                            ("discharge", ("died", "hospice", "home", "facility", "other", "missing")))},
            all_code_status=table(("any_record", "no_record"), COUNT))),
        "glucose": record(n_values=COUNT, over_1000=COUNT, over_2000=COUNT, over_5000=COUNT, samples_over_cap=COUNT,
                          mean_raw=FLOAT, sd_raw=FLOAT, mean_capped=FLOAT, sd_capped=FLOAT,
                          p50=FLOAT, p99=FLOAT, cap=META_INT_BIG),
    },
    # 06_shap: LightGBM TreeSHAP 변수 중요도 비율 (30개 변수 단위)
    "shap": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "n_explained": COUNT,
        "share": table(TASKS, table(FEATURES + ("age", "gender", "icu_los", "adm_to_icu"), RATE)),
    },
    # 04 tune 모드: holdout AUROC 만 (test 미사용)
    "tune": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "n_configs": table(("lgbm", "lstm"), META_INT),
        "holdout": table(TASKS, table(("lgbm_best", "lgbm_default", "lstm_best", "lstm_default"), RATE)),
        "chosen": table(TASKS, record(num_leaves=META_INT, min_child_samples=META_INT, lgbm_lr=RATE, spw=enum("ratio", "sqrt", "none"),
                                      d=META_INT, p=RATE, lstm_lr=RATE)),
    },
    # 08_calib: iREAD 보정 곡선(10분위) + 하위집단 성능
    "calib": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "curve": table(TASKS, table(tuple(f"d{i}" for i in range(1, 11)), record(mean_pred=RATE, obs=RATE, n=COUNT))),
        "subgroup": table(TASKS, table(("<65", "65-79", ">=80", "<2d", "2-7d", ">7d", "full", "none", "dnr", "cmo"),
                                       record(n=COUNT, events=COUNT, auroc=RATE))),
        "by_event_type": table(TASKS, table(EVENT_TYPES, record(events=COUNT, auroc=RATE))),
        "alert": table(TASKS, record(
            holdout_thr=record(flagged=COUNT, **{k: COUNT for k in ("readmission", "ward_death", "ward_dnr", "later_event", "none")},
                               **{f"capture_{k}": RATE for k in EVENT_TYPES}),
            top10=record(flagged=COUNT, **{k: COUNT for k in ("readmission", "ward_death", "ward_dnr", "later_event", "none")},
                         **{f"capture_{k}": RATE for k in EVENT_TYPES}),
            totals=table(("readmission", "ward_death", "ward_dnr", "later_event", "none"), COUNT))),
    },
    # 09_w6: 재입실 경로·분할 통계·사건별 지표 CI (재학습 없음)
    "w6": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "split_stats": table(("train", "holdout", "test", "other"), record(
            subjects=COUNT, samples=COUNT, early_readmission=COUNT, early_ward_death=COUNT,
            late_readmission=COUNT, late_ward_death=COUNT)),
        "w4_overlap": record(rand_test_subjects=COUNT, overlap_subjects=COUNT, overlap_samples=COUNT),
        "readmission_paths": table(("early", "late"), table(("ward_only", "via_procedure", "unclear"), COUNT)),
        "test_ci": table(TASKS, table(("ppv", "capture_readmission", "capture_ward_death", "auroc_readmission",
                                       "auroc_ward_death", "auroc_readm_ward_only", "auroc_readm_via_procedure",
                                       "auroc_readm_unclear"), record(est=RATE, lo=RATE, hi=RATE))),
        "test_readm_by_path": table(TASKS, table(("ward_only", "via_procedure", "unclear"), COUNT)),
    },
    # 11_disposition: ICU 퇴실 직후 위치(직접 퇴원 등) 집계 — 재학습·라벨 변경 없음
    "disposition": {
        "cohorts": table(VERSIONS, table(("all", "d1"), record(
            n=COUNT, test_n=COUNT,
            next=table(DISPO_CATS, COUNT), test_next=table(DISPO_CATS, COUNT),
            early_events=table(DISPO_CATS, COUNT), test_early_events=table(DISPO_CATS, COUNT),
            direct_discharge_location=table(DISCH_LOCS, COUNT),
            direct_discharge_gap_hours=record(p25=FLOAT, p50=FLOAT, p75=FLOAT),
            proxy_discharged_within_1h=COUNT, proxy_and_direct=COUNT))),
    },
    # 12_dd_sensitivity: 바로 퇴원 제외 민감도 — 저장된 test 예측 재집계 (재학습 없음)
    "ddsens": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "counts": table(TASKS, table(("all", "excl_dd"), record(
            n=COUNT, events=COUNT, readmission=COUNT, ward_death=COUNT, alerts=COUNT,
            direct_discharge=COUNT, direct_discharge_alerted=COUNT))),
        "metrics": table(TASKS, table(("all", "excl_dd"), table(
            ("auroc_composite", "ppv", "capture_readmission", "capture_ward_death", "auroc_readmission", "auroc_ward_death"),
            record(est=RATE, lo=RATE, hi=RATE)))),
    },
    # 13_table1: 조원 공유용 기술통계 (대표 분석 대상 B·①) — 집계만
    "table1": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "groups": table(("all", "train", "holdout", "test", "early_readm", "early_death", "no_early_event"), record(
            n_samples=COUNT, n_patients=COUNT, n_admissions=COUNT,
            age=T1_Q, age_group=table(("18-44", "45-64", "65-79", "80+"), COUNT), male=COUNT,
            first_unit=table(("micu_sicu", "cvicu", "tsicu", "neuro", "ccu", "sicu", "micu", "other", "missing"), COUNT),
            icu_los_days=T1_Q, adm_to_icu_days=T1_Q, from_ed=COUNT,
            code_status=table(("full", "none", "dnr", "cmo"), COUNT),
            next_place=table(DISPO_CATS, COUNT),
            discharge_location=table(DISCH_LOCS, COUNT),
            died_in_hospital=COUNT,
            events=table(("early_readm", "early_death", "late_readm", "late_death"), COUNT),
            last_value=table(FEATURES, T1_Q),
            missing_rate=table(FEATURES, RATE))),
    },
    # 14_short_readm: 짧은 간격 재입실의 성격 집계 (B·①, 재학습 없음)
    "shortreadm": {
        "groups": table(("lt1h", "h1_6", "h6_48"), record(
            n=COUNT,
            gap_minutes=table(("0-5", "5-15", "15-30", "30-60", "60-360", "360-2880"), COUNT),
            gap_unit_any=table(("icu_name", "procedure", "emergency", "intermediate_stepdown", "ward_other", "blank"), COUNT),
            gap_only_icu_or_blank=COUNT, gap_no_rows=COUNT,
            gap_rows=table(("0", "1", "2", "3+"), COUNT),
            same_unit=COUNT, unit_known=COUNT,
            next_icu_los_hours=table(("<4", "4-12", "12-24", "24-72", "72+"), COUNT))),
    },
    # 10_direct_readm (W6-d): 재입실 직접 모델 vs 복합 점수, 같은 집단·같은 경고 수
    "direct": {
        "version": enum(*VERSIONS),
        "outcome": enum(*OUTCOMES),
        "counts": table(TASKS, record(test=COUNT, alert_k=COUNT, readmission=COUNT, ward_death=COUNT,
                                      train_readm=COUNT, holdout_readm=COUNT)),
        "results": table(TASKS, table(("composite", "direct_lgbm", "direct_lr"), table(
            ("auroc_readm_vs_all", "auroc_readm_vs_nonevent", "auroc_death_vs_nonevent",
             "capture_readm_topk", "capture_death_topk", "readm_share_topk"), record(est=RATE, lo=RATE, hi=RATE)))),
    },
}
