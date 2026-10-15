"""내부 열람용 둘러보기(browse_gen.py) 시험용 합성 MIMIC 형식 데이터 (실제 자료 아님).

이슈 ad40b868 의 사례 유형을 모두 포함하도록 손으로 만든 입원들.
  S1 병동 전실 후 사건 없음      S2 ≤24h 병동 경유 재입실     S3 24–48h 병동 경유 재입실
  S4a ICU→회복실→ICU (B 병합)    S4b ICU→병동→수술실→ICU      S5 재입실 전 병동 사망(+퇴실 후 DNR)
  S6a 간격에 이동 기록 없음        S6b code status 출처 충돌·D/C·퇴실 후 저장   S7 퇴실 시점 CMO
  S8 ICU에서 바로 퇴원(시설)       S9 ICU 퇴실 30분 뒤 퇴원(자택)
각 유형 3개(시각을 조금씩 바꿈) + 분할용 test 연도군 환자.
측정값: 반복 측정, 범위 밖 값, 퇴실 후 저장 값, 체류 밖 값을 일부러 넣는다.
실행: python synth_browse.py <원자료 형식 출력 폴더> <WORK 출력 폴더>
"""
import sys
from pathlib import Path

import pandas as pd

D, W = Path(sys.argv[1]), Path(sys.argv[2])
T0 = pd.Timestamp("2150-03-01")
H = lambda h: T0 + pd.Timedelta(hours=h) if h is not None else pd.NaT
pat, adm, icu, xfer, chart, lab, poe, poed = [], [], [], [], [], [], [], []
ids = {"sid": 10000, "hadm": 20000, "stay": 30000, "poe": 0}
MICU = "Medical Intensive Care Unit (MICU)"


def new(kind):
    ids[kind] += 1
    return ids[kind]


def vitals(stay, a, b, k):
    """체류 [a, b] 동안 2시간 간격 hr/sbp/rr. k 로 값을 조금씩 바꾼다."""
    t = a + 0.5
    while t < b:
        chart.append((stay, 220045, H(t), H(t + 0.1), str(80 + k), 80.0 + k))
        chart.append((stay, 220179, H(t), H(t + 0.1), str(118 - k), 118.0 - k))
        chart.append((stay, 220210, H(t), H(t + 0.1), str(18 + k % 3), 18.0 + k % 3))
        t += 2
    chart.append((stay, 220045, H(b - 0.3), H(b - 0.2), "82", 82.0))          # 같은 시간대 반복 측정
    chart.append((stay, 220045, H(b - 3), H(b - 3), "250", 250.0))           # 범위 밖 (10–190) → 제외
    chart.append((stay, 220179, H(b - 0.2), H(b + 0.5), "95", 95.0))          # 퇴실 전 측정·퇴실 후 저장 → 제외
    chart.append((stay, 220210, H(a - 1), H(a - 1), "22", 22.0))             # 체류 밖 → 제외
    chart.append((stay, 220045, H(b - 1), H(b - 1), None, None))             # 값 없음 → 제외


def admission(stays, xfers, disch, death=None, loc="HOME", group="2008 - 2010", k=0):
    """stays: [(in_h, out_h, careunit)], xfers: [(careunit, in_h, out_h)] — 이동 기록은 그대로 기록한다."""
    sid, hadm = new("sid"), new("hadm")
    pat.append((sid, "F" if k % 2 else "M", 60 + k, 2150, group))
    adm.append((sid, hadm, H(-12), H(disch), H(death), "DIED" if death is not None else loc))
    xfer.append((sid, hadm, "ED", "Emergency Department", H(-12), H(stays[0][0])))
    stay_ids = []
    for a, b, cu in stays:
        st = new("stay"); stay_ids.append(st)
        icu.append((sid, hadm, st, cu, cu, H(a), H(b)))
        vitals(st, a, b, k)
        lab.append((sid, hadm, 50983, H(a + 1), H(a + 1.5), 138.0))
        lab.append((sid, hadm, 51006, H(a + 1), H(a + 2), 20.0 + k))
    for cu, a, b in xfers:
        xfer.append((sid, hadm, "admit" if cu is None else "transfer", cu, H(a), H(b)))
    xfer.append((sid, hadm, "discharge", None, H(disch), pd.NaT))
    return sid, hadm, stay_ids


def cs_chart(stay, t, v, stored=None):
    chart.append((stay, 223758, H(t), H(stored if stored is not None else t), v, None))


def poe_order(sid, hadm, t, value, ttype="New", disc_of=None):
    n = new("poe"); pid = f"{sid}-{n}"
    poe.append((pid, n, sid, hadm, H(t), "General Care", "Code status", ttype, disc_of, None, None, "Active"))
    if value is not None:
        poed.append((pid, n, sid, "Code status", value))
    return pid


for k in range(3):
    j = k * 0.5
    # S1 병동 전실 후 사건 없음 (퇴원)
    sid, hadm, s = admission([(0, 48 + j, MICU)], [(MICU, 0, 48 + j), ("Medicine", 48 + j, 150)], 150, k=k)
    cs_chart(s[0], 2, "Full code")
    # S2 ≤24h 병동 경유 재입실
    sid, hadm, s = admission([(0, 48, MICU), (60 + j, 100, MICU)],
                             [(MICU, 0, 48), ("Med/Surg", 48, 60 + j), (MICU, 60 + j, 100), ("Med/Surg", 100, 200)], 200, k=k)
    # S3 24–48h 병동 경유 재입실
    sid, hadm, s = admission([(0, 48, "Coronary Care Unit (CCU)"), (84 + j, 120, "Coronary Care Unit (CCU)")],
                             [("Coronary Care Unit (CCU)", 0, 48), ("Cardiology", 48, 84 + j),
                              ("Coronary Care Unit (CCU)", 84 + j, 120), ("Cardiology", 120, 220)], 220, k=k)
    poe_order(sid, hadm, 1, "Full code")
    # S4a ICU→회복실→ICU (간격 전체가 시술 유닛 → B 도 병합)
    sid, hadm, s = admission([(0, 48, "Surgical Intensive Care Unit (SICU)"), (52 + j, 100, "Surgical Intensive Care Unit (SICU)")],
                             [("Surgical Intensive Care Unit (SICU)", 0, 48), ("PACU", 48, 52 + j),
                              ("Surgical Intensive Care Unit (SICU)", 52 + j, 100), ("Surgery", 100, 160)], 160, k=k)
    # S4b ICU→병동→수술실→ICU (병동을 거쳤으므로 B 는 병합하지 않음, 경로 = 시술 경유)
    sid, hadm, s = admission([(0, 48, "Surgical Intensive Care Unit (SICU)"), (66 + j, 110, "Surgical Intensive Care Unit (SICU)")],
                             [("Surgical Intensive Care Unit (SICU)", 0, 48), ("Surgery", 48, 62), ("Operating Room", 62, 66 + j),
                              ("Surgical Intensive Care Unit (SICU)", 66 + j, 110), ("Surgery", 110, 170)], 170, k=k)
    # S5 재입실 전 병동 사망 (퇴실 후 DNR 오더 → d2 에서는 병동 DNR 이 먼저)
    sid, hadm, s = admission([(0, 48, MICU)], [(MICU, 0, 48), ("Medicine", 48, 70 + j)], 70 + j, death=70 + j, k=k)
    cs_chart(s[0], 3, "Full code")
    poe_order(sid, hadm, 60, "DNR/DNI")
    # S6a 간격에 이동 기록 없음 (2h 공백 → A 병합, B 는 1h 초과라 병합 안 함, 경로 불명확)
    sid, hadm, s = admission([(0, 48, MICU), (50 + j, 90, MICU)], [(MICU, 0, 48), (MICU, 50 + j, 90), ("Medicine", 90, 150)], 150, k=k)
    # S6b code status 출처 충돌: chart DNR → POE Full(나중) → POE D/C → 퇴실 전 측정·퇴실 후 저장된 chart DNR
    sid, hadm, s = admission([(0, 48, MICU)], [(MICU, 0, 48), ("Medicine", 48, 130)], 130, k=k)
    cs_chart(s[0], 10, "DNR / DNI")
    p = poe_order(sid, hadm, 20, "Full code")
    poe_order(sid, hadm, 30, None, ttype="D/C", disc_of=p)
    cs_chart(s[0], 47.5, "DNR (do not resuscitate)", stored=49)
    # S7 퇴실 시점 CMO (d1·d2 제외)
    sid, hadm, s = admission([(0, 48, MICU)], [(MICU, 0, 48), ("Medicine", 48, 90)], 90, death=90, k=k)
    cs_chart(s[0], 40, "Comfort measures only")

    # S8 ICU에서 바로 병원 퇴원 (요양시설) / S9 ICU 퇴실 30분 뒤 퇴원 (자택)
    admission([(0, 48, MICU)], [(MICU, 0, 48)], 48, loc="SKILLED NURSING FACILITY", k=k)
    admission([(0, 48, MICU)], [(MICU, 0, 48)], 48.5, loc="HOME", k=k)

for g in ["2017 - 2019", "2020 - 2022"]:      # 분할용
    for k in range(3):
        admission([(0, 48, MICU)], [(MICU, 0, 48), ("Medicine", 48, 150)], 150, group=g, k=k)

(D / "hosp").mkdir(parents=True, exist_ok=True); (D / "icu").mkdir(parents=True, exist_ok=True); W.mkdir(parents=True, exist_ok=True)
pd.DataFrame(pat, columns=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]).assign(dod=None).to_csv(D / "hosp/patients.csv.gz", index=False)
pd.DataFrame(adm, columns=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "discharge_location"]).to_csv(D / "hosp/admissions.csv.gz", index=False)
pd.DataFrame(icu, columns=["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime"]).to_csv(D / "icu/icustays.csv.gz", index=False)
pd.DataFrame(xfer, columns=["subject_id", "hadm_id", "eventtype", "careunit", "intime", "outtime"]).to_csv(D / "hosp/transfers.csv.gz", index=False)
pd.DataFrame(poe, columns=["poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime", "order_type", "order_subtype", "transaction_type",
                           "discontinue_of_poe_id", "discontinued_by_poe_id", "order_provider_id", "order_status"]).to_csv(D / "hosp/poe.csv.gz", index=False)
pd.DataFrame(poed, columns=["poe_id", "poe_seq", "subject_id", "field_name", "field_value"]).to_csv(D / "hosp/poe_detail.csv.gz", index=False)
pd.DataFrame([(220045, "Heart Rate", "bpm"), (220179, "Non Invasive Blood Pressure systolic", "mmHg"),
              (220210, "Respiratory Rate", "insp/min"), (223758, "Code Status", None)],
             columns=["itemid", "label", "unitname"]).to_csv(D / "icu/d_items.csv.gz", index=False)
pd.DataFrame(chart, columns=["stay_id", "itemid", "charttime", "storetime", "value", "valuenum"]).to_parquet(W / "chart_raw.parquet")
pd.DataFrame(lab, columns=["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum"]).to_parquet(W / "lab_raw.parquet")
print("synthetic ok:", len(adm), "admissions,", len(icu), "stays")
