"""조원 공유용 기술통계 문서 생성. 입력: results/aggregates_W8table1.jsonl (안전 래퍼 통과 집계) — 숫자는 이 파일에서만 가져온다.
실행: .venv/bin/python iread/make_table1_doc.py  → results/DESC_STATS_B1.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads(open(HERE / "results" / "aggregates_W8table1.jsonl").readline())
assert d["ok"], "집계 실패본"
G = d["result"]["groups"]
SPLIT = [("all", "전체"), ("train", "학습용"), ("holdout", "설정 선택용"), ("test", "평가용")]
OUTC = [("early_readm", "48시간 내 ICU 재입실"), ("early_death", "48시간 내 재입실 전 사망"), ("no_early_event", "48시간 사건 없음")]


def n(v):
    return v if isinstance(v, str) else f"{v:,}"


def pct(v, tot):
    if isinstance(v, str):
        return v
    return f"{v:,} ({100 * v / tot:.1f})" if tot else "0"


def med(qd, dec=1):
    return f"{qd['p50']:.{dec}f} [{qd['p25']:.{dec}f}–{qd['p75']:.{dec}f}]"


def table(cols, rows):
    head = "| 항목 | " + " | ".join(f"{lab}<br>(n={n(G[k]['n_samples'])})" for k, lab in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    body = []
    for lab, fn in rows:
        if fn is None:
            body.append(f"| **{lab}** |" + " |" * len(cols))
        else:
            body.append(f"| {lab} | " + " | ".join(fn(G[k]) for k, _ in cols) + " |")
    return "\n".join([head, sep] + body)


T = lambda g: g["n_samples"]
rows = [
    ("고유 환자 수", lambda g: n(g["n_patients"])),
    ("입원 수", lambda g: n(g["n_admissions"])),
    ("나이, 세 — 중앙값 [IQR]", lambda g: med(g["age"], 0)),
    ("나이, 세 — 평균 (SD)", lambda g: f"{g['age']['mean']:.1f} ({g['age']['sd']:.1f})"),
    ("나이군", None),
    *[(f"&emsp;{k}세", (lambda k: lambda g: pct(g["age_group"][k], T(g)))(k)) for k in ("18-44", "45-64", "65-79", "80+")],
    ("남성", lambda g: pct(g["male"], T(g))),
    ("첫 ICU 유형", None),
    *[(f"&emsp;{lab}", (lambda k: lambda g: pct(g["first_unit"][k], T(g)))(k)) for k, lab in
      [("micu", "내과 ICU(MICU)"), ("sicu", "외과 ICU(SICU)"), ("micu_sicu", "내·외과 ICU(MICU/SICU)"), ("cvicu", "심혈관 ICU(CVICU)"),
       ("ccu", "심장 ICU(CCU)"), ("tsicu", "외상 ICU(TSICU)"), ("neuro", "신경계 ICU·중간치료"), ("other", "기타")]],
    ("ICU 재실 기간, 일 — 중앙값 [IQR]", lambda g: med(g["icu_los_days"])),
    ("입원 → ICU 입실, 일 — 중앙값 [IQR]", lambda g: med(g["adm_to_icu_days"])),
    ("ICU 입실 직전 응급실", lambda g: pct(g["from_ed"], T(g))),
    ("퇴실 시점 치료목표(code status)", None),
    ("&emsp;명시적 full code", lambda g: pct(g["code_status"]["full"], T(g))),
    ("&emsp;기록 없음", lambda g: pct(g["code_status"]["none"], T(g))),
    ("&emsp;DNR·DNI", lambda g: pct(g["code_status"]["dnr"], T(g))),
    ("ICU 퇴실 직후 위치", None),
    ("&emsp;병동(이름 규칙상)", lambda g: pct(g["next_place"]["ward"], T(g))),
    ("&emsp;곧바로 병원 퇴원", lambda g: pct(g["next_place"]["discharge"], T(g))),
    ("&emsp;수술실·회복실·시술", lambda g: pct(g["next_place"]["procedure"], T(g))),
    ("&emsp;ICU 이름 유닛", lambda g: pct(g["next_place"]["icu_name"], T(g))),
    ("병원 퇴원 장소", None),
    *[(f"&emsp;{lab}", (lambda k: lambda g: pct(g["discharge_location"][k], T(g)))(k)) for k, lab in
      [("home", "자택"), ("home_health", "자택(방문간호)"), ("facility", "요양·재활 등 시설"), ("acute_hospital", "다른 급성기 병원"),
       ("hospice", "호스피스"), ("died", "사망"), ("against_advice", "권고에 반한 퇴원"), ("other", "기타"), ("missing", "기록 없음")]],
    ("원내 사망(같은 입원)", lambda g: pct(g["died_in_hospital"], T(g))),
    ("사건 (정의 ①)", None),
    ("&emsp;48시간 내 ICU 재입실", lambda g: pct(g["events"]["early_readm"], T(g))),
    ("&emsp;48시간 내 재입실 전 사망", lambda g: pct(g["events"]["early_death"], T(g))),
    ("&emsp;48시간 이후 ICU 재입실", lambda g: pct(g["events"]["late_readm"], T(g))),
    ("&emsp;48시간 이후 재입실 전 사망", lambda g: pct(g["events"]["late_death"], T(g))),
]
VARS = [("hr", "심박수", "회/분", 0), ("rr", "호흡수", "회/분", 0), ("sbp", "수축기 혈압", "mmHg", 0), ("dbp", "이완기 혈압", "mmHg", 0),
        ("temp", "체온", "°C", 1), ("spo2", "산소포화도", "%", 0), ("gcs_eye", "GCS 눈뜨기", "점", 0), ("gcs_verbal", "GCS 언어", "점", 0),
        ("gcs_motor", "GCS 운동", "점", 0), ("bun", "BUN", "mg/dL", 0), ("creatinine", "크레아티닌", "mg/dL", 2), ("sodium", "나트륨", "mEq/L", 0),
        ("potassium", "칼륨", "mEq/L", 1), ("hco3", "중탄산", "mEq/L", 0), ("glucose", "혈당", "mg/dL", 0), ("calcium", "칼슘", "mg/dL", 1),
        ("ion_calcium", "이온화칼슘", "mmol/L", 2), ("wbc", "백혈구", "K/µL", 1), ("platelet", "혈소판", "K/µL", 0), ("lymphocyte", "림프구", "%", 1),
        ("inr", "INR", "", 2), ("albumin", "알부민", "g/dL", 1), ("bilirubin", "빌리루빈", "mg/dL", 1), ("lactate", "젖산", "mmol/L", 1),
        ("ph", "pH", "", 2), ("po2", "pO2", "mmHg", 0)]
vrows = [(f"{lab}{' (' + u + ')' if u else ''}", (lambda f, dec: lambda g: f"{med(g['last_value'][f], dec)} · {100 * g['missing_rate'][f]:.0f}%")(f, dec))
         for f, lab, u, dec in VARS]

doc = f"""# 분석 대상 기술통계 (조원 공유용)

**대상**: MIMIC-IV v3.1, 대표 분석 대상 — 코호트 B·정의 ① (성인 ICU 퇴실 사례, 퇴실 시점 CMO 제외)

## 읽는 법
- **분석 단위는 "ICU 퇴실 사례"입니다.** 한 환자가 여러 번 들어갈 수 있어서 사례 수가 환자 수보다 많습니다.
- **값 표기**: 범주는 건수 (%)이고, %의 분모는 각 열의 사례 수입니다. 연속값은 중앙값 [사분위 범위]입니다.
- **소표본 표시**: 1–9건인 칸은 `<10`으로 가렸습니다.
- **분할**: 학습용·설정 선택용은 2008–2016년 시기군, 평가용은 2017–2022년 시기군입니다. 같은 환자는 한 곳에만 들어갑니다.
- **⚠️ 병동 전실은 확인되지 않았습니다.** 실제 일반병동 전실이 확인된 집단이 아닙니다. "ICU 퇴실 직후 위치"의 "병동"도 유닛 이름이 ICU·시술 이름이 아니라는 뜻일 뿐입니다.
- **출처**: 연구실 PC에서 원자료를 안전 래퍼로 집계했습니다(재학습 없음).
  - 집계 원본: `results/aggregates_W8table1.jsonl`
  - 코드 해시: `evidence/manifest_W8table1.txt`
  - 이 문서는 집계 파일에서 자동으로 만들었습니다: `make_table1_doc.py`

## 표 1. 분할별 특성

{table(SPLIT, rows)}

## 표 2. 48시간 결과별 특성 (전체 83,792건)

"48시간 사건 없음"에는 48시간 이후 재입실·사망과 ICU에서 곧바로 퇴원한 사례가 포함됩니다.

{table(OUTC, rows)}

## 표 3. 퇴실 전 마지막 측정값 — 중앙값 [IQR] · 결측률

**읽는 법**
- 각 변수에서 퇴실 전 ICU 체류 중 마지막으로 측정된 값입니다. 모델 입력 규칙과 같습니다.
  - 퇴실 후에 결과가 기록된 값과 허용 범위 밖의 값은 뺐습니다.
  - 혈당은 2,000 mg/dL을 상한으로 두었습니다.
- 결측률은 해당 체류 중 그 변수를 한 번도 재지 않은 사례의 비율입니다.
- 단위는 MIMIC-IV 해당 항목의 일반 단위입니다. 항목별 단위 열은 따로 대조하지 않았습니다.

### 표 3-1. 분할별

{table(SPLIT, vrows)}

### 표 3-2. 48시간 결과별

{table(OUTC, vrows)}

## 눈에 띄는 점 (기술적 관찰, 해석은 팀 논의 필요)
- **학습용(2008–2016)과 평가용(2017–2022) 사이에 대상 구성 차이가 있습니다.**
  - 첫 ICU가 신경계 ICU·중간치료 유닛인 비율: {{neuro_dev}} → {{neuro_test}}
  - 퇴실 시점 치료목표 "기록 없음": {{none_dev}} → {{none_test}}
  - ICU에서 곧바로 병원 퇴원: {{dd_dev}} → {{dd_test}}
  - 기록 체계나 유닛 운영이 바뀐 영향일 수 있지만, 원인은 확인하지 않았습니다. 평가 성능을 해석할 때 이 차이를 염두에 둬야 합니다.
- **"신경계 ICU·중간치료"에는 MIMIC-IV가 ICU 체류로 기록한 신경계 중간치료(stepdown·intermediate) 유닛이 섞여 있을 수 있습니다.** 이 집계로는 둘을 나누지 않았습니다.
- **48시간 내 재입실 전 사망 사례는 퇴실 직전 상태가 뚜렷이 나쁩니다**(표 3-2). 산소포화도와 GCS가 낮고 심박수가 높습니다. 반면 재입실 사례는 사건 없음과 차이가 작습니다. 모델이 사망은 잘 구별하고 재입실은 덜 구별하는 결과(보고서 5장)와 같은 방향입니다.
- **사망 집계의 차이**: 병원 퇴원 장소가 "사망"인 건수(표 1)와 "원내 사망"(사망 시각 기록 기준)은 조금 다릅니다. 기록 출처가 달라서입니다.

## 참고
- **치료목표 정의**
  - "명시적 full code": 기록상 full code로 지정된 경우
  - "기록 없음": 퇴실 시각까지 지정 기록이 없는 경우. full code로 확정한 것이 아닙니다.
  - DNR·DNI: 심폐소생술 또는 삽관을 하지 않기로 한 결정. 코드에서는 한 범주로 묶습니다.
- **퇴원 장소**: MIMIC-IV `discharge_location` 값을 9개 범주로 묶었습니다.
- **첫 ICU 유형**: 병합된 체류의 첫 ICU를 유닛 이름 규칙으로 묶었습니다.
"""
P = lambda g, sec, k: f"{100 * G[g][sec][k] / G[g]['n_samples']:.1f}%"
dev_pct = lambda sec, k: f"{100 * (G['train'][sec][k] + G['holdout'][sec][k]) / (G['train']['n_samples'] + G['holdout']['n_samples']):.1f}%"
doc = doc.format(neuro_dev=dev_pct("first_unit", "neuro"), neuro_test=P("test", "first_unit", "neuro"),
                 none_dev=dev_pct("code_status", "none"), none_test=P("test", "code_status", "none"),
                 dd_dev=dev_pct("next_place", "discharge"), dd_test=P("test", "next_place", "discharge"))
(HERE / "results" / "DESC_STATS_B1.md").write_text(doc)
print("written", len(doc))
