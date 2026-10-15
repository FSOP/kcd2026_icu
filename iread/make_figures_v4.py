"""조원용 보고서 v4 그림 (한글, 분모 표기). 입력은 안전 래퍼를 통과한 집계(results/aggregates_*.jsonl)뿐 — 새 계산 없음.
대표 분석 = 코호트 B · 정의 ① · test 23,542건 · 퇴실 후 48시간.
실행: .venv/bin/python iread/make_figures_v4.py   → iread/results/v4/fig*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v4"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
rows = [json.loads(l) for l in open(HERE / "results" / "aggregates_W3_W4.jsonl")]
w6 = [json.loads(l) for l in open(HERE / "results" / "aggregates_W6.jsonl")]


def get(step, version, outcome):
    for r in rows:
        x = r["result"]
        if r["step"] == step and x.get("version") == version and x.get("outcome") == outcome and x.get("split", "temporal") == "temporal":
            return x
    raise KeyError((step, version, outcome))


cal = get("calib", "B", "d1")
al = cal["alert"]["early"]["holdout_thr"]; tot = cal["alert"]["early"]["totals"]
ci = next(r["result"] for r in w6 if r["result"]["version"] == "B")["test_ci"]

# 1) 대표 결과: 실제 사건별 포착 + 경고 구성
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
names = ["48시간 이내\nICU 재입실", "재입실 전 48시간\n이내 사망"]
caught = [al["readmission"], al["ward_death"]]; alln = [tot["readmission"], tot["ward_death"]]
a1.bar(names, alln, color="#d0d7e2", label="실제 발생")
a1.bar(names, caught, color=["#2f6db5", "#c0392b"], label="경고로 포착")
for i, (c, n) in enumerate(zip(caught, alln)):
    a1.text(i, n + 15, f"{c:,} / {n:,}\n({c / n:.0%})", ha="center", fontsize=9)
a1.set_ylabel("건수 (퇴실 사례)"); a1.set_ylim(0, max(alln) * 1.3); a1.legend(frameon=False, fontsize=8, loc="upper right")
a1.set_title("(가) 실제 사건 중 경고가 울린 비율")
parts = [("재입실", al["readmission"], "#2f6db5"), ("재입실 전 사망", al["ward_death"], "#c0392b"),
         ("48시간 이후 사건", al["later_event"], "#f0ad4e"), ("퇴원까지 사건 없음", al["none"], "#bbbbbb")]
left = 0
for lab, n, col in parts:
    a2.barh([0], [n], left=left, color=col, label=f"{lab} {n:,}")
    left += n
a2.set_yticks([]); a2.set_xlabel(f"경고 {al['flagged']:,}건의 구성"); a2.set_xlim(0, al["flagged"])
a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
a2.set_title("(나) 경고가 울린 사례에서 실제로 일어난 일")
fig.suptitle("그림 1. 대표 분석: 퇴실 후 48시간, test 23,542건, 경고 3,059건 (holdout 기준 고정 임계값)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig1_alert.png", dpi=200); plt.close(fig)

# 2) 같은 점수의 사건별 AUROC (95% CI)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
items = [("early", "auroc_readmission", "≤48시간 재입실"), ("early", "auroc_ward_death", "≤48시간 재입실 전 사망"),
         ("late", "auroc_readmission", ">48시간 재입실"), ("late", "auroc_ward_death", ">48시간 재입실 전 사망")]
for i, (t, k, lab) in enumerate(items):
    e = ci[t][k]
    ax.errorbar([e["est"]], [i], xerr=[[e["est"] - e["lo"]], [e["hi"] - e["est"]]], fmt="o",
                color="#c0392b" if "사망" in lab else "#2f6db5", capsize=3)
    ax.text(e["hi"] + 0.01, i, f"{e['est']:.3f} ({e['lo']:.3f}–{e['hi']:.3f})", va="center", fontsize=8)
ax.axvline(0.5, color="grey", lw=.8, ls=":"); ax.set_xlim(0.45, 1.08)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[2] for x in items]); ax.invert_yaxis()
ax.set_xlabel("AUROC (해당 사건 vs 사건 없는 사례, 95% CI) — 0.5 = 무작위")
ax.set_title("그림 2. 같은 복합 점수가 사건 종류별로 얼마나 구별하는가 (코호트 B·정의 ①·test)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig2_event_auroc.png", dpi=200); plt.close(fig)

# 3) 모델 비교 (≤48시간 복합결과, 95% CI)
r = get("train", "B", "d1")
MODELS = [("iREAD", "iREAD 방식(결합)"), ("LSTM only", "시계열 모델만"), ("LightGBM only", "요약표 모델만"),
          ("RF", "랜덤포레스트"), ("LR", "로지스틱 회귀"), ("Cox", "Cox"), ("NEWS", "NEWS"), ("MEWS", "MEWS"), ("SWIFT", "SWIFT")]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for i, (m, lab) in enumerate(MODELS):
    y, lo, hi = r["auroc"][m]["early"], r["auroc_lo"][m]["early"], r["auroc_hi"][m]["early"]
    ax.errorbar([y], [i], xerr=[[y - lo], [hi - y]], fmt="o", capsize=3, color="#555" if m in ("NEWS", "MEWS", "SWIFT") else "#2f6db5")
    ax.text(hi + 0.005, i, f"{y:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([x[1] for x in MODELS]); ax.invert_yaxis()
ax.set_xlabel("AUROC, ≤48시간 복합결과(재입실 또는 재입실 전 사망), 95% CI")
ax.set_title("그림 3. 모델별 비교 (코호트 B·정의 ①·test 23,542건, 사건 1,279건)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig3_models.png", dpi=200); plt.close(fig)

# 4) 보정 (≤48시간, 10분위)
cv = cal["curve"]["early"]
xs = [cv[f"d{i}"]["mean_pred"] for i in range(1, 11)]; ys = [cv[f"d{i}"]["obs"] for i in range(1, 11)]
ns = [cv[f"d{i}"]["n"] for i in range(1, 11)]
fig, ax = plt.subplots(figsize=(5.2, 4))
m = max(xs + ys) * 1.1
ax.plot([0, m], [0, m], color="grey", lw=.8, ls=":", label="완벽한 보정")
ax.plot(xs, ys, "o-", ms=4, label="iREAD 방식(재보정 후)")
ax.set_xlabel("예측 확률 평균 (10분위, 분위당 약 2,350건)"); ax.set_ylabel("실제 사건 비율")
ax.legend(frameon=False, fontsize=8)
ax.set_title("그림 4. 예측 확률과 실제 비율 (≤48시간 복합결과)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig4_calibration.png", dpi=200); plt.close(fig)

# 5) 주요 변수 (LightGBM 부분, SHAP 비중)
KO = {"bun": "BUN(요소질소)", "hr": "심박수", "wbc": "백혈구", "adm_to_icu": "입원→ICU 입실 기간", "gcs_verbal": "GCS 언어",
      "rr": "호흡수", "gcs_eye": "GCS 눈뜨기", "spo2": "산소포화도", "albumin": "알부민", "age": "나이", "icu_los": "ICU 재실 기간",
      "sbp": "수축기 혈압", "dbp": "이완기 혈압", "platelet": "혈소판", "bilirubin": "빌리루빈", "inr": "INR", "temp": "체온",
      "gcs_motor": "GCS 운동", "lactate": "젖산", "creatinine": "크레아티닌", "glucose": "혈당", "hco3": "중탄산", "sodium": "나트륨",
      "potassium": "칼륨", "calcium": "칼슘", "ion_calcium": "이온화칼슘", "lymphocyte": "림프구", "ph": "pH", "po2": "pO2", "gender": "성별"}
s = get("shap", "B", "d1")["share"]["overall"]
top = sorted(s.items(), key=lambda kv: -kv[1])[:10][::-1]
fig, ax = plt.subplots(figsize=(5.8, 4))
ax.barh([KO.get(k, k) for k, _ in top], [v for _, v in top], color="#4a9a6a")
ax.set_xlabel("평균 |SHAP| 비중 (요약표 모델 부분)")
ax.set_title("그림 5. 점수에 많이 쓰인 변수 상위 10개\n(코호트 B·정의 ①·전체 기간)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig5_variables.png", dpi=200); plt.close(fig)
print("v4 figures:", sorted(p.name for p in OUT.glob("fig*.png")))
