"""발표용 그림. 입력은 안전 래퍼를 통과한 집계 결과(results/aggregates_W3_W4.jsonl)뿐이다 — 환자 단위 값 없음.
실행: .venv/bin/python iread/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
rows = [json.loads(l) for l in open(OUT / "aggregates_W3_W4.jsonl")]


def get(step, version, outcome, split="temporal"):
    for r in rows:
        x = r.get("result", {})
        if r["step"] == step and x.get("version") == version and x.get("outcome") == outcome \
                and x.get("split", "temporal") == split:
            return x
    raise KeyError((step, version, outcome, split))


TASKS = ["early", "late", "overall"]
LABEL = {"early": "≤48h", "late": ">48h", "overall": "overall"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1) 같은 점수로 본 사건별 포착률 (holdout 고정 임계값)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (v, title) in zip(axes, (("B", "Cohort B (clinical primary)"), ("A", "Cohort A (paper-style 24h merge)"))):
    c = get("calib", v, "d1")["alert"]
    xs = range(len(TASKS)); w = 0.38
    ax.bar([i - w / 2 for i in xs], [c[t]["holdout_thr"]["capture_readmission"] for t in TASKS], w, label="ICU readmission")
    ax.bar([i + w / 2 for i in xs], [c[t]["holdout_thr"]["capture_ward_death"] for t in TASKS], w, label="Ward death before readmission")
    for i, t in enumerate(TASKS):
        ax.text(i - w / 2, c[t]["holdout_thr"]["capture_readmission"] + .02, f'{c[t]["holdout_thr"]["capture_readmission"]:.0%}', ha="center", fontsize=8)
        ax.text(i + w / 2, c[t]["holdout_thr"]["capture_ward_death"] + .02, f'{c[t]["holdout_thr"]["capture_ward_death"]:.0%}', ha="center", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL[t] for t in TASKS]); ax.set_ylim(0, 1.08); ax.set_title(title)
axes[0].set_ylabel("Captured by alert\n(fixed holdout threshold)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=2)
fig.suptitle("Same iREAD score: which events does the alert catch? (definition ①, test 2017–22)")
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig1_event_capture.png", dpi=200); plt.close(fig)

# 2) 사건별 AUROC (같은 점수)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for k, (v, mk) in enumerate((("B", "o"), ("A", "s"))):
    b = get("calib", v, "d1")["by_event_type"]
    for j, (et, col) in enumerate((("readmission", "C0"), ("ward_death", "C1"))):
        ax.plot(range(3), [b[t][et]["auroc"] for t in TASKS], marker=mk, color=col, ls="-" if v == "B" else "--",
                label=f"{'Readmission' if et == 'readmission' else 'Ward death'} – cohort {v}")
ax.axhline(0.5, color="grey", lw=.8, ls=":")
ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[t] for t in TASKS]); ax.set_ylim(0.5, 1.0)
ax.set_ylabel("AUROC (event type vs non-events)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Event-type discrimination of the same composite score")
fig.tight_layout(); fig.savefig(OUT / "fig2_event_type_auroc.png", dpi=200); plt.close(fig)

# 3) 모델 비교 (코호트 B, ① 과 ③, 95% CI)
MODELS = ["iREAD", "LSTM only", "LightGBM only", "RF", "LR", "Cox", "NEWS", "MEWS", "SWIFT"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (d, title) in zip(axes, (("d1", "① readmission + ward death"), ("d3", "③ readmission only"))):
    r = get("train", "B", d)
    for j, t in enumerate(("early", "late")):
        y = [r["auroc"][m][t] for m in MODELS]
        lo = [r["auroc"][m][t] - r["auroc_lo"][m][t] for m in MODELS]
        hi = [r["auroc_hi"][m][t] - r["auroc"][m][t] for m in MODELS]
        ax.errorbar([i + (j - .5) * .3 for i in range(len(MODELS))], y, yerr=[lo, hi], fmt="o", ms=4, capsize=2,
                    label=LABEL[t])
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"Cohort B, {title}"); ax.set_ylim(0.5, 0.85)
axes[0].set_ylabel("AUROC (95% CI)"); axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig3_model_comparison.png", dpi=200); plt.close(fig)

# 4) 보정 곡선 (코호트 B ①, 10분위)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, t in zip(axes, TASKS):
    cv = get("calib", "B", "d1")["curve"][t]
    xs = [cv[f"d{i}"]["mean_pred"] for i in range(1, 11)]; ys = [cv[f"d{i}"]["obs"] for i in range(1, 11)]
    m = max(xs + ys) * 1.1
    ax.plot([0, m], [0, m], color="grey", lw=.8, ls=":"); ax.plot(xs, ys, "o-", ms=4)
    ax.set_title(f"{LABEL[t]}"); ax.set_xlabel("Mean predicted (decile)")
axes[0].set_ylabel("Observed event rate")
fig.suptitle("Calibration of recalibrated iREAD, cohort B ① (test)")
fig.tight_layout(); fig.savefig(OUT / "fig4_calibration.png", dpi=200); plt.close(fig)

# 5) SHAP 상위 변수 (코호트 B ① overall, LightGBM 부분)
s = get("shap", "B", "d1")["share"]["overall"]
top = sorted(s.items(), key=lambda kv: -kv[1])[:12][::-1]
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.barh([k for k, _ in top], [v for _, v in top], color="C2")
ax.set_xlabel("Share of mean |SHAP| (LightGBM part)"); ax.set_title("Top variables, cohort B ① overall")
fig.tight_layout(); fig.savefig(OUT / "fig5_shap.png", dpi=200); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("fig*.png")))
