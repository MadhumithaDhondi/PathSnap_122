"""Report figures. Every figure is drawn from the CSV/JSON files written by the experiment scripts (no hand-typed numbers)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathsnap import config as C

R, F = C.RESULTS, C.ROOT / "report" / "figs"; F.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.axisbelow": True, "figure.dpi": 140, "legend.frameon": False})
cmp_ = pd.read_csv(R / "model_comparison.csv")
order = cmp_.sort_values("cv_macro_f1_mean")["model"].tolist()
c = cmp_.set_index("model").loc[order]

# 1 CV macro F1 (mean +- std over 5 group folds)
fig, ax = plt.subplots(figsize=(7, 3.8)); y = np.arange(len(c))
ax.barh(y, c.cv_macro_f1_mean, xerr=c.cv_macro_f1_std, color=BLUE, height=.55, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
for i, v in enumerate(c.cv_macro_f1_mean): ax.text(v + .0045, i, f"{v:.4f}", va="center", fontsize=9, color=INK)
ax.set_yticks(y, c.index); ax.set_xlim(.82, .965); ax.set_xlabel("candidate-level macro F1, mean ± std over 5 group-CV folds (78 training trips)"); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(F / "cmp_cv_macro_f1.png"); plt.close()

# 2 student-test Top-1 and candidate macro F1
fig, ax = plt.subplots(figsize=(8.2, 3.9)); x = np.arange(len(c)); w = .38
ax.bar(x - w/2, c.test_top1, w, color=BLUE, label="Top-1 accuracy (per GPS fix)"); ax.bar(x + w/2, c.test_cand_macro_f1, w, color=ORANGE, label="macro F1 (candidate level)")
ax.set_xticks(x, c.index, rotation=25, ha="right"); ax.set_ylim(.8, .96); ax.set_ylabel("student-test (10 trips)"); ax.legend(loc="upper left", ncol=2); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(F / "cmp_test_top1_f1.png"); plt.close()

# 3 log loss
o2 = cmp_.sort_values("test_logloss", ascending=False); fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(o2.model, o2.test_logloss, color=BLUE, height=.55)
for i, v in enumerate(o2.test_logloss): ax.text(v + .006, i, f"{v:.4f}", va="center", fontsize=9, color=INK)
ax.set_xlabel("candidate-level log loss on student-test (lower is better)"); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(F / "cmp_logloss.png"); plt.close()

# 4 over/under-fitting evidence: train vs CV vs test Top-1
fig, ax = plt.subplots(figsize=(8.4, 3.9)); w = .27
for k, (col, lab, colr) in enumerate([("train_top1", "training fit set (in-sample)", BLUE), ("cv_top1_mean", "group-CV (held-out trips)", ORANGE), ("test_top1", "student-test", AQUA)]):
    ax.bar(x + (k - 1) * w, cmp_.set_index("model").loc[c.index, col], w, color=colr, label=lab)
ax.set_xticks(x, c.index, rotation=25, ha="right"); ax.set_ylim(.8, 1.0); ax.set_ylabel("Top-1 accuracy"); ax.legend(loc="upper left", ncol=3, fontsize=8.5); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(F / "cmp_overfit.png"); plt.close()

# 5 confusion matrices (student-test, candidate level, threshold 0.5)
_n = len(cmp_); _rows = -(-_n // 4)
fig, axs = plt.subplots(_rows, 4, figsize=(10, 2.5 * _rows))
for ax in axs.ravel()[_n:]: ax.axis("off")
for ax, m in zip(axs.ravel(), cmp_.model):
    cm = np.array(json.load(open(R / "model_runs" / f"{m}.json"))["confusion_matrix_student_test"]); pct = cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i,j]:,}\n({pct[i,j]:.1%})", ha="center", va="center", fontsize=8, color="white" if pct[i, j] > .5 else INK)
    ax.set_title(m, fontsize=9); ax.set_xticks([0, 1], ["not ref.", "ref."]); ax.set_yticks([0, 1], ["not ref.", "ref."], rotation=90, va="center")
fig.supxlabel("predicted", fontsize=9, color=MUTED); fig.supylabel("actual", fontsize=9, color=MUTED); plt.tight_layout(); plt.savefig(F / "cmp_confusion.png"); plt.close()

# 6 loss curves (only models with real iteration histories)
mh = json.load(open(R / "model_runs" / "MLP_history.json"))
lu, lt = (json.load(open(R / "model_runs" / f"LightGBM_{t}_history.json")) for t in ("untuned", "tuned"))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].plot(range(1, len(mh["loss_curve"]) + 1), mh["loss_curve"], color=BLUE, lw=2); axs[0].set_title("MLP training loss (per epoch)", fontsize=10); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("log loss")
for h, nm, ls in ((lu, "untuned", "-"), (lt, "tuned", "--")):
    axs[1].plot(h["train"], color=BLUE, lw=1.8, ls=ls, label=f"train ({nm})"); axs[1].plot(h["valid"], color=ORANGE, lw=1.8, ls=ls, label=f"validation ({nm})")
axs[1].set_title("LightGBM: CV fold 0 (validation = held-out trips)", fontsize=10); axs[1].set_xlabel("boosting round"); axs[1].set_ylabel("log loss"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(F / "learning_curves.png"); plt.close()

# 7 Bayesian optimisation history
tr = pd.read_csv(R / "bayes_opt_trials.csv"); fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].scatter(tr.number, tr.value, s=22, color=BLUE, zorder=3); axs[0].plot(tr.number, tr.value.cummax(), color=ORANGE, lw=2, label="best so far")
axs[0].set_xlabel("trial"); axs[0].set_ylabel("mean 3-fold CV macro F1"); axs[0].legend(); axs[0].set_title("Optimisation history", fontsize=10)
axs[1].scatter(tr["params_learning_rate"], tr.value, s=22, color=BLUE); axs[1].set_xscale("log"); axs[1].set_xlabel("learning_rate"); axs[1].set_title("objective vs learning rate", fontsize=10)
axs[2].scatter(tr["params_num_leaves"], tr.value, s=22, color=BLUE); axs[2].set_xlabel("num_leaves"); axs[2].set_title("objective vs num_leaves", fontsize=10)
plt.tight_layout(); plt.savefig(F / "bayes_opt_history.png"); plt.close()

# 8 HMM vs classifier-only (student-test, per trip) + ablation
ev = pd.read_csv(R / "hmm_eval_student_test_per_trip.csv"); a = ev[ev.variant == "classifier_only"].set_index("trip_id"); b = ev[ev.variant == "hmm_tuned"].set_index("trip_id")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1])); x = np.arange(len(a)); w = .38
axs[0].bar(x - w/2, a.top1_acc, w, color=BLUE, label="classifier only"); axs[0].bar(x + w/2, b.top1_acc, w, color=ORANGE, label="classifier + HMM (tuned)")
axs[0].set_xticks(x, [f"{int(t):03d}" for t in a.index]); axs[0].set_xlabel("student-test trip id (last 3 digits)"); axs[0].set_ylabel("Top-1 accuracy"); axs[0].set_ylim(.7, 1.02); axs[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2); axs[0].grid(axis="x", visible=False)
ab = pd.read_csv(R / "hmm_ablation_student_test.csv")[::-1]; axs[1].barh(ab.variant, ab.top1_acc, color=BLUE, height=.55)
for i, v in enumerate(ab.top1_acc): axs[1].text(v + .003, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
axs[1].set_xlim(.85, 1.0); axs[1].set_xlabel("student-test Top-1 accuracy"); axs[1].tick_params(axis="y", labelsize=8); axs[1].grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(F / "hmm_comparison.png"); plt.close()

# 9 feature-group ablation
fa = pd.read_csv(R / "feature_ablation.csv"); fig, ax = plt.subplots(figsize=(8.4, 3.4)); x = np.arange(len(fa)); w = .38
ax.bar(x - w/2, fa.cv_macro_f1, w, color=BLUE, label="CV macro F1 (train trips)"); ax.bar(x + w/2, fa.test_macro_f1, w, color=ORANGE, label="student-test macro F1")
ax.set_xticks(x, ["geometry", "+ motion", "+ temporal\ncontext", "+ topology"]); ax.set_ylim(.88, .94); ax.set_ylabel("candidate-level macro F1"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(F / "feature_ablation.png"); plt.close()
print("figures written:", sorted(p.name for p in F.glob("*.png")))
