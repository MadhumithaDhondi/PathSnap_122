"""Extra experiment: which feature groups matter? LightGBM (untuned config) trained on cumulative feature groups.
Same protocol as the model comparison (group-aware 5-fold CV on training trips + student-test)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.base import clone
from pathsnap import config as C
from pathsnap.experiment import load_cached, stack, evaluate
from pathsnap.features import FEATURES
from pathsnap.modelzoo import zoo
from pathsnap.protocol import cross_validate
from sklearn.model_selection import GroupKFold

groups = {
    "geometry (dist, position on arc, arc length, rank, #candidates)": ["dist", "t_along", "arc_len", "rank", "dist_gap", "dist_ratio", "n_cand"],
    "+ motion/heading (heading agreement, speed, time step)": ["cos_head_s", "cos_head_l", "ang_head_s", "head_valid_s", "speed", "dt_prev"],
    "+ temporal context (distance of neighbouring fixes to the arc)": ["w_mean", "w_max", "d_m3", "d_p3", "d_m6", "d_p6"],
    "+ topology (junction degrees)": ["head_out_deg", "tail_in_deg"],
}
train, test = load_cached("train"), load_cached("student_test")
Xf, yf, gf, bigf = stack(train, C.FIT_STRIDE, 0)
Xt, yt, gt, bigt = stack(test, 1)
est = zoo(C.SEED)["LightGBM"]["est"]
used, rows = [], []
for gname, cols in groups.items():
    used += cols
    idx = [FEATURES.index(c) for c in used]
    from sklearn.model_selection import GroupKFold
    cv = []
    for tr, va in GroupKFold(C.CV_FOLDS).split(Xf, yf, gf):
        m = clone(est).fit(Xf[tr][:, idx], yf[tr])
        p = m.predict_proba(Xf[va][:, idx])[:, 1]
        sub = bigf.iloc[va].copy(); tv = sorted(set(sub["trip"])); sub["trip"] = sub["trip"].map({t: i for i, t in enumerate(tv)})
        cv.append(evaluate([train[t] for t in tv], sub.reset_index(drop=True), p, C.FIT_STRIDE, 0)[0])
    cv = pd.DataFrame(cv)
    m = clone(est).fit(Xf[:, idx], yf)
    res, _ = evaluate(test, bigt, m.predict_proba(Xt[:, idx])[:, 1], 1)
    rows.append(dict(feature_set=gname, n_features=len(idx), cv_macro_f1=cv.cand_macro_f1.mean(), cv_top1=cv.top1_acc.mean(),
                     test_macro_f1=res["cand_macro_f1"], test_top1=res["top1_acc"], test_logloss=res["log_loss"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(C.RESULTS / "feature_ablation.csv", index=False)
