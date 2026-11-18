"""One evaluation protocol shared by the model comparison (03) and the tuned-model evaluation (04):
group-aware CV on the training trips, then development evaluation on the student-test trips."""
from __future__ import annotations
import time
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from . import config as C
from .experiment import evaluate
from .metrics import candidate_metrics, top1_predictions, route_continuity, sequence_metrics, fill_missing


def cross_validate(est, Xf, yf, gf, bigf, train, fit_mask, folds=C.CV_FOLDS, verbose=True):
    cv = []
    for k, (tr_i, va_i) in enumerate(GroupKFold(folds).split(Xf, yf, gf)):
        tr_i = tr_i[fit_mask[tr_i]]
        m = clone(est)
        t0 = time.time(); m.fit(Xf[tr_i], yf[tr_i]); tfit = time.time() - t0
        p = m.predict_proba(Xf[va_i])[:, 1]
        sub = bigf.iloc[va_i].copy()
        trips_va = sorted(set(sub["trip"]))
        sub["trip"] = sub["trip"].map({t: i for i, t in enumerate(trips_va)})
        res, _ = evaluate([train[t] for t in trips_va], sub.reset_index(drop=True), p, C.FIT_STRIDE, 0)
        res.update(fold=k, fit_seconds=tfit); cv.append(res)
        if verbose:
            print(f"  fold {k}: macroF1={res['cand_macro_f1']:.4f} top1={res['top1_acc']:.4f} logloss={res['log_loss']:.4f} ({tfit:.0f}s)", flush=True)
    return pd.DataFrame(cv)


def fit_and_test(est, Xf, yf, bigf, train, fit_mask, mult, test, bigt, Xt, yt, nets, refs):
    """Refit on all (sub-sampled) training trips; in-sample metrics; full student-test evaluation."""
    m = clone(est)
    t0 = time.time(); m.fit(Xf[fit_mask], yf[fit_mask]); fit_s = time.time() - t0
    p_fit = m.predict_proba(Xf[fit_mask])[:, 1]
    cm_fit = candidate_metrics(yf[fit_mask], p_fit)
    sub = bigf[fit_mask].reset_index(drop=True)
    ok = tot = 0
    for ti, d in enumerate(train):
        mm = sub["trip"].to_numpy() == ti
        if mm.sum() == 0:
            continue
        pred = top1_predictions(sub[mm], p_fit[mm], d["obs"]["n_obs"])
        sel = np.arange(d["obs"]["n_obs"]) % (C.FIT_STRIDE * mult) == 0
        ok += (pred[sel] == d["label"][sel]).sum(); tot += sel.sum()
    t0 = time.time(); p_t = m.predict_proba(Xt)[:, 1]; inf_s = time.time() - t0
    res_t, _ = evaluate(test, bigt, p_t, 1)
    cont, seq = [], []
    for ti, d in enumerate(test):
        mm = bigt["trip"].to_numpy() == ti
        pred = fill_missing(top1_predictions(bigt[mm], p_t[mm], d["obs"]["n_obs"]))
        cont.append(route_continuity(nets[d["trip_id"]], pred))
        seq.append(sequence_metrics(pred, refs[d["trip_id"]]))
    n_obs_test = sum(d["obs"]["n_obs"] for d in test)
    rec = dict(train_macro_f1=cm_fit["cand_macro_f1"], train_top1=ok / tot, train_logloss=cm_fit["log_loss"],
               test_top1=res_t["top1_acc"], test_cand_macro_f1=res_t["cand_macro_f1"], test_obs_macro_f1=res_t["obs_macro_f1"],
               test_precision=res_t["cand_precision"], test_recall=res_t["cand_recall"], test_logloss=res_t["log_loss"],
               test_cand_accuracy=res_t["cand_accuracy"], test_route_continuity=float(np.mean(cont)),
               test_route_f1=float(np.mean([s["route_f1"] for s in seq])),
               fit_seconds=fit_s, inference_seconds=inf_s, inference_ms_per_obs=1000 * inf_s / n_obs_test, fit_rows=int(fit_mask.sum()))
    cmx = confusion_matrix(yt, (p_t >= 0.5).astype(int), labels=[0, 1]).tolist()
    return m, rec, cmx
