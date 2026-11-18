"""Stage 4/5: train and compare >= 5 models. Group-aware 5-fold CV on the training trips (split BY TRIP), then a
development evaluation on the 10 student-test trips. Everything is logged to results/ .
Usage: python scripts/03_compare_models.py [ModelName ...]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from pathsnap import config as C
from pathsnap.experiment import load_cached, stack
from pathsnap.network import RoadNetwork
from pathsnap.data import load_trip
from pathsnap.modelzoo import zoo, describe
from pathsnap.protocol import cross_validate, fit_and_test

np.random.seed(C.SEED)
(C.RESULTS / "model_runs").mkdir(parents=True, exist_ok=True)
train, test = load_cached("train"), load_cached("student_test")
print(f"train trips used: {len(train)} (of 80 after label-QC); student-test trips: {len(test)}", flush=True)
Xf, yf, gf, bigf = stack(train, C.FIT_STRIDE, 0)           # fit / CV pool (every 3rd observation)
Xt, yt, gt, bigt = stack(test, 1)                           # student-test, every observation
print("fit rows", len(yf), "positives", int(yf.sum()), " student-test rows", len(yt), flush=True)
refs = {d["trip_id"]: load_trip(C.DATA_ROOT / "student_test" / d["trip_id"], with_route=True) for d in test}
nets = {k: RoadNetwork(v) for k, v in refs.items()}
refs = {k: v.route for k, v in refs.items()}

summary, fold_rows, only = [], [], sys.argv[1:]
for name, spec in zoo(C.SEED).items():
    if only and name not in only:
        continue
    est, mult = spec["est"], spec["fit_stride_mult"]
    fit_mask = (bigf["obs"].to_numpy() % (C.FIT_STRIDE * mult)) == 0
    print(f"\n=== {name} ({spec['family']})  fit rows {fit_mask.sum()}", flush=True)
    cvdf = cross_validate(est, Xf, yf, gf, bigf, train, fit_mask)
    cvdf["model"] = name; fold_rows.append(cvdf)
    m, rec, cmx = fit_and_test(est, Xf, yf, bigf, train, fit_mask, mult, test, bigt, Xt, yt, nets, refs)
    rec.update(model=name, family=spec["family"], cv_macro_f1_mean=cvdf.cand_macro_f1.mean(), cv_macro_f1_std=cvdf.cand_macro_f1.std(),
               cv_top1_mean=cvdf.top1_acc.mean(), cv_top1_std=cvdf.top1_acc.std(), cv_logloss_mean=cvdf.log_loss.mean(),
               cv_fit_seconds_mean=cvdf.fit_seconds.mean())
    summary.append(rec)
    json.dump(dict(summary=rec, params=describe(est), confusion_matrix_student_test=cmx, cv_folds=cvdf.to_dict("records")),
              open(C.RESULTS / "model_runs" / f"{name}.json", "w"), indent=1, default=float)
    print(f"  CV macroF1 {rec['cv_macro_f1_mean']:.4f}±{rec['cv_macro_f1_std']:.4f}  CV top1 {rec['cv_top1_mean']:.4f} | train top1 {rec['train_top1']:.4f} | "
          f"TEST top1 {rec['test_top1']:.4f} macroF1 {rec['test_cand_macro_f1']:.4f} logloss {rec['test_logloss']:.4f} "
          f"cont {rec['test_route_continuity']:.4f} inf {rec['inference_seconds']:.1f}s fit {rec['fit_seconds']:.0f}s", flush=True)
    last = m.steps[-1][1] if hasattr(m, "steps") else m
    if hasattr(last, "loss_curve_"):
        json.dump(dict(loss_curve=last.loss_curve_, val_acc=last.validation_scores_), open(C.RESULTS / "model_runs" / f"{name}_history.json", "w"))

suffix = "" if not only else "_partial"
pd.DataFrame(summary).to_csv(C.RESULTS / f"model_comparison{suffix}.csv", index=False)
pd.concat(fold_rows).to_csv(C.RESULTS / f"cv_folds{suffix}.csv", index=False)
print("saved results")
