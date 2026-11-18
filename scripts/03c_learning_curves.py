"""Training/validation loss curves for the models that really have epoch/iteration histories (LightGBM, MLP).
LightGBM: fold 0 of the group-CV (validation = held-out trips). MLP history is saved by 03_compare_models.py."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from pathsnap import config as C
from pathsnap.experiment import load_cached, stack

train = load_cached("train")
Xf, yf, gf, _ = stack(train, C.FIT_STRIDE, 0)
tr, va = next(iter(GroupKFold(C.CV_FOLDS).split(Xf, yf, gf)))
for tag, params in (("untuned", dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8)),
                    ("tuned", json.load(open(C.RESULTS / "bayes_opt_best.json"))["best_params"])):
    m = LGBMClassifier(**params, subsample_freq=1, n_jobs=2, random_state=C.SEED, verbose=-1)
    m.fit(Xf[tr], yf[tr], eval_set=[(Xf[tr], yf[tr]), (Xf[va], yf[va])], eval_names=["train", "valid"], eval_metric="binary_logloss")
    ev = m.evals_result_
    json.dump({"train": ev["train"]["binary_logloss"], "valid": ev["valid"]["binary_logloss"]}, open(C.RESULTS / "model_runs" / f"LightGBM_{tag}_history.json", "w"))
    print(tag, "final train/valid logloss", ev["train"]["binary_logloss"][-1], ev["valid"]["binary_logloss"][-1])
