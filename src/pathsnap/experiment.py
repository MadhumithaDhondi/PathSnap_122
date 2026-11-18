"""Helpers shared by the experiment scripts: load cached per-trip datasets and assemble model matrices."""
from __future__ import annotations
import joblib, numpy as np, pandas as pd
from . import config as C
from .features import FEATURES


def load_cached(partition: str, qc: bool = True) -> list[dict]:
    out = []
    for f in sorted(C.CACHE.glob(f"{partition}_*.joblib")):
        d = joblib.load(f)
        if qc and partition == "train" and np.median(d["label_dist"]) > C.LABEL_QC_MAX_MEDIAN_M:
            continue
        out.append(d)
    return out


def stack(trips: list[dict], stride: int = 1, offset: int = 0):
    """Concatenate candidate rows of the observations obs % stride == offset. Returns X, y, trip_index."""
    parts = []
    for ti, d in enumerate(trips):
        df = d["cand"]
        if stride > 1:
            df = df[(df["obs"].to_numpy() % stride) == offset]
        parts.append(df.assign(trip=ti))
    big = pd.concat(parts, ignore_index=True)
    return big[FEATURES].to_numpy(np.float32), big["y"].to_numpy(), big["trip"].to_numpy(), big


# ---------------------------------------------------------------------------------------------------------------
from .metrics import candidate_metrics, top1_predictions, obs_metrics, candidate_miss_rate


def evaluate(trips, big, p, stride=1, offset=0):
    """Pooled candidate metrics + observation-level metrics for the trips in `trips`.
    `big` = stacked candidate rows (with a `trip` column indexing into `trips`), p = probabilities aligned with `big`."""
    sel_rows = np.ones(len(big), bool) if stride == 1 else (big["obs"].to_numpy() % stride) == offset
    cm = candidate_metrics(big["y"].to_numpy()[sel_rows], p[sel_rows])
    n_ok = n_tot = 0
    mf1 = []
    trip_col = big["trip"].to_numpy()
    for ti, d in enumerate(trips):
        m = (trip_col == ti) & sel_rows
        sub = big[m]
        pred = top1_predictions(sub, p[m], d["obs"]["n_obs"])
        sel = np.arange(d["obs"]["n_obs"]) % stride == offset if stride > 1 else np.ones(d["obs"]["n_obs"], bool)
        om = obs_metrics(pred, d["label"], sel)
        n_ok += om["top1_acc"] * sel.sum(); n_tot += sel.sum(); mf1.append(om["obs_macro_f1"])
    cm.update(top1_acc=n_ok / n_tot, obs_macro_f1=float(np.mean(mf1)))
    return cm, mf1
