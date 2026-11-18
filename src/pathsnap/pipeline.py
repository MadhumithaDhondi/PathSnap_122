"""End-to-end inference: raw trip files (track + nodes + arcs ONLY) -> predicted arc for every GPS fix.
The reference route is never an input of this module, so it cannot leak into predictions or the UI."""
from __future__ import annotations
import time
import joblib, numpy as np, pandas as pd
from . import config as C
from .data import Trip
from .features import FEATURES, build_trip_features
from .hmm import HMMParams, viterbi_decode
from .metrics import top1_predictions, fill_missing
from .network import RoadNetwork


class PathSnapPipeline:
    def __init__(self, model, hmm_params: HMMParams, radius: float = C.CAND_RADIUS_M, k: int = C.CAND_MAX_K, meta: dict | None = None):
        self.model, self.hmm, self.radius, self.k, self.meta = model, hmm_params, radius, k, meta or {}

    def save(self, path):
        joblib.dump(dict(model=self.model, hmm=self.hmm.to_dict(), radius=self.radius, k=self.k, features=FEATURES, meta=self.meta), path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        assert d["features"] == FEATURES, "feature list changed since the model was trained"
        return cls(d["model"], HMMParams(**d["hmm"]), d["radius"], d["k"], d.get("meta"))

    def predict(self, trip: Trip, net: RoadNetwork | None = None) -> dict:
        t0 = time.time()
        net = net or RoadNetwork(trip)
        df, obs = build_trip_features(trip, net, self.radius, self.k, label_arc=None)
        t1 = time.time()
        p = self.model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1] if len(df) else np.zeros(0)
        t2 = time.time()
        clf = fill_missing(top1_predictions(df, p, obs["n_obs"]))
        hmm = fill_missing(viterbi_decode(net, df["obs"].to_numpy(), df["arc"].to_numpy(), p, obs["t"], obs["n_obs"], self.hmm))
        t3 = time.time()
        return dict(rows=obs["rows"], t=obs["t"], lonlat=obs["lonlat"], clf_arc=clf, hmm_arc=hmm, cand=df, prob=p,
                    timing=dict(features_s=t1 - t0, model_s=t2 - t1, hmm_s=t3 - t2, total_s=t3 - t0), net=net)
