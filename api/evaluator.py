"""SERVER-SIDE evaluation. The reference route is read here, AFTER predictions exist, and never goes back into the
prediction pipeline or the response (the UI only receives metrics, not labels)."""
import numpy as np
from pathsnap import config as C
from pathsnap.data import load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.metrics import obs_metrics, route_continuity, sequence_metrics

_label_cache: dict[str, np.ndarray] = {}


def evaluate(trip_id: str, result: dict) -> dict:
    trip = load_trip(C.EVAL_ROOT / trip_id, with_route=True)         # reference labels enter only here
    net = result["net"]
    if trip_id not in _label_cache:
        P = net.project_points(trip.track[trip.valid, 0], trip.track[trip.valid, 1])
        _label_cache[trip_id] = align_points_to_route(net, P, trip.route, C.ALIGN_SKIP_PENALTY_M)[0]
    label = _label_cache[trip_id]
    out = {}
    for key, name in (("clf_arc", "classifier_only"), ("hmm_arc", "classifier_plus_hmm")):
        pred = result[key]
        m = obs_metrics(pred, label)
        m.update(route_continuity=route_continuity(net, pred), **sequence_metrics(pred, trip.route))
        out[name] = {k: round(float(v), 4) if not isinstance(v, int) else v for k, v in m.items()}
    return out
