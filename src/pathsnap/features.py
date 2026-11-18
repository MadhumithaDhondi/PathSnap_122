"""Candidate generation + feature engineering for one trip.

One row = (GPS observation i, candidate arc a). Target y = 1 if a is the reference arc of observation i.
All features use only the trip's own track and road network (never the reference route), so the same code runs
at training time and inside the API.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import Trip
from .geo import point_segment_distance, angle_diff
from .network import RoadNetwork

FEATURES = [
    "dist", "t_along", "arc_len",
    "cos_head_s", "cos_head_l", "ang_head_s", "head_valid_s",
    "speed", "dt_prev",
    "rank", "dist_gap", "dist_ratio", "n_cand",
    "w_mean", "w_max", "d_m3", "d_p3", "d_m6", "d_p6",
    "head_out_deg", "tail_in_deg",
]
OFFSETS = (-6, -3, -1, 1, 3, 6)


def _heading(P: np.ndarray, k: int):
    """Heading and displacement of the track over a +-k window (clipped at the ends)."""
    n = len(P)
    i = np.arange(n)
    a, b = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
    d = P[b] - P[a]
    return np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1]), a, b


def build_trip_features(trip: Trip, net: RoadNetwork, radius: float, k: int, label_arc: np.ndarray | None = None):
    """Returns (cand DataFrame, obs dict). obs index = position among *valid* GPS rows; `row` maps back to the file."""
    rows = np.flatnonzero(trip.valid)
    lon, lat, t = trip.track[rows, 0], trip.track[rows, 1], trip.track[rows, 2]
    P = net.project_points(lon, lat)
    N = len(P)

    oi, ai, dist, talong = net.candidates(P, radius, k)

    # ---- per-observation motion features
    hs, ds, a1, b1 = _heading(P, 1)
    hl, dl, _, _ = _heading(P, 4)
    dts = np.maximum(t[b1] - t[a1], 1e-6)
    speed = ds / dts
    valid_s = (ds > 3.0)
    valid_l = (dl > 8.0)
    dt_prev = np.concatenate([[1.0], np.diff(t)])

    n_cand = np.bincount(oi, minlength=N)
    min_d = np.full(N, np.inf)
    np.minimum.at(min_d, oi, dist)
    first = np.searchsorted(oi, oi, side="left")
    rank = np.arange(len(oi)) - first

    bear = net.bearing[ai]
    cos_s = np.where(valid_s[oi], np.cos(angle_diff(hs[oi], bear)), 0.0)
    cos_l = np.where(valid_l[oi], np.cos(angle_diff(hl[oi], bear)), 0.0)
    ang_s = np.where(valid_s[oi], np.abs(angle_diff(hs[oi], bear)), np.pi / 2)

    # ---- temporal context: how close are neighbouring GPS points to this candidate arc?
    A, B = net.A[ai], net.B[ai]
    offs = []
    for o in OFFSETS:
        j = np.clip(oi + o, 0, N - 1)
        d_o, _ = point_segment_distance(P[j], A, B)
        offs.append(d_o)
    offs = np.stack(offs, axis=1)

    df = pd.DataFrame({
        "obs": oi.astype(np.int32), "arc": ai.astype(np.int64),
        "dist": dist, "t_along": talong, "arc_len": net.length[ai],
        "cos_head_s": cos_s, "cos_head_l": cos_l, "ang_head_s": ang_s, "head_valid_s": valid_s[oi].astype(float),
        "speed": speed[oi], "dt_prev": dt_prev[oi],
        "rank": rank, "dist_gap": dist - min_d[oi], "dist_ratio": dist / (min_d[oi] + 1.0), "n_cand": n_cand[oi],
        "w_mean": offs.mean(1), "w_max": offs.max(1),
        "d_m3": offs[:, 1], "d_p3": offs[:, 4], "d_m6": offs[:, 0], "d_p6": offs[:, 5],
        "head_out_deg": net.out_deg[net.head[ai]], "tail_in_deg": net.in_deg[net.tail[ai]],
    })
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    if label_arc is not None:
        df["y"] = (df["arc"].to_numpy() == label_arc[df["obs"].to_numpy()]).astype(np.int8)
    obs = {"rows": rows, "t": t, "lonlat": np.stack([lon, lat], 1), "n_obs": N, "n_cand": n_cand}
    return df, obs
