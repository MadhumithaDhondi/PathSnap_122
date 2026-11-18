"""Stage 7/12: interactive Folium maps (mandatory deliverable). Layers: raw GPS, road network, reference route,
classifier-only prediction, classifier+HMM prediction, with legend + layer control.
Usage: python scripts/06_make_maps.py [partition/trip_id ...]   (default: a few student-test trips)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, folium
from pathsnap import config as C
from pathsnap.data import load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.pipeline import PathSnapPipeline
from pathsnap.metrics import obs_metrics

C.MAPS.mkdir(exist_ok=True)
pipe = PathSnapPipeline.load(C.MODELS / "pathsnap_final.joblib")
targets = sys.argv[1:] or ["student_test/00000000", "student_test/00000040", "student_test/00000056"]

LEGEND = """<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 12px;border:1px solid #888;
border-radius:6px;font:12px/1.5 sans-serif"><b>PathSnap legend</b><br>
<span style="color:#1f77b4">&#9679;</span> raw GPS fixes<br><span style="color:#2ca02c">&#9644;</span> reference route<br>
<span style="color:#ff9f1c">&#9644;</span> classifier-only prediction<br><span style="color:#d62728">&#9644;</span> classifier + HMM prediction<br>
<span style="color:#999">&#9644;</span> road network (near the trip)</div>"""


def seg(net, nodes, a):
    t, h = net.tail[a], net.head[a]
    return [[nodes[t, 1], nodes[t, 0]], [nodes[h, 1], nodes[h, 0]]]


for target in targets:
    part, tid = target.split("/")
    trip = load_trip(C.DATA_ROOT / part / tid, with_route=True)
    res = pipe.predict(trip); net = res["net"]
    P = net.project_points(res["lonlat"][:, 0], res["lonlat"][:, 1])
    label = align_points_to_route(net, P, trip.route, C.ALIGN_SKIP_PENALTY_M)[0]
    m_clf, m_hmm = obs_metrics(res["clf_arc"], label), obs_metrics(res["hmm_arc"], label)
    lat, lon = res["lonlat"][:, 1], res["lonlat"][:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    # road network inside the trip's bounding box (+ margin), capped for file size
    lo, hi = P.min(0) - 200, P.max(0) + 200
    inb = np.flatnonzero((((net.A > lo) & (net.A < hi)).all(1)) | (((net.B > lo) & (net.B < hi)).all(1)))
    if len(inb) > 6000:
        inb = np.random.default_rng(C.SEED).choice(inb, 6000, replace=False)
    g_net = folium.FeatureGroup("Road network", show=False)
    for a in inb: folium.PolyLine(seg(net, trip.nodes, a), color="#999999", weight=1, opacity=.6).add_to(g_net)
    g_ref = folium.FeatureGroup("Reference route", show=True)
    for a in dict.fromkeys(trip.route.tolist()): folium.PolyLine(seg(net, trip.nodes, a), color="#2ca02c", weight=7, opacity=.55).add_to(g_ref)
    def route_layer(name, arcs, color, weight, dash=None):
        g = folium.FeatureGroup(name, show=True)
        prev = None
        for a in arcs.tolist():
            if a != prev and a >= 0:
                folium.PolyLine(seg(net, trip.nodes, a), color=color, weight=weight, opacity=.9, dash_array=dash).add_to(g)
            prev = a
        return g
    g_clf = route_layer("Prediction: classifier only", res["clf_arc"], "#ff9f1c", 4, "6 6")
    g_hmm = route_layer("Prediction: classifier + HMM", res["hmm_arc"], "#d62728", 3)
    g_gps = folium.FeatureGroup("Raw GPS", show=True)
    step = max(1, len(lat) // 1500)
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, weight=1, color="#1f77b4", fill=True, fill_opacity=.6,
                            tooltip=f"fix {i}  t={res['t'][i]:.0f}s").add_to(g_gps)
    for g in (g_net, g_ref, g_gps, g_clf, g_hmm): g.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND + f"<div style='position:fixed;bottom:24px;right:12px;z-index:9999;background:white;padding:8px 10px;border:1px solid #888;border-radius:6px;font:12px sans-serif'>"
        f"{part}/{tid} (whole trip)<br>Top-1 classifier-only: {m_clf['top1_acc']:.3f}<br>Top-1 + HMM: {m_hmm['top1_acc']:.3f}</div>"))
    # initial view: the 300-fix window where the classifier-only prediction makes the most mistakes (zoom out for the whole trip)
    err = (res["clf_arc"] != label).astype(int)
    W = min(300, len(err)); csum = np.concatenate([[0], np.cumsum(err)])
    s0 = int(np.argmax(csum[W:] - csum[:-W])); sl = slice(s0, s0 + W)
    padl = 0.0004
    m.fit_bounds([[lat[sl].min() - padl, lon[sl].min() - padl], [lat[sl].max() + padl, lon[sl].max() + padl]])
    out = C.MAPS / f"{part}_{tid}.html"
    m.save(out)
    print("saved", out, f"top1 clf={m_clf['top1_acc']:.3f} hmm={m_hmm['top1_acc']:.3f}", f"({out.stat().st_size/1e6:.1f} MB)")
