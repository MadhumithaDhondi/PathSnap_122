"""PathSnap FastAPI backend (runs on Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

GET  /health              liveness + model info
GET  /trips               ids of the evaluation trips found in PATHSNAP_EVAL_DATA
POST /predict {trip_id}   full pipeline: candidates -> features -> classifier -> HMM/Viterbi; accuracy is computed
                          server-side from the reference route, which is never sent to the client.
"""
import logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathsnap import config as C
from pathsnap.data import load_trip, list_trips
from pathsnap.pipeline import PathSnapPipeline
import evaluator

C.ROOT.joinpath("logs").mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
for h in (logging.StreamHandler(), logging.FileHandler(C.ROOT / "logs" / "api_requests.log")):
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s")); log.addHandler(h)

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
PIPE = PathSnapPipeline.load(C.MODELS / "pathsnap_final.joblib")
_nets: dict = {}


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info(f"{request.client.host if request.client else '-'} {request.method} {request.url.path} -> {resp.status_code} ({time.time()-t0:.2f}s)")
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": PIPE.meta.get("model_name"), "hmm": PIPE.hmm.to_dict(), "eval_root": str(C.EVAL_ROOT)}


@app.get("/trips")
def trips():
    ids = [p.name for p in sorted(C.EVAL_ROOT.iterdir()) if p.is_dir()]
    return {"trips": ids}


def _route_segments(net, arcs, nodes_lonlat):
    """Predicted route as a list of [[lat,lon],[lat,lon]] segments (consecutive duplicate arcs collapsed)."""
    dedup = [int(a) for i, a in enumerate(arcs) if (i == 0 or a != arcs[i - 1]) and a >= 0]
    segs = []
    for a in dedup:
        t, h = net.tail[a], net.head[a]
        segs.append([[float(nodes_lonlat[t, 1]), float(nodes_lonlat[t, 0])], [float(nodes_lonlat[h, 1]), float(nodes_lonlat[h, 0])]])
    return dedup, segs


@app.post("/predict")
def predict(req: PredictRequest):
    tdir = C.EVAL_ROOT / req.trip_id
    if not tdir.is_dir():
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    trip = load_trip(tdir, with_route=False)                   # prediction side: NO labels are loaded
    res = PIPE.predict(trip, _nets.get(req.trip_id))
    _nets[req.trip_id] = res["net"]
    metrics = evaluator.evaluate(req.trip_id, res)             # labels used only here, server-side
    clf_arcs, clf_segs = _route_segments(res["net"], res["clf_arc"], trip.nodes)
    hmm_arcs, hmm_segs = _route_segments(res["net"], res["hmm_arc"], trip.nodes)
    log.info(f"predict trip={req.trip_id} obs={len(res['rows'])} arcs(hmm)={len(hmm_arcs)} "
             f"top1 clf={metrics['classifier_only']['top1_acc']} hmm={metrics['classifier_plus_hmm']['top1_acc']} "
             f"pipeline={res['timing']['total_s']:.2f}s")
    return {
        "trip_id": req.trip_id, "n_obs": int(len(res["rows"])),
        "gps": [[float(la), float(lo)] for lo, la in res["lonlat"]],
        "predicted_route_hmm": hmm_segs, "predicted_route_classifier_only": clf_segs,
        "predicted_arc_ids_hmm": hmm_arcs,
        "metrics": metrics,
        "timing_seconds": {k: round(v, 3) for k, v in res["timing"].items()} | {"request_total_s": round(time.time() - t0, 3)},
    }
