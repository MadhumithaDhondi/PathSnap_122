"""End-to-end API check: call /predict for every evaluation trip over HTTP and compare with the offline evaluation.
Run while the API is up:  python scripts/07_test_api.py [http://localhost:8000]"""
import sys, json, time
import requests, pandas as pd
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
ids = requests.get(f"{base}/trips").json()["trips"]
rows = []
for tid in ids:
    t0 = time.time()
    r = requests.post(f"{base}/predict", json={"trip_id": tid}); r.raise_for_status()
    d = r.json(); assert "reference" not in json.dumps(d).lower(), "reference data must not be returned"
    m = d["metrics"]
    rows.append(dict(trip=tid, n_obs=d["n_obs"], clf_top1=m["classifier_only"]["top1_acc"], hmm_top1=m["classifier_plus_hmm"]["top1_acc"],
                     hmm_cont=m["classifier_plus_hmm"]["route_continuity"], pipeline_s=d["timing_seconds"]["total_s"], http_s=round(time.time() - t0, 2)))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv("results/api_test_results.csv", index=False)
print(df.round(4).to_string()); print("mean clf top1 %.4f  mean hmm top1 %.4f (obs-weighted %.4f)" % (df.clf_top1.mean(), df.hmm_top1.mean(), (df.hmm_top1 * df.n_obs).sum() / df.n_obs.sum()))
