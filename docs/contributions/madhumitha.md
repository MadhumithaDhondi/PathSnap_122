# Madhumitha — Inference pipeline, API, UI, maps, report; models XGBoost + LightGBM (report Sections 3 and 6-8)

## What this part covers

- `./` — `.gitignore`, `README.md`, `requirements.txt`
- `api/` — `__init__.py`, `evaluator.py`, `main.py`
- `logs/` — `api_requests.log`, `api_stdout.log`, `ui_stdout.log`
- `maps/` — `student_test_00000000.html`, `student_test_00000065.html`, `student_test_00000077.html`
- `report/` — `PathSnap_Technical_Report_DRAFT.pdf`
- `report/figs/` — `bayes_opt_history.png`, `cmp_confusion.png`, `cmp_cv_macro_f1.png`, `cmp_logloss.png`, `cmp_overfit.png`, `cmp_test_top1_f1.png`, `feature_ablation.png`, `folium_student_test_00000000.png`, `folium_student_test_00000065.png`, `folium_student_test_00000077.png`, `hmm_comparison.png`, `learning_curves.png`
- `results/` — `api_test_results.csv`
- `scripts/` — `03d_add_xgboost.py`, `06_make_maps.py`, `07_test_api.py`, `08_make_figures.py`, `09_build_report.py`
- `src/pathsnap/` — `pipeline.py`
- `ui/` — `index.html`, `serve.py`
- `ui/vendor/leaflet/images/` — `layers-2x.png`, `layers.png`, `marker-icon-2x.png`, `marker-icon.png`, `marker-shadow.png`
- `ui/vendor/leaflet/` — `leaflet-src.esm.js`, `leaflet-src.js`, `leaflet.css`, `leaflet.js`

## Reproduce this part

```
python scripts/06_make_maps.py student_test/00000000
python scripts/07_test_api.py http://<laptop-1-ip>:8000
python scripts/08_make_figures.py
python scripts/09_build_report.py
```

(Needs the dataset in `data/` or `PATHSNAP_DATA`; see the top-level README.)

## Be ready to explain in the viva

- End-to-end inference path and why it never reads .route (pipeline.py); labels only appear in api/evaluator.py after prediction.
- FastAPI endpoints GET /trips and POST /predict, request logging (api/main.py).
- Two-laptop deployment: API on laptop 1, static Leaflet UI on laptop 2 (ui/), firewall/port 8000, how to run both.
- Folium maps and what each layer shows (06_make_maps.py, maps/).
- How figures and the PDF are generated (08_make_figures.py, 09_build_report.py) and the report limitations / leakage-control statements.
- YOUR MODELS — LightGBM (final model, 300 trees, lr 0.05, 31 leaves; CV macro F1 0.9332, lowest log loss 0.0760; Bayesian-tuned version barely differs) and XGBoost (300 trees, lr 0.05, depth 6, hist; NEW — see below). Know report 3.2.7 and why LightGBM is shipped.
- XGBOOST (DONE) — run with the same protocol as every other model: CV macro F1 0.9336 ± 0.0148, CV Top-1 0.8876, student-test Top-1 0.8996, macro F1 0.9363, log loss 0.0748, route continuity 0.9738, fit 7.9 s, 0.0227 ms/fix (results/model_runs/XGBoost.json, report 3.2.9). It edges LightGBM on CV macro F1 by +0.0004, far inside the fold std of 0.015, so LightGBM stays the shipped model (tuning and HMM parameters were built on it; the decoder was not re-tuned for XGBoost). Be ready to say this honestly at the viva.
- The report is a DRAFT: replace [TEAM TO FILL IN] and the two [SCREENSHOT] placeholders, rewrite analysis in your own words, and finalise the AI-assistance disclosure.

## Note

Parts of this code and the draft report were produced with AI assistance (Claude), as disclosed in the README and report. Re-run this part, read the code, and check every number against `results/` before the viva.
