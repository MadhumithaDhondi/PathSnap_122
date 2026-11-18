# PathSnap — Map-Matching & Route Reconstruction

Course: Machine Learning, PES University · ML Hackathon

PathSnap turns a noisy GPS trace into the sequence of road arcs a vehicle actually drove. A GPS fix on one street can easily appear on a parallel street, so each fix is snapped to a road arc and the whole route is then reconstructed so that it is consistent with the directed road network.

## Team

| Member | Part | Owns |
|---|---|---|
| Chidvilas | Data, labels and evaluation metrics; models: **DecisionTree, KNN** | `src/pathsnap/{data,geo,network,labels,metrics,config}.py`, EDA, dataset build |
| Tejaswini | Features and model comparison; model: **MLP** | `src/pathsnap/{features,modelzoo,protocol,experiment}.py`, 8-model comparison, ablation, learning curves |
| Yogish | Bayesian tuning and HMM/Viterbi; model: **RandomForest** | `src/pathsnap/hmm.py`, Optuna tuning, decoder tuning, saved models |
| Madhumitha | Inference pipeline, API, UI, maps, report; models: **XGBoost, LightGBM (final)** | `src/pathsnap/pipeline.py`, `api/`, `ui/`, Folium maps, figures, report |

Each member's notes (files, how to re-run, what to be able to explain) are in [`docs/contributions/`](docs/contributions/).

## Approach

```
GPS fix ──► candidate arcs ──► 21 features ──► LightGBM ──► HMM / Viterbi ──► route
            (45 m, ≤12 arcs)   per fix–arc     P(arc is     emissions from     (one arc
                               pair            reference)   P, transitions     per fix)
                                                            from the directed
                                                            road network
```

1. **Candidate generation** – the nearest road arcs within 45 m of each fix (at most 12). The radius widens to 120 m, then 300 m, for fixes with no arc nearby.
2. **Feature engineering** – 21 numeric features per (fix, arc) pair in four groups: geometry, motion/heading, temporal context, topology.
3. **Classifier** – Bayesian-tuned LightGBM outputs the probability that a candidate arc is the reference arc.
4. **HMM + Viterbi** – hidden states are each fix's candidate arcs; emissions are `α · log P_clf`; transitions reward staying on the arc or moving along connected arcs (hop cost), and penalise U-turns and jumps.

Labels for training are derived by a monotone dynamic-programming alignment of each fix to the reference route (3 m skip penalty), so they are an approximation of the true arc.

## Headline results (10 student-test trips; the evaluator partition was never used)

| | Classifier only | Classifier + HMM (tuned) |
|---|---|---|
| Top-1 accuracy | 89.6% | **95.9%** |
| Observation-level macro F1 | 0.8386 | **0.9520** |
| Route continuity | 97.4% | **99.9%** |

Nine models were compared with 5-fold, trip-grouped cross-validation (LogisticRegression, GaussianNB, KNN, DecisionTree, RandomForest, HistGradientBoosting, LightGBM, XGBoost, MLP). LightGBM was selected by the pre-fixed rule (mean CV macro F1 0.9332) and then tuned. XGBoost, added afterwards, scores 0.9336 CV macro F1 and 0.0748 student-test log loss (LightGBM: 0.0760), a difference far below the fold-to-fold standard deviation (0.015), so we kept LightGBM, which the tuning and HMM parameters are built on. Bayesian tuning (Optuna TPE, 36 trials) changed CV macro F1 by only +0.0006, so we do not claim a meaningful improvement from it. The HMM improved all 10 student-test trips (+1.5 to +15.4 points Top-1). Full tables and discussion are in `report/PathSnap_Technical_Report_DRAFT.pdf`.

## Models and ownership

| Model | Family | Owner |
|---|---|---|
| DecisionTree | Tree-based | Chidvilas |
| KNN | Instance-based | Chidvilas |
| MLP | Neural network | Tejaswini |
| RandomForest | Bagging ensemble | Yogish |
| XGBoost | Boosting (level-wise) | Madhumitha |
| LightGBM (Bayesian-tuned) | Boosting (leaf-wise), **final model** | Madhumitha |

LogisticRegression, GaussianNB and HistGradientBoosting are also trained and reported in the comparison table as extra baselines.

## Repository layout

| Folder | Contents |
|---|---|
| `src/pathsnap/` | data loading, geometry, road network and candidate search, label alignment, features, metrics, model zoo, evaluation protocol, HMM/Viterbi, inference pipeline |
| `scripts/` | numbered experiment scripts (run in the order below) |
| `api/` | FastAPI service (`main.py`) and server-side evaluator (`evaluator.py`) |
| `ui/` | evaluator UI (`index.html`, `serve.py`, vendored Leaflet) |
| `models/` | saved final pipeline (`pathsnap_final.joblib`) plus tuned and untuned LightGBM |
| `results/` | every metric table, CV folds, Bayesian-optimisation trials, HMM tuning path, EDA tables and figures |
| `maps/` | interactive Folium HTML maps for three student-test trips |
| `report/` | report figures and the draft technical report PDF |
| `logs/` | run logs of the scripts and the API |
| `docs/contributions/` | per-member notes |

## Setup

```
pip install -r requirements.txt
```

The dataset is **not** committed. Put it in `data/` (it must contain `train/` and `student_test/`), or point to it with `PATHSNAP_DATA=/path/to/folder`. Each trip folder holds four header-less files: `.track` (lon, lat, timestamp), `.nodes`, `.arcs` (directed) and `.route` (reference arc ids).

## Reproduce the experiments (about 40 minutes on 2 CPU cores)

```
python scripts/02_build_dataset.py      # label alignment, candidates, features (cached in cache/)
python scripts/01_eda.py                # dataset inventory, data-quality tables, figures
python scripts/03_compare_models.py     # 8 models, group-CV + student-test
python scripts/03b_feature_ablation.py  # feature-group ablation
python scripts/03d_add_xgboost.py       # XGBoost, same protocol, merged into model_comparison.csv (already run; results committed)
python scripts/04_bayes_opt.py          # Optuna TPE on LightGBM, tuned vs untuned
python scripts/03c_learning_curves.py   # train/validation loss curves
python scripts/05_hmm_decode.py         # OOF probabilities, decoder tuning, HMM vs classifier, saves models/pathsnap_final.joblib
python scripts/06_make_maps.py student_test/00000000   # Folium maps -> maps/
python scripts/08_make_figures.py && python scripts/09_build_report.py   # figures + report PDF
```

Seeds are fixed (42). Configuration lives in `src/pathsnap/config.py` and `src/pathsnap/modelzoo.py`.

## Run the live system (two laptops)

**Laptop 1 – API.** Needs the 10 evaluation trips in a folder laid out like `student_test/`.

```
# Windows:      set PATHSNAP_EVAL_DATA=<folder with the 10 trips>
# Linux/macOS:  export PATHSNAP_EVAL_DATA=<folder with the 10 trips>
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Allow port 8000 through the firewall. Every request is printed to the console and written to `logs/api_requests.log`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /trips` | ids of the evaluation trips |
| `POST /predict` | body `{"trip_id": ...}`; runs candidates → features → LightGBM → HMM and returns GPS points, predicted routes and metrics |

**Laptop 2 – UI.**

```
python ui/serve.py 8080
```

Open http://localhost:8080, enter `http://<laptop-1-ip>:8000`, press *Load evaluation trips*, then click a trip. Map tiles need internet; GPS points and routes are drawn without it.

**Check the API:** `python scripts/07_test_api.py http://<laptop-1-ip>:8000`

## Integrity notes

- The prediction path (`src/pathsnap/pipeline.py`, `features.py`) never reads `.route`. Reference labels are loaded only in `api/evaluator.py` after the prediction exists and are never returned to the UI.
- Splits are by trip. Evaluator data is never used for training, tuning, or model or threshold choice. No predictions are hard-coded.
- Scaling is fitted on training data only. The label-QC rule and model-selection rule were fixed before looking at student-test results.
- Two training trips (00000033 and 00000054) are excluded from fitting because their reference route does not follow the GPS track (rule in `config.py`, documented in the report).

## Limitations

1. Per-fix labels come from our own alignment, and the reference route itself may be imperfect.
2. Candidate Miss Rate is measured against these derived labels, so it is optimistic.
3. Models are fitted on every 3rd fix (KNN on every 9th) for speed.
4. Training uses 78 trips from many places; generalisation to the evaluator trips is untested.
5. The student-test set has only 10 trips, so differences of about one point between models are within noise.
6. HMM parameters were tuned on out-of-fold training probabilities.
7. Route continuity uses a 3-hop tolerance, which is our own definition.
8. The HMM lowered Top-1 on 5 of the 78 training trips (worst: trip 57); the cause has not been diagnosed.

## AI-assistance disclosure

Parts of the code and the draft report were written with AI assistance (Claude). All reported numbers come from running the submitted code. *(Edit this paragraph to state accurately what the team did before submission.)*
