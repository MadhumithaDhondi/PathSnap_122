# Tejaswini — Features & model comparison (report Sections 2.3 and 3)

## What this part covers

- `logs/` — `03_compare_models.log`, `03_test_dt.log`, `03b_feature_ablation.log`, `03c_learning_curves.log`
- `results/` — `confusion_lightgbm_tuned.json`, `confusion_lightgbm_untuned.json`, `cv_folds.csv`, `feature_ablation.csv`, `model_comparison.csv`, `model_selection.json`
- `results/model_runs/` — `DecisionTree.json`, `GaussianNB.json`, `HistGradientBoosting.json`, `KNN.json`, `LightGBM.json`, `LightGBM_tuned_history.json`, `LightGBM_untuned_history.json`, `LogisticRegression.json`, `MLP.json`, `MLP_history.json`, `RandomForest.json`
- `scripts/` — `03_compare_models.py`, `03b_feature_ablation.py`, `03c_learning_curves.py`
- `src/pathsnap/` — `experiment.py`, `features.py`, `modelzoo.py`, `protocol.py`

## Reproduce this part

```
python scripts/03_compare_models.py
python scripts/03b_feature_ablation.py
python scripts/03c_learning_curves.py
```

(Needs the dataset in `data/` or `PATHSNAP_DATA`; see the top-level README.)

## Be ready to explain in the viva

- The 21 features per (fix, arc) pair and the four groups: geometry, motion/heading, temporal context, topology (features.py).
- Feature-group ablation: geometry alone 56.2% Top-1 -> 88.4% with heading (Figure 3).
- Evaluation protocol: 5-fold GroupKFold by trip, every 3rd fix (KNN every 9th), seed 42, selection by CV macro F1 (protocol.py, experiment.py).
- The eight models in modelzoo.py, their hyper-parameters, and the over/under-fitting reading of each (KNN overfits, LogReg/NB underfit).
- Why LightGBM was chosen even though its margin is within fold-to-fold noise.
- Figures 5-10 (CV F1, test Top-1, log loss, over-fit, confusion matrices, learning curves).

## Note

Parts of this code and the draft report were produced with AI assistance (Claude), as disclosed in the README and report. Re-run this part, read the code, and check every number against `results/` before the viva.
