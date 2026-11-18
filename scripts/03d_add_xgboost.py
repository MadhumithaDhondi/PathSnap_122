"""Train XGBoost with exactly the same protocol as the other models (5-fold group CV + student-test) and merge the
result into results/model_comparison.csv and results/cv_folds.csv, so figures/report can be rebuilt with 9 models.
Usage (needs the dataset and the cache from scripts/02_build_dataset.py):  python scripts/03d_add_xgboost.py
Then: python scripts/08_make_figures.py && python scripts/09_build_report.py"""
import subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT / "scripts" / "03_compare_models.py"), "XGBoost"], check=True)
R = ROOT / "results"
for base in ("model_comparison", "cv_folds"):
    full, part = pd.read_csv(R / f"{base}.csv"), pd.read_csv(R / f"{base}_partial.csv")
    full = full[full["model"] != "XGBoost"]
    pd.concat([full, part], ignore_index=True).to_csv(R / f"{base}.csv", index=False)
    (R / f"{base}_partial.csv").unlink()
print("merged XGBoost into results/model_comparison.csv and results/cv_folds.csv")
