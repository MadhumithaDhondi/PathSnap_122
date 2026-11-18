"""The candidate models compared in scripts/03_compare_models.py (one per algorithm family)."""
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


def zoo(seed: int):
    sc = lambda: StandardScaler()
    return {
        "LogisticRegression": dict(family="Linear / probabilistic", fit_stride_mult=1,
            est=Pipeline([("sc", sc()), ("m", LogisticRegression(C=1.0, max_iter=500))])),
        "GaussianNB": dict(family="Probabilistic (generative)", fit_stride_mult=1,
            est=Pipeline([("sc", sc()), ("m", GaussianNB())])),
        "KNN": dict(family="Instance-based", fit_stride_mult=3,   # fit on every 9th point: inference cost grows with n
            est=Pipeline([("sc", sc()), ("m", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2))])),
        "DecisionTree": dict(family="Tree-based", fit_stride_mult=1,
            est=DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=seed)),
        "RandomForest": dict(family="Bagging ensemble", fit_stride_mult=1,
            est=RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=20, max_features="sqrt",
                                       max_samples=0.5, n_jobs=2, random_state=seed)),
        "HistGradientBoosting": dict(family="Boosting (sklearn)", fit_stride_mult=1,
            est=HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)),
        "LightGBM": dict(family="Boosting (leaf-wise GBDT)", fit_stride_mult=1,
            est=LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=2, random_state=seed, verbose=-1)),
        "MLP": dict(family="Neural network", fit_stride_mult=1,
            est=Pipeline([("sc", sc()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                                            max_iter=40, early_stopping=True, random_state=seed))])),
    }


def describe(est) -> dict:
    """Key hyper-parameters as a flat dict for the experiment log."""
    m = est.steps[-1][1] if isinstance(est, Pipeline) else est
    return {k: (v if isinstance(v, (int, float, str, bool, type(None), tuple)) else str(v))
            for k, v in m.get_params().items()}
