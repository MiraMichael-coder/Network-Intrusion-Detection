"""Model registry: make_pipeline(name, X) returns an unfitted preprocessor + model Pipeline."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import CAT_COLS, SEED
from src.preprocessing import create_preprocessor

MODELS = {
    "lr": lambda: LogisticRegression(solver="saga", tol=1e-3, max_iter=1000, random_state=SEED),  # tol 1e-3 (default 1e-4): saga + l1 is too slow to converge on 175k rows otherwise
    "dt": lambda: DecisionTreeClassifier(random_state=SEED),
}


def make_pipeline(name, X, scale=None):
    """Step names are "prep" and "model" so grid keys look like model__C."""
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    if scale is None:
        scale = name not in {"dt", "rf"}
    return Pipeline([("prep", create_preprocessor(num_cols, scale=scale)), ("model", MODELS[name]())])
