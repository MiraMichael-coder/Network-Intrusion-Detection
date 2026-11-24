"""Model registry: make_pipeline(name, X) returns an unfitted preprocessor + model Pipeline."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import CAT_COLS, SEED
from src.preprocessing import create_preprocessor

MODELS = {
    "lr": lambda: LogisticRegression(solver="saga", tol=1e-3, max_iter=1000, random_state=SEED),  # tol 1e-3 (default 1e-4): saga + l1 is too slow to converge on 175k rows otherwise
}


def make_pipeline(name, X):
    """Step names are "prep" and "model" so grid keys look like model__C."""
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return Pipeline([("prep", create_preprocessor(num_cols)), ("model", MODELS[name]())])
