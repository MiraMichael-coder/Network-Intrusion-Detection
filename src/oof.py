"""Out-of-fold probabilities for a tuned model: configs/<model>.json -> results/oof_<model>.csv."""
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import ROOT, SEED
from src.data import load_train
from src.models import make_pipeline
from src.preprocessing import EXCLUDED_COLS


def save_oof(model):
    """Load best params, run 5-fold cross_val_predict(predict_proba), save P(attack) per row."""
    params = json.loads((ROOT / "configs" / f"{model}.json").read_text())
    df = load_train()
    X, y = df.drop(columns=EXCLUDED_COLS), df["label"]
    pipe = make_pipeline(model, X).set_params(**params)
    # same folds as cv_evaluate, so OOF scores line up with the reported CV metrics
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    out = ROOT / "results" / f"oof_{model}.csv"
    pd.DataFrame({"id": df["id"], "label": y, "proba": proba}).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    import sys
    print(save_oof(sys.argv[1]))
