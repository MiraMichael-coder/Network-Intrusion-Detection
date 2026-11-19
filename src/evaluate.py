"""Score one fitted model on one split: metrics row in results/metrics.csv + confusion-matrix PNG."""
import csv
import time
import matplotlib
matplotlib.use("Agg")  # save files only, no window
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             make_scorer, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd

from src.config import ROOT, SEED

RESULTS = ROOT / "results"


def evaluate(name, fitted_model, X, y, split):
    """Evaluate a trained classifier on one dataset split and persist its score summary.

    The function records the model's predictive performance, writes a confusion-matrix
    plot to disk, and appends a single metrics row to the results CSV so that each
    fitted model/split can be compared consistently.
    """
    # Measure how long inference takes for this split so we can compare runtime cost.
    start = time.perf_counter()
    pred = fitted_model.predict(X)
    predict_seconds = time.perf_counter() - start

    # Some scikit-learn models expose class probabilities; others expose a decision
    # score. We need a continuous score to compute ROC and PR AUC.
    if hasattr(fitted_model, "predict_proba"):
        scores = fitted_model.predict_proba(X)[:, 1]
    else:
        scores = fitted_model.decision_function(X)

    # Build the confusion matrix with the binary labels used for attack detection.
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp = cm[0]
    row = {
        "model": name,
        "split": split,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "predict_seconds": predict_seconds,
    }

    # Save a visual summary for manual inspection of classification errors.
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"]).plot(cmap="Blues")
    plt.title(f"{name} ({split})")
    plt.savefig(figures / f"cm_{name}_{split}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Append this row so the experiment log grows as more models are evaluated.
    metrics_path = RESULTS / "metrics.csv"
    new_file = not metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


def _fpr(y, pred):
    """False positive rate: share of normal traffic (label 0) wrongly flagged as attack."""
    # Row 0 of the confusion matrix holds the true negatives and false positives.
    tn, fp = confusion_matrix(y, pred, labels=[0, 1])[0]
    # Guard against a fold with no normal samples (would divide by zero).
    return fp / (fp + tn) if fp + tn else 0.0


def cv_evaluate(name, pipeline, X, y):
    """5-fold stratified CV of an unfitted pipeline; appends mean/std as a split="cv" row."""
    # Metrics to compute on each fold. Strings are built-in sklearn scorers;
    # FPR has no built-in scorer, so we wrap our own function with make_scorer.
    # Keys match the column names evaluate() uses so rows line up in metrics.csv.
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "fpr": make_scorer(_fpr), "roc_auc": "roc_auc",
               "pr_auc": "average_precision"}
    # Stratified folds keep the attack/normal ratio the same in every fold;
    # shuffling with the shared SEED makes the folds repeatable.
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    # Fits a fresh copy of the pipeline on each fold (so preprocessing is learned
    # only from that fold's training part) and scores it on the held-out part.
    # n_jobs=-1 runs the folds in parallel on all CPU cores.
    res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    # Summarise the 5 fold scores: the mean is the estimate, the std shows how stable it is.
    row = {"model": name, "split": "cv"}
    for key in scoring:
        row[key] = res[f"test_{key}"].mean()
        row[f"{key}_std"] = res[f"test_{key}"].std()
    row["fit_seconds"] = res["fit_time"].mean()
    row["fit_seconds_std"] = res["fit_time"].std()

    # cv rows add *_std columns, so rewrite with the union of columns instead of a plain csv append
    metrics_path = RESULTS / "metrics.csv"
    RESULTS.mkdir(exist_ok=True)
    old = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(metrics_path, index=False)
    return row
