"""Score one fitted model on one split: metrics row in results/metrics.csv + confusion-matrix PNG."""
import csv
import time
import matplotlib
matplotlib.use("Agg")  # save files only, no window
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from src.config import ROOT

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
