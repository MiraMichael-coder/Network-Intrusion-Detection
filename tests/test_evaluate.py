"""Run with: python -m pytest tests  (or: python -m tests.test_evaluate)"""
import csv
import tempfile
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier

from src import evaluate as ev


def test_evaluate():
    # Use a temporary results directory so the real output folder stays untouched.
    ev.RESULTS = Path(tempfile.mkdtemp())  # keep the real results/ clean

    # Generate a small synthetic dataset that matches the binary-classification flow.
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(100, 3)), rng.integers(0, 2, 100)
    model = DummyClassifier(strategy="stratified", random_state=0).fit(X, y)

    # Run the evaluator twice to confirm it appends a new row and still creates artifacts.
    ev.evaluate("dummy", model, X, y, "test")
    row = ev.evaluate("dummy", model, X, y, "test")

    # The summary metrics should stay in a valid range for binary classification.
    assert 0 <= row["accuracy"] <= 1 and 0 <= row["fpr"] <= 1

    # A confusion-matrix plot should be created for each evaluated split.
    assert (ev.RESULTS / "figures" / "cm_dummy_test.png").exists()

    # CSV output should include a header once and two metric rows in total.
    with open(ev.RESULTS / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["model"] == "dummy"  # header written once


if __name__ == "__main__":
    test_evaluate()
    print("ok")
