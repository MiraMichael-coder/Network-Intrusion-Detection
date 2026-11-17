
"""Functions for loading the UNSW-NB15 datasets."""

import pandas as pd

from src.config import TRAIN_PATH, TEST_PATH


def load_train():
    """Load and return the training dataset."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    return pd.read_csv(TRAIN_PATH)


def load_test():
    """Load and return the testing dataset."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_PATH}"
        )

    return pd.read_csv(TEST_PATH)
