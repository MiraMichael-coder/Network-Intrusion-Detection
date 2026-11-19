"""Shared settings. Every other file imports from here so we change things in one place."""
from pathlib import Path

# One seed used everywhere (splits, models, SMOTE) so results are repeatable.
SEED = 42

# Project root = the folder that contains src/ (this file is src/config.py).
ROOT = Path(__file__).resolve().parent.parent

# The official UNSW-NB15 files. Put them in data/raw/ (the data/ folder is not in git).
TRAIN_PATH = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"  # 175,341 rows
TEST_PATH = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"    # 82,332 rows


# Text columns that must be one-hot encoded (inside the Pipeline, never before it).
CAT_COLS = ["proto", "service", "state"]
