# Network Intrusion Detection on UNSW-NB15

Binary classification (normal vs attack) with Logistic Regression, SVM,
Decision Tree and Random Forest, using the official UNSW-NB15 training
(175,341 rows) and testing (82,332 rows) CSVs.

## Install

Needs Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data is not in git. Download the two official CSVs and put them in `data/raw/`:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

(The paths are set in `src/config.py`.)

## Run the notebooks

From the project root, with the virtual environment active:

```bash
jupyter notebook
```

Open a notebook from `notebooks/` and run the cells top to bottom.
Notebooks import shared code with `from src.config import SEED, TRAIN_PATH`,
so start Jupyter from the project root (not from inside `notebooks/`).
If that import fails, add this at the top of the first cell:

```python
import sys; sys.path.append("..")
```

## Layout

| Folder | What is in it |
|---|---|
| `data/raw/` | the two official CSVs (not in git) |
| `notebooks/` | exploration, training, final evaluation |
| `src/` | reusable code (config, data loading, preprocessing, models, evaluation) |
| `configs/` | model/tuning settings |
| `results/` | `metrics.csv` and `figures/` |

## Rules we follow

- `id` and `attack_cat` are never used as features.
- All encoding and scaling happens inside a scikit-learn Pipeline.
- SMOTE is used only inside an imbalanced-learn Pipeline.
- Tuning uses `StratifiedKFold(5, shuffle=True, random_state=SEED)` on the training file only.
- The test file is used once, in the final run.
